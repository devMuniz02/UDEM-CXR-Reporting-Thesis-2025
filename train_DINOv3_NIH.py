import os
import json
import gc
import copy
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.data import (
    load_data, split_data, get_label_columns, build_transforms,
    compute_class_pos_weights, ChestXrayDataset
)
from utils.models import load_base_model, ChestXrayClassifier
from utils.torch_train import train_model, eval_model

# -----------------------------
# Static config (no argparse)
# -----------------------------
CONFIG = {
    "data_dir": r"C:\Users\emman\Desktop\PROYECTOS_VS_CODE\PRUEBAS_DE_PYTHON\CNN_PEF",                 # root containing images and CSV
    "csv_file": "data_clean.csv",
    "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "batch_size": 64,
    "batch_per_epoch_train": 20,      # how many batches to process per epoch (None to use full epoch)
    "batch_per_epoch_val": 10,        # how many batches to process per epoch (None to use full epoch)
    "batch_per_epoch_test": 10,       # how many batches to process per epoch (None to use full epoch)
    "image_size": 516,
    "epochs_per_run": 10,            # how many epochs to train each execution
    "lr": 1e-3,
    "cap_pos_weight": 500.0,         # cap for positive class weights (None to disable)
    "device": "cuda"                 # "cuda" or "cpu"
}

# -----------------------------
# Fixed output/ckpt locations
# -----------------------------
OUTPUT_DIR = "train/NIH/DINOv3_516px"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
LATEST_CKPT = os.path.join(MODELS_DIR, "latest.pt")
LAST_EPOCH_JSON = os.path.join(OUTPUT_DIR, "last_epoch.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

def save_epoch(epoch, path):
    with open(path, "w") as f:
        json.dump({"last_epoch": int(epoch)}, f)

def load_epoch(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            d = json.load(f)
            return int(d.get("last_epoch", 0))
    return 0

def free_gpu_resources(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _to_serializable_labels(labels):
    try:
        return list(map(str, list(labels)))  # handles pd.Index, np arrays, etc.
    except Exception:
        return [str(x) for x in labels]

def save_checkpoint(path, epoch, model, optimizer, scheduler, run_name, label_cols, best_metric=None):
    ckpt = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "run_name": run_name,
        "label_cols": _to_serializable_labels(label_cols),  # <-- sanitize here
        "config_model_id": CONFIG["model_id"],
        "best_metric": best_metric,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    torch.save(ckpt, path)
    save_epoch(epoch, LAST_EPOCH_JSON)


def load_checkpoint_if_any(path, device, model, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        return None
    print(f"[resume] Loading checkpoint from {path}")
    try:
        ckpt = torch.load(path, map_location=device)  # PyTorch 2.6 default: weights_only=True
    except Exception as e:
        print(f"[resume] Safe load failed ({e}). Retrying with weights_only=False")
        ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt

def main():
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    df = load_data(CONFIG["data_dir"], CONFIG["csv_file"])
    train_df, val_df, test_df = split_data(df)
    label_cols = get_label_columns(df)

    pos_w = compute_class_pos_weights(train_df, label_cols, cap=CONFIG["cap_pos_weight"])
    pos_weights = torch.tensor(pos_w, dtype=torch.float32, device=device)

    # Datasets & loaders
    train_tf, test_tf = build_transforms(CONFIG["image_size"])
    train_dataset = ChestXrayDataset(train_df, transform=train_tf,  label_cols=label_cols)
    val_dataset   = ChestXrayDataset(val_df,   transform=test_tf,   label_cols=label_cols)
    test_dataset  = ChestXrayDataset(test_df,  transform=test_tf,   label_cols=label_cols)

    num_workers = 0 if os.name == "nt" else 4
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=num_workers)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=True, num_workers=num_workers)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=num_workers)

    # Model (fresh init)
    base_model = load_base_model(CONFIG["model_id"], device=device)
    model = ChestXrayClassifier(
        base_model=base_model,
        num_classes=len(label_cols),
        freeze_base=True,
        head_dims=(1024, 512),
        activation="gelu",
        dropout=0.2,
        bn=False
    ).to(device)

    # Loss & optim
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    def _to_list_str(x):
        if x is None:
            return []
        # pandas Index / numpy array have .tolist()
        if hasattr(x, "tolist"):
            x = x.tolist()
        # tuples → list
        if not isinstance(x, list):
            x = list(x)
        return [str(v) for v in x]


    # -------- Resume if latest.ckpt exists --------
    start_epoch = 0
    maybe_ckpt = load_checkpoint_if_any(LATEST_CKPT, device, model, optimizer, scheduler)
    if maybe_ckpt is not None:
        start_epoch = int(maybe_ckpt.get("epoch", load_epoch(LAST_EPOCH_JSON)))
        ckpt_labels = maybe_ckpt.get("label_cols")
        ckpt_labels = _to_list_str(ckpt_labels)
        list_labels = _to_list_str(label_cols)
        if ckpt_labels and ckpt_labels != list_labels:
            print("[warn] Label columns differ from checkpoint; continuing anyway.")
        if maybe_ckpt.get("config_model_id") and maybe_ckpt["config_model_id"] != CONFIG["model_id"]:
            print(f"[warn] model_id differs (ckpt={maybe_ckpt['config_model_id']} vs current={CONFIG['model_id']}).")

        print(f"[resume] Continuing from epoch {start_epoch}")

    # TensorBoard
    run_name = f"DINOv3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(RUNS_DIR, "DINOv3"))

    # Keep a copy for AUC-before vs after
    model_before = copy.deepcopy(model)

    # -------- Train --------
    epochs_to_train = CONFIG["epochs_per_run"]
    history = train_model(
        model, device, train_loader, val_loader, criterion, optimizer,
        class_names=label_cols, epochs=epochs_to_train, verbose=2, scheduler=scheduler,
        save_dir=MODELS_DIR, batch_per_epoch_train=CONFIG["batch_per_epoch_train"], batch_per_epoch_val=CONFIG["batch_per_epoch_val"], writer=writer, start_epoch=start_epoch,
    )

    # Save checkpoint (latest)
    end_epoch = start_epoch + epochs_to_train
    save_checkpoint(LATEST_CKPT, end_epoch, model, optimizer, scheduler, run_name, label_cols)

    # Save a frozen copy of weights for archival (optional)
    final_model_path = os.path.join(MODELS_DIR, f"{run_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"[save] Model weights saved to {final_model_path}")
    print(f"[save] Latest checkpoint updated at {LATEST_CKPT} (epoch {end_epoch})")

    free_gpu_resources(model_before)

    print("Evaluating on test set:")
    eval_model(model, test_loader, device, criterion, label_cols, batch_per_epoch=CONFIG["batch_per_epoch_test"], writer=writer, current_epoch=end_epoch)

    writer.close()

if __name__ == "__main__":
    main()
