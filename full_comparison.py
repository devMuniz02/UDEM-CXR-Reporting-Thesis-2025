import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
from itertools import islice
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.text_metrics import evaluate_all_metrics, save_metrics_to_json
from utils.train_comparison import *
from utils.processing import image_transform
from utils.models.gpt_models import DinoGPTCaptioner, DinoGPT2Captioner
from utils.models.lstm_models import DinoLSTMAttnCaptioner, DinoBiLSTMAttnCaptioner
from utils.data.chexpert_dataset import CheXpertDataset
from utils.data.padchest_dataset import PadChestGRDataset
from utils.data.dataloaders import create_dataloaders
from utils.models.complete_model import create_complete_model
from utils.training import train

# CheXpert
CHEXPERT_DIR = "Datasets/CheXpertPlus"
chexpert_paths = {
    "chexpert_data_path": f"{CHEXPERT_DIR}/PNG",  # base PNG folder
    "chexpert_data_csv": f"{CHEXPERT_DIR}/df_chexpert_plus_240401_findings.csv",
}

# MIMIC
MIMIC_DIR = "Datasets/MIMIC"
mimic_paths = {
    "mimic_data_path": MIMIC_DIR,
    "mimic_splits_csv": f"{MIMIC_DIR}/mimic-cxr-2.0.0-split.csv.gz",
    "mimic_metadata_csv": f"{MIMIC_DIR}/mimic-cxr-2.0.0-metadata-findings-only.csv",
    "mimic_reports_path": f"{MIMIC_DIR}/cxr-record-list.csv.gz",  # must contain 'path'
    "mimic_images_dir": f"{MIMIC_DIR}/matched_images_and_masks_mimic_224/images",
}

def train_old(model, train_loader, valid_loader, optimizer, device, pad_id, num_epochs=5, num_batches=100, grad_clip=1.0, model_name="Model", findings_or_impression="findings"):
    writer = SummaryWriter(log_dir=f"lstm-vs-gpt/runs/{model_name}_{findings_or_impression}")
    time_start = time.time()
    best_val_ppl = float("inf")
    for epoch in range(num_epochs):
        sliced_train_loader = islice(train_loader, num_batches)
        sliced_valid_loader = islice(valid_loader, num_batches)
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_metrics = train_one_epoch(
            model, sliced_train_loader, optimizer, device, pad_id, num_batches, grad_clip=grad_clip, loss_fn=sequence_ce_loss
        )
        print(f" Train Loss: {train_metrics['loss']:.4f}, Train PPL: {train_metrics['ppl']:.4f}")
        val_metrics = evaluate(
            model, sliced_valid_loader, device, pad_id, num_batches, loss_fn=sequence_ce_loss
        )
        print(f" Val Loss: {val_metrics['val_loss']:.4f}, Val PPL: {val_metrics['val_ppl']:.4f}")
        if val_metrics['val_ppl'] < best_val_ppl:
            best_val_ppl = val_metrics['val_ppl']
            torch.save(model.state_dict(), f"lstm-vs-gpt/models/best_model_{model_name}_{findings_or_impression}.pth")
            print("  Saved Best Model")
        writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
        writer.add_scalar("Loss/Validation", val_metrics['val_loss'], epoch)
    time_end = time.time()
    total_time = time_end - time_start
    print(f"Training completed in {total_time/60:.2f} minutes.")

    return total_time

def train_new(model, train_loader, valid_loader, optimizer, device, pad_id, num_epochs=5, num_batches=100, grad_clip=1.0, model_name="ModGPT2", findings_or_impression="findings"):
    time_start = time.time()
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        epochs=num_epochs,              # total target; not "remaining"
        device=device,
        log_dir=f"lstm-vs-gpt/runs/{model_name}_{findings_or_impression}",       # SAME dir to keep appending
        checkpoint_path=f"lstm-vs-gpt/models/best_model_{model_name}_{findings_or_impression}.pth",
        validate_every=1,
        ckpt_every=2,
        scheduler=None,
        scheduler_step_on="step",
        early_stopping=None,
        resume_from=None,  # or model_best.pth if you prefer to start from best weights
    )

    time_end = time.time()
    total_time = time_end - time_start
    print(f"Training completed in {total_time/60:.2f} minutes.")

    return total_time

def test_old():
    generated_text = []
    target_text = []
    iteration = 0

    with torch.no_grad():
        for pixel_values, ids_loader, paths, raw_labels in tqdm(test_loader):
            iteration += 1
            pixel_values = pixel_values.to(device)

            info = model.generate_with_logging(
                pixel_values=pixel_values,             # [B, C, H, W]
                bos_id=tokenizer.bos_token_id,
                eos_id=tokenizer.eos_token_id,
                tokenizer=tokenizer,
                preset="safe_sample",
                stop_sequences=None, #["\n\n", "Impression:"],
                max_new_tokens=128,
            )
            generated_text.extend([s["text"]["generated"] for s in info["per_sample"]])
            target_text.extend(raw_labels)
    return generated_text, target_text

def test_new(model, test_loader, pad_id, eos_id):
    generated_text, target_text = [], []
    iteration = 0

    with torch.inference_mode():
        for pixel_values, ids_loader, paths, raw_labels in tqdm(test_loader):
            iteration += 1
            
            pixel_values = pixel_values.to(model.device, non_blocking=True)

            # Visual path
            patches = model.encoder(pixel_values)                           # [B,Np,Cenc]
            projected_patches = model.linear_projection(patches)            # [B,Np,n_embd]

            # Segmentation path per layer
            segmented_layers = model.segmenter(pixel_values, model.num_layers) # [B,n_layers,H,W] (per current decoder)


            # Generate (disable all plotting/diagnostics for speed)
            gen_ids = model.decoder.generate(
                inputs_embeds=projected_patches,
                max_new_tokens=128,
                do_sample=False,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                repetition_penalty=1.2,
                num_beams=1,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                use_cache=True,
                segmentation_mask=segmented_layers,
                prefix_allowed_length=0,
                plot_attention_mask=False,
                plot_attention_mask_layer=[],
                plot_attention_map=False,
                plot_attention_map_layer=[],
                plot_attention_map_generation=0,
            )
            # Move only the ids needed for decoding to CPU
            texts = model.tokenizer.batch_decode(gen_ids.detach().cpu(), skip_special_tokens=True)

            # Accumulate for final metric pass (metrics often run on CPU/strings anyway)
            generated_text.extend(texts)
            target_text.extend(ids_loader)
    return generated_text, target_text

def save_and_evaluate_test(generated_text, target_text, training_time, epochs, key, use_mimic=True):
    if use_mimic:
        dataset_test = "MIMIC"
    else:
        dataset_test = "Chexpert"
    eval_results = evaluate_all_metrics(generated_text, target_text, evaluation_mode="CheXagent")
    for metric, scores in eval_results.items():
        print(f"{metric}: {scores}")
    eval_results["training_time_seconds"] = training_time
    save_metrics_to_json(eval_results, f"lstm-vs-gpt/results_complete/{key}_{epochs}_{dataset_test}.json")

def save_model(model, path):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs("lstm-vs-gpt/models", exist_ok=True)
    os.makedirs("lstm-vs-gpt/runs", exist_ok=True)
    os.makedirs("lstm-vs-gpt/results_complete", exist_ok=True)
    NUM_BATCH = 4
    EPOCHS = 20

    SEGMENTER_MODEL_PATH_LUNG = f"models/dino_unet_decoder_finetuned.pth"
    SEGMENTER_MODEL_PATH_HEART = f"models/dino_unet_organos_best.pth"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    models = {
        "LSTM": DinoLSTMAttnCaptioner(
                vocab_size=tokenizer.vocab_size,
                d_img=384,
                d_h=512,
                pad_id=pad_id,
                dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
                freeze_dino=True,
            ).to(device),
        "BiLSTM": DinoBiLSTMAttnCaptioner(
                vocab_size=tokenizer.vocab_size,
                d_img=384,
                d_h=512,
                pad_id=pad_id,
                dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
                freeze_dino=True,
            ).to(device),
        "GPT": DinoGPTCaptioner(
                vocab_size=tokenizer.vocab_size,
                d_img=384,
                pad_id=pad_id,
                d_model=768,
                n_layer=12,
                n_head=12,
                n_prefix=1024,           # number of visual prefix tokens
                max_seq_len=512,
                dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
                freeze_dino=True,
            ).to(device),
        "GPT2": DinoGPT2Captioner(
                d_img=384,
                num_prefix_tokens=512,
                gpt2_name="gpt2",
                dino_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
                freeze_dino=True
            ).to(device),
        "ModGPT2": create_complete_model(device=device, SEGMENTER_MODEL_PATH_LUNG=SEGMENTER_MODEL_PATH_LUNG, SEGMENTER_MODEL_PATH_HEART=SEGMENTER_MODEL_PATH_HEART),
        "ModGPT2HIDDEN": create_complete_model(device=device, SEGMENTER_MODEL_PATH_LUNG=SEGMENTER_MODEL_PATH_LUNG, SEGMENTER_MODEL_PATH_HEART=SEGMENTER_MODEL_PATH_HEART, mask_implementation="hidden"),
        "ModGPT2HIDDENDINO": create_complete_model(device=device, SEGMENTER_MODEL_PATH_LUNG=SEGMENTER_MODEL_PATH_LUNG, SEGMENTER_MODEL_PATH_HEART=SEGMENTER_MODEL_PATH_HEART, freeze_encoder=False, mask_implementation="hidden"),
        "ModGPT2DINO": create_complete_model(device=device, SEGMENTER_MODEL_PATH_LUNG=SEGMENTER_MODEL_PATH_LUNG, SEGMENTER_MODEL_PATH_HEART=SEGMENTER_MODEL_PATH_HEART, freeze_encoder=False),
        "ModGPT2NOMASK": create_complete_model(device=device, SEGMENTER_MODEL_PATH_LUNG=SEGMENTER_MODEL_PATH_LUNG, SEGMENTER_MODEL_PATH_HEART=SEGMENTER_MODEL_PATH_HEART, use_segmentation_mask=False),
        "ModGPT2NOMASKDINO": create_complete_model(device=device, SEGMENTER_MODEL_PATH_LUNG=SEGMENTER_MODEL_PATH_LUNG, SEGMENTER_MODEL_PATH_HEART=SEGMENTER_MODEL_PATH_HEART, use_segmentation_mask=False, freeze_encoder=False),
    }
    datasets = ["Chexpert", "MIMIC"]

    for dataset in datasets:
        if dataset == "Chexpert":
            USE_MIMIC = False
            FINDINGS_OR_IMPRESSION = "impression"
        else:
            USE_MIMIC = True
            FINDINGS_OR_IMPRESSION = "findings"
        print(f"\n\n=== USING DATASET: {dataset} ===\n\n")
        print(f"Using findings or impression: {FINDINGS_OR_IMPRESSION}")

        train_loader = create_dataloaders(
            chexpert_paths, 
            mimic_paths, 
            batch_size=NUM_BATCH,
            split="train", 
            sampling_ratio=0.7,
            findings_or_impression=FINDINGS_OR_IMPRESSION,
        )

        valid_loader = create_dataloaders(
            chexpert_paths, 
            mimic_paths,
            batch_size=NUM_BATCH,
            split="valid",
            sampling_ratio=0.7,
            findings_or_impression=FINDINGS_OR_IMPRESSION,
        )

        test_loader = create_dataloaders(
            chexpert_paths, 
            mimic_paths, 
            batch_size=NUM_BATCH,
            split="test", 
            sampling_ratio=0.7,
            findings_or_impression=FINDINGS_OR_IMPRESSION,
        )

        for key, model in models.items():
            if os.path.exists(f"lstm-vs-gpt/models/{key}_final_model_{FINDINGS_OR_IMPRESSION}.pth"):
                print(f"Model {key} already trained for {FINDINGS_OR_IMPRESSION}, skipping...")
                if os.path.exists(f"lstm-vs-gpt/results_complete/{key}_{EPOCHS}_{dataset}.json"):
                    print(f"Results for model {key} already exist for {FINDINGS_OR_IMPRESSION}, skipping...")
                    del model
                    torch.cuda.empty_cache()
                    continue
                print(f"Loading model {key} for evaluation...")
                model.load_state_dict(torch.load(f"lstm-vs-gpt/models/{key}_final_model_{FINDINGS_OR_IMPRESSION}.pth"))
                model.to(device)
                if key in ["LSTM", "BiLSTM", "GPT", "GPT2"]:
                    generated_text, target_text = test_old()
                else:
                    generated_text, target_text = test_new(model, test_loader, pad_id, eos_id)
                save_and_evaluate_test(generated_text, target_text, training_time=0, epochs=EPOCHS, key=key, use_mimic=USE_MIMIC)
                del model
                torch.cuda.empty_cache()
                
                continue
            
            print(f"Training model: {key}")
            print(f"Model {key} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2
            )

            if key in ["LSTM", "BiLSTM", "GPT", "GPT2"]:
                time_training = train_old(model, train_loader, valid_loader, optimizer, device, pad_id, num_epochs=EPOCHS, num_batches=100, grad_clip=1.0, model_name=key, findings_or_impression=FINDINGS_OR_IMPRESSION)
                generated_text, target_text = test_old()

            else:
                time_training = train_new(model, train_loader, valid_loader, optimizer, device, pad_id, num_epochs=EPOCHS, num_batches=100, grad_clip=1.0, model_name=key, findings_or_impression=FINDINGS_OR_IMPRESSION)
                generated_text, target_text = test_new(model, test_loader, pad_id, eos_id)

            save_model(model, f"lstm-vs-gpt/models/{key}_final_model_{FINDINGS_OR_IMPRESSION}.pth")
            save_and_evaluate_test(generated_text, target_text, time_training, epochs=EPOCHS, key=key, use_mimic=USE_MIMIC)

            del model
            torch.cuda.empty_cache()

