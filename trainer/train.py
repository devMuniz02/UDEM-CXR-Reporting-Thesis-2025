#!/usr/bin/env python
# coding: utf-8

# Model creation and loading

# In[6]:


import torch
from pathlib import Path
from utils.models.complete_model import create_complete_model, save_complete_model, load_complete_model, save_checkpoint, load_checkpoint


# ALL PATHS
MODELS_DIR = "models/"
SEGMENTER_MODEL_PATH = f"{MODELS_DIR}dino_unet_decoder_finetuned.pth"
save_path = f"{MODELS_DIR}complete_model.pth"
checkpoint_path = f"{MODELS_DIR}model_checkpoint.pth"

# Example usage
device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_complete_model(device=device, SEGMENTER_MODEL_PATH=SEGMENTER_MODEL_PATH)

# Load the model
if Path(save_path).exists():
    model = load_complete_model(model, save_path, device=device, strict=True)


# Data loader creation

# In[ ]:


from utils.data.dataloaders import create_dataloaders

# CheXpert
CHEXPERT_DIR = "Datasets/CheXpertPlus"
chexpert_paths = {
    "chexpert_data_path": f"{CHEXPERT_DIR}/PNG",  # base PNG folder
    "chexpert_data_csv": f"{CHEXPERT_DIR}/df_chexpert_plus_240401.csv",
}

# MIMIC
MIMIC_DIR = "Datasets/MIMIC"
mimic_paths = {
    "mimic_data_path": MIMIC_DIR,
    "mimic_splits_csv": f"{MIMIC_DIR}/mimic-cxr-2.0.0-split.csv.gz",
    "mimic_metadata_csv": f"{MIMIC_DIR}/mimic-cxr-2.0.0-metadata.csv",
    "mimic_reports_path": f"{MIMIC_DIR}/cxr-record-list.csv.gz",  # must contain 'path'
    "mimic_images_dir": f"{MIMIC_DIR}/matched_images_and_masks_mimic_224/images",
}

import os
kwargs = {
    # "num_workers": os.cpu_count() // 2 if os.cpu_count() else 4,  # adjust on your VM
    # "persistent_workers": True,           # reuses workers between iterations
    # "prefetch_factor": 4,                 # each worker prefetches batches
    # "pin_memory": True,                   # if using CUDA
    # "drop_last": False
}

train_loader = create_dataloaders(
    chexpert_paths, 
    mimic_paths, 
    batch_size=4,
    split="train", 
    sampling_ratio=0.7,
    **kwargs
)

valid_loader = create_dataloaders(
    chexpert_paths,
    mimic_paths,
    batch_size=4,
    split="valid",
    sampling_ratio=0.7,
    **kwargs
)

images, findings, image_paths, _ = next(iter(train_loader))
print("Batch image tensor shape:", getattr(images, "shape", "N/A"))
print("Batch findings shape:", getattr(findings, "shape", len(findings)))
print("Batch image paths shape:", getattr(image_paths, "shape", len(image_paths)))


# Training

# In[ ]:


# train_loop.py (resumable + seeded best_metric + cumulative tokens_cum)

from __future__ import annotations
import os
import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Literal, Dict, Any, Tuple
from tqdm import tqdm
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# Checkpoint utilities (enhanced + global_step + tokens_cum)
# ============================================================
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: str,
    *,
    epoch: int | None = None,
    global_step: int | None = None,
    tokens_cum: int | None = None,           # <- NEW: persist cumulative tokens
    best_metric: float | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra: dict | None = None,
) -> None:
    """
    Saves model/optimizer states with your original key names plus metadata.
    Keeps: 'model_state_dict', 'optimizer_state_dict'
    Adds: epoch/global_step/tokens_cum/best_metric/scheduler/scaler/extra
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "tokens_cum": tokens_cum,  # <- NEW
        "best_metric": best_metric,  # ES-internal scale (loss for "min", negated metric for "max")
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": (scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None),
        "checkpoint_version": 3,
    }
    if extra:
        checkpoint["extra"] = extra

    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    load_path: str,
    device: str = "cpu",
    *,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    strict: bool = False,
) -> tuple[nn.Module, torch.optim.Optimizer, dict]:
    """
    Loads weights/optimizer and, if present, scheduler/scaler metadata.
    Returns (model, optimizer, meta_dict) with 'epoch', 'global_step', 'tokens_cum' if present.
    """
    if not os.path.exists(load_path):
        print(f"No checkpoint found at {load_path}")
        model.to(device)
        return model, optimizer, {}

    ckpt = torch.load(load_path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[WARN] Could not load scheduler state: {e}")

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as e:
            print(f"[WARN] Could not load AMP scaler state: {e}")

    model.to(device)

    meta = {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step", 0),
        "tokens_cum": ckpt.get("tokens_cum", 0),  # <- NEW
        "best_metric": ckpt.get("best_metric"),
        "extra": ckpt.get("extra", {}),
        "checkpoint_version": ckpt.get("checkpoint_version", 0),
    }
    print(
        f"Loaded checkpoint from {load_path} "
        f"(epoch={meta['epoch']}, global_step={meta['global_step']}, tokens_cum={meta['tokens_cum']}, best_metric={meta['best_metric']})"
    )
    return model, optimizer, meta


# =======================
# Early Stopping machinery
# =======================
@dataclass
class EarlyStoppingConfig:
    patience: int = 5
    min_delta: float = 0.0
    mode: Literal["min", "max"] = "min"
    restore_best: bool = True
    best_ckpt_path: str = "model_checkpoint_best.pth"


class EarlyStopping:
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best = float("inf") if cfg.mode == "min" else -float("inf")
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_better(self, metric: float) -> bool:
        if self.cfg.mode == "min":
            return (self.best - metric) > self.cfg.min_delta
        else:
            return (metric - self.best) > self.cfg.min_delta

    def step(self, metric: float) -> bool:
        """
        Returns True if improved (and resets patience), else False.
        """
        if self._is_better(metric):
            self.best = metric
            self.num_bad_epochs = 0
            return True
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.cfg.patience:
                self.should_stop = True
            return False


# =========================
# Utility helpers for logging
# =========================
def _count_nonpad_tokens(tgt_ids: torch.Tensor, pad_token_id: int) -> int:
    return int((tgt_ids != pad_token_id).sum().item())


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0))


# ===============
# Main train loop
# ===============
def train(
    model: nn.Module,
    train_loader: Iterable,
    valid_loader: Optional[Iterable],
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    device: str = "cpu",
    log_dir: str = "runs/exp1",
    checkpoint_path: str = "model_checkpoint.pth",
    is_on_cloud: bool = False,
    max_grad_norm: float = 1.0,
    validate_every: int = 1,
    ckpt_every: int = 5,
    use_amp: Optional[bool] = None,
    grad_accum_steps: int = 1,
    # Scheduler
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    scheduler_step_on: Literal["epoch", "step", "val_metric"] = "epoch",
    # Early stopping
    early_stopping: Optional[EarlyStopping] = None,
    # ---- Resume controls ----
    resume_from: str | None = None,      # path to .pth to resume from
    start_epoch: int | None = None,      # overrides checkpoint epoch if provided
    start_global_step: int | None = None, # overrides checkpoint global_step if provided
    start_tokens_cum: int | None = None,  # overrides checkpoint tokens_cum if provided
):
    """
    Full training loop with AMP, grad accumulation, LR scheduler, early stopping, and TensorBoard logging.
    Supports resuming from checkpoint with continued TensorBoard steps and cumulative tokens.

    Expected batch format:
      (images: FloatTensor[B,C,H,W], findings: list[str], image_path, report_path)

    Model interface:
      model.forward(pixel_values, tgt_ids) -> object with `.loss` when labels are provided.
      model.tokenizer must provide `pad_token_id`.
    """
    torch.cuda.empty_cache()
    device = torch.device(device)
    model.to(device)

    # AMP setup
    if use_amp is None:
        use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
    autocast = torch.amp.autocast(device.type, enabled=use_amp, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

    # --------- RESUME STATE ---------
    _epoch0 = 0
    global_step = 0
    tokens_cum = 0  # <- NEW: global cumulative tokens that never reset
    best_metric = float("inf") if (early_stopping and early_stopping.cfg.mode == "min") else -float("inf")

    if resume_from is not None and os.path.exists(resume_from):
        print(f"🔁 Resuming from checkpoint: {resume_from}")
        _, _, meta = load_checkpoint(
            model,
            optimizer,
            resume_from,
            device=str(device),
            scheduler=scheduler,
            scaler=scaler,
            strict=False,
        )
        _epoch0 = int(meta.get("epoch") or 0)
        global_step = int(meta.get("global_step") or 0)
        tokens_cum = int(meta.get("tokens_cum") or 0)  # <- NEW
        if meta.get("best_metric") is not None:
            best_metric = meta["best_metric"]
            if early_stopping is not None:
                early_stopping.best = best_metric  # ✅ Seed ES best for correct comparisons

    # Optional manual overrides
    if start_epoch is not None:
        _epoch0 = int(start_epoch)
    if start_global_step is not None:
        global_step = int(start_global_step)
    if start_tokens_cum is not None:
        tokens_cum = int(start_tokens_cum)

    # TensorBoard: same log_dir → new events append; steps continue from global_step
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(
        f"TensorBoard logging to: {log_dir} "
        f"(starting global_step={global_step}, tokens_cum={tokens_cum}, start_epoch={_epoch0+1})"
    )

    best_epoch = _epoch0

    # ------------------ EPOCHS ------------------
    for epoch in range(_epoch0 + 1, epochs + 1):
        if is_on_cloud:
            print("[CLOUD CHECK] Ensure credits > $10 USD before continuing…")

        # ----------------- TRAIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_start = time.time()
        running_loss = 0.0
        step = 0
        tokens_this_epoch = 0
        wall_start = time.time()

        print(f"\n🟡 Epoch {epoch}/{epochs} — training")
        for batch in tqdm(train_loader):
            if step > 10:
                break
            images, findings, *_ = batch

            # Tokenize target strings
            tok = model.tokenizer(
                findings,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            tgt_ids = tok["input_ids"].to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)

            with autocast:
                out = model(pixel_values=images, tgt_ids=tgt_ids)
                if not hasattr(out, "loss") or out.loss is None:
                    raise RuntimeError("Model forward did not return .loss. Ensure labels are built inside model.forward.")
                loss = out.loss / grad_accum_steps

            # Backward
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update counters
            step += 1
            global_step += 1
            running_loss += loss.item() * grad_accum_steps

            ntoks = _count_nonpad_tokens(tgt_ids, model.pad_token_id)
            tokens_this_epoch += ntoks
            tokens_cum += ntoks  # <- NEW: global cumulative count never resets

            # Optimizer step on accumulation boundary
            if step % grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                total_norm = clip_grad_norm_(model.parameters(), max_grad_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Scheduler stepping per-step
                if scheduler is not None and scheduler_step_on == "step":
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(running_loss / max(1, step))
                    else:
                        scheduler.step()

                # TB per-step
                writer.add_scalar("train/loss_step", running_loss / max(1, step), global_step)
                writer.add_scalar("train/lr", _get_lr(optimizer), global_step)
                writer.add_scalar("train/grad_norm", float(total_norm), global_step)
                writer.add_scalar("train/tokens_cum", tokens_cum, global_step)  # <- NEW: log global cumulative

        # End epoch: aggregate train stats
        train_time = time.time() - epoch_start
        avg_train_loss = running_loss / max(1, step)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")
        toks_per_s = tokens_this_epoch / max(1e-6, (time.time() - wall_start))

        print(
            f"✅ Train — loss: {avg_train_loss:.4f} | ppl≈ {train_ppl:.2f} | "
            f"steps: {step} | tokens_epoch: {tokens_this_epoch} | tokens_cum: {tokens_cum} | "
            f"time: {train_time:.1f}s | toks/s≈ {toks_per_s:.1f}"
        )
        writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
        writer.add_scalar("train/ppl_epoch", train_ppl if math.isfinite(train_ppl) else 0.0, epoch)
        writer.add_scalar("train/tokens_epoch", tokens_this_epoch, epoch)
        writer.add_scalar("train/tokens_cum_epoch_close", tokens_cum, epoch)  # epoch-close snapshot of global tokens
        writer.add_scalar("time/train_epoch_s", train_time, epoch)

        # Scheduler per-epoch (non-plateau)
        if scheduler is not None and scheduler_step_on == "epoch" and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # ----------------- VALID -----------------
        do_validate = (valid_loader is not None) and (epoch % validate_every == 0)
        avg_val_loss = None
        if do_validate:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            val_tokens = 0
            print(f"🔵 Epoch {epoch}/{epochs} — validating")
            with torch.no_grad():
                for batch in tqdm(valid_loader):
                    if val_steps > 10:
                        break
                    v_images, v_findings, *_ = batch

                    tok = model.tokenizer(
                        v_findings,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    v_tgt_ids = tok["input_ids"].to(device, non_blocking=True)
                    v_images = v_images.to(device, non_blocking=True)

                    with autocast:
                        out = model(pixel_values=v_images, tgt_ids=v_tgt_ids)
                        if not hasattr(out, "loss") or out.loss is None:
                            raise RuntimeError("Model forward did not return .loss on validation.")
                        v_loss = out.loss

                    val_loss_sum += float(v_loss.item())
                    val_steps += 1
                    val_tokens += _count_nonpad_tokens(v_tgt_ids, model.pad_token_id)

            avg_val_loss = val_loss_sum / max(1, val_steps)
            val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
            print(f"📘 Valid — loss: {avg_val_loss:.4f} | ppl≈ {val_ppl:.2f} | steps: {val_steps} | tokens_valid: {val_tokens}")

            writer.add_scalar("valid/loss_epoch", avg_val_loss, epoch)
            writer.add_scalar("valid/ppl_epoch", val_ppl if math.isfinite(val_ppl) else 0.0, epoch)
            writer.add_scalar("valid/tokens_epoch", val_tokens, epoch)

            # Scheduler on validation metric (ReduceLROnPlateau)
            if scheduler is not None and scheduler_step_on == "val_metric" and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)

            # Early stopping & best checkpoint
            if early_stopping is not None:
                # ES metric: for "min" we use avg_val_loss; for "max" you could pass a positive metric.
                metric_for_es = avg_val_loss if early_stopping.cfg.mode == "min" else -avg_val_loss
                improved = early_stopping.step(metric_for_es)

                if improved:
                    best_metric = metric_for_es
                    best_epoch = epoch
                    try:
                        print("💾 New best model — saving best checkpoint…")
                        save_checkpoint(
                            model,
                            optimizer,
                            early_stopping.cfg.best_ckpt_path,
                            epoch=epoch,
                            global_step=global_step,
                            tokens_cum=tokens_cum,  # <- NEW
                            best_metric=best_metric,  # internal ES scale
                            scheduler=scheduler,
                            scaler=scaler,
                            extra={
                                "phase": "best",
                                "best_epoch": epoch,
                                "best_val_loss": float(avg_val_loss),  # human-friendly raw loss
                            },
                        )
                    except Exception as e:
                        print(f"[WARN] Could not save best checkpoint: {e}")

                if early_stopping.should_stop:
                    print(f"⛳ Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}.")
                    if early_stopping.cfg.restore_best:
                        try:
                            print("🔁 Restoring best checkpoint weights…")
                            load_checkpoint(
                                model,
                                optimizer,
                                early_stopping.cfg.best_ckpt_path,
                                device=str(device),
                                scheduler=scheduler,
                                scaler=scaler,
                                strict=False,
                            )
                        except Exception as e:
                            print(f"[WARN] Failed to restore best checkpoint: {e}")
                    writer.close()
                    return  # graceful stop

        # Periodic checkpoint
        if ckpt_every and (epoch % ckpt_every == 0):
            try:
                print("💾 Periodic checkpoint — saving…")
                save_checkpoint(
                    model,
                    optimizer,
                    checkpoint_path,
                    epoch=epoch,
                    global_step=global_step,
                    tokens_cum=tokens_cum,  # <- NEW
                    best_metric=best_metric,
                    scheduler=scheduler,
                    scaler=scaler,
                    extra={
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "lr": _get_lr(optimizer),
                    },
                )
            except Exception as e:
                print(f"[WARN] Could not save periodic checkpoint: {e}")

    writer.close()
    print("🎉 Training complete.")


# In[9]:


from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

early = EarlyStopping(EarlyStoppingConfig(
    patience=4, min_delta=1e-4, mode="min", restore_best=True,
    best_ckpt_path="checkpoints/model_best.pth"
))

train(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    epochs=20,                       # total target; not "remaining"
    device=device,
    log_dir="runs/chestx_exp1",       # SAME dir to keep appending
    checkpoint_path="checkpoints/model_epoch.pth",
    validate_every=1,
    ckpt_every=2,
    scheduler=scheduler,
    scheduler_step_on="val_metric",
    early_stopping=early,
    resume_from="checkpoints/model_epoch.pth",  # or model_best.pth if you prefer to start from best weights
    # start_epoch=...,                 # optional override
    # start_global_step=...,           # optional override
)


# Saving model

# In[10]:


# Save the model
save_complete_model(model, save_path, device=device)


# In[ ]:




