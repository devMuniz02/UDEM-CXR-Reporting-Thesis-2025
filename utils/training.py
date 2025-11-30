# train_loop.py (resumable + seeded best_metric + cumulative tokens_cum + TB local→GCS sync)

from __future__ import annotations
import os
import math
import time
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional, Literal, Dict, Any, Tuple
from tqdm import tqdm
import torch
from torch import device, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# Checkpoint utilities (enhanced + global_step + tokens_cum)
# ============================================================
import os
from typing import Tuple, Dict
import torch
import torch.nn as nn
import fsspec
from fsspec.core import url_to_fs


def _is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")


def _fs_exists(path: str) -> bool:
    fs, p = url_to_fs(path)
    try:
        return fs.exists(p)
    except FileNotFoundError:
        return False


def _ensure_parent_dir(path: str) -> None:
    # Only make local directories; GCS has no concept of local dirs
    if not _is_gcs(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: str,
    *,
    epoch: int | None = None,
    global_step: int | None = None,
    tokens_cum: int | None = None,           # persist cumulative tokens
    best_metric: float | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra: dict | None = None,
) -> None:
    """
    Saves model/optimizer states to local disk or GCS (gs://).
    Keeps: 'model_state_dict', 'optimizer_state_dict'
    Adds: epoch/global_step/tokens_cum/best_metric/scheduler/scaler/extra
    """
    _ensure_parent_dir(save_path)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "tokens_cum": tokens_cum,
        "best_metric": best_metric,  # ES-internal scale (loss for "min", negated metric for "max")
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": (scaler.state_dict() if (scaler is not None and getattr(scaler, "is_enabled", lambda: True)()) else None),
        "checkpoint_version": 3,
    }
    if extra:
        checkpoint["extra"] = extra

    # Use fsspec so this works for both local files and gs:// URIs
    with fsspec.open(save_path, mode="wb") as f:
        torch.save(checkpoint, f)

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
) -> Tuple[nn.Module, torch.optim.Optimizer, Dict]:
    """
    Loads weights/optimizer and, if present, scheduler/scaler metadata from local disk or GCS.
    Returns (model, optimizer, meta_dict) with 'epoch', 'global_step', 'tokens_cum' if present.
    """
    if not _fs_exists(load_path):
        print(f"No checkpoint found at {load_path}")
        model.to(device)
        return model, optimizer, {}

    # Read via fsspec so it works for both backends
    with fsspec.open(load_path, mode="rb") as f:
        ckpt = torch.load(f, map_location="cpu")

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
        "tokens_cum": ckpt.get("tokens_cum", 0),
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


def run_train_batch(model, images, findings, optimizer, scheduler, scaler, autocast, use_amp):
    """
    Runs a single training step: forward, backward, optimizer step, scheduler step.
    Returns loss, learning rate, and number of non-pad tokens in targets.
    """
    tok = model.tokenizer(
        findings,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    tgt_ids = tok["input_ids"].to(model.device, non_blocking=True)
    images = images.to(model.device, non_blocking=True)

    with autocast:
        output = model(pixel_values=images, tgt_ids=tgt_ids)
        loss = output.loss

    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau expects a metric
            scheduler.step(loss.item())
        else:
            scheduler.step()

    return loss.item(), _get_lr(optimizer), _count_nonpad_tokens(tgt_ids, model.pad_token_id)

# def run_train_batch(model, images, findings, scaler, autocast, use_amp, accumulation_steps):
#     """
#     Runs a single forward and backward pass for a training step, 
#     with loss scaled for gradient accumulation.
#     Returns scaled loss and number of non-pad tokens in targets.
#     """
#     tok = model.tokenizer(
#         findings,
#         padding=True,
#         truncation=True,
#         return_tensors="pt"
#     )
#     tgt_ids = tok["input_ids"].to(model.device, non_blocking=True)
#     images = images.to(model.device, non_blocking=True)

#     with autocast:
#         output = model(pixel_values=images, tgt_ids=tgt_ids)
#         # 🔥 CRITICAL FIX: Scale the loss for accumulation
#         loss = output.loss / accumulation_steps 

#     if use_amp:
#         scaler.scale(loss).backward()
#     else:
#         loss.backward()

#     # NOTE: optimizer.step() and optimizer.zero_grad() are moved to run_epoch
    
#     # Return the scaled loss, not the raw one
#     return loss.item(), _count_nonpad_tokens(tgt_ids, model.pad_token_id)

@torch.inference_mode()  # or @torch.no_grad()
def run_valid_batch(model, images, findings, autocast):
    """
    Runs a single validation step: forward only.
    Returns loss and number of non-pad tokens in targets.
    """
    tok = model.tokenizer(
        findings,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    tgt_ids = tok["input_ids"].to(model.device, non_blocking=True)
    images = images.to(model.device, non_blocking=True)

    with autocast:
        output = model(pixel_values=images, tgt_ids=tgt_ids)
        loss = output.loss

    return loss.item(), _count_nonpad_tokens(tgt_ids, model.pad_token_id)

def run_epoch( 
        model,
        epoch, 
        train_loader,
        valid_loader,
        do_validate,
        optimizer, 
        scheduler,
        scheduler_step_on: Literal["epoch", "step", "val_metric"],
        scaler,
        autocast, 
        use_amp, 
        writer, 
        start_step,
        cumulative_tokens
    ):
    """
    Runs a single epoch over the training data.
    """
    total_loss = 0.0
    total_tokens = cumulative_tokens
    steps_taken = start_step
    model.train()
    batches_per_epoch = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training", unit="batch"):
        if batches_per_epoch > 100:
            break
        batches_per_epoch += 1
        images, findings, *_ = batch
        loss, lr, tokens = run_train_batch(model, images, findings, optimizer, scheduler if scheduler_step_on == "step" else None, scaler, autocast, use_amp)
        total_loss += loss
        total_tokens += tokens
        writer.add_scalar("train/loss_step", loss, steps_taken)
        writer.add_scalar("train/lr", lr, steps_taken)
        writer.add_scalar("train/tokens_cum", total_tokens, steps_taken)
        steps_taken += 1

    avg_loss = total_loss / 100 #len(train_loader)
    avg_lr = _get_lr(optimizer)
    if scheduler is not None and scheduler_step_on == "epoch":
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau expects a metric
            scheduler.step(avg_loss)
        else:
            scheduler.step()
    writer.add_scalar("train/loss_epoch", avg_loss, epoch)
    writer.add_scalar("train/lr_epoch", avg_lr, epoch)
    writer.add_scalar("train/tokens_cum_epoch", total_tokens, epoch)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {avg_lr:.4f} | Tokens: {total_tokens}")

    if do_validate and valid_loader is not None:
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        batches_per_epoch = 0
        for batch in tqdm(valid_loader, desc=f"Epoch {epoch} Validation", unit="batch"):
            if batches_per_epoch > 100:
                break
            batches_per_epoch += 1
            images, findings, *_ = batch
            loss, tokens = run_valid_batch(model, images, findings, autocast)
            val_loss += loss
            val_tokens += tokens

        avg_val_loss = val_loss / 100 #len(valid_loader)
        
        # Scheduler on validation metric (ReduceLROnPlateau expects a metric)
        # Only step on val_metric when validation actually ran this epoch
        if scheduler is not None and scheduler_step_on == "val_metric" and do_validate:
            try:
                # Prefer passing the validation metric for ReduceLROnPlateau-like schedulers
                if hasattr(torch.optim.lr_scheduler, "ReduceLROnPlateau") and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    # Try calling with the metric; if scheduler doesn't accept it, fall back to stepping without args
                    try:
                        scheduler.step(avg_val_loss)
                    except TypeError:
                        scheduler.step()
            except Exception as e:
                print(f"[WARN] Scheduler step on val_metric failed: {e}")
        writer.add_scalar("valid/loss_epoch", avg_val_loss, epoch)
        writer.add_scalar("valid/tokens_epoch", val_tokens, epoch)
        print(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f} | Validation Tokens: {val_tokens}")

    return total_tokens, steps_taken, avg_val_loss

# def run_epoch( 
#         model,
#         epoch, 
#         train_loader,
#         valid_loader,
#         do_validate,
#         optimizer, 
#         scheduler,
#         scheduler_step_on: Literal["epoch", "step", "val_metric"],
#         scaler,
#         autocast, 
#         use_amp, 
#         writer, 
#         start_step,
#         cumulative_tokens,
#         accumulation_steps: int = 64 # 💡 NEW ARGUMENT
#     ):
#     """
#     Runs a single epoch over the training data with gradient accumulation.
#     """
#     total_loss_raw = 0.0 # Will store unscaled loss for logging
#     total_tokens = cumulative_tokens
#     steps_taken = start_step
#     model.train()
    
#     # The actual number of optimization steps performed
#     opt_steps_taken = start_step // accumulation_steps 

#     # Zero gradients at the start of the epoch for safety
#     optimizer.zero_grad(set_to_none=True) 

#     for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training", unit="batch")):
#         images, findings, *_ = batch
        
#         # NOTE: run_train_batch now returns SCALED loss
#         scaled_loss, tokens = run_train_batch(model, images, findings, scaler, autocast, use_amp, accumulation_steps)
        
#         # Calculate the raw loss for logging purposes
#         loss_raw = scaled_loss * accumulation_steps
        
#         total_loss_raw += loss_raw
#         total_tokens += tokens
        
#         # --- Optimization Checkpoint ---
#         # Perform optimization step only after accumulation_steps have passed
#         if (step + 1) % accumulation_steps == 0:
#             # Get current LR before step
#             lr = _get_lr(optimizer) 

#             # Step 1: Perform weight update (and unscale/update scaler if using AMP)
#             if use_amp:
#                 scaler.unscale_(optimizer) # Unscale before clipping/stepping if you had clipping
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 optimizer.step()
                
#             # Step 2: Clear accumulated gradients
#             optimizer.zero_grad(set_to_none=True)
            
#             # Step 3: Step the scheduler if set to 'step'
#             if scheduler is not None and scheduler_step_on == "step":
#                  # Use a placeholder metric (like step loss) if ReduceLROnPlateau is used for step-wise update
#                 if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler.step(loss_raw)
#                 else:
#                     scheduler.step()
            
#             # Logging after an optimization step
#             writer.add_scalar("train/loss_step", loss_raw, opt_steps_taken)
#             writer.add_scalar("train/lr", lr, opt_steps_taken)
#             writer.add_scalar("train/tokens_cum", total_tokens, opt_steps_taken)
#             opt_steps_taken += 1
            
#         # Handle the case where the loop finishes mid-accumulation
#         elif step == len(train_loader) - 1:
#             # Perform final step and clear gradients
#             if use_amp:
#                 scaler.unscale_(optimizer)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 optimizer.step()
#             optimizer.zero_grad(set_to_none=True)
            
#             # Logging for the final partial step (optional, but good practice)
#             lr = _get_lr(optimizer)
#             writer.add_scalar("train/loss_step", loss_raw, opt_steps_taken)
#             writer.add_scalar("train/lr", lr, opt_steps_taken)
#             writer.add_scalar("train/tokens_cum", total_tokens, opt_steps_taken)
#             opt_steps_taken += 1


#     # --- End of Epoch Logging and Scheduling ---
#     avg_loss = total_loss_raw / len(train_loader) # Use total batches for avg loss
#     avg_lr = _get_lr(optimizer)
    
#     # Scheduler logic for 'epoch' and 'val_metric' steps (remains similar)
#     if scheduler is not None and scheduler_step_on == "epoch":
#         if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#             scheduler.step(avg_loss)
#         else:
#             scheduler.step()
            
#     writer.add_scalar("train/loss_epoch", avg_loss, epoch)
#     writer.add_scalar("train/lr_epoch", avg_lr, epoch)
#     writer.add_scalar("train/tokens_cum_epoch", total_tokens, epoch)
#     print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {avg_lr:.4f} | Tokens: {total_tokens}")

#     # ... Validation logic (unchanged) ...
#     # (The existing validation logic below is complex due to the scheduler handling, 
#     # but the core loss calculation for validation does not change.)

#     if do_validate and valid_loader is not None:
#         model.eval()
#         val_loss = 0.0
#         val_tokens = 0
#         batches_per_epoch = 0
#         for batch in tqdm(valid_loader, desc=f"Epoch {epoch} Validation", unit="batch"):
#             batches_per_epoch += 1
#             images, findings, *_ = batch
#             loss, tokens = run_valid_batch(model, images, findings, autocast)
#             val_loss += loss
#             val_tokens += tokens

#         avg_val_loss = val_loss / len(valid_loader)
        
#         # Scheduler on validation metric (ReduceLROnPlateau expects a metric)
#         if scheduler is not None and scheduler_step_on == "val_metric" and do_validate:
#             try:
#                 if hasattr(torch.optim.lr_scheduler, "ReduceLROnPlateau") and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler.step(avg_val_loss)
#                 else:
#                     try:
#                         scheduler.step(avg_val_loss)
#                     except TypeError:
#                         scheduler.step()
#             except Exception as e:
#                 print(f"[WARN] Scheduler step on val_metric failed: {e}")
#         writer.add_scalar("valid/loss_epoch", avg_val_loss, epoch)
#         writer.add_scalar("valid/tokens_epoch", val_tokens, epoch)
#         print(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f} | Validation Tokens: {val_tokens}")

#     return total_tokens, steps_taken, avg_val_loss

def _sync_tb_to_gcs(src: str, dst: str) -> None:
    """Sync TensorBoard logs to GCS using gsutil rsync."""
    if not dst:
        return
    try:
        # -m parallel; -r recursive; we do NOT pass -d (no deletions in GCS)
        subprocess.run(["gsutil", "-m", "rsync", "-r", src, dst], check=True)
        print(f"[TB] Synced logs → {dst}")
    except Exception as e:
        print(f"[WARN] TensorBoard rsync to {dst} failed: {e}")

def new_train(
        start_epoch: int,
        total_epochs: int,
        start_global_step: int,
        start_tokens_cum: int,
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        scheduler_step_on: Literal["epoch", "step", "val_metric"],
        scaler,
        autocast: Optional[torch.cuda.amp.autocast] = None,
        use_amp: Optional[bool] = None,
        device: torch.device = torch.device("cpu"),
        log_dir: str = "",
        checkpoint_path: str = "",
        validate_every: int = 1,
        ckpt_every: int = 1,
        # Early stopping
        early_stopping: Optional[EarlyStopping] = None,
        ):
    
    if start_epoch is None:
        start_epoch = 0
    if start_epoch >= total_epochs:
        print("Training already completed.")
        return
    
    if start_global_step is None:
        start_global_step = 0
    if start_tokens_cum is None:
        start_tokens_cum = 0

    if use_amp is None:
        use_amp = (device.type == "cuda")
    if autocast is None:
        autocast = torch.cuda.amp.autocast(device.type, enabled=use_amp, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

    if _is_gcs(log_dir):
        # Stable local directory per run (based on the GCS path)
        safe = log_dir.replace("gs://", "").replace("/", "_")
        local_log_dir = os.path.join(tempfile.gettempdir(), "tb_runs", safe)
        os.makedirs(local_log_dir, exist_ok=True)
        gcs_dst = log_dir
        print(f"[TB] Using local log dir: {local_log_dir} (will sync → {gcs_dst})")
    else:
        local_log_dir = log_dir
        gcs_dst = ""

    writer = SummaryWriter(log_dir=local_log_dir, purge_step=start_global_step)
    
    for epoch in range(start_epoch + 1, total_epochs + 1):
        # Determine if we should validate this epoch
        do_validate = (valid_loader is not None) and (epoch % validate_every == 0)
        # Run epoch
        cumulative_tokens, steps_taken, avg_val_loss = run_epoch(
            model,
            epoch,
            train_loader,
            valid_loader,
            do_validate,
            optimizer,
            scheduler,
            scheduler_step_on=scheduler_step_on,
            scaler=scaler,
            autocast=autocast,
            use_amp=use_amp,
            writer=writer,
            start_step=start_global_step,
            cumulative_tokens=start_tokens_cum
        )
        # Update starting points for next epoch
        start_global_step = steps_taken
        start_tokens_cum = cumulative_tokens

        # Early stopping check
        if early_stopping is not None and do_validate:
            metric_for_es = avg_val_loss if early_stopping.cfg.mode == "min" else -avg_val_loss
            improved = early_stopping.step(metric_for_es)
            if improved:
                best_metric = metric_for_es
                best_epoch = epoch
                print("New best model found during early stopping.")
                save_checkpoint(
                    model,
                    optimizer,
                    early_stopping.cfg.best_ckpt_path,
                    epoch=epoch,
                    global_step=steps_taken,
                    tokens_cum=cumulative_tokens,
                    best_metric=metric_for_es,
                    scheduler=scheduler
                )
            if early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} with val loss: {best_metric}.")
                break
        
        # Periodic checkpointing
        if ckpt_every and (epoch % ckpt_every == 0):
            print("Saving periodic checkpoint.")
            save_checkpoint(
                model,
                optimizer,
                checkpoint_path,
                epoch=epoch,
                global_step=steps_taken,
                tokens_cum=cumulative_tokens,
                scheduler=scheduler
            )

        writer.flush()
        if gcs_dst:
            _sync_tb_to_gcs(local_log_dir, gcs_dst)
    writer.flush()
    writer.close()

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
    validate_every: int = 1,
    ckpt_every: int = 5,
    use_amp: Optional[bool] = None,
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

    if resume_from is not None and _fs_exists(resume_from):
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
        start_epoch = int(meta.get("epoch") or 0)
        start_global_step = int(meta.get("global_step") or 0)
        start_tokens_cum = int(meta.get("tokens_cum") or 0)
        if meta.get("best_metric") is not None:
            best_metric = meta["best_metric"]
            if early_stopping is not None:
                early_stopping.best = best_metric  # ✅ Seed ES best for correct comparisons
    else:
        start_epoch = 0 if start_epoch is None else start_epoch
        start_global_step = 0 if start_global_step is None else start_global_step
        start_tokens_cum = 0 if start_tokens_cum is None else start_tokens_cum
        print("🚀 Starting training from scratch.")

    new_train(start_epoch,
        total_epochs=epochs,
        start_global_step=start_global_step,
        start_tokens_cum=start_tokens_cum,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_on=scheduler_step_on,
        scaler=scaler,
        autocast=autocast,
        use_amp=use_amp,
        device=device,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path,
        validate_every=validate_every,
        ckpt_every=ckpt_every,
        early_stopping=early_stopping,
        )
    print("🎉 Training complete.")
