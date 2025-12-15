#!/usr/bin/env python
# coding: utf-8

import os
# os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HUGGING_FACE_HUB_TOKEN_HERE"  # if needed


from utils.training import train, EarlyStopping, EarlyStoppingConfig
import torch
from utils.models.complete_model import create_complete_model, load_complete_model, save_complete_model
from utils.data.dataloaders import create_dataloaders
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path arguments
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--log_dir", default="lstm-vs-gpt/runs")
    parser.add_argument("--num_trial", default="16")
    parser.add_argument("--best_model_path")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--resume_from")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sampling_ratio", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    # LR scheduler parameters
    parser.add_argument("--scaler_min_lr", type=float, default=1e-7)
    parser.add_argument("--scaler_factor", type=float, default=0.5)
    parser.add_argument("--scaler_patience", type=int, default=10)
    parser.add_argument("--scaler_warmup", type=float, default=0.05)
    parser.add_argument("--scaler_warmup_cosine", type=bool, default=True)

    # Early stopping parameters
    parser.add_argument("--early_stopping_patience", type=int, default=2500000)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    # Validation and checkpointing frequency
    parser.add_argument("--validate_every", type=int, default=1)
    parser.add_argument("--ckpt_every", type=int, default=2)

    # Override sys.argv to avoid Jupyter's internal args
    args = parser.parse_args()
    args.log_dir = f"{args.log_dir}/train{args.num_trial}"
    model_path = f"{args.models_dir}/train{args.num_trial}/checkpoints"
    os.makedirs(model_path, exist_ok=True)
    args.best_model_path = f"{model_path}/model_best.pth"
    args.checkpoint_path = f"{model_path}/model_epoch.pth"
    args.resume_from = f"{model_path}/model_epoch.pth"

    # ALL PATHS
    MODELS_DIR = args.models_dir

    SEGMENTER_MODEL_PATH_LUNG = f"models/dino_unet_decoder_finetuned.pth"
    SEGMENTER_MODEL_PATH_HEART = f"models/dino_unet_organos_best.pth"

    # CheXpert
    CHEXPERT_DIR = "datasets/CheXpertPlus"
    chexpert_paths = {
        "chexpert_data_path": "datasets/CHEXPERT516",  # base PNG folder
        "chexpert_data_csv": f"{CHEXPERT_DIR}/df_chexpert_plus_240401_findings.csv",
    }

    # MIMIC
    MIMIC_DIR = "datasets/MIMIC"
    mimic_paths = {
        "mimic_data_path": MIMIC_DIR,
        "mimic_splits_csv": f"{MIMIC_DIR}/mimic-cxr-2.0.0-split.csv.gz",
        "mimic_metadata_csv": f"{MIMIC_DIR}/mimic-cxr-2.0.0-metadata-findings-only.csv",
        "mimic_reports_path": f"{MIMIC_DIR}/cxr-record-list.csv.gz",  # must contain 'path'
        "mimic_images_dir": "datasets/MIMIC516/datos",
    }

    # Example usage
    try:
        print(os.environ['AIP_TENSORBOARD_LOG_DIR'])
    except KeyError:
        print("AIP_TENSORBOARD_LOG_DIR not found")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = create_complete_model(device=device, SEGMENTER_MODEL_PATH_LUNG=SEGMENTER_MODEL_PATH_LUNG, SEGMENTER_MODEL_PATH_HEART=SEGMENTER_MODEL_PATH_HEART, freeze_encoder=False, mask_implementation="hidden")

    # # Load the model
    # if storage.exists(save_path):
    #     model = load_complete_model(model, save_path, device=device, strict=True)

    # Data loader creation

    kwargs = {
        # "num_workers": os.cpu_count() // 2 if os.cpu_count() else 4,  # adjust on your VM
        # "persistent_workers": True,           # reuses workers between iterations
        # "prefetch_factor": 4,                 # each worker prefetches batches
        # "pin_memory": True,                   # if using CUDA
        # "drop_last": False
    }
    FINDINGS_OR_IMPRESSION = "findings"  # "findings" or "impression"
    train_loader = create_dataloaders(
        chexpert_paths, 
        mimic_paths, 
        batch_size=args.batch_size,
        split="train", 
        sampling_ratio=args.sampling_ratio,
        findings_or_impression=FINDINGS_OR_IMPRESSION,
        **kwargs
    )

    valid_loader = create_dataloaders(
        chexpert_paths,
        mimic_paths,
        batch_size=args.batch_size,
        split="valid",
        sampling_ratio=args.sampling_ratio,
        findings_or_impression=FINDINGS_OR_IMPRESSION,
        **kwargs
    )

    optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
            )
    if args.scaler_warmup_cosine:
        from transformers import get_cosine_schedule_with_warmup
        total_steps = args.epochs * len(train_loader)
        warmup_steps = max(1, int(args.scaler_warmup * total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        print(f"Using ReduceLROnPlateau scheduler, learning rate: {args.lr}, factor: {args.scaler_factor}, patience: {args.scaler_patience}, min_lr: {args.scaler_min_lr}")
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.scaler_factor, patience=args.scaler_patience, min_lr=args.scaler_min_lr)

    early = EarlyStopping(EarlyStoppingConfig(
        patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta, mode="min", restore_best=True,
        best_ckpt_path=args.best_model_path
    ))

    # args.epochs = (args.epochs * len(train_loader)) // 100  # convert to steps
    print(f"Current memory before training: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB") if device == "cuda" else None
    torch.cuda.empty_cache() if device == "cuda" else None
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        epochs=args.epochs,                       # total target; not "remaining"
        device=device,
        log_dir=args.log_dir,       # SAME dir to keep appending
        checkpoint_path=args.checkpoint_path,
        validate_every=args.validate_every,
        ckpt_every=args.ckpt_every,
        scheduler=scheduler,
        scheduler_step_on="step",
        early_stopping=early,
        resume_from=args.resume_from,  # or model_best.pth if you prefer to start from best weights
        # start_epoch=...,                 # optional override
        # start_global_step=...,           # optional override
    )
    save_complete_model(model, args.best_model_path, device=device)
