
"""
PyTorch training utilities for chest X-ray classification.
Includes AUC computation, training/validation loops, and evaluation helpers.
"""

# Standard library imports
from itertools import islice
import os

# Third-party imports
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.nn.functional import sigmoid
from tqdm import tqdm

def compute_auc(y_true, y_pred):
    """
    Compute AUC for binary or multi-label classification.
    Args:
        y_true (array-like): True labels (shape: [num_samples, num_classes] or [num_samples]).
        y_pred (array-like): Predicted scores (same shape as y_true).
    Returns:
        tuple: (average_auc, aucs) where average_auc is the mean AUC and aucs is a list per class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    aucs = []
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        # Binary case
        y_true_flat = y_true.ravel()
        if np.any(y_true_flat == 1) and np.any(y_true_flat == 0):
            fpr, tpr, _ = roc_curve(y_true_flat, y_pred.ravel())
            aucs.append(auc(fpr, tpr))
        else:
            aucs.append(np.nan)
    else:
        # Multiclass/multilabel case
        for i in range(y_true.shape[1]):
            y_true_class = y_true[:, i]
            if np.any(y_true_class == 1) and np.any(y_true_class == 0):
                try:
                    fpr, tpr, _ = roc_curve(y_true_class, y_pred[:, i])
                    aucs.append(auc(fpr, tpr))
                except ValueError:
                    aucs.append(np.nan)
            else:
                aucs.append(np.nan)
    average_auc = np.nanmean(aucs)
    return average_auc, aucs

def epoch_step(model,
               device,
               data_loader,
               criterion,
               usage = "Training",
               batch_per_epoch=None,
               optimizer=None,
               writer=None,
               current_epoch=None,
               class_names=None):
    """
    Run a training or validation step for one epoch.
    Args:
        model (torch.nn.Module): Model to train/evaluate.
        device (torch.device): Device to use.
        data_loader (DataLoader): DataLoader for the data.
        criterion (torch.nn.Module): Loss function.
        usage (str): 'Training', 'Validation', or 'Evaluating'.
        batch_per_epoch (int, optional): Number of batches per epoch.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training.
        writer (SummaryWriter, optional): TensorBoard writer.
        current_epoch (int, optional): Current epoch number.
        class_names (list, optional): List of class names.
    Returns:
        tuple: (mean_loss, mean_auc)
    """
    if usage == "Training":
        model.train()
    else:
        model.eval()
    y_true, y_pred, total_loss = [], [], 0
    if batch_per_epoch:
        data_loader = islice(data_loader, batch_per_epoch)
        total_batches = batch_per_epoch
    else:
        total_batches = len(data_loader)
    with torch.set_grad_enabled(optimizer is not None):
        for (x, y) in tqdm(data_loader, total=total_batches, desc=usage):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append(sigmoid(logits).detach().cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mean_auc, list_auc = compute_auc(y_true, y_pred)
    mean_loss = total_loss / total_batches
    if writer is not None and current_epoch is not None:
        # Section for usage (grouping metrics under usage)
        # AUC section
        writer.add_scalar(f'{usage}/AUC/Mean', mean_auc, current_epoch)
        for i, auc_value in enumerate(list_auc):
            writer.add_scalar(f'{usage}/AUC/{class_names[i]}', auc_value, current_epoch)
        # Loss section
        writer.add_scalar(f'{usage}/Loss/Mean', mean_loss, current_epoch)
    return mean_loss, mean_auc

def train_model(model,
                device,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                class_names,
                epochs=10,
                batch_per_epoch_train=None,
                batch_per_epoch_val=None,
                verbose=2,
                scheduler=None,
                save_dir="models",
                writer=None,
                start_epoch=0):
    """
    Train a model for multiple epochs, saving the best model by validation loss.
    Args:
        model (torch.nn.Module): Model to train.
        device (torch.device): Device to use.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        class_names (list): List of class names.
        epochs (int): Number of epochs to train.
        batch_per_epoch_train (int, optional): Batches per training epoch.
        batch_per_epoch_val (int, optional): Batches per validation epoch.
        verbose (int): Verbosity level.
        scheduler (torch.optim.lr_scheduler, optional): LR scheduler.
        save_dir (str): Directory to save best model.
        writer (SummaryWriter, optional): TensorBoard writer.
        start_epoch (int): Starting epoch number.
    Returns:
        None
    """
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_mean_loss, train_mean_auc = epoch_step(model,
                                                     device,
                                                     train_loader,
                                                     criterion,
                                                     usage="Training",
                                                     batch_per_epoch=batch_per_epoch_train,
                                                     optimizer=optimizer,
                                                     writer=writer,
                                                     current_epoch=epoch+start_epoch,
                                                     class_names=class_names)
        val_mean_loss, val_mean_auc = epoch_step(model,
                                                 device,
                                                 val_loader,
                                                 criterion,
                                                 usage="Validation",
                                                 batch_per_epoch=batch_per_epoch_val,
                                                 writer=writer,
                                                 current_epoch=epoch+start_epoch,
                                                 class_names=class_names)
        if scheduler is not None:
            scheduler.step(val_mean_loss)
        if verbose != 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Mean Loss: {train_mean_loss:.4f} - Train Mean AUC: {train_mean_auc:.4f} - Val Mean Loss: {val_mean_loss:.4f} - Val Mean AUC: {val_mean_auc:.4f}")
        if val_mean_loss < best_val_loss:
            best_val_loss = val_mean_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

def eval_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    class_names: list,
    batch_per_epoch=1,
    writer=None,
    current_epoch=None
):
    """
    Evaluate a trained model on a dataset and print metrics.
    Args:
        model (torch.nn.Module): Model to evaluate.
        loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to use.
        criterion (torch.nn.Module): Loss function.
        class_names (list): List of class names.
        batch_per_epoch (int, optional): Batches per evaluation.
        writer (SummaryWriter, optional): TensorBoard writer.
        current_epoch (int, optional): Current epoch number.
    Returns:
        None
    """
    eval_mean_loss, eval_mean_auc = epoch_step(model,
                                               device,
                                               loader,
                                               criterion,
                                               usage="Evaluating",
                                               batch_per_epoch=batch_per_epoch,
                                               writer=writer,
                                               current_epoch=current_epoch,
                                               class_names=class_names)
    print(f"Eval Mean Loss: {eval_mean_loss:.4f} - Eval Mean AUC: {eval_mean_auc:.4f}")
