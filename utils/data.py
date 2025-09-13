
"""
Data utilities for chest X-ray classification datasets.
Includes dataset classes, data loading, splitting, transforms, and normalization helpers.
"""

# Standard library imports
import os

# Third-party imports
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch

def find_valid_image_path(image_path, root_path):
    """
    Search for a valid image file in numbered subfolders.
    Args:
        image_path (str): Relative image path.
        root_path (str): Root directory containing image folders.
    Returns:
        str or None: Full path to image if found, else None.
    """
    image_dirs = [f"{root_path}/images{i}" if i > 1 else f"{root_path}/images" for i in range(1, 13)]
    for folder in image_dirs:
        full_path = os.path.join(folder, image_path)
        if os.path.isfile(full_path):
            return full_path
    return None

def load_data(folders_path, data_path):
    """
    Load and clean chest X-ray data from CSV, resolving image paths.
    Args:
        folders_path (str): Root folder containing images and CSV.
        data_path (str): CSV filename.
    Returns:
        pd.DataFrame: Cleaned dataframe with valid image paths.
    """
    df = pd.read_csv(os.path.join(folders_path, data_path))
    df.columns = df.columns.str.strip()
    df['Image Index'] = df['Image Index'].str.strip()
    df['Image Index'] = df['Image Index'].apply(find_valid_image_path, root_path=folders_path)
    return df[df['Image Index'].notnull()]

def split_data(df):
    """
    Split dataframe into train, validation, and test sets by patient ID.
    Args:
        df (pd.DataFrame): Input dataframe.
    Returns:
        tuple: train_df, val_df, test_df
    """
    train_ids, split_ids = train_test_split(df['Patient ID'].unique(), test_size=0.3, random_state=42)
    train_df = df[df['Patient ID'].isin(train_ids)]
    val_ids, test_ids = train_test_split(split_ids, test_size=0.5, random_state=42)
    val_df = df[df['Patient ID'].isin(val_ids)]
    test_df = df[df['Patient ID'].isin(test_ids)]
    return train_df, val_df, test_df

def get_label_columns(df):
    """
    Get label columns for multi-label classification, excluding 'No Finding'.
    Args:
        df (pd.DataFrame): Input dataframe.
    Returns:
        pd.Index: Label columns.
    """
    return df.columns[6:][df.columns[6:] != 'No Finding']

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray images and associated labels.
    """
    def __init__(self, dataframe, transform=None, label_cols=None):
        """
        Initialize ChestXrayDataset.
        Args:
            dataframe (pd.DataFrame): DataFrame with image paths and labels.
            transform (callable, optional): Image transform.
            label_cols (list or pd.Index, optional): Label columns.
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_cols = label_cols if label_cols is not None else get_label_columns(dataframe)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset by index.
        Loads and transforms the image, returns image and label array.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (image, labels)
        """
        row = self.df.iloc[idx]
        img_path = row['Image Index']
        image = Image.open(img_path).convert("RGB")
        labels = np.array(row[self.label_cols].values.astype(float), dtype=np.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels

def build_transforms(image_size: int):
    """
    Build image transforms for training and testing.
    Args:
        image_size (int): Target image size.
    Returns:
        tuple: (train_transform, test_transform)
    """
    train_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_t, test_t

def compute_class_pos_weights(df, label_cols, cap=None):
    """
    Compute positive class weights for imbalanced multi-label classification.
    Args:
        df (pd.DataFrame): DataFrame with labels.
        label_cols (list or pd.Index): Label columns.
        cap (float, optional): Maximum allowed weight value.
    Returns:
        np.ndarray: Array of positive class weights.
    """
    n_pos = df[label_cols].sum().astype(float).values
    N = len(df)
    w_pos = (N - n_pos) / (n_pos + 1e-8)
    if cap is not None:
        w_pos = np.minimum(w_pos, cap)
    return w_pos

def get_dataloaders(train_df, val_df, test_df, label_cols, batch_size, image_size):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    Args:
        train_df, val_df, test_df (pd.DataFrame): DataFrames for each split.
        label_cols (list or pd.Index): Label columns.
        batch_size (int): Batch size.
        image_size (int): Target image size.
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform, test_transform = build_transforms(image_size)
    train_dataset = ChestXrayDataset(train_df, transform=train_transform, label_cols=label_cols)
    val_dataset   = ChestXrayDataset(val_df,   transform=test_transform,  label_cols=label_cols)
    test_dataset  = ChestXrayDataset(test_df,  transform=test_transform,  label_cols=label_cols)
    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def denormalize_image(tensor):
    """
    Denormalize an image tensor normalized with torchvision.transforms.Normalize.
    Args:
        tensor (torch.Tensor): Normalized image tensor.
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean
