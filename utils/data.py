import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch

def find_valid_image_path(image_path, root_path):
    image_dirs = [f"{root_path}/images{i}" if i > 1 else f"{root_path}/images" for i in range(1, 13)]
    for folder in image_dirs:
        full_path = os.path.join(folder, image_path)
        if os.path.isfile(full_path):
            return full_path
    return None

def load_data(folders_path, data_path):
    df = pd.read_csv(os.path.join(folders_path, data_path))
    df.columns = df.columns.str.strip()
    df['Image Index'] = df['Image Index'].str.strip()
    df['Image Index'] = df['Image Index'].apply(find_valid_image_path, root_path=folders_path)
    return df[df['Image Index'].notnull()]

def split_data(df):
    train_ids, split_ids = train_test_split(df['Patient ID'].unique(), test_size=0.3, random_state=42)
    train_df = df[df['Patient ID'].isin(train_ids)]
    val_ids, test_ids = train_test_split(split_ids, test_size=0.5, random_state=42)
    val_df = df[df['Patient ID'].isin(val_ids)]
    test_df = df[df['Patient ID'].isin(test_ids)]
    return train_df, val_df, test_df

def get_label_columns(df):
    return df.columns[6:][df.columns[6:] != 'No Finding']

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None, label_cols=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_cols = label_cols if label_cols is not None else get_label_columns(dataframe)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['Image Index']
        image = Image.open(img_path).convert("RGB")
        labels = np.array(row[self.label_cols].values.astype(float), dtype=np.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels

def build_transforms(image_size: int):
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
    n_pos = df[label_cols].sum().astype(float).values
    N = len(df)
    w_pos = (N - n_pos) / (n_pos + 1e-8)
    if cap is not None:
        w_pos = np.minimum(w_pos, cap)
    return w_pos

def get_dataloaders(train_df, val_df, test_df, label_cols, batch_size, image_size):
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
    Desnormaliza una imagen normalizada con torchvision.transforms.Normalize
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean