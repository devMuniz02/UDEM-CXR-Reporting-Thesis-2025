
"""PyTorch Dataset for CheXpert chest X-ray images and associated reports."""

# Standard library imports
import os
import re
import string

# Third-party imports
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Configure PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True

def clean_text(text: str) -> str:
    """
    Clean and normalize radiology report text for NLP tasks.
    - Lowercases text
    - Removes enumerators like "1." but keeps decimals
    - Removes punctuation except periods
    - Normalizes spaces around periods
    - Collapses multiple spaces
    Args:
        text (str): Input text string.
    Returns:
        str: Cleaned text string.
    """
    # lowercase
    text = text.lower()

    # remove enumerators like "1." or "23." but KEEP decimals like "2.5"
    # (?<!\d) ensures no digit right before; (?!\d) ensures no digit right after the dot
    text = re.sub(r'(?<!\d)\b\d+\.(?!\d)', ' ', text)

    # remove all punctuation EXCEPT "."
    punctuation = string.punctuation.replace('.', '')
    text = text.translate(str.maketrans('', '', punctuation))

    # normalize spaces around periods to " . " → ". "
    text = re.sub(r'\s*\.\s*', '. ', text)

    # collapse multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()

    return text

class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for CheXpert chest X-ray images and associated reports.
    Loads images and cleans associated text findings for use in deep learning models.
    """
    def __init__(self, img_root, csv_path, split="train", transform=None, text_col="section_impression", path_col="path_to_image", enforce_exists=True):
        """
        Initialize CheXpertDataset.
        Args:
            img_root (str): Root directory containing images.
            csv_path (str): Path to CSV file with metadata.
            split (str): Data split ('train', 'valid', 'test').
            transform (callable, optional): Image transform.
            text_col (str): Column name for findings text.
            path_col (str): Column name for image path.
            enforce_exists (bool): If True, only keep rows with existing images.
        """
        df = pd.read_csv(csv_path)
        keep_idx = []
        for i, rel in enumerate(df[path_col].tolist()):
            p = os.path.join(img_root, rel.replace("\\", "/")).replace(".jpg", ".png")
            if os.path.exists(p):
                keep_idx.append(i)
        csv = df.iloc[keep_idx].reset_index(drop=True)
        print(f"[INFO] Kept {len(csv)}/{len(df)} rows with existing PNGs")

        self.img_root = os.path.abspath(img_root)
        if split in {"train", "valid", "test"}:
            csv = csv[csv["split"] == "train"]
            patients = csv["deid_patient_id"].unique().tolist()
            train_ids, temp_ids = train_test_split(patients, test_size=0.1, random_state=42)
            valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
            if split == "train":
                csv = csv[csv["deid_patient_id"].isin(train_ids)]
            elif split == "valid":
                csv = csv[csv["deid_patient_id"].isin(valid_ids)]
            else:
                csv = csv[csv["deid_patient_id"].isin(test_ids)]
        self.df = csv.reset_index(drop=True)
        self.transform = transform
        self.text_col = text_col
        self.enforce_exists = enforce_exists
        self.path_col = path_col

    def __len__(self):
        """
        Return the number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return len(self.df)

    def _full_png_path(self, rel):
        """
        Get the absolute path to a PNG image given a relative path.
        Args:
            rel (str): Relative image path from CSV.
        Returns:
            str: Absolute path to PNG image.
        """
        p = rel.replace("\\", "/")
        return os.path.join(self.img_root, p).replace(".jpg", ".png")

    def __getitem__(self, idx):
        """
        Get a sample from the dataset by index.
        Loads and transforms the image, cleans the findings text.
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: Dictionary with keys 'image', 'label', and 'path'.
        """
        row = self.df.iloc[idx]
        rel = row[self.path_col]
        full_path = self._full_png_path(rel)
        if self.enforce_exists and not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        with Image.open(full_path) as im:
            im = im.convert("RGB")
            image = self.transform(im) if self.transform else im
        findings = row[self.text_col]
        findings = "" if pd.isna(findings) else str(findings)
        findings = clean_text(findings)
        return {"image": image, "label": findings, "path": full_path}
