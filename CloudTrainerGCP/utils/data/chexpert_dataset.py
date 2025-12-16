
"""PyTorch Dataset for CheXpert chest X-ray images and associated reports."""

# Standard library imports
import os
import re
import string
import gcsfs

# Third-party imports
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from utils.processing import is_gcs, join_uri, pil_from_path

# Configure PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True

def clean_text(text: str) -> str:
    """
    Clean and normalize radiology report text for NLP tasks.
    - Lowercases text
    - Removes enumerators like "1." but keeps decimals like "2.5"
    - Converts intra-word hyphens to spaces ("follow-up" -> "follow up")
    - Removes punctuation except periods
    - Normalizes spaces around non-decimal periods
    - Collapses multiple spaces and repeated periods
    """
    # Guard: accept None or non-string inputs
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    t = text.lower()

    # 1) Remove enumerators like "1." or "23." when NOT part of a decimal (no digit after the dot)
    #    Does not affect "2.5", "1.23", etc.
    t = re.sub(r'\b\d+\.(?!\d)', ' ', t)

    # 2) Convert intra-word hyphens to spaces (so "follow-up" -> "follow up")
    t = re.sub(r'(?<=\w)-(?=\w)', ' ', t)

    # 3) Remove all punctuation EXCEPT periods
    #    (string.punctuation includes '.', so remove it from the deletion list)
    punctuation = string.punctuation.replace('.', '')
    t = t.translate(str.maketrans('', '', punctuation))

    # 4) Collapse repeated periods ("..." -> ".")
    t = re.sub(r'\.{2,}', '.', t)

    # 5) Remove spaces before a period ("word ." -> "word.")
    t = re.sub(r'\s+\.', '.', t)

    # 6) Ensure a space after non-decimal periods
    #    Only if the period doesn't have a digit on either side, to avoid "2. 5"
    t = re.sub(r'(?<!\d)\.(?!\d)', '. ', t)

    # 7) Collapse multiple spaces and trim
    t = re.sub(r'\s+', ' ', t).strip()

    return t

def clean_text_for_training(
    text: str,
    cutoff: str = "physician to physician",
    cleaner=clean_text
) -> str:
    """
    Prepare report text for training:

    1) Cut everything after (and excluding) `cutoff` (case-insensitive).
    2) Clean the remaining fragment using `cleaner` (defaults to `clean_text`).

    Args:
        text: Input text (can be None; will be cast to str).
        cutoff: Delimiter phrase used to cut the text (case-insensitive).
        cleaner: Cleaning function to apply (signature: (str) -> str). Defaults to `clean_text`.

    Returns:
        Cleaned text suitable for training.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Case-insensitive split before `cutoff`
    pattern = re.compile(re.escape(cutoff), flags=re.IGNORECASE)
    parts = pattern.split(text, maxsplit=1)
    before = parts[0].strip() if parts else text

    # Reuse the main cleaner to keep a single source of truth
    return cleaner(before)

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
            csv = csv[csv["split"] == "train"] #### TEMPORARY HACK #### To be removed when using official splits
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
        self.split = split

    def __len__(self):
        """
        Return the number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return len(self.df)

    def _full_png_path(self, rel):
        """
        Get the absolute path to a PNG image given a relative .jpg path.
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
        findings = clean_text(findings) if self.split == "test" else clean_text_for_training(findings)
        return {"image": image, "label": findings, "path": full_path}

class CHEXPERTDataset(Dataset):
    """
    Expects a DataFrame with columns:
      - text_col: the report/impression text (default 'section_impression')
      - path_col: relative path to image (default 'path_to_image', usually .jpg)
    Provide images_dir as the base folder (local or gs://) of CheXpertPlus/PNG.
    Automatically resolves split subfolder (train|valid|test) and swaps .jpg -> .png.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: str,
        split: str = "train",
        transform=None,
        text_col: str = 'section_impression',  # "section_findings",
        path_col: str = "path_to_image"
    ):
        self.df = dataframe.reset_index(drop=True)
        self.split = "train" if split in ["train", "valid", "validate"] else "test"
        self.img_root = images_dir
        self.transform = transform
        self.text_col = text_col
        self.path_col = path_col

        print("Filtering rows with missing PNGs...")

        is_gcs = images_dir.startswith("gs://")
        fs = gcsfs.GCSFileSystem() if is_gcs else None
        keep_idx = []

        for i, rel in enumerate(self.df[self.path_col].tolist()):
            rel_path = rel.replace("\\", "/").replace(".jpg", ".png")
            full_path = f"{images_dir.rstrip('/')}/{rel_path}" if is_gcs else os.path.join(images_dir, rel_path)

            exists = fs.exists(full_path) if is_gcs else os.path.exists(full_path)
            if exists:
                keep_idx.append(i)

        csv = self.df.iloc[keep_idx].reset_index(drop=True)
        print(f"[INFO] Kept {len(csv)}/{len(self.df)} rows with existing PNGs")
        self.df = csv

    def __len__(self):
        return len(self.df)

    def _full_png_path(self, rel: str) -> str:
        p = str(rel).replace("\\", "/")
        # Incoming rel often ends with .jpg; dataset is PNG
        p_png = p.replace(".jpg", ".png")
        return join_uri(self.img_root, p_png)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = row[self.path_col]
        image_path = self._full_png_path(rel)

        # Load image from local/GCS
        im = pil_from_path(image_path)
        image = self.transform(im) if self.transform else im

        # Clean text
        findings = row[self.text_col]
        findings = "" if pd.isna(findings) else str(findings)
        findings = clean_text(findings) if self.split == "test" else clean_text_for_training(findings)

        return image, findings, image_path, ""  # no report_path for CheXpert