import torch
import torchvision.transforms as T
import fsspec
import io
import os
from pathlib import Path, PurePosixPath
from typing import Tuple

import pandas as pd
from PIL import Image
import sklearn.model_selection as ym
from fsspec.core import url_to_fs

def image_transform(img_size=512):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def reverse_image_transform(tensor):
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    tensor = inv_normalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")

def _fs_and_path(path: str):
    """Return (fs, path_in_fs) using fsspec.url_to_fs for both gs:// and local paths."""
    fs, fs_path = url_to_fs(path)
    return fs, fs_path

def exists(path: str) -> bool:
    fs, p = _fs_and_path(path)
    try:
        return fs.exists(p)
    except Exception:
        # Fallback: for local filesystem, use os.path
        if getattr(fs, "protocol", None) == "file":
            return os.path.exists(p)
        raise

def join_uri(base: str, *parts: str) -> str:
    """
    Join path segments for both local and gs:// URIs (PurePosix for URIs).
    """
    if is_gcs(base):
        return "gs://" + str(PurePosixPath(base.replace("gs://", "")).joinpath(*parts))
    else:
        return str(Path(base).joinpath(*parts))

def open_binary(path: str):
    """
    Open any (local or gs://) file for binary reading.
    Returns a file-like object (context manager).
    """
    return fsspec.open(path, mode="rb").open()

def open_text(path: str, encoding="utf-8"):
    return fsspec.open(path, mode="rt", encoding=encoding).open()

def read_csv_any(path: str, **kwargs) -> pd.DataFrame:
    """
    Read CSV or CSV.GZ from local/GCS using fsspec.
    """
    return pd.read_csv(path, storage_options=None, **kwargs)

def pil_from_path(path: str) -> Image.Image:
    """
    Load an image from local or GCS; returns a PIL image in RGB.
    """
    with open_binary(path) as f:
        img_bytes = f.read()
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return im

def loader(chexpert_paths, mimic_paths, split: str = "train") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split not in ["train", "valid", "validate", "test"]:
        raise ValueError("Invalid split name. Choose from 'train', 'valid', 'validate', 'test'.")

    mimic_split = "validate" if split == "valid" else split   # MIMIC uses "validate"
    chexpert_split = "train" if split in ["train", "valid", "validate"] else split  # CheXpert uses "train"
    chexpert_split = "valid" if split == "test" else chexpert_split                 # CheXpert uses "valid"

    # Read CSVs directly (works with gs:// or local)
    mimic_splits_csv  = read_csv_any(mimic_paths["mimic_splits_csv"])
    mimic_metadata_csv = read_csv_any(mimic_paths["mimic_metadata_csv"])
    mimic_reports_csv  = read_csv_any(mimic_paths["mimic_reports_path"])
    full_mimic_metadata = mimic_splits_csv.merge(
        mimic_metadata_csv[["dicom_id", "ViewPosition"]],
        on="dicom_id",
        how="left"
    ).merge(
        mimic_reports_csv[["dicom_id", "path"]],
        on="dicom_id",
        how="left"
    )
    full_mimic_metadata = full_mimic_metadata[full_mimic_metadata["ViewPosition"].isin(["AP", "PA"])]

    chexpert_csv = read_csv_any(chexpert_paths["chexpert_data_csv"])
    chexpert_csv = chexpert_csv[chexpert_csv["frontal_lateral"] == "Frontal"]

    MIMIC = full_mimic_metadata.loc[full_mimic_metadata["split"] == mimic_split].copy()
    CHEXPERT = chexpert_csv.loc[chexpert_csv["split"] == chexpert_split].copy()

    # Optional fine split to approximate MIMIC's validate size
    if split in ["train", "valid", "validate"]:
        if split in ["valid", "validate"]:
            _, CHEXPERT = ym.train_test_split(CHEXPERT, test_size=0.01, random_state=42)
        else:
            CHEXPERT, _ = ym.train_test_split(CHEXPERT, test_size=0.01, random_state=42)

    return MIMIC.reset_index(drop=True), CHEXPERT.reset_index(drop=True)