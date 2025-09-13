
"""
PadChestGRDataset: PyTorch Dataset for PadChest-GR chest X-ray images and associated reports.
Includes text cleaning and image preprocessing utilities.
"""

# Standard library imports
import os
import json
import re
import string

# Third-party imports
from typing import List, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

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
    text = text.lower()
    text = re.sub(r'(?<!\d)\b\d+\.(?!\d)', ' ', text)
    punctuation = string.punctuation.replace('.', '')
    text = text.translate(str.maketrans('', '', punctuation))
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class PadChestGRDataset(Dataset):
    """
    Minimal, fast dataset for PadChest-GR report generation.
    - Precomputes image paths.
    - Pre-tokenizes reports once with BioBERT (or any HF tokenizer).
    - No logging, no prints.
    """
    def __init__(
        self,
        dataframe,
        root_dir: str,
        json_file: str,
        image_size: int = 1024,
        normalize: bool = True,
        transform=None,
        return_paths: bool = False,
        sentence_key: str = "sentence_en",
    ):
        """
        Initialize PadChestGRDataset.
        Args:
            dataframe (pd.DataFrame): DataFrame with image IDs.
            root_dir (str): Root directory containing images.
            json_file (str): Path to JSON file with findings.
            image_size (int): Size to resize images to.
            normalize (bool): Whether to normalize images.
            transform (callable, optional): Image transform.
            return_paths (bool): If True, include image paths in output.
            sentence_key (str): Key for findings sentences in JSON.
        """
        self.root_dir = root_dir
        self.img_ids: List[str] = dataframe["ImageID"].tolist()
        self.img_paths: List[str] = [os.path.join(root_dir, x) for x in self.img_ids]
        self.return_paths = return_paths

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        by_id: Dict[str, List[Dict[str, Any]]] = {
            d["ImageID"]: d.get("findings", []) for d in data
        }
        texts: List[str] = []
        for img_id in self.img_ids:
            findings = by_id.get(img_id, [])
            joined = " ".join(
                (f.get(sentence_key) or "").strip()
                for f in findings if f.get(sentence_key)
            ).strip()
            texts.append(joined)

        tfs = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        if normalize:
            tfs.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]))
        self.transform = transform or transforms.Compose(tfs)
        self.texts = texts

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset by index.
        Loads and transforms the image, cleans the findings text.
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: Dictionary with keys 'image', 'label', and 'path'.
        """
        with Image.open(self.img_paths[idx]).convert("RGB") as im:
            image = self.transform(im)
        findings = self.texts[idx]
        findings = clean_text(findings)
        full_path = self.img_paths[idx]
        return {"image": image, "label": findings, "path": full_path}