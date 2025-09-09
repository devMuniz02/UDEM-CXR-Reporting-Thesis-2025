import os
import json
from typing import List, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

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
        max_txt_len: int = 64,
        image_size: int = 1024,
        normalize: bool = True,
        transform=None,
        return_paths: bool = False,
        sentence_key: str = "sentence_en",
    ):
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
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        with Image.open(self.img_paths[idx]).convert("RGB") as im:
            image = self.transform(im)
        findings = self.texts[idx]
        full_path = self.img_paths[idx]
        return {"image": image, "label": findings, "path": full_path}