import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CheXpertDataset(Dataset):
    def __init__(self, img_root, csv_path, split="train", transform=None, text_col="section_impression", path_col="path_to_image", enforce_exists=True):
        df = pd.read_csv(csv_path)
        keep_idx = []
        for i, rel in enumerate(df[path_col].tolist()):
            p = os.path.join(img_root, rel.replace("\\", "/")).replace(".jpg", ".png")
            if os.path.exists(p):
                keep_idx.append(i)
        csv = df.iloc[keep_idx].reset_index(drop=True)
        print(f"[INFO] Kept {len(csv)}/{len(df)} rows with existing PNGs under {img_root}")

        self.img_root = os.path.abspath(img_root)
        if split in {"train", "valid", "test"}:
            csv = csv[csv["split"] == "train"]
            patients = csv["deid_patient_id"].unique().tolist()
            from sklearn.model_selection import train_test_split
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
        return len(self.df)

    def _full_png_path(self, rel):
        p = rel.replace("\\", "/")
        return os.path.join(self.img_root, p).replace(".jpg", ".png")

    def __getitem__(self, idx):
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
        return {"image": image, "label": findings, "path": full_path}