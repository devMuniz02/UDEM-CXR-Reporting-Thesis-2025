import os
import re
from torch.utils.data import Dataset
from utils.processing import join_uri, pil_from_path, open_text

def extract_findings(report_text: str) -> str:
    """
    Extract the FINDINGS section from a radiology report.
    If not found, return the whole report text.
    """
    match = re.search(r"FINDINGS:\s*(.*?)\s*(IMPRESSION:|$)", report_text, re.S | re.I)
    findings = match.group(1).strip() if match else report_text.strip()
    findings = re.sub(r"\s+", " ", findings).strip()  # Clean up whitespace
    return findings

class MIMICDataset(Dataset):
    """
    Expects a DataFrame with at least a 'path' (relative DICOM path) and
    you provide:
      - images_dir: base dir (local or gs://) where your pre-rendered JPG/PNG live
      - reports_dir: base dir (local or gs://) where report TXT files live
    It will map '.../dXXXXX.dcm' -> '.../dXXXXX.jpg' (adjust if needed).
    """
    def __init__(self, dataframe, images_dir: str, reports_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir  # can be local or gs://
        self.reports_dir = reports_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row.get("path", ""))  # e.g. p10/p100XXXX/sXXXXX/dXXXXX.dcm
        if not rel_path:
            raise ValueError("Row is missing 'path' column required to locate files.")

        # Image path: swap extension .dcm -> .jpg (or .png) according to your preprocessing
        image_name = os.path.basename(rel_path).replace(".dcm", "")
        # image_path = join_uri(self.images_dir, image_name)
        image_dir = os.path.dirname(rel_path)[10:]
        image_path = join_uri(self.images_dir, f"{image_dir}/{image_name}.png")

        # Report path: derive folder path and append .txt
        rel_dir = os.path.dirname(rel_path)  # p10/p100XXXX/sXXXXX
        report_path = join_uri(self.reports_dir, f"{rel_dir}.txt")

        # Load image
        # Load image
        try:
            im = pil_from_path(image_path)
        except FileNotFoundError:
            print(f"[WARN] Image file not found: {image_path}, skipping index {idx}")
            
            # Evitar recursión infinita si hay muchos archivos faltantes
            next_idx = (idx + 1) % len(self.df)
            if next_idx == idx:
                # Solo quedaba este elemento y también falta → sí lanzamos error
                raise FileNotFoundError(f"No valid images found in dataset.")
            
            return self.__getitem__(next_idx)
        image = self.transform(im) if self.transform else im

        # Load & clean report text (best-effort)
        findings = ""
        try:
            with open_text(report_path, encoding="utf-8") as f:
                full_report = f.read()
            # Robust FINDINGS extraction; fallback to whole report
            findings = extract_findings(full_report)
        except FileNotFoundError:
            findings = ""

        return image, findings, image_path, report_path