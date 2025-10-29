import os
import re
from pathlib import PurePosixPath
from torch.utils.data import Dataset
from utils.processing import is_gcs, join_uri, pil_from_path, open_text

def extract_findings(report_text: str) -> str:
    """
    Extract the FINDINGS section from a radiology report.
    If not found, return the whole report text.
    """
    match = re.search(r"FINDINGS:\s*(.*?)\s*(IMPRESSION:|$)", report_text, re.S | re.I)
    findings = match.group(1).strip() if match else report_text.strip()
    findings = re.sub(r"\s+", " ", findings).strip()
    return findings


class MIMICDataset(Dataset):
    """
    Expects a DataFrame with at least a 'path' column containing a *relative*
    DICOM path like 'p10/p1000001/s12345678/d1234567.dcm'.

    You provide:
      - images_dir: base dir (local or gs://) where your pre-rendered images live
      - reports_dir: base dir (local or gs://) where report TXT files live

    Path mapping:
      rel = 'p10/p100.../s.../dXXXXX.dcm'
      image_path  = join_uri(images_dir, 'p10/p100.../s.../dXXXXX.<image_ext>')
      report_path = join_uri(reports_dir, 'p10/p100.../s... .txt')   # per-series txt

    Notes
    -----
    - Works for local paths and GCS URIs seamlessly via `join_uri`.
    - Uses `pil_from_path` and `open_text` to support fsspec/GCS.
    - Set `image_ext` to 'png' if your renderer produced PNGs.
    """

    def __init__(
        self,
        dataframe,
        images_dir: str,
        reports_dir: str,
        transform=None,
        image_ext: str = "jpg",
        report_ext: str = "txt",
    ):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir           # local path or gs://...
        self.reports_dir = reports_dir         # local path or gs://...
        self.transform = transform
        self.image_ext = image_ext.lstrip(".").lower()
        self.report_ext = report_ext.lstrip(".").lower()

    def __len__(self):
        return len(self.df)

    def _paths_from_rel(self, rel_path: str) -> tuple[str, str]:
        """
        Build (image_path, report_path) from the DICOM relative path.
        Preserves subdirectories regardless of local or GCS usage.
        """
        # Normalize using PurePosixPath to avoid OS-dependent separators
        rel = PurePosixPath(rel_path)
        rel_dir = rel.parent.as_posix()            # 'p10/p100.../s...'
        stem = rel.stem                             # 'dXXXXX' (without .dcm)

        # Image lives alongside the DICOM (same rel_dir) with new extension
        image_rel = PurePosixPath(rel_dir) / f"{stem}.{self.image_ext}"
        image_path = join_uri(self.images_dir, image_rel.as_posix())

        # Report is per series (rel_dir) with chosen report_ext
        report_rel = f"{rel_dir}.{self.report_ext}" if rel_dir else f"{stem}.{self.report_ext}"
        report_path = join_uri(self.reports_dir, report_rel)

        return image_path, report_path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row.get("path", "")).strip()  # e.g. 'p10/p100.../s.../dXXXXX.dcm'
        if not rel_path:
            raise ValueError("Row is missing 'path' column required to locate files.")

        image_path, report_path = self._paths_from_rel(rel_path)

        # Load image via fsspec-aware helper
        im = pil_from_path(image_path)
        image = self.transform(im) if self.transform else im

        # Load & clean report text (best effort)
        findings = ""
        try:
            with open_text(report_path, encoding="utf-8") as f:
                full_report = f.read()
            findings = extract_findings(full_report)
        except FileNotFoundError:
            findings = ""

        return image, findings, image_path, report_path
