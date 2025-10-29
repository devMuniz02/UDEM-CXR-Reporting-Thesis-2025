# test_mimic_dataset.py
from pathlib import Path
import pandas as pd
import pytest
from PIL import Image

import utils.data.mimic_dataset as mimod
from utils.data.mimic_dataset import extract_findings, MIMICDataset

# ---------------------------
# extract_findings tests
# ---------------------------

def test_extract_findings_basic():
    text = """
    HISTORY: cough and fever
    FINDINGS:  The heart is normal in size.
               Lungs are clear.
    IMPRESSION: No acute cardiopulmonary process.
    """
    out = extract_findings(text)
    assert out == "The heart is normal in size. Lungs are clear."

def test_extract_findings_no_section_returns_whole():
    text = "No sections here, just free text of the report."
    out = extract_findings(text)
    assert out == "No sections here, just free text of the report."

def test_extract_findings_case_insensitive_and_whitespace():
    text = "FiNdInGs:   mild  atelectasis   at  bases \n\n IMPRESSION: ok"
    out = extract_findings(text)
    assert out == "mild atelectasis at bases"

# ---------------------------
# MIMICDataset tests
# ---------------------------

def test_getitem_local_paths_with_real_io(tmp_path: Path):
    """
    Local test: create an image file and a report file on disk.
    No monkeypatching needed—paths are local and safe.
    """
    # Dataframe row with relative DICOM path
    rel_path = "p10/p1001/s1/d123.dcm"
    df = pd.DataFrame({"path": [rel_path]})

    # images_dir: dataset uses only the basename and swaps .dcm->.jpg
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_name = "d123.jpg"
    img_path = images_dir / image_name
    Image.new("RGB", (11, 7), color=(9, 9, 9)).save(img_path)

    # reports_dir: dataset joins reports_dir with "<rel_dir>.txt"
    rel_dir = "p10/p1001/s1"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{rel_dir}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "HISTORY: none\nFINDINGS: clear lungs.\nIMPRESSION: normal",
        encoding="utf-8",
    )

    # Simple transform to prove it's applied
    def transform(im):
        return ("XFORM", im.size)

    ds = MIMICDataset(
        dataframe=df,
        images_dir=str(images_dir),
        reports_dir=str(reports_dir),
        transform=transform,
    )

    image, findings, image_path, got_report_path = ds[0]
    assert image == ("XFORM", (11, 7))
    assert findings == "clear lungs."
    assert image_path == str(images_dir / image_name)
    assert got_report_path == str(report_path)

def test_getitem_gs_paths_no_io(monkeypatch):
    """
    GCS-style paths: avoid I/O by monkeypatching the module-under-test's
    pil_from_path and open_text.
    """
    rel_path = "p10/p2002/s9/d8888.dcm"
    df = pd.DataFrame({"path": [rel_path]})

    images_dir = "gs://bucket/images"
    reports_dir = "gs://bucket/reports"

    # Fake image loader: return a known-size image without reading disk
    def fake_pil_from_path(_):
        return Image.new("RGB", (5, 5))

    # Fake open_text: return a context manager that yields a crafted report
    class _FakeCtx:
        def __init__(self, txt): self.txt = txt
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
        def read(self): return self.txt

    def fake_open_text(path, encoding=None):
        assert path == f"{reports_dir}/p10/p2002/s9.txt"
        return _FakeCtx("FINDINGS: mild cardiomegaly. IMPRESSION: correlate clinically.")

    # Patch on the module under test (IMPORTANT)
    monkeypatch.setattr(mimod, "pil_from_path", fake_pil_from_path, raising=True)
    monkeypatch.setattr(mimod, "open_text", fake_open_text, raising=True)

    # Transform to verify it’s applied
    def transform(im):
        return ("OK", im.size)

    ds = MIMICDataset(df, images_dir=images_dir, reports_dir=reports_dir, transform=transform)
    image, findings, image_path, report_path = ds[0]

    assert image == ("OK", (5, 5))
    assert findings == "mild cardiomegaly."
    # Only basename joined to images_dir
    assert image_path == f"{images_dir}/d8888.jpg"
    # Full rel_dir appended (with .txt) to reports_dir
    assert report_path == f"{reports_dir}/p10/p2002/s9.txt"

def test_getitem_missing_report_returns_empty(monkeypatch):
    """
    If open_text raises FileNotFoundError, dataset should return '' for findings.
    """
    rel_path = "p1/p2/s3/d9999.dcm"
    df = pd.DataFrame({"path": [rel_path]})
    images_dir = "gs://x/images"
    reports_dir = "gs://x/reports"

    monkeypatch.setattr(
        mimod,
        "pil_from_path",
        lambda _: Image.new("RGB", (3, 4)),
        raising=True,
    )

    def missing_open_text(_path, encoding=None):
        raise FileNotFoundError

    monkeypatch.setattr(mimod, "open_text", missing_open_text, raising=True)

    ds = MIMICDataset(df, images_dir=images_dir, reports_dir=reports_dir, transform=None)
    image, findings, image_path, report_path = ds[0]

    assert isinstance(image, Image.Image)
    assert image.size == (3, 4)
    assert findings == ""  # fallback on missing report
    assert image_path.endswith("/d9999.jpg")
    assert report_path.endswith("/p1/p2/s3.txt")

def test_getitem_raises_if_path_missing():
    df = pd.DataFrame([{"other": "no path col"}])
    ds = MIMICDataset(df, images_dir="/x", reports_dir="/y")
    with pytest.raises(ValueError):
        _ = ds[0]
