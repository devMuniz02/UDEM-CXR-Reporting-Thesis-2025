# test_chexpert_dataset.py
from pathlib import Path
import csv
from PIL import Image
import pytest

from utils.data.chexpert_dataset import (
    clean_text,
    clean_text_for_training,
    CheXpertDataset,
)

# ----------------------------
# clean_text tests
# ----------------------------

@pytest.mark.parametrize(
    "inp,expected",
    [
        (
            "This is a SAMPLE finding! It includes numbers 123 and symbols #$%.",
            "this is a sample finding it includes numbers 123 and symbols.",
        ),
        (
            "1. First 2.5 measurement 2. Another.",
            "first 2.5 measurement another.",
        ),
        (
            "Follow-up recommended... Next steps.",
            "follow up recommended. next steps.",
        ),
    ],
)
def test_clean_text_parametrized(inp, expected):
    assert clean_text(inp) == expected


# ----------------------------
# clean_text_for_training tests
# ----------------------------

def test_clean_text_for_training_cuts_before_cutoff_case_insensitive():
    raw = "1. Normal lungs. 2. No effusion. Physician to Physician: internal msg."
    expected = "normal lungs. no effusion."
    out = clean_text_for_training(raw)
    assert "internal msg" not in out.lower()
    assert "normal lungs." in out
    assert "no effusion." in out
    assert out == expected


def test_clean_text_for_training_without_cutoff_returns_cleaned_all():
    raw = "1. Stable 2.5 cm mass. Recommend follow-up."
    out = clean_text_for_training(raw)  # cutoff not present
    expected = "stable 2.5 cm mass. recommend follow up."
    assert out == expected


def test_clean_text_for_training_with_custom_cleaner():
    # Custom cleaner to verify injection is honored
    def custom_cleaner(s: str) -> str:
        return s.strip().upper()

    raw = "Alpha beta. Physician To Physician: remove this."
    out = clean_text_for_training(raw, cleaner=custom_cleaner)
    assert out == "ALPHA BETA."


# ----------------------------
# Helpers
# ----------------------------

def _write_csv(csv_path: Path, rows):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path_to_image", "section_impression", "split", "deid_patient_id"])
        for r in rows:
            writer.writerow(r)


# ----------------------------
# CheXpertDataset tests (dataset-only)
# ----------------------------

def test_chexpertdataset_paths_and_getitem(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create a small PNG image that should be resolved from a .jpg reference
    (images_dir / "foo.png").resolve()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(images_dir / "foo.png")

    # CSV references .jpg; dataset should map it to foo.png
    csv_path = tmp_path / "meta.csv"
    rel_path = "images/foo.jpg"
    findings = "Findings: Heart size normal."
    split = "train"
    _write_csv(csv_path, [(rel_path, findings, split)])

    ds = CheXpertDataset(img_root=str(tmp_path), csv_path=str(csv_path), split="none", transform=None)

    expected_png = (tmp_path / "images" / "foo.png").resolve()
    got = Path(ds._full_png_path(r"images\foo.jpg")).resolve()
    assert got == expected_png

    assert len(ds) == 1
    item = ds[0]
    assert {"image", "label", "path"} <= set(item.keys())
    assert Path(item["path"]).resolve() == expected_png

    # Expect lowercase + punctuation normalized by dataset's internal cleaning
    assert item["label"] == "findings heart size normal."


def test_chexpertdataset_uppercase_ext_and_mixed_separators(tmp_path: Path):
    subdir = tmp_path / "images" / "sub dir"
    subdir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(subdir / "bar.png")

    # Mixed slashes and uppercase extension in CSV
    csv_path = tmp_path / "meta.csv"
    rel_path = r"images\sub dir/bar.jpg"
    findings = "1. Normal lungs. 2.5 cm nodule? 2. No pleural effusion."
    split = "none"
    _write_csv(csv_path, [(rel_path, findings, split, "patient_001")])

    ds = CheXpertDataset(img_root=str(tmp_path), csv_path=str(csv_path), split="none", transform=None)

    expected_png = (subdir / "bar.png").resolve()
    got = Path(ds._full_png_path(rel_path)).resolve()
    assert got == expected_png

    sample = ds[0]
    assert Path(sample["path"]).resolve() == expected_png

    # Expect enumerators removed, question mark stripped, decimals preserved, spacing normalized
    expected_label = "normal lungs. 2.5 cm nodule no pleural effusion."
    assert sample["label"] == expected_label


def test_chexpertdataset_missing_impression_graceful(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (2, 2), color=(0, 0, 255)).save(images_dir / "baz.png")

    # Empty impression in CSV
    csv_path = tmp_path / "meta.csv"
    rel_path = "images/baz.jpg"
    split = "none"
    _write_csv(csv_path, [(rel_path, "", split, "patient_002")])

    ds = CheXpertDataset(img_root=str(tmp_path), csv_path=str(csv_path), split="none", transform=None)

    assert len(ds) == 1
    item = ds[0]
    # Expect empty string after cleaning
    assert item["label"] == ""
    assert Path(item["path"]).exists()

def test_chexpertdataset_split_cleaning_behavior(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (2, 2), color=(123, 123, 123)).save(images_dir / "split_test.png")

    # Impression with physician-to-physician note
    csv_path = tmp_path / "meta.csv"
    rel_path = "images/split_test.jpg"
    findings = "1. Mild cardiomegaly. Physician to Physician: internal note."
    split = "train"
    _write_csv(csv_path, [(rel_path, findings, split, f"patient_00{i}") for i in range(1, 14)])

    # Test split
    ds_test = CheXpertDataset(img_root=str(tmp_path), csv_path=str(csv_path), split="test", transform=None)
    item_test = ds_test[0]
    expected_test_label = "mild cardiomegaly. physician to physician internal note." ## We cleaned the physician note only for train/valid to avoid model hallucinations
    assert item_test["label"] == expected_test_label

    # Train split
    ds_train = CheXpertDataset(img_root=str(tmp_path), csv_path=str(csv_path), split="train", transform=None)
    item_train = ds_train[0]
    expected_train_label = "mild cardiomegaly."
    assert item_train["label"] == expected_train_label

    # Valid split
    ds_valid = CheXpertDataset(img_root=str(tmp_path), csv_path=str(csv_path), split="valid", transform=None)
    item_valid = ds_valid[0]
    expected_valid_label = "mild cardiomegaly."
    assert item_valid["label"] == expected_valid_label
