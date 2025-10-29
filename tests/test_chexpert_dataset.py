# test_chexpert_dataset.py
import sys
import types
import io
import contextlib

from pathlib import Path
import csv
from PIL import Image
import pandas as pd
import pytest

from utils.data.chexpert_dataset import (
    clean_text,
    clean_text_for_training,
    CheXpertDataset,
    CHEXPERTDataset,
    is_gcs,
    join_uri,
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
    expected_test_label = "mild cardiomegaly. physician to physician internal note."  # cleaned less on test
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


def _make_png(path: Path, size=(8, 6), color=(9, 8, 7)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size=size, color=color).save(path)


def test_init_filters_local_existing_pngs(tmp_path: Path, monkeypatch):
    """
    When images_dir is LOCAL (not gs://), the constructor should filter rows
    to those whose corresponding PNG files exist (JPG -> PNG swap).
    """
    images_dir = tmp_path / "PNG" / "train"
    # Create two images expected to exist and one missing
    _make_png(images_dir / "p/a.png")
    _make_png(images_dir / "q/b.png")
    # No file for r/c.png (should be filtered out)

    df = pd.DataFrame(
        {
            "path_to_image": ["p/a.jpg", "q/b.jpg", "r/c.jpg"],
            "section_impression": ["t1", "t2", "t3"],
            "frontal_lateral": ["Frontal", "Frontal", "Frontal"],
            "split": ["train", "train", "train"],
        }
    )

    # Ensure the module's is_gcs returns False (local path)
    assert is_gcs(str(images_dir)) is False

    ds = CHEXPERTDataset(
        dataframe=df,
        images_dir=str(images_dir),
        split="train",
        transform=None,
        text_col="section_impression",
        path_col="path_to_image",
    )

    # Only the two existing PNGs should remain
    assert len(ds) == 2
    kept_paths = set(ds.df["path_to_image"].tolist())
    assert kept_paths == {"p/a.jpg", "q/b.jpg"}


def test__full_png_path_conversion_and_join(tmp_path: Path):
    """
    _full_png_path must replace .jpg -> .png and join correctly with images_dir root.
    """
    images_dir = tmp_path / "PNG" / "train"
    ds = CHEXPERTDataset(
        dataframe=pd.DataFrame({"path_to_image": ["folder/sample.jpg"], "section_impression": ["x"]}),
        images_dir=str(images_dir),
        split="train",
        transform=None,
    )
    out = ds._full_png_path("folder/sample.jpg")
    assert out == str(images_dir / "folder" / "sample.png")


def test_getitem_train_uses_clean_text_for_training_and_transform(tmp_path: Path, monkeypatch):
    """
    For split 'train' (or 'valid'/'validate'), the class should call clean_text_for_training.
    We monkeypatch clean_text_for_training to a sentinel function.
    Also verify that a provided transform is applied.
    """
    images_dir = tmp_path / "PNG" / "train"
    _make_png(images_dir / "z/k.png", size=(16, 12))

    df = pd.DataFrame(
        {
            "path_to_image": ["z/k.jpg"],
            "section_impression": ["Some raw  text\nwith  spaces"],
            "split": ["train"],
        }
    )

    # Monkeypatch the text cleaner on the module under test
    import utils.data.chexpert_dataset as chexmod

    def fake_clean_train(s: str) -> str:
        return f"TRAIN::{s.strip()}"

    monkeypatch.setattr(chexmod, "clean_text_for_training", fake_clean_train, raising=True)

    # Provide a simple transform that tags output
    def transform(img: Image.Image):
        return ("XFORMED", img.size)

    ds = CHEXPERTDataset(
        dataframe=df,
        images_dir=str(images_dir),
        split="train",
        transform=transform,
    )

    image, findings, img_path, report_path = ds[0]
    # Transform applied
    assert image[0] == "XFORMED"
    assert image[1] == (16, 12)
    # Cleaner used
    assert findings.startswith("TRAIN::")
    # Path correctness
    assert img_path == str(images_dir / "z" / "k.png")
    assert report_path == ""

def _patch_io(monkeypatch):
    """
    Prevent *any* real cloud access by stubbing fsspec/gcsfs at import level,
    and (if present) the helpers inside utils.data.chexpert_dataset.
    This works whether chexpert_dataset imports these directly or uses helpers
    that import them internally.
    """
    # --- Build fake fsspec module ---
    class _FakeOpenObj:
        def __init__(self, path=None, mode="rb", **kwargs):
            self.mode = mode
        def open(self):
            @contextlib.contextmanager
            def _cm():
                if "b" in self.mode:
                    yield io.BytesIO(b"")    # pretend binary file
                else:
                    yield io.StringIO("")    # pretend text file
            return _cm()

    class _FakeFS:
        def exists(self, path): return True
        def glob(self, pattern): return []

    fake_fsspec = types.SimpleNamespace(
        open=lambda *a, **k: _FakeOpenObj(mode=k.get("mode", "rb")),
        filesystem=lambda *a, **k: _FakeFS()
    )

    # --- Build fake gcsfs module ---
    class _NoopGCSFS:
        def __init__(self, *a, **k): ...
        def open(self, *a, **k): return io.BytesIO(b"")
        def exists(self, *a, **k): return True
        def glob(self, *a, **k): return []

    fake_gcsfs = types.SimpleNamespace(GCSFileSystem=_NoopGCSFS)

    # Patch import table so any import fsspec/gcsfs sees our fakes
    monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)
    monkeypatch.setitem(sys.modules, "gcsfs", fake_gcsfs)

    # If chexmod already imported gcsfs/fsspec, override those too (no error if absent)
    try:
        import utils.data.chexpert_dataset as chexmod
        if hasattr(chexmod, "gcsfs"):
            monkeypatch.setattr(chexmod, "gcsfs", fake_gcsfs, raising=False)
        if hasattr(chexmod, "fsspec"):
            monkeypatch.setattr(chexmod, "fsspec", fake_fsspec, raising=False)

        # If your module exposes helpers that call fsspec/gcsfs internally, stub them as well.
        if hasattr(chexmod, "open_binary"):
            def _fake_open_binary(path):
                return _FakeOpenObj(mode="rb").open()
            monkeypatch.setattr(chexmod, "open_binary", _fake_open_binary, raising=False)

        if hasattr(chexmod, "open_text"):
            def _fake_open_text(path, encoding="utf-8"):
                return _FakeOpenObj(mode="rt").open()
            monkeypatch.setattr(chexmod, "open_text", _fake_open_text, raising=False)
    except Exception:
        # If import path changes or module isn’t present yet, it’s fine;
        # the sys.modules overrides above still protect later imports.
        pass
