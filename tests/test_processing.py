import io
import os
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
import torch
import pytest
from PIL import Image
import torchvision.transforms as T

# >>> Patch targets must be the module under test, not sklearn directly
import utils.processing as procmod

from utils.processing import (
    image_transform, 
    reverse_image_transform,
    is_gcs,
    join_uri,
    open_binary,
    open_text,
    read_csv_any,
    pil_from_path,
    loader,
)

def _set_seeds(seed: int = 1234):
    torch.manual_seed(seed)
    np.random.seed(seed)

def _rand_pil_rgb(w: int, h: int) -> Image.Image:
    """Random uint8 RGB PIL image."""
    arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def _resize_to_tensor_baseline(img: Image.Image, size: int) -> torch.Tensor:
    """Baseline: Resize + ToTensor (no normalize)."""
    baseline = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    return baseline(img)  # (3, size, size) in [0,1]

@pytest.mark.parametrize("img_size_in", [96, 321])
@pytest.mark.parametrize("img_size_out", [128, 512])
def test_transform_shapes_and_dtype(img_size_in, img_size_out):
    _set_seeds()
    img = _rand_pil_rgb(img_size_in, img_size_in)

    tfm = image_transform(img_size=img_size_out)
    t = tfm(img)  # (3,H,W) normalized
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, img_size_out, img_size_out)
    assert t.dtype == torch.float32

    # reverse → [0,1], shape preserved
    x = reverse_image_transform(t)
    assert x.shape == (3, img_size_out, img_size_out)
    assert (x >= 0).all() and (x <= 1).all()

@pytest.mark.parametrize("img_size_out", [128, 256, 512])
def test_reverse_matches_baseline_resize_totensor(img_size_out):
    """
    image_transform = Resize → ToTensor → Normalize
    reverse_image_transform should invert Normalize and clamp to [0,1],
    thus matching the plain (Resize→ToTensor) baseline up to numerical tol.
    """
    _set_seeds()
    img = _rand_pil_rgb(173, 219)  # non-square to exercise resize

    tfm = image_transform(img_size=img_size_out)
    normed = tfm(img)  # (3,S,S) normalized

    reversed_img = reverse_image_transform(normed)  # (3,S,S) in [0,1]

    baseline = _resize_to_tensor_baseline(img, img_size_out)  # (3,S,S) in [0,1]

    # Should be extremely close (no clipping expected for random inputs)
    assert torch.allclose(reversed_img, baseline, atol=1e-6, rtol=0)

def test_reverse_supports_batched_tensors():
    """
    torchvision.transforms.Normalize supports (C,H,W) and (B,C,H,W).
    Ensure reverse_image_transform handles batched input.
    """
    _set_seeds()
    B, S = 4, 224
    # create a normalized batch using the same stats as image_transform
    # Start from [0,1] images:
    batch = torch.rand(B, 3, S, S)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    normed = normalize(batch)

    denorm = reverse_image_transform(normed)
    assert denorm.shape == (B, 3, S, S)
    assert denorm.dtype == torch.float32
    # Values must be clamped to [0,1]
    assert (denorm >= 0).all() and (denorm <= 1).all()

    # Since we constructed normed by normalizing batch, reversing should recover it closely
    assert torch.allclose(denorm, batch, atol=1e-6, rtol=0)

def test_identity_on_edge_values_with_clamp():
    """
    If input after inverse-normalize slightly exceeds [0,1], clamp should bring it back.
    We simulate this by crafting tensors near the extremes.
    """
    # Create values that, after inverse normalization, will exceed [0,1] for some channels
    # We'll just directly pass slightly out-of-range values to reverse and check clamp.
    # reverse_image_transform expects a normalized tensor; for this test we only check clamping behavior.
    t = torch.tensor([
        [[-10.0, 0.5], [2.0, 10.0]],   # channel 0
        [[-1.0, 0.2], [0.8, 1.5]],     # channel 1
        [[-3.0, 0.1], [0.9, 3.0]],     # channel 2
    ], dtype=torch.float32)  # (3,2,2)

    out = reverse_image_transform(t)
    assert out.shape == (3, 2, 2)
    assert (out >= 0).all() and (out <= 1).all()

# ---------------------------
# Local-only mock for procmod.fsspec.open
# ---------------------------
class _MockFSOpen:
    """
    Mimics fsspec.open(...).open() -> fileobj, but *never* uses network.
    If path is 'gs://bucket/...', it is mapped to <tmp>/bucket/... (PurePosix).
    """
    def __init__(self, tmp_root: Path, path: str, mode: str, encoding=None):
        self.tmp_root = tmp_root
        self.path = path
        self.mode = mode
        self.encoding = encoding

    def open(self):
        if str(self.path).startswith("gs://"):
            rel = self.path.replace("gs://", "")
            local_path = self.tmp_root / PurePosixPath(rel)
        else:
            local_path = Path(self.path)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        if "b" in self.mode:
            return open(local_path, self.mode)
        else:
            return open(local_path, self.mode, encoding=self.encoding)

def test_is_gcs():
    assert is_gcs("gs://bucket/file.txt") is True
    assert is_gcs("gs://bucket") is True
    assert is_gcs("/local/path") is False
    assert is_gcs("C:\\windows\\path") is False

def test_join_uri_local_and_gs(tmp_path: Path):
    # local
    base_local = str(tmp_path / "base")
    res_local = join_uri(base_local, "a", "b", "c.txt")
    assert Path(res_local) == Path(base_local) / "a" / "b" / "c.txt"

    # gs
    base_gs = "gs://mybucket/dir"
    res_gs = join_uri(base_gs, "x", "y", "z.png")
    assert res_gs == "gs://mybucket/dir/x/y/z.png"

def test_open_text_and_binary_local(tmp_path: Path, monkeypatch):
    # Prepare local files
    txt_p = tmp_path / "note.txt"
    bin_p = tmp_path / "raw.bin"
    txt_p.write_text("hola mundo\n", encoding="utf-8")
    bin_p.write_bytes(b"\x01\x02abc")

    # Mock fsspec.open to ensure no external I/O
    def _fake_open(path, mode="rb", encoding=None):
        return _MockFSOpen(tmp_path, path, mode, encoding)

    # >>> Patch the exact object used inside utils.processing
    monkeypatch.setattr(procmod.fsspec, "open", _fake_open, raising=True)

    with open_text(str(txt_p), encoding="utf-8") as f:
        assert f.read() == "hola mundo\n"

    with open_binary(str(bin_p)) as f:
        assert f.read() == b"\x01\x02abc"

def test_open_text_and_binary_gs_mapped_to_tmp(tmp_path: Path, monkeypatch):
    # Create files where the mock will map gs:// paths
    (tmp_path / "mybucket/dir").mkdir(parents=True, exist_ok=True)
    (tmp_path / "mybucket/dir/note.txt").write_text("cloud line", encoding="utf-8")
    (tmp_path / "mybucket/dir/blob.bin").write_bytes(b"\x00\xff")

    def _fake_open(path, mode="rb", encoding=None):
        return _MockFSOpen(tmp_path, path, mode, encoding)

    monkeypatch.setattr(procmod.fsspec, "open", _fake_open, raising=True)

    with open_text("gs://mybucket/dir/note.txt") as f:
        assert f.read() == "cloud line"
    with open_binary("gs://mybucket/dir/blob.bin") as f:
        assert f.read() == b"\x00\xff"

def test_pil_from_path_local(tmp_path: Path, monkeypatch):
    # Save a small image
    local_img = tmp_path / "img.png"
    Image.new("RGB", (16, 8), color=(10, 20, 30)).save(local_img)

    # Route through mocked fsspec to avoid any accidental remote access
    def _fake_open(path, mode="rb", encoding=None):
        return _MockFSOpen(tmp_path, path, mode, encoding)

    monkeypatch.setattr(procmod.fsspec, "open", _fake_open, raising=True)

    img = pil_from_path(str(local_img))
    assert img.mode == "RGB"
    assert img.size == (16, 8)

def test_pil_from_path_gs_mapped_to_tmp(tmp_path: Path, monkeypatch):
    # Create a "bucket" image in tmp for the mock
    mapped = tmp_path / "bucket/a/b.png"
    mapped.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (7, 5), color=(1, 2, 3)).save(mapped)

    def _fake_open(path, mode="rb", encoding=None):
        return _MockFSOpen(tmp_path, path, mode, encoding)

    monkeypatch.setattr(procmod.fsspec, "open", _fake_open, raising=True)

    img = pil_from_path("gs://bucket/a/b.png")
    assert img.mode == "RGB"
    assert img.size == (7, 5)

def test_read_csv_any_local(tmp_path: Path):
    csvp = tmp_path / "small.csv"
    df_in = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    df_in.to_csv(csvp, index=False)

    df = read_csv_any(str(csvp))
    assert df.equals(df_in)

def test_loader_basic_filters_and_splits(tmp_path: Path, monkeypatch):
    """
    Validates:
      - split name mapping
      - ViewPosition filtering (AP/PA only)
      - CheXpert Frontal-only filtering
      - Basic train/valid split behavior
    Uses only local CSVs and a fake train_test_split; no network access.
    """
    # --- create tiny CSVs ---
    mimic_splits = pd.DataFrame({
        "dicom_id": ["d1", "d2", "d3", "d4"],
        "split":     ["train", "validate", "test", "train"],
    })
    mimic_meta = pd.DataFrame({
        "dicom_id": ["d1", "d2", "d3", "d4"],
        "ViewPosition": ["AP", "LATERAL", "PA", "AP"],
    })
    mimic_reports = pd.DataFrame({
        "dicom_id": ["d1", "d2", "d3", "d4"],
        "path": ["p1/s1/d1.dcm", "p1/s2/d2.dcm", "p2/s1/d3.dcm", "p3/s3/d4.dcm"],
    })
    chexpert = pd.DataFrame({
        "path_to_image": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "frontal_lateral": ["Frontal", "Lateral", "Frontal", "Frontal"],
        "split": ["train", "train", "valid", "test"],
        "section_impression": ["t1", "t2", "t3", "t4"],
    })

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "mimic_splits.csv").write_text(mimic_splits.to_csv(index=False))
    (data_dir / "mimic_meta.csv").write_text(mimic_meta.to_csv(index=False))
    (data_dir / "mimic_reports.csv").write_text(mimic_reports.to_csv(index=False))
    (data_dir / "chexpert.csv").write_text(chexpert.to_csv(index=False))

    chexpert_paths = {"chexpert_data_csv": str(data_dir / "chexpert.csv")}
    mimic_paths = {
        "mimic_splits_csv": str(data_dir / "mimic_splits.csv"),
        "mimic_metadata_csv": str(data_dir / "mimic_meta.csv"),
        "mimic_reports_path": str(data_dir / "mimic_reports.csv"),
    }

    # Fake TTS and patch the symbol actually used inside utils.processing
    def _fake_tts(df, test_size=0.01, random_state=None):
        n = len(df)
        cut = max(1, int(round(n * (1 - test_size))))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    from sklearn import model_selection as ym
    monkeypatch.setattr(ym, "train_test_split", _fake_tts, raising=True)

    # Case A: split='test' => chexpert_split='valid', no tts invoked for chexpert
    MIMIC_df, CHEXPERT_df = loader(chexpert_paths, mimic_paths, split="test")
    # MIMIC: AP/PA only & split=='test' -> only 'd3' (PA)
    assert set(MIMIC_df["dicom_id"]) == {"d3"}
    # CheXpert: Frontal & split=='valid' -> only 'c.jpg'
    assert set(CHEXPERT_df["path_to_image"]) == {"c.jpg"}

    # Case B: split='train' => chexpert_split='train' and tts is called
    MIMIC_df2, CHEXPERT_df2 = loader(chexpert_paths, mimic_paths, split="train")
    assert set(MIMIC_df2["dicom_id"]) == {"d1", "d4"}  # AP only & split train
    assert set(CHEXPERT_df2["frontal_lateral"]) == {"Frontal"}
    assert len(CHEXPERT_df2) >= 1  # non-empty after fake split

# --- Additional tests for missing functions in utils.processing ---

import gzip
import io as _pyio
import pandas as _pd
from types import SimpleNamespace

# _fs_and_path ---------------------------------------------------------------

def test__fs_and_path_monkeypatched_url_to_fs(monkeypatch):
    # Arrange: fake url_to_fs that echoes protocol/path
    def _fake_url_to_fs(path):
        # Return (fs_like, path_in_fs)
        fs = SimpleNamespace(protocol="gs", exists=lambda p: True)
        return fs, "bucket/dir/file.txt"

    monkeypatch.setattr(procmod, "url_to_fs", _fake_url_to_fs, raising=True)

    # Act
    fs, p = procmod._fs_and_path("gs://bucket/dir/file.txt")

    # Assert
    assert getattr(fs, "protocol", None) == "gs"
    assert p == "bucket/dir/file.txt"


# exists --------------------------------------------------------------------

def test_exists_uses_fs_exists_true(monkeypatch):
    # fs.exists returns True
    class _FS:
        protocol = "gs"
        def exists(self, p): return True

    monkeypatch.setattr(procmod, "url_to_fs", lambda path: (_FS(), "x/y/z"), raising=True)
    assert procmod.exists("gs://any/thing") is True

def test_exists_uses_fs_exists_false(monkeypatch):
    # fs.exists returns False
    class _FS:
        protocol = "gs"
        def exists(self, p): return False

    monkeypatch.setattr(procmod, "url_to_fs", lambda path: (_FS(), "x/y/z"), raising=True)
    assert procmod.exists("gs://any/thing") is False

def test_exists_fallback_to_os_path_when_fs_errors(tmp_path, monkeypatch):
    # When fs.exists raises AND protocol=='file', function should fallback to os.path.exists
    target = tmp_path / "a/b/c.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ok")

    class _FS:
        protocol = "file"
        def exists(self, p):  # simulate broken fs.exists
            raise RuntimeError("boom")

    # Map the incoming path to the real local path in url_to_fs result
    local_path = str(target)
    monkeypatch.setattr(procmod, "url_to_fs", lambda path: (_FS(), local_path), raising=True)

    # Should fallback to os.path.exists(local_path) -> True
    assert procmod.exists("file://" + local_path) is True

    # And for a non-existent path -> False
    missing_local = str(target.parent / "missing.txt")
    monkeypatch.setattr(procmod, "url_to_fs", lambda path: (_FS(), missing_local), raising=True)
    assert procmod.exists("file://" + missing_local) is False


# read_csv_any --------------------------------------------------------------

def test_read_csv_any_reads_gz_local(tmp_path: Path):
    # Create a gzipped CSV and ensure read_csv_any can read it
    df_in = _pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    gz_path = tmp_path / "table.csv.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        df_in.to_csv(f, index=False)

    df = read_csv_any(str(gz_path))
    assert df.equals(df_in)

def test_read_csv_any_kwargs_passthrough(monkeypatch, tmp_path: Path):
    # Verify that read_csv_any forwards kwargs to pandas.read_csv
    called = {}

    def _fake_read_csv(path, storage_options=None, **kwargs):
        called["path"] = path
        called["storage_options"] = storage_options
        called["kwargs"] = kwargs
        # Return a trivial frame
        return _pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(_pd, "read_csv", _fake_read_csv, raising=True)

    p = tmp_path / "dummy.csv"
    p.write_text("ok\n1\n")

    df = read_csv_any(str(p), dtype={"ok": int}, nrows=1)
    assert "ok" in df.columns and df.shape == (1, 1)
    # Ensure kwargs + storage_options were passed as defined in the helper
    assert called["path"] == str(p)
    assert called["storage_options"] is None
    assert called["kwargs"].get("dtype") == {"ok": int}
    assert called["kwargs"].get("nrows") == 1
