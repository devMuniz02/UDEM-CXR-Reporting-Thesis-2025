# tests/test_create_dataloaders.py
import types
from typing import Callable, Optional
import torch
import pandas as pd
import pytest

from utils.data.dataloaders import create_dataloaders


class _DummyDataset(torch.utils.data.Dataset):
    """
    Minimal torch Dataset that returns (tensor_image, label)
    and exposes `transform` for assertions.
    """
    def __init__(self, n: int, transform: Optional[Callable] = None, name: str = ""):
        self._n = n
        self.transform = transform
        self.name = name

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        x = torch.zeros(3, 8, 8, dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(0, dtype=torch.long)
        return x, y


@pytest.fixture
def monkeypatched_env(monkeypatch):
    """
    Monkeypatches inside the target module:
      - MIMICDataset, CHEXPERTDataset -> dummy datasets
      - loader -> returns trivial DataFrames
      - image_transform -> identity transform
    """
    import utils.data.dataloaders as target_mod  # same module as create_dataloaders

    ns = types.SimpleNamespace(n_mimic=10, n_chexpert=20)

    def _mock_loader(chexpert_paths, mimic_paths, split="train"):
        return pd.DataFrame({"id": list(range(ns.n_mimic))}), pd.DataFrame({"id": list(range(ns.n_chexpert))})

    def _mock_image_transform(img_size=512):
        return lambda t: t

    def _mock_mimic_dataset(df, images_dir, data_path, transform=None):
        return _DummyDataset(n=len(df), transform=transform, name="MIMIC")

    def _mock_chexpert_dataset(df, data_path, split="train", transform=None):
        return _DummyDataset(n=len(df), transform=transform, name="CHEXPERT")

    monkeypatch.setattr(target_mod, "loader", _mock_loader, raising=True)
    monkeypatch.setattr(target_mod, "image_transform", _mock_image_transform, raising=True)
    monkeypatch.setattr(target_mod, "MIMICDataset", _mock_mimic_dataset, raising=True)
    monkeypatch.setattr(target_mod, "CHEXPERTDataset", _mock_chexpert_dataset, raising=True)

    return ns


def _count_indices_from_dataset0(indices, n0):
    return sum(1 for i in indices if i < n0)


def _collect_sampler_indices(sampler):
    return list(iter(sampler))


def test_basic_construction_and_weights(monkeypatched_env):
    # Arrange
    n1, n2 = 12, 18
    monkeypatched_env.n_mimic = n1
    monkeypatched_env.n_chexpert = n2

    chexpert_paths = {"chexpert_data_path": "/dev/null"}
    mimic_paths = {"mimic_images_dir": "/dev/null", "mimic_data_path": "/dev/null"}
    batch_size = 8
    ratio = 0.7

    # Act
    dl = create_dataloaders(
        chexpert_paths=chexpert_paths,
        mimic_paths=mimic_paths,
        batch_size=batch_size,
        split="train",
        sampling_ratio=ratio,
        num_workers=0,
        drop_last=False,
    )

    # Assert DataLoader structure
    assert isinstance(dl.dataset, torch.utils.data.ConcatDataset)
    ds0, ds1 = dl.dataset.datasets
    assert isinstance(ds0, _DummyDataset) and ds0.name == "MIMIC"
    assert isinstance(ds1, _DummyDataset) and ds1.name == "CHEXPERT"
    assert len(ds0) == n1 and len(ds1) == n2

    # Assert sampler
    from torch.utils.data import WeightedRandomSampler
    assert isinstance(dl.sampler, WeightedRandomSampler)

    # Build expectations with the SAME dtype as the sampler's weights
    weights = dl.sampler.weights
    assert weights.numel() == n1 + n2
    p1, p2 = ratio, 1 - ratio
    expected_w1 = torch.full((n1,), fill_value=p1 / n1, dtype=weights.dtype)
    expected_w2 = torch.full((n2,), fill_value=p2 / n2, dtype=weights.dtype)

    assert torch.allclose(weights[:n1], expected_w1, atol=1e-7)
    assert torch.allclose(weights[n1:], expected_w2, atol=1e-7)

    # Sampler length equals total dataset size per function (n1 + n2)
    indices = _collect_sampler_indices(dl.sampler)
    assert len(indices) == n1 + n2

    # Quick smoke run: iterate one batch
    batch = next(iter(dl))
    x, y = batch
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.dim() == 4  # (B, C, H, W)
    assert y.dim() == 1  # (B,)
    assert x.shape[0] <= batch_size


def test_transform_passthrough_when_not_provided(monkeypatched_env):
    monkeypatched_env.n_mimic = 3
    monkeypatched_env.n_chexpert = 2

    chexpert_paths = {"chexpert_data_path": "/dev/null"}
    mimic_paths = {"mimic_images_dir": "/dev/null", "mimic_data_path": "/dev/null"}

    dl = create_dataloaders(chexpert_paths, mimic_paths, batch_size=2, split="train", sampling_ratio=0.5, num_workers=0)

    ds0, ds1 = dl.dataset.datasets
    assert callable(ds0.transform)
    assert callable(ds1.transform)

    x0, _ = ds0[0]
    x1, _ = ds1[0]
    assert torch.is_tensor(x0) and torch.is_tensor(x1)


def test_custom_transform_wired(monkeypatched_env):
    monkeypatched_env.n_mimic = 4
    monkeypatched_env.n_chexpert = 6

    def custom_transform(t):
        return t + 1.0

    chexpert_paths = {"chexpert_data_path": "/dev/null"}
    mimic_paths = {"mimic_images_dir": "/dev/null", "mimic_data_path": "/dev/null"}

    dl = create_dataloaders(
        chexpert_paths,
        mimic_paths,
        batch_size=3,
        split="train",
        transform=custom_transform,
        sampling_ratio=0.6,
        num_workers=0,
    )

    ds0, ds1 = dl.dataset.datasets
    assert ds0.transform is custom_transform
    assert ds1.transform is custom_transform

    x0, _ = ds0[0]
    x1, _ = ds1[0]
    assert torch.allclose(x0, torch.ones_like(x0))
    assert torch.allclose(x1, torch.ones_like(x1))


def test_kwargs_passthrough(monkeypatched_env):
    monkeypatched_env.n_mimic = 5
    monkeypatched_env.n_chexpert = 5

    chexpert_paths = {"chexpert_data_path": "/dev/null"}
    mimic_paths = {"mimic_images_dir": "/dev/null", "mimic_data_path": "/dev/null"}

    dl = create_dataloaders(
        chexpert_paths,
        mimic_paths,
        batch_size=4,
        split="train",
        sampling_ratio=0.5,
        drop_last=True,
        num_workers=0,
    )

    assert dl.batch_size == 4
    assert dl.drop_last is True


def test_edge_case_small_imbalanced(monkeypatched_env):
    n1, n2 = 1, 9
    monkeypatched_env.n_mimic = n1
    monkeypatched_env.n_chexpert = n2

    chexpert_paths = {"chexpert_data_path": "/dev/null"}
    mimic_paths = {"mimic_images_dir": "/dev/null", "mimic_data_path": "/dev/null"}

    ratio = 0.9
    dl = create_dataloaders(
        chexpert_paths,
        mimic_paths,
        batch_size=5,
        split="train",
        sampling_ratio=ratio,
        num_workers=0,
    )

    from torch.utils.data import WeightedRandomSampler
    assert isinstance(dl.sampler, WeightedRandomSampler)
    weights = dl.sampler.weights
    assert weights.numel() == n1 + n2

    w0 = weights[0].item()
    w1_each = weights[1:].mean().item()
    assert w0 > w1_each

    indices = _collect_sampler_indices(dl.sampler)
    assert len(indices) == n1 + n2

    taken_from_ds0 = _count_indices_from_dataset0(indices, n1)
    assert taken_from_ds0 >= 1
