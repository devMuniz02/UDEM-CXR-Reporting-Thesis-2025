import os
import numpy as np
import torch
import pytest
from PIL import Image
import torchvision.transforms as T

from utils.processing import image_transform, reverse_image_transform


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
