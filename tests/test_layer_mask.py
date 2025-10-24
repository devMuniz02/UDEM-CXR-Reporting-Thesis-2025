import os
import math
import numpy as np
import torch
import pytest

# --- Add these imports near the top of your test file ---
import matplotlib
matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt

from utils.layer_mask import gaussian_layer_stack_pipeline, plot_layers_any


def _set_seeds(seed: int = 1234):
    torch.manual_seed(seed)
    np.random.seed(seed)


@pytest.mark.parametrize("shape", [(8, 256, 256), (8, 1, 256, 256)])
@pytest.mark.parametrize("n_layers,base_ksize,ksize_growth", [(5, 3, 2), (3, 1, 2)])
def test_pipeline_shapes_and_values(shape, n_layers, base_ksize, ksize_growth):
    """
    Validates:
      - Accepts (B,H,W) and (B,1,H,W)
      - Returns stacked:(B,L,32,32), flat:(B,L,1024), tiled:(B,L,1024,1024)
      - All outputs are in [0,1]
      - `tiled` is a broadcasted view of `flat` across the extra axis
    """
    _set_seeds()
    x = torch.tensor(np.random.rand(*shape)).float()  # CPU
    # Use the user's pattern: if 4D, take channel 0, else already BH W
    x_in = x if x.ndim == 3 else x[:, 0, :, :]

    stacked, flat, tiled = gaussian_layer_stack_pipeline(
        x_in,
        n_layers=n_layers,
        base_ksize=base_ksize,
        ksize_growth=ksize_growth,
        sigma=None,
    )

    B = shape[0]
    assert stacked.shape == (B, n_layers, 32, 32)
    assert flat.shape == (B, n_layers, 1024)
    assert tiled.shape == (B, n_layers, 1024, 1024)

    # Range checks
    assert torch.isfinite(stacked).all()
    assert torch.isfinite(flat).all()
    assert torch.isfinite(tiled).all()
    assert (stacked >= 0).all() and (stacked <= 1).all()
    assert (flat >= 0).all() and (flat <= 1).all()
    assert (tiled >= 0).all() and (tiled <= 1).all()

    # Broadcasted equality: tiled == flat expanded over the extra axis
    # Do a single vectorized check (faster than looping 1024 times)
    expected = flat.unsqueeze(-2).expand_as(tiled)
    assert torch.allclose(tiled, expected)

    # Spot-check a few rows (including boundaries) for strict equality
    for l in [0, 1, 17, 511, 1023]:
        assert torch.equal(tiled[:, :, l, :], flat), f"Tiled row {l} != flat"


def test_order_and_basic_monotonic_blur():
    """
    Weak behavioral check using an impulse:
      - Later layers (constructed with larger kernels) should be at least as smooth
        as earlier ones (max value should not increase).
    """
    _set_seeds()
    B, H, W = 2, 256, 256
    x = torch.zeros((B, H, W), dtype=torch.float32)
    # Put an impulse at center
    x[:, H // 2, W // 2] = 1.0

    n_layers = 5
    stacked, flat, tiled = gaussian_layer_stack_pipeline(
        x, n_layers=n_layers, base_ksize=3, ksize_growth=2, sigma=None
    )

    # stacked is in [0,1]; check that max per-layer does not increase with larger kernels
    # Recall: layers are ordered L..1 (descending i), so index 0 has the largest kernel.
    max_per_layer = stacked.view(B, n_layers, -1).max(dim=-1).values  # (B, L)
    # For each sample, ensure non-increasing across layers index 0..L-1
    diff = max_per_layer[:, :-1] - max_per_layer[:, 1:]
    assert (diff >= -1e-6).all(), "Later layers should not be peakier than earlier ones"


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_device_consistency(device_str):
    """
    Run on CPU and (if available) CUDA, verifying shape and dtype/device are consistent.
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    _set_seeds()
    device = torch.device(device_str)

    B, C, H, W = 4, 1, 192, 192
    x = torch.rand((B, C, H, W), dtype=torch.float32, device=device)

    # Follow user's invocation pattern: pass (B,H,W)
    x_in = x[:, 0, :, :]

    stacked, flat, tiled = gaussian_layer_stack_pipeline(
        x_in, n_layers=4, base_ksize=3, ksize_growth=2, sigma=None
    )

    assert stacked.device == x.device
    assert flat.device == x.device
    assert tiled.device == x.device

    assert stacked.dtype == torch.float32
    assert flat.dtype == torch.float32
    assert tiled.dtype == torch.float32

    assert stacked.shape == (B, 4, 32, 32)
    assert flat.shape == (B, 4, 1024)
    assert tiled.shape == (B, 4, 1024, 1024)

    # Range check
    assert (stacked >= 0).all() and (stacked <= 1).all()

def _axes_shape(axes):
    """Return (rows, cols) for a 2D ndarray of axes."""
    # axes is created with squeeze=False, so should be 2D
    return axes.shape


def _count_drawn_images(ax):
    # Matplotlib stores rendered images in ax.images
    return len(ax.images)

def test_plot_layers_any_stacked_grid_and_titles():
    B, L, H, W = 2, 8, 32, 32
    x_stacked = torch.rand(B, L, H, W)

    figs = plot_layers_any(
        x_stacked,
        max_batches=None,   # plot all batches
        vlim=(0, 1),
        one_indexed=False,  # titles: Layer 0..L-1 (ascending per provided impl)
        max_cols=6,
    )

    # Expect one figure per batch
    assert isinstance(figs, list) and len(figs) == B

    rows_expected = (L + 6 - 1) // 6  # ceil(L/6) -> for L=8 => 2
    cols_expected = 6

    for b, (fig, axes) in enumerate(figs):
        # grid shape
        r, c = _axes_shape(axes)
        assert r == rows_expected and c == cols_expected

        # suptitle
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == f"Masks for input {b} out of {B}"

        # First 8 axes should have an image; the rest should be empty/hidden
        filled = 0
        for idx in range(r * c):
            rr, cc = divmod(idx, cols_expected)
            ax = axes[rr, cc]
            if idx < L:
                # Has one image drawn
                assert _count_drawn_images(ax) == 1
                # Column titles should be set (function sets on every tile)
                expected_title = f"Layer {idx}"
                assert ax.get_title() == expected_title
                # Axes are turned off for visuals
                # assert not ax.xaxis.get_visible() and not ax.yaxis.get_visible()
                filled += 1
            # else:
                # Unused slot: axis turned off
                # assert not ax.xaxis.get_visible() and not ax.yaxis.get_visible()
        assert filled == L

        plt.close(fig)

def test_plot_layers_any_one_indexed_titles_and_batch_limit():
    B, L, H, W = 3, 5, 32, 32
    x_stacked = torch.rand(B, L, H, W)

    # Limit to first 2 batches, one-indexed labels
    figs = plot_layers_any(
        x_stacked,
        max_batches=2,
        vlim=(0, 1),
        one_indexed=True,   # titles: Layer 1..L
        max_cols=6,
    )

    assert len(figs) == 2
    for b, (fig, axes) in enumerate(figs):
        # suptitle
        assert fig._suptitle.get_text() == f"Masks for input {b} out of {B}"

        # Titles should be 1..L in ascending order per column
        for l in range(L):
            rr, cc = divmod(l, 6)
            ax = axes[rr, cc]
            assert ax.get_title() == f"Layer {l+1}"
            # image drawn
            assert _count_drawn_images(ax) == 1
            # axis off
            # assert not ax.xaxis.get_visible() and not ax.yaxis.get_visible()

        plt.close(fig)

def test_plot_layers_any_flat_and_tiled_inputs_and_vlim_none():
    B, L = 2, 7
    # flat: (B, L, 1024) -> 32x32 inferred
    flat = torch.rand(B, L, 32 * 32)
    # tiled: (B, L, 1024, 1024) -> should collapse a row to flat internally
    # (use expand to keep memory reasonable)
    base = torch.rand(B, L, 32 * 32)
    tiled = base.unsqueeze(-2).expand(-1, -1, 32 * 32, -1).contiguous()

    # Flat input
    figs_flat = plot_layers_any(flat, max_batches=None, vlim=None, max_cols=6)
    assert len(figs_flat) == B
    for fig, axes in figs_flat:
        # Images placed; vlim=None should not error
        # Just sanity-check at least one image exists
        assert any(_count_drawn_images(ax) == 1 for ax in axes.ravel())
        plt.close(fig)

    # Tiled input
    figs_tiled = plot_layers_any(tiled, max_batches=1, vlim=(0, 1), max_cols=6)
    assert len(figs_tiled) == 1  # max_batches=1 respected
    fig, axes = figs_tiled[0]
    # Should draw L images across rows/cols
    drawn = sum(_count_drawn_images(ax) == 1 for ax in axes.ravel())
    assert drawn == L
    plt.close(fig)
