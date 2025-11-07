import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def gaussian_layer_stack_pipeline(
    x: torch.Tensor,
    n_layers: int,
    base_ksize: int = 3,
    ksize_growth: int = 2,
    sigma: float | None = None,
    eps: float = 1e-8,
):
    """
    All-in-one GPU batch pipeline:
      1) Per-sample min-max normalize to [0,1]
      2) Resize to (32,32)
      3) Apply L Gaussian blurs with increasing kernel size in a single
         horizontal conv + single vertical conv using depthwise groups
         (via a shared max kernel padded with zeros)
      4) Renormalize each layer to [0,1]
      5) Return stacked (B,L,32,32), flat (B,L,1024), tiled (B,L,1024,1024 view)

    Args:
      x: (B,H,W) or (B,1,H,W) tensor (any device/dtype)
      n_layers: number of layers
      base_ksize: starting odd kernel size (e.g., 3)
      ksize_growth: increment per layer (e.g., 2) -> ensures odd sizes
      sigma: if None, uses (ksize-1)/6 per layer; else fixed sigma for all
      eps: small number for safe division

    Returns:
      stacked: (B, n_layers, 32, 32)  float on x.device
      flat:    (B, n_layers, 1024)
      tiled:   (B, n_layers, 1024, 1024)  (expand view; memory-cheap)
    """
    assert n_layers >= 1, "n_layers must be >= 1"

    # ---- Ensure 4D, 1 channel; cast to float (stay on same device) ----
    if x.ndim == 3:
        x = x.unsqueeze(1)  # (B,1,H,W)
    elif x.ndim != 4 or x.shape[1] not in (1,):
        raise ValueError(f"Expected (B,H,W) or (B,1,H,W); got {tuple(x.shape)}")
    x = x.float()

    B, _, H, W = x.shape

    # ---- Per-sample min-max normalize to [0,1] ----
    xmin = x.amin(dim=(2, 3), keepdim=True)
    xmax = x.amax(dim=(2, 3), keepdim=True)
    denom = (xmax - xmin).clamp_min(eps)
    x = (x - xmin) / denom  # (B,1,H,W) in [0,1]

    # ---- Resize to 32x32 on GPU ----
    x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)  # (B,1,32,32)

    # ---- Prepare per-layer kernel sizes (odd) ----
    ksizes = []
    for i in range(n_layers, 0, -1):  # to keep your original ordering: L...1
        k = base_ksize + i * ksize_growth
        k = int(k)
        if k % 2 == 0:
            k += 1
        k = max(k, 1)
        ksizes.append(k)

    Kmax = max(ksizes)
    pad = Kmax // 2

    # ---- Build per-layer 1D Gaussian vectors and embed into shared Kmax kernel ----
    # We create horizontal weights of shape (L,1,1,Kmax) and vertical (L,1,Kmax,1)
    device, dtype = x.device, x.dtype
    weight_h = torch.zeros((n_layers, 1, 1, Kmax), device=device, dtype=dtype)
    weight_v = torch.zeros((n_layers, 1, Kmax, 1), device=device, dtype=dtype)

    for idx, k in enumerate(ksizes):
        # choose sigma
        sig = sigma if (sigma is not None and sigma > 0) else (k - 1) / 6.0
        r = k // 2
        xp = torch.arange(-r, r + 1, device=device, dtype=dtype)
        g = torch.exp(-(xp * xp) / (2.0 * sig * sig))
        g = g / g.sum()  # (k,)

        # center g into Kmax with zeros around
        start = (Kmax - k) // 2
        end = start + k

        # horizontal row
        weight_h[idx, 0, 0, start:end] = g  # (1 x Kmax)

        # vertical column
        weight_v[idx, 0, start:end, 0] = g  # (Kmax x 1)

    # ---- Duplicate input across L channels (depthwise groups) ----
    xL = x.expand(B, n_layers, 32, 32).contiguous()  # (B,L,32,32)

    # ---- Separable Gaussian blur with a single pass per axis (groups=L) ----
    # Horizontal
    xh = F.pad(xL, (pad, pad, 0, 0), mode="reflect")
    xh = F.conv2d(xh, weight=weight_h, bias=None, stride=1, padding=0, groups=n_layers)  # (B,L,32,32)

    # Vertical
    xv = F.pad(xh, (0, 0, pad, pad), mode="reflect")
    yL = F.conv2d(xv, weight=weight_v, bias=None, stride=1, padding=0, groups=n_layers)  # (B,L,32,32)

    # ---- Renormalize each layer to [0,1] (per-sample, per-layer) ----
    y_min = yL.amin(dim=(2, 3), keepdim=True)
    y_max = yL.amax(dim=(2, 3), keepdim=True)
    y_den = (y_max - y_min).clamp_min(eps)
    stacked = (yL - y_min) / y_den  # (B,L,32,32) in [0,1]

    # ---- Flatten + tile (expand view; caution w/ later materialization) ----
    flat = stacked.reshape(B, n_layers, 32 * 32)               # (B,L,1024)
    tiled = flat.unsqueeze(-2).expand(-1, -1, 32 * 32, -1)     # (B,L,1024,1024) view

    return stacked, flat, tiled

def plot_layers_any(
    x,
    *,
    max_batches=None,
    vlim=(0, 1),
    one_indexed: bool = False,
    max_cols: int = 6,
):
    """
    Plot layers for each batch sample in separate figures.

    Accepts:
      - stacked: (B, L, H, W)
      - flat:    (B, L, HW)
      - tiled:   (B, L, HW, HW)

    Behavior:
      - Creates one figure PER BATCH (up to `max_batches`).
      - At most `max_cols` layers per row (default 6).
      - Column headers: 'Layer {i}' descending from n-1 -> 0 (or n -> 1 if one_indexed=True).
      - Figure title per batch: 'Masks for input {i} out of {B}'.

    Returns:
      A list of (fig, axes) tuples, one per plotted batch.
    """
    # ---- Normalize input to torch ----
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")

    if x.ndim not in (3, 4):
        raise ValueError(f"Expected ndim 3 or 4, got shape {tuple(x.shape)}")

    # ---- Convert to (B, L, H, W) 'stacked' ----
    if x.ndim == 4:
        B, L, A, B_ = x.shape
        if A == B_:
            # Could be stacked (H==W) or tiled (HW x HW). Heuristic: if A is a perfect square
            # and reasonably large (e.g., 1024), treat as tiled and collapse to flat.
            s = int(math.isqrt(A))
            if s * s == A and A >= 64:
                flat = x[..., 0, :].detach()  # (B, L, HW)
                H = W = s
                stacked = flat.reshape(B, L, H, W)
            else:
                stacked = x.detach()
        else:
            stacked = x.detach()
    else:
        # x.ndim == 3 -> (B, L, HW)
        B, L, HW = x.shape
        s = int(math.isqrt(HW))
        if s * s != HW:
            if HW != 32 * 32:
                raise ValueError(
                    f"Cannot infer square image size from HW={HW}. "
                    f"Provide stacked (B,L,H,W) or flat with square HW."
                )
            s = 32
        H = W = s
        stacked = x.detach().reshape(B, L, H, W)

    # Ensure float & CPU for plotting
    stacked = stacked.to(torch.float32).cpu().numpy()

    # ---- Batch selection ----
    B, L, H, W = stacked.shape
    plot_B = B if max_batches is None else max(1, min(B, int(max_batches)))

    # ---- Layout params ----
    cols = max(1, int(max_cols))
    rows_needed = lambda L: (L + cols - 1) // cols

    figs = []
    for b in range(plot_B):
        # number of rows for this batch
        r = rows_needed(L)
        fig, axes = plt.subplots(r, cols, figsize=(cols * 3, r * 3), squeeze=False)
        fig.suptitle(f"Masks for input {b} out of {B}", fontsize=12, y=1.02)

        for l in range(L):
            rr = l // cols
            cc = l % cols
            ax = axes[rr, cc]
            if vlim is None:
                ax.imshow(stacked[b, l], cmap="gray")
            else:
                ax.imshow(stacked[b, l], cmap="gray", vmin=vlim[0], vmax=vlim[1])
            ax.axis("off")

            # Set column titles only on the first row of the grid
            label_num = (l + 1) if one_indexed else l
            ax.set_title(f"Layer {label_num}", fontsize=10)

        # Hide any unused axes (when L is not a multiple of cols)
        total_slots = r * cols
        for empty_idx in range(L, total_slots):
            rr = empty_idx // cols
            cc = empty_idx % cols
            axes[rr, cc].axis("off")

        plt.tight_layout()
        plt.show()
        figs.append((fig, axes))
    return figs
