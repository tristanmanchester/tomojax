from __future__ import annotations

"""Differentiable alignment loss functions (per-view).

All losses here take projected images (pred) and measurements (target) and return
per-view scalar losses suitable for summation across views. They are JAX-friendly
and keep computation in `jnp.float32`.

Notes
-----
- For non L2 losses (e.g., ZNCC/SSIM), Gauss–Newton is not appropriate. Use
  gradient descent in the alignment loop.
- Otsu-masked L2 uses a mask derived from the target only (kept constant), with a
  soft sigmoid for differentiability.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class LossState:
    kind: str
    params: Dict[str, float]
    # Optional per-view mask (n, nv, nu) for masked losses
    mask: Optional[jnp.ndarray] = None


def _safe_epsilon(p: Dict[str, float], key: str, default: float) -> float:
    v = float(p.get(key, default))
    return max(v, 1e-12)


def _per_view_sum(resid2d: jnp.ndarray) -> jnp.ndarray:
    """Sum of elements for each view (resid2d is (nv, nu))."""
    return jnp.sum(resid2d)


def _loss_l2(pred: jnp.ndarray, tar: jnp.ndarray, _: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    return 0.5 * jnp.sum(r * r)


def _loss_charbonnier(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    eps = _safe_epsilon(st.params, "eps", 1e-3)
    return jnp.sum(jnp.sqrt(r * r + eps * eps))


def _loss_huber(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    delta = float(st.params.get("delta", 1.0))
    absr = jnp.abs(r)
    quad = jnp.minimum(absr, delta)
    # 0.5 r^2 for |r|<=delta; delta(|r|-0.5 delta) otherwise
    return jnp.sum(0.5 * quad * quad + delta * (absr - quad))


def _loss_cauchy(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    """Welsch/Cauchy loss: (c^2 / 2) * (1 - exp(-(r/c)^2))."""
    r = (pred - tar).astype(jnp.float32)
    c = _safe_epsilon(st.params, "c", 1.0)
    z = (r / c) ** 2
    return jnp.sum(0.5 * (c * c) * (1.0 - jnp.exp(-z)))


def _loss_zncc(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    """Negative zero-mean normalized cross-correlation per view.

    Returns: (1 - NCC) * N where N is number of pixels, so magnitude roughly matches
    L2 in scale for easier optimizer tuning.
    """
    eps = _safe_epsilon(st.params, "eps", 1e-5)
    p = pred.astype(jnp.float32)
    y = tar.astype(jnp.float32)
    pm = jnp.mean(p)
    ym = jnp.mean(y)
    p0 = p - pm
    y0 = y - ym
    num = jnp.sum(p0 * y0)
    den = jnp.sqrt(jnp.sum(p0 * p0) * jnp.sum(y0 * y0) + eps)
    ncc = num / den
    N = p.size
    return (1.0 - ncc) * jnp.float32(N)


def _avgpool2(x: jnp.ndarray, k: int) -> jnp.ndarray:
    """Average pool with SAME padding using NHWC conv.

    x: (H, W)
    """
    x4 = x[None, :, :, None]
    kernel = jnp.ones((k, k, 1, 1), dtype=x4.dtype) / float(k * k)
    y4 = jax.lax.conv_general_dilated(
        x4,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y4[0, :, :, 0]


def _loss_ssim(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    """1 - SSIM per view (single-scale, uniform window)."""
    K1 = float(st.params.get("K1", 0.01))
    K2 = float(st.params.get("K2", 0.03))
    win = int(st.params.get("window", 7))
    win = max(3, win | 1)  # ensure odd >= 3
    C1 = (K1 * 1.0) ** 2
    C2 = (K2 * 1.0) ** 2
    x = pred.astype(jnp.float32)
    y = tar.astype(jnp.float32)
    mu_x = _avgpool2(x, win)
    mu_y = _avgpool2(y, win)
    sigma_x2 = _avgpool2(x * x, win) - mu_x * mu_x
    sigma_y2 = _avgpool2(y * y, win) - mu_y * mu_y
    sigma_xy = _avgpool2(x * y, win) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2))
    # 1 - mean SSIM, scaled by N pixels
    N = x.size
    return (1.0 - jnp.mean(ssim_map)) * jnp.float32(N)


def _compute_otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    """Classic Otsu threshold on a single 2D image (numpy side)."""
    x = np.asarray(x, np.float32)
    if not np.isfinite(x).any():
        return float(np.nan)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax <= vmin:
        return float(vmin)
    hist, bin_edges = np.histogram(x, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    p = hist / max(1e-12, hist.sum())
    omega = np.cumsum(p)
    mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = np.nanargmax(sigma_b2)
    # Threshold at the center between edges
    return float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)


def _loss_l2_otsu_soft(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    """L2 applied on a soft mask derived from Otsu threshold of target.

    Parameters (st.params):
      - temp: temperature for the sigmoid (default 0.5)
      - polarity: 'fg' or 'bg' (default 'fg'): keep foreground or background
    """
    assert st.mask is not None, "Otsu mask is required for l2_otsu"
    temp = _safe_epsilon(st.params, "temp", 0.5)
    pol = st.params.get("polarity", "fg")
    base = st.mask.astype(jnp.float32)
    # base is a soft probability in [0,1] if we precomputed with sigmoid; if hard
    # binary, smooth here.
    if base.dtype != jnp.float32:
        base = base.astype(jnp.float32)
    mask = base if pol == "fg" else (1.0 - base)
    r = (pred - tar).astype(jnp.float32)
    # An extra smoothing to ensure gradients exist even if mask is {0,1}
    w = jax.nn.sigmoid((base - 0.5) / temp)
    return 0.5 * jnp.sum((r * w) ** 2)


def build_loss(kind: str, params: Optional[Dict[str, float]], targets: jnp.ndarray) -> Tuple:
    """Return (per_view_fn, state) for the requested loss.

    per_view_fn signature: (pred_chunk, tar_chunk, state_chunk) -> losses_per_view (b,)
    where `state_chunk` can be None for losses that do not need per-view state.
    """
    k = str(kind).lower()
    p = {} if params is None else {str(a): float(b) for a, b in params.items()}
    state = LossState(kind=k, params=p, mask=None)

    if k == "l2":
        f = _loss_l2
    elif k in ("charbonnier", "charb"):
        f = _loss_charbonnier
    elif k == "huber":
        f = _loss_huber
    elif k in ("cauchy", "welsch"):
        f = _loss_cauchy
    elif k in ("zncc", "ncc"):
        f = _loss_zncc
    elif k == "ssim":
        f = _loss_ssim
    elif k in ("l2_otsu", "l2-otsu", "otsu-l2"):
        # Precompute soft mask per view from targets
        temp = _safe_epsilon(p, "temp", 0.5)
        # Compute thresholds on host (numpy), shape (n,)
        Ys = np.asarray(targets)
        thr = np.array([_compute_otsu_threshold(Ys[i]) for i in range(Ys.shape[0])], dtype=np.float32)
        # Soft mask via sigmoid((y-tau)/temp)
        base = jax.device_put(jax.nn.sigmoid((targets - jnp.asarray(thr)[:, None, None]) / temp))
        state.mask = base
        f = _loss_l2_otsu_soft
    else:
        raise ValueError(f"Unknown loss kind: {kind}")

    def per_view_fn(pred_chunk: jnp.ndarray, tar_chunk: jnp.ndarray, mask_chunk: Optional[jnp.ndarray]) -> jnp.ndarray:
        # pred_chunk, tar_chunk: (b, nv, nu)
        if mask_chunk is None:
            return jax.vmap(lambda a, b: f(a, b, state))(pred_chunk, tar_chunk)
        return jax.vmap(lambda a, b, m: f(a, b, LossState(kind=state.kind, params=state.params, mask=m)))(
            pred_chunk, tar_chunk, mask_chunk
        )

    return per_view_fn, state
