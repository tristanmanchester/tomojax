"""Projection-domain L2, foreground-weighted L2 and Huber losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .loss_state import LossState


def _safe_epsilon(p: dict[str, float], key: str, default: float) -> float:
    v = float(p.get(key, default))
    return max(v, 1e-12)


def _loss_l2(pred: jnp.ndarray, tar: jnp.ndarray, _: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    return 0.5 * jnp.sum(r * r)


def _loss_huber(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    r = (pred - tar).astype(jnp.float32)
    delta = float(st.params.get("delta", 1.0))
    absr = jnp.abs(r)
    quad = jnp.minimum(absr, delta)
    return jnp.sum(0.5 * quad * quad + delta * (absr - quad))


def _required_state_precompute[T](value: T | None, message: str) -> T:
    if value is None:
        raise ValueError(message)
    return value


def _compute_otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    """Choose a foreground threshold by maximizing between-class variance.

    Histogram edges need more precision than the measured values: even a
    valid FP32 view can span only two adjacent representable intensities,
    while the histogram subdivides that interval into hundreds of bins.
    Use float64 for both the histogram edges and the class moments rather
    than collapsing distinct measurements or changing the requested bins.
    """
    x = np.asarray(x, np.float64)
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
    return float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)


def _loss_l2_otsu_soft(pred: jnp.ndarray, tar: jnp.ndarray, st: LossState) -> jnp.ndarray:
    mask = _required_state_precompute(st.mask, "Otsu mask required")
    temp = _safe_epsilon(st.params, "temp", 0.5)
    base = mask.astype(jnp.float32)
    r = (pred - tar).astype(jnp.float32)
    w = jax.nn.sigmoid((base - 0.5) / temp)
    return 0.5 * jnp.sum((r * w) ** 2)


__all__ = [
    "_compute_otsu_threshold",
    "_loss_huber",
    "_loss_l2",
    "_loss_l2_otsu_soft",
    "_safe_epsilon",
]
