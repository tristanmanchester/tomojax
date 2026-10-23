"""Quicklook image extraction and PNG writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v3 as iio
import numpy as np

if TYPE_CHECKING:
    import jax

type PathLike = str | Path


def extract_central_slice(volume: np.ndarray | jax.Array) -> np.ndarray:
    """Return a display-oriented central reconstruction slice.

    Reconstruction volumes use ``(z, y, x)`` order. PNG images use row/column
    order, so the returned slice displays as ``(y, x)``.
    Only the selected plane of a device volume is transferred to the host.
    """
    arr = volume if hasattr(volume, "shape") else np.asarray(volume)
    if arr.ndim == 2:
        return np.asarray(arr)
    if arr.ndim == 3:
        return np.asarray(arr[arr.shape[0] // 2])
    raise ValueError(f"quicklook expects a 2D image or 3D volume, got shape {arr.shape}")


def scale_to_uint8(
    image: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Scale a 2D image to uint8 using finite-value percentile limits."""
    lower = float(lower_percentile)
    upper = float(upper_percentile)
    if not (0.0 <= lower < upper <= 100.0):
        raise ValueError(
            "percentile bounds must satisfy 0 <= lower_percentile < upper_percentile <= 100"
        )

    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if not np.any(finite):
        return out

    finite_values = arr[finite]
    lo, hi = np.percentile(finite_values, [lower, upper])
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return out

    scaled = (finite_values - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    out[finite] = np.round(scaled * 255.0).astype(np.uint8)
    return out


def save_quicklook_png(path: PathLike, volume: np.ndarray | jax.Array) -> Path:
    """Write a percentile-scaled central reconstruction slice as a PNG."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = scale_to_uint8(extract_central_slice(volume))
    iio.imwrite(out_path, image)
    return out_path
