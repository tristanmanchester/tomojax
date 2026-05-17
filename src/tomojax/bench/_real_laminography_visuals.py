"""Reusable visual helpers for real-laminography developer artifacts."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy import ndimage


def load_volume_array(path: Path, *, key: str) -> np.ndarray:
    """Load a 3D volume from `.npy` or keyed `.npz` files."""
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"{path} does not contain key {key!r}; keys={data.files}")
            arr = data[key]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def scale_uint8(
    image: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if lo is None or hi is None:
        lo_v, hi_v = np.percentile(finite, [1.0, 99.0])
    else:
        lo_v, hi_v = float(lo), float(hi)
    if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v:
        lo_v, hi_v = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi_v <= lo_v:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (np.clip(arr, lo_v, hi_v) - lo_v) / (hi_v - lo_v)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return np.asarray(np.round(255.0 * scaled), dtype=np.uint8)


def save_uint8_png(
    path: Path,
    image: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> str:
    """Scale an image to uint8 and write it as a PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, scale_uint8(image, lo=lo, hi=hi))
    return str(path)


def resize_nearest_2d(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D image by deterministic nearest-neighbor sampling."""
    arr = np.asarray(image, dtype=np.float32)
    if tuple(arr.shape) == tuple(shape):
        return arr
    y_idx = np.rint(np.linspace(0, arr.shape[0] - 1, int(shape[0]))).astype(np.int64)
    x_idx = np.rint(np.linspace(0, arr.shape[1] - 1, int(shape[1]))).astype(np.int64)
    return arr[np.ix_(y_idx, x_idx)]


def window_normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.asarray(np.clip((image - lo) / (hi - lo), 0.0, 1.0), dtype=np.float32)


def center_crop(shape_yx: tuple[int, int], *, size: int) -> tuple[slice, slice]:
    crop = max(1, min(int(size), int(shape_yx[0]), int(shape_yx[1])))
    y0 = (int(shape_yx[0]) - crop) // 2
    x0 = (int(shape_yx[1]) - crop) // 2
    return slice(y0, y0 + crop), slice(x0, x0 + crop)


def largest_centered_square_inside_rotated_frame(
    shape_yx: tuple[int, int],
    *,
    angle_deg: float,
    margin: int,
) -> tuple[slice, slice]:
    """Largest centered square guaranteed to avoid same-shape rotation padding."""
    n = min(int(shape_yx[0]), int(shape_yx[1]))
    theta = abs(float(angle_deg)) % 90.0
    if theta > 45.0:
        theta = 90.0 - theta
    radians = np.deg2rad(theta)
    side = int(np.floor(n / (abs(np.cos(radians)) + abs(np.sin(radians)))))
    side = max(1, min(n, side - 2 * max(0, int(margin))))
    return center_crop(shape_yx, size=side)


def grid_aligned_xy(
    volume_xyz: np.ndarray,
    *,
    z_index: int,
    angle_deg: float,
    crop: tuple[slice, slice],
) -> np.ndarray:
    vol_yxz = np.transpose(volume_xyz, (1, 0, 2))
    rotated = ndimage.rotate(
        vol_yxz,
        angle_deg,
        axes=(1, 0),
        reshape=False,
        order=1,
        mode="nearest",
    )
    z = int(np.clip(z_index, 0, rotated.shape[2] - 1))
    return np.asarray(rotated[:, :, z][crop], dtype=np.float32)


__all__ = [
    "center_crop",
    "grid_aligned_xy",
    "largest_centered_square_inside_rotated_frame",
    "load_volume_array",
    "resize_nearest_2d",
    "save_uint8_png",
    "scale_uint8",
    "window_normalize",
]
