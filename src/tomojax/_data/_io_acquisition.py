"""Shared acquisition-axis validation before metadata conversions lose information."""

from __future__ import annotations

import numpy as np


def checked_image_key(values: np.ndarray, *, frames: int, path: str) -> np.ndarray:
    """Return validated integer labels without truncating or overflowing their values."""
    if values.shape != (frames,):
        raise ValueError(f"{path}: image_key must have shape ({frames},), not {values.shape}")
    if values.dtype.kind not in "iu":
        raise ValueError(f"{path}: image_key must use integer dtype, not {values.dtype}")
    if not np.isin(values, (0, 1, 2, 3)).all():
        raise ValueError(f"{path}: image_key values must be in {{0, 1, 2, 3}}")
    return values.astype(np.int32)


def angles_in_degrees(
    values: np.ndarray,
    units: str,
    *,
    frames: int,
    path: str,
    image_key: np.ndarray | None = None,
) -> np.ndarray:
    """Return angles in degrees; only sample frames need finite physical angles."""
    if values.shape != (frames,):
        raise ValueError(f"{path}: angles must have shape ({frames},), not {values.shape}")
    sample_angles = values if image_key is None else values[image_key == 0]
    if values.dtype.kind not in "iuf" or not np.isfinite(sample_angles).all():
        raise ValueError(f"{path}: sample angles must contain only finite real numbers")
    unit = units.strip().lower()
    if unit in {"rad", "radian", "radians"}:
        degrees = np.degrees(values.astype(np.float64))
    elif unit in {"", "deg", "degree", "degrees"}:
        degrees = values.astype(np.float64)
    else:
        raise ValueError(f"{path}: unsupported angle units {units!r}; expected degrees or radians")
    sample_degrees = degrees if image_key is None else degrees[image_key == 0]
    if not np.isfinite(sample_degrees).all() or np.any(
        np.abs(sample_degrees) > np.finfo(np.float32).max
    ):
        raise ValueError(
            f"{path}: sample angles in degrees must be finite and representable in float32"
        )
    return degrees
