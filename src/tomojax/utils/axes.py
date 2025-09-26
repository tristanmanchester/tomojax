"""Utilities for reconciling internal (xyz) and disk (zyx) axis orders."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - JAX might be absent in build docs
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    if hasattr(jax, "Array"):
        _JAX_ARRAY_TYPES = (jax.Array,)  # type: ignore[attr-defined]
    else:  # Fallback for older JAX
        from jax.interpreters.xla import DeviceArray  # type: ignore

        _JAX_ARRAY_TYPES = (DeviceArray,)  # type: ignore[assignment]
except Exception:  # pragma: no cover - non-JAX contexts
    jnp = None  # type: ignore[assignment]
    _JAX_ARRAY_TYPES: Tuple[type, ...] = ()


INTERNAL_VOLUME_AXES = "xyz"
DISK_VOLUME_AXES = "zyx"
VOLUME_AXES_ATTR = "volume_axes_order"


def _norm_axes(axes: str) -> str:
    axes = axes.lower()
    if len(axes) != 3:
        raise ValueError(f"axes '{axes}' must be length 3")
    if set(axes) != {"x", "y", "z"}:
        raise ValueError(f"axes '{axes}' must be a permutation of xyz")
    return axes


def axes_to_perm(src: str, dst: str) -> Tuple[int, int, int]:
    """Return permutation bringing `src` axis order into `dst` order."""

    s = _norm_axes(src)
    d = _norm_axes(dst)
    return tuple(s.index(axis) for axis in d)


def transpose_volume(volume: np.ndarray, src: str, dst: str):
    """Transpose a volume from `src` axis order to `dst` order.

    Keeps numpy arrays as numpy and JAX arrays as JAX when possible.
    """

    perm = axes_to_perm(src, dst)
    if perm == (0, 1, 2):
        return volume

    if _JAX_ARRAY_TYPES and isinstance(volume, _JAX_ARRAY_TYPES):  # pragma: no cover - exercised in GPU envs
        assert jnp is not None
        return jnp.transpose(volume, axes=perm)

    if isinstance(volume, np.ndarray):
        return np.transpose(volume, axes=perm)

    arr = np.asarray(volume)
    return np.transpose(arr, axes=perm)


def _grid_dims(grid: Optional[object]) -> Optional[Tuple[int, int, int]]:
    if grid is None:
        return None
    if hasattr(grid, "nx") and hasattr(grid, "ny") and hasattr(grid, "nz"):
        try:
            return (int(grid.nx), int(grid.ny), int(grid.nz))  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            return None
    if isinstance(grid, dict):
        try:
            return (int(grid["nx"]), int(grid["ny"]), int(grid["nz"]))
        except Exception:
            return None
    return None


def is_shape_xyz(vol_shape: Sequence[int], grid: Optional[object]) -> bool:
    dims = _grid_dims(grid)
    if dims is None:
        return False
    if len(vol_shape) != 3:
        return False
    return tuple(int(s) for s in vol_shape) == dims


def is_shape_zyx(vol_shape: Sequence[int], grid: Optional[object]) -> bool:
    dims = _grid_dims(grid)
    if dims is None:
        return False
    if len(vol_shape) != 3:
        return False
    return tuple(int(s) for s in vol_shape) == (dims[2], dims[1], dims[0])


def infer_disk_axes(vol_shape: Sequence[int], grid: Optional[object]) -> Optional[str]:
    """Infer on-disk axis order using grid heuristics.

    Returns "xyz", "zyx", or None if ambiguous.
    """

    if len(vol_shape) != 3:
        return None
    if is_shape_xyz(vol_shape, grid):
        return INTERNAL_VOLUME_AXES
    if is_shape_zyx(vol_shape, grid):
        return DISK_VOLUME_AXES
    # If grid unknown and shape is non-cubic, try to guess by assuming nz == grid.nz
    if grid is None:
        a, b, c = (int(s) for s in vol_shape)
        if a != c:
            # Heuristic: treat longest edge as x (common for laminography volumes)
            if a > c:
                return INTERNAL_VOLUME_AXES
            return DISK_VOLUME_AXES
    return None


__all__ = [
    "INTERNAL_VOLUME_AXES",
    "DISK_VOLUME_AXES",
    "VOLUME_AXES_ATTR",
    "axes_to_perm",
    "transpose_volume",
    "infer_disk_axes",
    "is_shape_xyz",
    "is_shape_zyx",
]
