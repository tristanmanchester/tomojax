"""Utilities for reconciling internal (xyz) and disk (zyx) axis orders."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np

from tomojax.core.geometry.base import Grid, GridDict

if TYPE_CHECKING:
    from collections.abc import Sequence

INTERNAL_VOLUME_AXES = "xyz"
DISK_VOLUME_AXES = "zyx"
VOLUME_AXES_ATTR = "volume_axes_order"
CORE_X_AXIS = 0
CORE_Y_AXIS = 1
CORE_Z_AXIS = 2
GridLike = Grid | GridDict


def _norm_axes(axes: str) -> str:
    axes = axes.lower()
    if len(axes) != 3:
        raise ValueError(f"axes '{axes}' must be length 3")
    if set(axes) != {"x", "y", "z"}:
        raise ValueError(f"axes '{axes}' must be a permutation of xyz")
    return axes


def axes_to_perm(src: str, dst: str) -> tuple[int, int, int]:
    """Return permutation bringing `src` axis order into `dst` order.

    Parameters
    ----------
    src, dst : str
        Source and destination permutations of xyz, case-insensitive.

    Returns
    -------
    tuple of int
        Axis permutation for numpy.transpose or jax.numpy.transpose.

    Raises
    ------
    ValueError
        If an axis order is not a permutation of xyz.
    """
    s = _norm_axes(src)
    d = _norm_axes(dst)
    return cast("tuple[int, int, int]", tuple(s.index(axis) for axis in d))


@overload
def transpose_volume(volume: np.ndarray, src: str, dst: str) -> np.ndarray: ...


@overload
def transpose_volume(volume: object, src: str, dst: str) -> object: ...


def transpose_volume(volume: object, src: str, dst: str) -> object:
    """Transpose a volume from `src` axis order to `dst` order.

    Keeps numpy arrays as numpy and JAX arrays as JAX when possible.

    Parameters
    ----------
    volume : array-like
        Three-dimensional volume to transpose.
    src, dst : str
        Source and destination permutations of xyz, case-insensitive.

    Returns
    -------
    numpy.ndarray or jax.Array
        Volume in the requested order, preserving NumPy/JAX storage where possible.

    Raises
    ------
    ValueError
        If axis orders or volume dimensions are invalid.
    """
    perm = axes_to_perm(src, dst)

    if isinstance(volume, np.ndarray):
        if perm == (0, 1, 2):
            return cast("object", volume)
        return cast("object", volume.transpose(perm))

    import jax
    import jax.numpy as jnp

    if isinstance(volume, jax.Array):
        return volume if perm == (0, 1, 2) else jnp.transpose(volume, axes=perm)

    arr = np.asarray(volume)
    if perm == (0, 1, 2):
        return cast("object", arr)
    return cast("object", np.transpose(arr, axes=perm))


def _grid_dims(grid: GridLike | None) -> tuple[int, int, int] | None:
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


def infer_disk_axes(vol_shape: Sequence[int], grid: GridLike | None) -> str | None:
    """Infer on-disk axis order using grid heuristics.

    Returns "xyz", "zyx", or None if ambiguous.

    Parameters
    ----------
    vol_shape : sequence of int
        On-disk volume dimensions.
    grid : Grid, dict or None
        Grid dimensions used to distinguish xyz from zyx.

    Returns
    -------
    str or None
        Inferred xyz or zyx order; None when missing, inconsistent or ambiguous.
    """
    if len(vol_shape) != 3:
        return None
    dims = _grid_dims(grid)
    if dims is None:
        return None
    shape = tuple(int(s) for s in vol_shape)
    matches_xyz = shape == dims
    matches_zyx = shape == dims[::-1]
    if matches_xyz and matches_zyx:
        return None
    if matches_xyz:
        return INTERNAL_VOLUME_AXES
    if matches_zyx:
        return DISK_VOLUME_AXES
    # Without grid metadata, inferring axis order from shape alone is unreliable.
    return None


__all__ = [
    "CORE_X_AXIS",
    "CORE_Y_AXIS",
    "CORE_Z_AXIS",
    "DISK_VOLUME_AXES",
    "INTERNAL_VOLUME_AXES",
    "VOLUME_AXES_ATTR",
    "axes_to_perm",
    "infer_disk_axes",
    "transpose_volume",
]
