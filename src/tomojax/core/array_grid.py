"""Compare array registrations before combining samples by index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import numpy as np

from tomojax.core.registration import volume_axes
from tomojax.errors import InvalidParameter
from tomojax.wire.arrays import Axis, Coordinates, Fixed, Npy, Uniform

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.core.geometry.base import Grid as VolumeGrid
    from tomojax.wire.arrays import Array, Storage

_ATOL = 1e-6  # absolute, in the axis's declared units


class Grid(msgspec.Struct, frozen=True):
    """Physical part of a worker layout. Axes follow storage order."""

    shape: tuple[int, ...]
    axes: tuple[Axis, ...]
    unit: str  # value units must also agree for subtraction
    fixed: tuple[Fixed, ...] = ()

    @classmethod
    def volume(cls, grid: VolumeGrid, unit: str) -> Grid:
        """Register a volume in canonical voxel order."""
        return cls(grid.shape, volume_axes(grid, unit), f"1/{unit}")


def published(ref: Storage) -> np.ndarray:
    """Read a published coordinate vector."""
    if not isinstance(ref, Npy):
        raise ValueError(f"coordinate vector {ref} is not a published array")
    return np.load(ref.path, mmap_mode="r", allow_pickle=False)


def coordinate_values(axis: Axis, count: int, read: Callable[[Storage], np.ndarray]) -> np.ndarray:
    """Resolve sample centers in the axis unit from either coordinate encoding."""
    c = axis.coordinates
    if isinstance(c, Coordinates):
        return np.asarray(read(c.values), dtype=np.float64)
    return c.origin + np.arange(count, dtype=np.float64) * c.spacing


def require_same_grid(
    left: Array | Grid,
    right: Array | Grid,
    *,
    field: str,
    read_left: Callable[[Storage], np.ndarray] = published,
    read_right: Callable[[Storage], np.ndarray] = published,
) -> None:
    """Require the same physical position at every index, including fixed axes."""

    def mismatch(detail: str) -> InvalidParameter:
        return InvalidParameter(
            field,
            f"{field} grid mismatch ({detail}). Choose arrays on the same grid or resample "
            "onto the target grid before combining them.",
        )

    names = tuple(a.name for a in left.axes)
    other = tuple(a.name for a in right.axes)
    if names != other:
        raise mismatch(f"axes in storage order {names} vs {other}")
    if left.shape != right.shape:
        raise mismatch(f"shape {left.shape} vs {right.shape}")
    if not same_fixed_axes(left.fixed, right.fixed):
        raise mismatch(f"fixed axes {left.fixed} vs {right.fixed}")
    for a, b, n in zip(left.axes, right.axes, left.shape, strict=True):
        if a.frame != b.frame:
            raise mismatch(f"{a.name} frame {a.frame} vs {b.frame}")
        if a.unit != b.unit:
            raise mismatch(f"{a.name} units {a.unit} vs {b.unit}")
        reason = sampling_difference(a, b, n, read_left, read_right)
        if reason:
            raise mismatch(reason)


def same_fixed_axes(left: tuple[Fixed, ...], right: tuple[Fixed, ...]) -> bool:
    """Compare physical slice locations, independent of their original indices."""
    return len(left) == len(right) and all(
        (a.name, a.frame, a.unit) == (b.name, b.frame, b.unit)
        and abs(a.coordinate - b.coordinate) <= _ATOL
        for a, b in zip(left, right, strict=True)
    )


def sampling_difference(
    a: Axis,
    b: Axis,
    n: int,
    read_left: Callable[[Storage], np.ndarray],
    read_right: Callable[[Storage], np.ndarray],
) -> str | None:
    """Describe the first differing position, checking uniform axes in constant space."""
    u, v = a.coordinates, b.coordinates
    if isinstance(u, Uniform) and isinstance(v, Uniform):
        delta = u.origin - v.origin
        step = u.spacing - v.spacing
        if not abs(delta) <= _ATOL:
            return f"{a.name} origin {u.origin} vs {v.origin} {a.unit}"
        if not (abs(step) <= _ATOL and abs(delta + max(0, n - 1) * step) <= _ATOL):
            return f"{a.name} pitch {u.spacing} vs {v.spacing} {a.unit}"
    else:
        x, y = coordinate_values(a, n, read_left), coordinate_values(b, n, read_right)
        if x.shape != (n,) or y.shape != (n,):
            return f"{a.name} coordinate vector lengths must match axis length {n}"
        different = np.flatnonzero(~(np.abs(x - y) <= _ATOL))
        if different.size:
            i = different[0]
            return f"{a.name} coordinate[{i}] {x[i]} vs {y[i]} {a.unit}"
    return None
