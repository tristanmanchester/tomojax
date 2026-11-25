"""Physical registration and on-device selection of orthogonal preview planes."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from tomojax import progress
from tomojax.core.geometry.base import grid_volume_origin
from tomojax.wire.arrays import Axis, Column, Fixed, PoseColumn, Uniform

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from tomojax.core.geometry.base import Grid


length_unit = ContextVar("tomojax_preview_length", default="scan_length")


source_geometry = ContextVar[Any]("tomojax_preview_geometry", default=None)
output_cylinder = ContextVar[float | None]("tomojax_output_cylinder", default=None)
image_scale = ContextVar("tomojax_preview_scale", default=1.0)


@contextmanager
def registration(source: Any, *, cylinder_radius: float | None = None) -> Iterator[None]:
    """Inherit the acquisition's units and coordinates for numerical checkpoints.

    With ``cylinder_radius``, every plane is zeroed outside that cylinder about the
    rotation axis, on its own grid: an alignment's working levels included.
    """
    if not progress.previewing():
        yield
        return
    unit = recorded_length_unit(source.source.geometry_metadata) if source.source else None
    u = length_unit.set(unit or "scan_length")
    g = source_geometry.set(source.geometry)
    support = output_cylinder.set(cylinder_radius)
    try:
        yield
    finally:
        output_cylinder.reset(support)
        source_geometry.reset(g)
        length_unit.reset(u)


@contextmanager
def scaled_images(scale: float) -> Iterator[None]:
    """Restore the working image to its acquisition scale after slicing."""
    token = image_scale.set(scale)
    try:
        yield
    finally:
        image_scale.reset(token)


def recorded_length_unit(meta: Mapping[str, Any]) -> str | None:
    """Read the geometry's calibration unit from current or acquisition metadata."""
    unit = meta.get("length_unit") or (meta.get("acquisition") or {}).get("length_unit")
    return str(unit) if unit else None


def volume_axes(grid: Grid, unit: str) -> tuple[Axis, ...]:
    """Voxel centres in public storage order."""
    return tuple(
        Axis(name, "volume", unit, Uniform(start, pitch))
        for name, start, pitch in zip(
            "zyx", reversed(grid_volume_origin(grid)), (grid.vz, grid.vy, grid.vx), strict=True
        )
    )


def pose_columns(unit: str) -> tuple[Column, ...]:
    """Acquisition rotations followed by x, z and y translations."""
    return tuple(
        Column(name, "degree" if i < 3 else unit) for i, name in enumerate(get_args(PoseColumn))
    )


def planes(
    volume: Any,
    grid: Grid,
    *,
    reversed_axes: bool = False,
    observed: Any = None,
    unit: str | None = None,
) -> dict[str, progress.Sample]:
    """Index first so only three planes leave the device, including private xyz storage."""
    unit = unit or length_unit.get()
    axes = volume_axes(grid, unit)
    images = {}
    for i, axis in enumerate(axes):
        index = grid.shape[i] // 2
        key: list[slice | int] = [slice(None)] * 3
        key[2 - i if reversed_axes else i] = index
        plane = volume[tuple(key)]
        if hasattr(plane, "get"):  # a CuPy plane copies to the host
            plane = plane.get()
        plane = np.asarray(plane)
        if observed is not None:
            plane = np.where(np.asarray(observed[tuple(key)]), plane, 0)
        if reversed_axes:
            plane = plane.T
        if image_scale.get() != 1.0:
            plane = plane / np.float32(image_scale.get())
        images[axis.name] = registered_plane(plane, axes, i, index, unit)
    return images


def registered_plane(
    plane: Any,
    axes: tuple[Axis, ...],
    i: int,
    index: int,
    unit: str,
    *,
    leading_axes: tuple[Axis, ...] = (),
) -> progress.Sample:
    """Register a plane cut from a volume, retaining its missing coordinate."""
    axis = axes[i]
    coord = axis.coordinates
    assert isinstance(coord, Uniform)
    if (radius := output_cylinder.get()) is not None:
        plane = np.where(
            _inside(radius, axes, i, coord.origin + index * coord.spacing, plane), plane, 0
        )
    fixed = Fixed(axis.name, axis.frame, axis.unit, index, coord.origin + index * coord.spacing)
    return progress.Sample(
        np.array(plane, copy=True),
        leading_axes + axes[:i] + axes[i + 1 :],
        f"1/{unit}",
        "attenuation",
        (fixed,),
    )


def _inside(radius: float, axes: tuple[Axis, ...], i: int, fixed: float, plane: Any) -> np.ndarray:
    """True where ``plane`` (axis ``i`` of zyx fixed at ``fixed``) lies inside the cylinder."""

    def centres(k: int, count: int) -> np.ndarray:
        coordinates = axes[k].coordinates
        assert isinstance(coordinates, Uniform)
        return coordinates.origin + np.arange(count) * coordinates.spacing

    rows, columns = np.shape(plane)[-2:]
    x = np.asarray(fixed) if i == 2 else centres(2, columns)
    y = np.asarray(fixed) if i == 1 else centres(1, rows if i == 0 else columns)
    tolerance = 8 * np.finfo(np.float64).eps * radius
    # hypot avoids squaring physical lengths, which can underflow or overflow.
    inside = np.hypot(x[None, :], y[:, None]) if i == 0 else np.hypot(x, y)[None, :]
    return inside <= radius + tolerance


class SlabPlanes:
    """At most three planes of an axial reconstruction consumed one slab at a time."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.rows = np.empty((grid.nz, grid.nx), np.float32)
        self.columns = np.empty((grid.nz, grid.ny), np.float32)
        self.axial = None

    def add(self, start: int, stop: int, volume: Any) -> None:
        """Collect the intersections of the three planes with this slab."""
        grid = self.grid
        self.rows[start:stop] = np.asarray(volume[:, grid.ny // 2, :])
        self.columns[start:stop] = np.asarray(volume[:, :, grid.nx // 2])
        if start <= grid.nz // 2 < stop:
            self.axial = np.array(volume[grid.nz // 2 - start], copy=True)

    def images(self) -> dict[str, progress.Sample]:
        """Register all three complete planes on the original grid."""
        assert self.axial is not None
        unit = length_unit.get()
        axes = volume_axes(self.grid, unit)
        return {
            axis.name: registered_plane(data, axes, i, self.grid.shape[i] // 2, unit)
            for i, (axis, data) in enumerate(
                zip(axes, (self.axial, self.rows, self.columns), strict=True)
            )
        }


def correction_images(
    before: Any,
    after: np.ndarray,
    kept: np.ndarray,
    factor: int,
    domain: str,
    *,
    first_output_row: int = 0,
) -> dict[str, progress.Sample]:
    """Copy central native and corrected detector rows at their own registered coordinates."""
    geom = source_geometry.get()
    unit = length_unit.get() if geom is not None else "pixel"
    detector = geom.detector if geom is not None else None
    n, rows, cols = before.shape
    du, dv = (detector.du, detector.dv) if detector else (1.0, 1.0)
    u0, v0 = detector.center if detector else (0.0, 0.0)
    u0 -= (cols - 1) * du / 2
    v0 -= (rows - 1) * dv / 2
    images = {}
    for name, data, binning, ids, value_domain in (
        ("before", before, 1, np.arange(n), domain),
        ("after", after, factor, kept, "line_integrals"),
    ):
        row = data.shape[1] // 2
        window = getattr(data, "window", None)
        if window is not None:  # a lazy frame source reads only this row
            pixels = window(slice(row, row + 1), slice(None))[:][:, 0, :]
        else:
            pixels = data[:, row, :]
        if name == "after":
            row += first_output_row
        angles = None if geom is None else np.asarray(geom.angles)[ids]
        view = Axis("view", "acquisition", "index" if angles is None else "degree", Uniform(0, 1))
        coords = ids if angles is None else angles
        column = Axis("u", "detector", unit, Uniform(u0 + (binning - 1) * du / 2, du * binning))
        fixed = Fixed("v", "detector", unit, row, v0 + (row * binning + (binning - 1) / 2) * dv)
        images[name] = progress.Sample(
            np.array(pixels, copy=True),
            (view, column),
            "count" if value_domain == "counts" else "1",
            value_domain,
            (fixed,),
            coordinates=((0, coords),),
        )
    return images
