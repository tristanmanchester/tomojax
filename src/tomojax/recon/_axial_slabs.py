"""Bounded host storage for exactly independent axial Joseph reconstruction planes."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.geometry import ParallelGeometry, grid_volume_origin

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from tomojax.geometry import Detector, Geometry, Grid

    type Array = jax.Array | np.ndarray
    type Solve = Callable[
        [ParallelGeometry, Array, Array | None, Array | None],
        tuple[Array, dict[str, object]],
    ]


def independent_planes(geometry: Geometry, grid: Grid, detector: Detector) -> bool:
    """Whether axial Joseph sampling is one identical 2D operator per plane."""
    return (
        type(geometry) is ParallelGeometry
        and grid.nz == detector.nv
        and grid.vz == detector.dv
        and grid_volume_origin(grid)[2] == detector.center[1] - (detector.nv - 1) * detector.dv / 2
    )


def axial_slabs(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data: Array,
    *,
    initial: Array | None,
    support: Array | None,
    volume_buffers: int,
    projection_buffers: float,
    solve: Solve,
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Solve in slabs if storage is large and Joseph planes are independent.

    Call only for the Joseph model and canonical detector. Returns ``None``
    without calling ``solve`` when the geometry is not exactly separable or
    storage fits. ``solve(geometry, data, initial, support)`` receives one slab;
    it must preserve its method, subsets, bounds and iteration budget. Inputs
    stay caller-owned. The result is a new host array; no volume is written to
    disk. Buffer counts estimate this solver's peak workspace conservatively.
    """
    if not independent_planes(geometry, grid, detector) or grid.nz < 2:
        return None
    origin = grid_volume_origin(grid)
    from tomojax.backends import device_free_memory_bytes

    free = device_free_memory_bytes()
    per_row = 4 * (
        grid.nx * grid.ny * volume_buffers + data.shape[0] * detector.nu * projection_buffers
    )
    if free is None or per_row * grid.nz <= 0.5 * free:
        return None
    fitting = max(1, int(0.5 * free / per_row))
    depth = 2 ** (fitting.bit_length() - 1)
    output = np.empty((grid.nz, grid.ny, grid.nx), np.float32)
    info = {}
    for start in range(0, grid.nz, depth):
        stop = min(start + depth, grid.nz)
        nz = stop - start
        # Translate detector and volume together: every equal-sized slab has
        # the same compiled geometry, not one executable per z offset.
        local_grid = replace(grid, nz=nz, vol_origin=(*origin[:2], -(nz - 1) * grid.vz / 2))
        local_detector = replace(detector, nv=nz, center=(detector.center[0], 0.0))
        local_geometry = ParallelGeometry(
            local_grid, local_detector, cast("ParallelGeometry", geometry).angles
        )
        volume, info = solve(
            local_geometry,
            data[:, start:stop, :],
            None if initial is None else initial[start:stop, :, :],
            None if support is None else support[start:stop, :, :],
        )
        output[start:stop, :, :] = np.asarray(volume)
        # Do not retain the previous device slab while allocating the next.
        del volume
    return output, {**info, "host_slabs": True, "slices_per_batch": depth}
