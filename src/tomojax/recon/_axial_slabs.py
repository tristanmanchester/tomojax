"""Bounded host storage for exactly independent axial Joseph reconstruction planes."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax import progress
from tomojax.core import checkpoint
from tomojax.core.registration import planes
from tomojax.geometry import ParallelGeometry, grid_volume_origin

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from tomojax.geometry import Detector, Geometry, Grid

    from .mlem import MLEMConfig
    from .sirt import SIRTConfig

    type Array = jax.Array | np.ndarray
    type Solve = Callable[
        [ParallelGeometry, Array, Array | None, Array | None],
        tuple[Array, dict[str, object]],
    ]


def independent_planes(geometry: Geometry, grid: Grid, detector: Detector) -> bool:
    """Whether axial Joseph sampling is one identical 2D operator per plane."""
    return (
        type(geometry) is ParallelGeometry
        and not geometry.detector_roll
        and grid.nz == detector.nv
        and grid.vz == detector.dv
        and grid_volume_origin(grid)[2] == detector.center[1] - (detector.nv - 1) * detector.dv / 2
    )


def axial_workspace(cfg: SIRTConfig | MLEMConfig, volume: int, projections: int) -> float:
    """FP32 workspace in bytes for SIRT or EM over the given volume and projections."""
    from .sirt import SIRTConfig

    if isinstance(cfg, SIRTConfig):
        return (3 + min(cfg.subsets, 16) + 2 * cfg.accelerated) * volume + (
            2 + 2 / cfg.subsets
        ) * projections
    # EM keeps four volume buffers; axial sensitivities have the shape of one plane.
    return 4 * volume + 2 * projections


def axial_slab_depth(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    views: int,
    cfg: SIRTConfig | MLEMConfig,
    free: int | None,
) -> int | None:
    """Depth of independent z slabs under the half-memory policy, or None for a resident solve."""
    if free is None or grid.nz < 2 or not independent_planes(geometry, grid, detector):
        return None
    per_row = axial_workspace(cfg, 4 * grid.nx * grid.ny, 4 * views * detector.nu)
    if per_row * grid.nz <= 0.5 * free:
        return None
    fitting = max(1, int(0.5 * free / per_row))
    return 2 ** (fitting.bit_length() - 1)


def axial_slabs(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data: Array,
    *,
    initial: Array | None,
    support: Array | None,
    config: SIRTConfig | MLEMConfig,
    solve: Solve,
    summarize: Callable[[checkpoint.Slabs], dict[str, object]] | None = None,
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Solve in slabs if storage is large and Joseph planes are independent.

    Call only for the Joseph model and canonical detector. Returns ``None``
    without calling ``solve`` when the geometry is not exactly separable or
    storage fits. ``solve(geometry, data, initial, support)`` receives one slab;
    it must preserve its method, subsets, bounds and iteration budget. Inputs
    stay caller-owned. The result is a new host array; no volume is written to
    disk. Buffer counts estimate this solver's peak workspace conservatively.
    """
    from tomojax.backends import device_free_memory_bytes

    depth = axial_slab_depth(
        geometry, grid, detector, data.shape[0], config, device_free_memory_bytes()
    )
    if summarize is not None and checkpoint.active():
        # Statistical stops depend on the rows solved together. Restore that
        # partition even if free device memory has changed since interruption.
        with checkpoint.loop("partition") as partition_save:
            saved = partition_save.slab_depth(depth or grid.nz)
            depth = None if saved == grid.nz else saved
    if depth is None:
        return None
    origin = grid_volume_origin(grid)
    output = np.empty((grid.nz, grid.ny, grid.nx), np.float32)
    info = {}
    from ._chunks import partial_sweep

    with checkpoint.slabs(output) as finished:
        for start in range(0, grid.nz, depth):
            progress.check_canceled()
            stop = min(start + depth, grid.nz)
            if finished.finished(start, stop):
                continue
            nz = stop - start
            # Translate detector and volume together: every equal-sized slab has
            # the same compiled geometry, not one executable per z offset.
            local_grid = replace(grid, nz=nz, vol_origin=(*origin[:2], -(nz - 1) * grid.vz / 2))
            local_detector = replace(detector, nv=nz, center=(detector.center[0], 0.0))
            local_geometry = ParallelGeometry(
                local_grid, local_detector, cast("ParallelGeometry", geometry).angles
            )
            partition = partial_sweep.set(partial_sweep.get() or stop < grid.nz)
            try:
                with checkpoint.scope(f"z{start}"):
                    volume, info = solve(
                        local_geometry,
                        data[:, start:stop, :],
                        None if initial is None else initial[start:stop, :, :],
                        None if support is None else support[start:stop, :, :],
                    )
            finally:
                partial_sweep.reset(partition)
            output[start:stop, :, :] = np.asarray(volume)
            # Do not retain the previous device slab while allocating the next.
            del volume
            finished.add(start, stop, info=info if summarize is not None else None)
        if summarize is not None:
            info = summarize(finished)
    if info.get("effective_iterations") and partial_sweep.get() is None:
        progress.preview(lambda: planes(output, grid), final=True)
    return output, {**info, "host_slabs": True, "slices_per_batch": depth}
