"""A reusable Fourier executor consuming owned, bounded projection-row blocks."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import closing
from typing import Any

import jax
import numpy as np

from tomojax.core.validation import validate_detector, validate_grid
from tomojax.geometry import Detector, Grid, ParallelGeometry, grid_volume_origin

from ._fourier_backend import (
    FourierPlan,
    PendingSlab,
    fourier_slab_depth,
    is_cuda_oom,
    resolve_fourier_backend,
)
from ._fourier_grid import sample_projection_rows, sorted_half_turn
from .fourier import FourierConfig
from .types import ReconstructionSlab


def _source_rows(positions: np.ndarray, rows: int) -> tuple[int, int]:
    positions = np.clip(positions, -1, rows)
    indices = np.floor(positions).astype(np.int64)
    upper = indices + 1
    used = np.concatenate(
        (
            indices[(indices >= 0) & (indices < rows)],
            upper[(upper >= 0) & (upper < rows) & (positions != indices)],
        )
    )
    if len(used):
        return int(used.min()), int(used.max()) + 1
    first = int(np.clip(indices[0], 0, rows - 1))
    return first, first + 1


def _finish[T](item: tuple[ReconstructionSlab[T], PendingSlab]) -> ReconstructionSlab[T]:
    slab, pending = item
    slab.volume = pending.wait().transpose(2, 1, 0)
    if not np.isfinite(slab.volume).all():
        raise ValueError("fourier reconstruction must be finite in FP32")
    return slab


def fourier_slabs[T](
    geometry: ParallelGeometry,
    grid: Grid,
    detector: Detector,
    read: Callable[[int, int], tuple[np.ndarray | jax.Array, T]],
    *,
    config: FourierConfig,
) -> Generator[ReconstructionSlab[T], None, None]:
    """Yield independent Fourier slabs; closing drains all outstanding device work.

    ``read`` supplies all views of a measured detector-row interval and an opaque
    producer context, retained until that output is complete. Returned arrays
    are never reused or donated. Input geometry and arrays must remain unchanged.

    Parameters
    ----------
    geometry : ParallelGeometry
        Unposed axial parallel-beam scan geometry.
    grid : Grid
        Reconstruction grid, with physical spacing in ``(x, y, z)`` order.
    detector : Detector
        Detector shape and placement in the same length unit as the grid.
    read : callable
        Reader accepting a half-open measured row interval ``(first, stop)`` and
        returning ``(projections, context)``. Projections contain every view of
        those rows; context is retained until the slab completes.
    config : FourierConfig
        Reconstruction settings.

    Yields
    ------
    ReconstructionSlab
        Owned float32 host volume slab, its output z interval, measured row
        interval, source projections and producer context.

    Raises
    ------
    ValueError
        If geometry is not unposed axial parallel geometry, or shapes, sampling
        or settings are incompatible with this reconstruction.
    """
    if type(geometry) is not ParallelGeometry:
        raise ValueError("reconstruct_slabs requires unposed axial ParallelGeometry")
    validate_grid(grid, "reconstruct_slabs grid")
    validate_detector(detector, "reconstruct_slabs detector")
    backend = resolve_fourier_backend(config.backend)
    depth = fourier_slab_depth(grid, detector, config.slices_per_batch, backend)
    origin = grid_volume_origin(grid)
    if not np.isfinite([*origin, *detector.center]).all():
        raise ValueError("Fourier grid and detector placement must be finite")
    order, flipped, angles = sorted_half_turn(np.asarray(geometry.angles))
    positions = (origin[2] + np.arange(grid.nz) * grid.vz - detector.center[1]) / detector.dv + (
        detector.nv - 1
    ) / 2
    start = 0
    while start < grid.nz:
        plan = None
        try:
            plan = FourierPlan(
                grid,
                detector,
                len(order),
                flipped,
                angles,
                depth,
                backend,
                async_transfers=backend == "cupy",
                transpose_download=True,
            )
            with closing(_run_plan(plan, read, order, positions, start)) as completed:
                for slab in completed:
                    start = slab.stop
                    yield slab
        except Exception as error:
            if backend != "cupy" or depth == 1 or not is_cuda_oom(error):
                raise
            depth = max(1, depth // 2)
        finally:
            if plan is not None:
                plan.synchronize()
            plan = None


def _run_plan[T](
    plan: FourierPlan,
    read: Callable[[int, int], tuple[np.ndarray | jax.Array, T]],
    order: np.ndarray,
    positions: np.ndarray,
    start: int,
) -> Generator[ReconstructionSlab[T], None, None]:
    pending: list[tuple[ReconstructionSlab[T], PendingSlab]] = []
    device_order: Any = order
    if plan.cuda:
        assert plan.compute is not None
        with plan.compute:
            device_order = plan.xp.asarray(order)
    try:
        for at in range(start, plan.grid.nz, plan.depth):
            stop = min(at + plan.depth, plan.grid.nz)
            requested = positions[at:stop]
            first, last = _source_rows(requested, plan.detector.nv)
            data, context = read(first, last)
            slab = ReconstructionSlab(at, stop, slice(first, last), data, np.empty(0), context)
            if plan.cuda:
                job = plan.enqueue_rows(data, device_order, requested - first)
                pending.append((slab, job))
                if len(pending) >= 2:
                    yield _finish(pending.pop(0))
            else:
                raw = sample_projection_rows(np.asarray(data), order, requested - first)
                if stop - at < plan.depth:
                    raw = np.pad(raw, ((0, plan.depth - (stop - at)), (0, 0), (0, 0)))
                slab.volume = plan.reconstruct(raw, count=stop - at).transpose(2, 1, 0)
                if not np.isfinite(slab.volume).all():
                    raise ValueError("fourier reconstruction must be finite in FP32")
                yield slab
        for item in pending:
            yield _finish(item)
    finally:
        plan.synchronize()
        pending.clear()
