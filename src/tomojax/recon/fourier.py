"""Fourier-slice reconstruction for uniform parallel scans with host storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING, Literal

import numpy as np

from tomojax.core.geometry.cone import require_parallel_beam
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.geometry import ParallelGeometry, grid_volume_origin

from ._fourier_backend import FourierPlan
from ._fourier_grid import sample_projection_rows, uniform_half_turn
from ._host_arrays import validate_host_arrays

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from tomojax.geometry import Detector, Grid

    from ._fourier_backend import _PendingSlab

_HOST_PIPELINE_MIN_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True, kw_only=True)
class FourierConfig:
    """Choose a portable NumPy reference or CUDA execution with fixed axial slabs.

    ``cupy`` uses single-precision FFTs and an original CUDA interpolation
    kernel. It requires the optional ``fourier-cuda12`` extra. ``numpy`` uses
    double-precision FFTs and a vectorized interpolation reference. Both return
    float32 host arrays. Slab size bounds device volume storage; FFT plans and
    allocator caches consume additional memory. Large CUDA calls overlap host
    row preparation, device transfers, FFT execution and output writes, retaining
    at most two pending device slabs. A bounded cache retains immutable small
    geometry arrays; each call owns its working buffers.
    """

    slices_per_batch: int = 16
    backend: Literal["numpy", "cupy"] = "numpy"


def fourier_reconstruct(
    geometry: ParallelGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    *,
    config: FourierConfig | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct a uniform half-turn parallel scan using the Fourier slice theorem.

    Projections have shape ``(views, nv, nu)`` in physical line-integral units.
    Angles may be reordered, reversed, offset or replaced by opposing views,
    but must be uniformly spaced and unique modulo 180 degrees. Only built-in
    ``ParallelGeometry`` is supported. Detector offsets, anisotropic voxel
    sizes, shifted/cropped output grids and fractional detector-v rows retain
    their physical coordinates.

    This approximate inverse uses six-point Kaiser--Bessel radial interpolation,
    linear angular interpolation, a detector-Nyquist disk cutoff and a padded
    Cartesian inverse FFT. Missing detector data are zero. Truncated scans and
    insufficient angular sampling can produce artifacts; no amplitude fitting,
    positivity clipping or truncation correction is applied.

    Input and output stay in NumPy arrays or memmaps, with float32 output in
    ``(nx, ny, nz)`` order. Their storage must not overlap. Completed slabs are
    written immediately, so a later failure may leave partial output. Input
    storage must remain unchanged during the call. This routine is not
    differentiable and does not replace the default FBP reconstruction.
    """
    require_parallel_beam(geometry, "fourier_reconstruct")
    cfg = FourierConfig() if config is None else config
    slices = operator.index(cfg.slices_per_batch)
    if slices < 1:
        raise ValueError("fourier_reconstruct: slices_per_batch must be positive")
    if cfg.backend not in {"numpy", "cupy"}:
        raise ValueError("fourier_reconstruct: backend must be 'numpy' or 'cupy'")
    if type(geometry) is not ParallelGeometry:
        raise ValueError("fourier_reconstruct: requires built-in ParallelGeometry")
    shape = validate_grid(grid, "fourier_reconstruct grid")
    nviews, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fourier_reconstruct projections"
    )
    origin = grid_volume_origin(grid)
    if not np.isfinite([*origin, *detector.det_center]).all():
        raise ValueError("fourier_reconstruct: grid and detector placement must be finite")
    order, flipped, theta0 = uniform_half_turn(np.asarray(geometry.thetas_deg))
    output = validate_host_arrays(projections, out, shape, "fourier_reconstruct")
    depth = min(slices, grid.nz)
    host_bytes = 4 * (grid.nx * grid.ny * grid.nz + nviews * detector.nu * grid.nz)
    overlap = cfg.backend == "cupy" and grid.nz > depth and host_bytes >= _HOST_PIPELINE_MIN_BYTES
    plan = FourierPlan(
        grid, detector, nviews, flipped, theta0, depth, cfg.backend, async_transfers=overlap
    )
    positions = (
        origin[2] + np.arange(grid.nz) * grid.vz - detector.det_center[1]
    ) / detector.dv + (detector.nv - 1) / 2
    if not np.isfinite(positions).all():
        raise ValueError("fourier_reconstruct: detector-row coordinates must be finite")

    def prepare(start: int) -> np.ndarray:
        stop = min(start + depth, grid.nz)
        data = sample_projection_rows(projections, order, positions[start:stop])
        if not np.isfinite(data).all():
            raise ValueError("fourier_reconstruct: sampled projection rows must be finite in FP32")
        if stop - start < depth:
            data = np.pad(data, ((0, depth - (stop - start)), (0, 0), (0, 0)))
        return data

    def write(start: int, volume: np.ndarray) -> None:
        stop = min(start + depth, grid.nz)
        if not np.isfinite(volume).all():
            raise ValueError("fourier_reconstruct: reconstructed slab must be finite in FP32")
        output[:, :, start:stop] = volume.transpose(1, 2, 0)

    if overlap:
        _run_host_pipeline(plan, grid.nz, depth, prepare, write)
    else:
        for start in range(0, grid.nz, depth):
            count = min(depth, grid.nz - start)
            write(start, plan.reconstruct(prepare(start))[:count])
    return output


def _run_host_pipeline(
    plan: FourierPlan,
    nz: int,
    depth: int,
    prepare: Callable[[int], np.ndarray],
    write: Callable[[int, np.ndarray], None],
) -> None:
    """Overlap bounded staging and transfers; CUDA stays on the calling thread."""
    pending: list[tuple[int, _PendingSlab]] = []
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tomojax-fourier-read") as reader,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tomojax-fourier-write") as writer,
    ):
        written: Future[None] | None = None

        def finish(start: int, job: _PendingSlab) -> None:
            nonlocal written
            volume = job.wait()[: min(depth, nz - start)]
            if written is not None:
                written.result()
            written = writer.submit(write, start, volume)

        prepared = reader.submit(prepare, 0)
        try:
            for start in range(0, nz, depth):
                try:
                    data = prepared.result()
                except BaseException:
                    # A later invalid input must still commit completed slabs.
                    for first, job in pending:
                        finish(first, job)
                    pending.clear()
                    raise
                stop = min(start + depth, nz)
                if stop < nz:
                    prepared = reader.submit(prepare, stop)
                pending.append((start, plan.enqueue(data)))
                if len(pending) >= 2:
                    finish(*pending.pop(0))
            for first, job in pending:
                finish(first, job)
            pending.clear()
            if written is not None:
                written.result()
        finally:
            # Drain GPU work even if an output write fails. Buffer owners stay
            # alive through their completion event; executors join host workers.
            for _, job in pending:
                job.wait()
