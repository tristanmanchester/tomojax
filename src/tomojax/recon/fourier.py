"""Fourier-slice reconstruction for measured half-turn parallel scans with host storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING, Literal

import numpy as np

from tomojax.core.geometry.cone import require_parallel_beam
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.geometry import ParallelGeometry, grid_volume_origin

from ._fourier_backend import FourierPlan, is_cuda_oom, release_cuda_fft_cache
from ._fourier_grid import sample_projection_rows, sorted_half_turn
from ._fourier_rows import ProjectionRows, prepare_projection_rows
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
    float32 host arrays. Slab size bounds device volume storage and is halved
    after CUDA allocation exhaustion, down to one slice. Reclaiming this
    thread/device's cached FFT workspaces permits one retry at the same size.
    FFT plans and allocator caches consume additional memory. Large CUDA calls overlap host
    row preparation, device transfers, FFT execution and output writes, retaining
    at most two pending device slabs. A bounded cache retains immutable
    geometry arrays; each call owns its working buffers.
    Fractional detector rows are interpolated on CUDA from at most two measured
    source rows per requested slice; entire projection stacks are not uploaded.
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
    """Reconstruct a half-turn parallel scan using the Fourier slice theorem.

    Projections have shape ``(views, nv, nu)`` in physical line-integral units.
    Angles may be reordered, reversed, offset or replaced by opposing views,
    with uniform or nonuniform spacing. Redundant half-turn directions are
    averaged in physical Fourier coordinates after detector phase/orientation.
    Detector offsets, anisotropic voxels, shifted/cropped grids and fractional
    detector-v rows retain their physical coordinates.

    This approximate inverse uses six-point Kaiser--Bessel radial interpolation,
    linear interpolation in measured angle, a detector-Nyquist disk cutoff and
    a padded Cartesian inverse FFT. DC uses periodic angular trapezoid weights.
    Missing detector data are zero. Truncation or angular undersampling can cause
    artifacts; no amplitude fitting, clipping or truncation correction is applied.

    Input and output stay in NumPy arrays or memmaps; FP32 output is
    ``(nx, ny, nz)`` with contiguous axial slices by default. Caller output
    strides are retained. Storage must not overlap or change during the call.
    Completed slabs are written immediately; later failures may leave partial
    output. This nondifferentiable routine does not replace default FBP.
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
    if not np.isfinite([*origin, *detector.center]).all():
        raise ValueError("fourier_reconstruct: grid and detector placement must be finite")
    order, flipped, angles = sorted_half_turn(np.asarray(geometry.angles))
    output = _fourier_output(projections, out, shape)
    depth = min(slices, grid.nz)
    host_bytes = 4 * (grid.nx * grid.ny * grid.nz + nviews * detector.nu * grid.nz)
    positions = (origin[2] + np.arange(grid.nz) * grid.vz - detector.center[1]) / detector.dv + (
        detector.nv - 1
    ) / 2
    if not np.isfinite(positions).all():
        raise ValueError("fourier_reconstruct: detector-row coordinates must be finite")

    def prepare(start: int) -> np.ndarray | ProjectionRows:
        stop = min(start + depth, grid.nz)
        data = (prepare_projection_rows if cfg.backend == "cupy" else sample_projection_rows)(
            projections, order, positions[start:stop]
        )
        if isinstance(data, ProjectionRows):
            return data
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

    retried = False
    while True:
        plan = None
        try:
            overlap = (
                cfg.backend == "cupy" and grid.nz > depth and host_bytes >= _HOST_PIPELINE_MIN_BYTES
            )
            plan = FourierPlan(
                grid, detector, nviews, flipped, angles, depth, cfg.backend, async_transfers=overlap
            )
            _run_slabs(plan, grid.nz, depth, prepare, write, overlap=overlap)
            return output
        except Exception as error:
            if cfg.backend != "cupy" or not is_cuda_oom(error):
                raise
            # Transfer events are drained. Retry from zero to replace dirty slabs.
            plan = None
            retry = _retry_after_exhaustion(depth, retried=retried)
            if retry is None:
                raise
            depth, retried = retry


def _fourier_output(
    projections: np.ndarray, out: np.ndarray | None, shape: tuple[int, int, int]
) -> np.ndarray:
    """Keep default axial slab writes contiguous without changing caller storage."""
    output = validate_host_arrays(projections, out, shape, "fourier_reconstruct")
    if out is None:
        # Reinterpret the uninitialized owner; allocate no second volume.
        return output.reshape((shape[2], shape[0], shape[1])).transpose(1, 2, 0)
    return output


def _retry_after_exhaustion(depth: int, *, retried: bool) -> tuple[int, bool] | None:
    """Evict stale FFT workspaces once at each depth, then reduce axial storage."""
    if release_cuda_fft_cache() and not retried:
        return depth, True
    if depth > 1:
        return max(1, depth // 2), False
    return None


def _run_slabs(
    plan: FourierPlan,
    nz: int,
    depth: int,
    prepare: Callable[[int], np.ndarray | ProjectionRows],
    write: Callable[[int, np.ndarray], None],
    *,
    overlap: bool,
) -> None:
    """Execute complete slabs either serially or through the bounded pipeline."""
    if overlap:
        _run_host_pipeline(plan, nz, depth, prepare, write)
    else:
        for start in range(0, nz, depth):
            count = min(depth, nz - start)
            write(start, plan.reconstruct(prepare(start), count=count))


def _run_host_pipeline(
    plan: FourierPlan,
    nz: int,
    depth: int,
    prepare: Callable[[int], np.ndarray | ProjectionRows],
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
                pending.append((start, plan.enqueue(data, count=stop - start)))
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
