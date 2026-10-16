"""Fourier-slice reconstruction for measured half-turn parallel scans with host storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from tomojax.core.geometry.cone import require_parallel_beam
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.geometry import ParallelGeometry, grid_volume_origin

from ._fourier_backend import FourierPlan, fourier_slab_depth, is_cuda_oom, resolve_fourier_backend
from ._fourier_grid import sample_projection_rows, sorted_half_turn
from ._fourier_rows import ProjectionRows, prepare_projection_rows
from ._host_arrays import validate_host_arrays

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from tomojax.geometry import Detector, Grid

    from ._fourier_backend import PendingSlab

_HOST_PIPELINE_MIN_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True, kw_only=True)
class FourierConfig:
    """Choose a portable NumPy reference or CUDA execution with fixed axial slabs.

    ``auto`` uses CuPy when its current CUDA device matches the selected JAX
    device and NumPy otherwise, including when CuPy is not installed. Explicit
    ``numpy`` and ``cupy`` never switch backends or change the current device.
    ``cupy`` uses single-precision FFTs and an original CUDA interpolation
    kernel. It requires the optional ``fourier-cuda12`` extra. ``numpy`` uses
    double-precision FFTs and a vectorized interpolation reference. Both return
    float32 host arrays. ``slices_per_batch=None`` chooses up to 16 slices,
    targeting a 256 MiB padded CUDA FP32 plane stack (at least one slice).
    NumPy defaults to 16; an explicit integer fixes the requested depth.
    CUDA allocation exhaustion still halves it down to one slice. FFT plans and
    allocator caches consume additional memory. Large CUDA calls overlap host
    row preparation, device transfers, FFT execution and output writes, retaining
    at most two pending device slabs. A bounded cache retains immutable
    geometry arrays; each call owns its working buffers.
    Fractional detector rows are interpolated on CUDA from at most two measured
    source rows per requested slice; entire projection stacks are not uploaded.
    """

    slices_per_batch: int | None = None
    backend: Literal["auto", "numpy", "cupy"] = "auto"


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

    Input and output stay in NumPy arrays or memmaps, with float32 output in
    ``(nx, ny, nz)`` order. CUDA downloads match Fortran-order output storage.
    New outputs are plane-contiguous; supplied output storage is preserved.
    Their storage must not overlap. Completed slabs are
    written immediately, so a later failure may leave partial output. Input
    storage must remain unchanged during the call. This routine is not
    differentiable; use the matched projector pair for iterative solves.
    """
    require_parallel_beam(geometry, "fourier_reconstruct")
    cfg = FourierConfig() if config is None else config
    backend = resolve_fourier_backend(cfg.backend)
    if type(geometry) is not ParallelGeometry:
        raise ValueError("fourier_reconstruct: requires built-in ParallelGeometry")
    shape = validate_grid(grid, "fourier_reconstruct grid")
    nviews, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fourier_reconstruct projections"
    )
    origin = grid_volume_origin(grid)
    if not np.isfinite([*origin, *detector.center]).all():
        raise ValueError("fourier_reconstruct: grid and detector placement must be finite")
    depth = fourier_slab_depth(grid, detector, cfg.slices_per_batch, backend)
    order, flipped, angles = sorted_half_turn(np.asarray(geometry.angles))
    output = validate_host_arrays(projections, out, shape, "fourier_reconstruct", order="F")
    transpose_download = backend == "cupy" and output.flags.f_contiguous
    host_bytes = 4 * (grid.nx * grid.ny * grid.nz + nviews * detector.nu * grid.nz)
    positions = (origin[2] + np.arange(grid.nz) * grid.vz - detector.center[1]) / detector.dv + (
        detector.nv - 1
    ) / 2
    if not np.isfinite(positions).all():
        raise ValueError("fourier_reconstruct: detector-row coordinates must be finite")

    def prepare(start: int) -> np.ndarray | ProjectionRows:
        rows = positions[start : start + depth]
        return _prepare_slab(projections, order, rows, depth, backend)

    def write(start: int, volume: np.ndarray) -> None:
        stop = min(start + depth, grid.nz)
        if not np.isfinite(volume).all():
            raise ValueError("fourier_reconstruct: reconstructed slab must be finite in FP32")
        axes = (2, 1, 0) if transpose_download else (1, 2, 0)
        output[:, :, start:stop] = volume.transpose(axes)

    while True:
        try:
            overlap = (
                backend == "cupy" and grid.nz > depth and host_bytes >= _HOST_PIPELINE_MIN_BYTES
            )
            plan = FourierPlan(
                grid,
                detector,
                nviews,
                flipped,
                angles,
                depth,
                backend,
                async_transfers=overlap,
                transpose_download=transpose_download,
            )
            if overlap:
                _run_host_pipeline(plan, grid.nz, depth, prepare, write)
            else:
                for start in range(0, grid.nz, depth):
                    count = min(depth, grid.nz - start)
                    write(start, plan.reconstruct(prepare(start), count=count))
            return output
        except Exception as error:
            if backend != "cupy" or depth == 1 or not is_cuda_oom(error):
                raise
            # The pipeline has drained its transfer events. Drop this attempt's
            # buffers and the exception traceback before allocating a smaller
            # slab; rewriting from slice zero also replaces any completed output.
            plan = None
        depth = max(1, depth // 2)


def _prepare_slab(
    projections: np.ndarray,
    order: np.ndarray,
    positions: np.ndarray,
    depth: int,
    backend: Literal["numpy", "cupy"],
) -> np.ndarray | ProjectionRows:
    """Finite measured rows for a fixed-depth FFT, padded only after row sampling."""
    data = (prepare_projection_rows if backend == "cupy" else sample_projection_rows)(
        projections, order, positions
    )
    if isinstance(data, ProjectionRows):
        return data
    if not np.isfinite(data).all():
        raise ValueError("fourier_reconstruct: sampled projection rows must be finite in FP32")
    if len(positions) < depth:
        data = np.pad(data, ((0, depth - len(positions)), (0, 0), (0, 0)))
    return data


def _run_host_pipeline(
    plan: FourierPlan,
    nz: int,
    depth: int,
    prepare: Callable[[int], np.ndarray | ProjectionRows],
    write: Callable[[int, np.ndarray], None],
) -> None:
    """Overlap bounded staging and transfers; CUDA stays on the calling thread."""
    pending: list[tuple[int, PendingSlab]] = []
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tomojax-fourier-read") as reader,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tomojax-fourier-write") as writer,
    ):
        written: Future[None] | None = None

        def finish(start: int, job: PendingSlab) -> None:
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
