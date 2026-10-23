"""Fourier-slice reconstruction for measured half-turn parallel scans with host storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
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

    from tomojax.geometry import Detector, Grid, ScanGeometry

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

    Parameters
    ----------
    slices_per_batch : int | None, optional
        Positive slab depth, or None to choose automatically as described above.
    backend : Literal['auto', 'numpy', 'cupy'], optional
        Execution backend: "auto", "numpy" or "cupy". Explicit choices never change
        backends. Default is 'auto'.
    """

    slices_per_batch: int | None = None
    backend: Literal["auto", "numpy", "cupy"] = "auto"


def fourier_reconstruct(
    geometry: ScanGeometry,
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
    Per-view translations are honored in either detector or object coordinates.
    Rotational pose corrections and tilted axes are not supported.
    Detector offsets, anisotropic voxels, shifted/cropped grids and fractional
    detector-v rows retain their physical coordinates.

    This approximate inverse uses six-point Kaiser--Bessel radial interpolation,
    linear interpolation in measured angle, a detector-Nyquist disk cutoff and
    a padded Cartesian inverse FFT. DC uses periodic angular trapezoid weights.
    Missing detector data are zero. Truncation or angular undersampling can cause
    artifacts; no amplitude fitting, clipping or truncation correction is applied.

    Input and output stay in NumPy arrays or memmaps, with float32 output in
    ``(nz, ny, nx)`` order. New outputs are C-contiguous for downstream volume
    operations; supplied output storage is preserved.
    Their storage must not overlap. Completed slabs are
    written immediately, so a later failure may leave partial output. Input
    storage must remain unchanged during the call. This routine is not
    differentiable; use the matched projector pair for iterative solves.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry, including view angles and any per-view poses.
    grid : Grid
        Reconstruction grid with named physical dimensions and voxel spacing.
    detector : Detector
        Detector dimensions and placement in the same length unit as the grid.
    projections : array-like
        Measured line integrals with shape ``(views, detector.nv, detector.nu)``.
    config : FourierConfig or None, optional
        Solver settings. None uses FourierConfig defaults.
    out : numpy.ndarray or None, optional
        Writable float32 host storage of shape ``(nz, ny, nx)``, including a memmap.
        Must not overlap input storage. None allocates a new output.

    Returns
    -------
    numpy.ndarray
        Float32 reconstruction of shape ``(nz, ny, nx)``. Returns out itself when supplied.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.
    ImportError
        If the requested CuPy backend is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> import tomojax as tj
    >>> from tomojax.recon import fourier_reconstruct, FourierConfig, project
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = project(geometry, np.ones((1, 4, 4), np.float32))
    >>> volume = fourier_reconstruct(geometry, geometry.grid, geometry.detector, np.asarray(data))
    >>> volume.shape
    (1, 4, 4)
    """
    require_parallel_beam(geometry, "fourier_reconstruct")
    cfg = FourierConfig() if config is None else config
    backend = resolve_fourier_backend(cfg.backend)
    shape = validate_grid(grid, "fourier_reconstruct grid")[::-1]
    nviews, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fourier_reconstruct projections"
    )
    origin = grid_volume_origin(grid)
    if not np.isfinite([*origin, *detector.center]).all():
        raise ValueError("fourier_reconstruct: grid and detector placement must be finite")
    order, flipped, angles = sorted_half_turn(np.asarray(geometry.angles))
    plan_detector, shifts, positions = _sampling_coordinates(geometry, grid, detector, order)
    depth = fourier_slab_depth(grid, plan_detector, cfg.slices_per_batch, backend)
    output = validate_host_arrays(projections, out, shape, "fourier_reconstruct")
    host_bytes = 4 * (grid.nx * grid.ny * grid.nz + nviews * detector.nu * grid.nz)

    def prepare(start: int) -> np.ndarray | ProjectionRows:
        rows = positions[start : start + depth]
        return _prepare_slab(projections, order, rows, depth, backend)

    def write(start: int, volume: np.ndarray) -> None:
        stop = min(start + depth, grid.nz)
        if not np.isfinite(volume).all():
            raise ValueError("fourier_reconstruct: reconstructed slab must be finite in FP32")
        output[start:stop] = volume

    while True:
        try:
            overlap = (
                backend == "cupy" and grid.nz > depth and host_bytes >= _HOST_PIPELINE_MIN_BYTES
            )
            plan = FourierPlan(
                grid,
                plan_detector,
                nviews,
                flipped,
                angles,
                depth,
                backend,
                async_transfers=overlap,
                shifts=shifts,
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


def _sampling_coordinates(
    geometry: ScanGeometry, grid: Grid, detector: Detector, order: np.ndarray
) -> tuple[Detector, np.ndarray | None, np.ndarray]:
    """Padded Fourier field and physical source rows for all requested slices."""
    translations = _detector_translations(geometry, len(order))
    plan_detector, shifts = detector, None
    origin = grid_volume_origin(grid)
    positions = (origin[2] + np.arange(grid.nz) * grid.vz - detector.center[1]) / detector.dv + (
        detector.nv - 1
    ) / 2
    if translations is not None:
        translations = translations[order]
        # Cover every translated detector field while retaining the physical
        # detector phase through a compensating per-view shift.
        extent = float(np.max(np.abs(detector.center[0] - translations[:, 0])))
        if extent > abs(detector.center[0]):
            plan_detector = replace(detector, center=(extent, detector.center[1]))
        if np.any(translations[:, 0]):
            shifts = (
                translations[:, 0] + plan_detector.center[0] - detector.center[0]
            ) / detector.du
        positions = positions[:, None] + translations[None, :, 1] / detector.dv
    if not np.isfinite(positions).all():
        raise ValueError("fourier_reconstruct: detector-row coordinates must be finite")
    return plan_detector, shifts, positions


def _detector_translations(geometry: ScanGeometry, nviews: int) -> np.ndarray | None:
    """Extract physical detector-u/v translations without importing alignment.

    Nominal angles remain exact, including repeated/opposing directions; a
    float32 pose's trigonometric rounding must not split their angular cells.
    Translations along the parallel beam have no effect on its line integral.
    """
    if type(geometry) is ParallelGeometry:
        return None
    angles = np.deg2rad(np.asarray(geometry.angles, dtype=np.float64))
    poses = np.asarray([geometry.pose_for_view(i) for i in range(nviews)])
    expected = np.zeros((nviews, 3, 3))
    expected[:, 0, 0] = expected[:, 1, 1] = np.cos(angles)
    expected[:, 1, 0] = np.sin(angles)
    expected[:, 0, 1] = -np.sin(angles)
    expected[:, 2, 2] = 1
    if not np.allclose(poses[:, :3, :3], expected, atol=2e-7, rtol=0):
        raise ValueError(
            "fourier_reconstruct: requires axial rotations with translation-only poses"
        )
    translations = poses[:, [0, 2], 3]
    return translations if np.any(translations) else None


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
