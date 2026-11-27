"""Fourier-slice reconstruction for measured half-turn parallel scans with host storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import operator
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from tomojax import _parameters, progress
from tomojax.backends.api import is_oom_error
from tomojax.core import checkpoint
from tomojax.core.geometry.cone import require_parallel_beam
from tomojax.core.validation import validate_projection_stack
from tomojax.errors import InvalidParameter
from tomojax.geometry import ParallelGeometry, grid_volume_origin

from ._fourier_backend import FourierPlan, fourier_slab_depth, resolve_fourier_backend
from ._fourier_grid import (
    check_samples,
    sample_projection_rows,
    sample_rolled_rows,
    sorted_half_turn,
)
from ._fourier_rows import ProjectionRows, prepare_projection_rows
from ._host_arrays import validate_host_arrays

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from concurrent.futures import Future

    import jax

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
    bounding its padded CUDA FP32 plane stack to 1/32 of free memory and
    512 MiB (at least one slice).
    NumPy defaults to 16; an explicit integer fixes the requested depth.
    CUDA allocation exhaustion still halves it down to one slice. FFT plans and
    allocator caches consume additional memory. Large CUDA calls overlap host
    row preparation, device transfers, FFT execution and output writes, retaining
    at most two pending device slabs, and check finiteness on the GPU before
    download. A bounded cache retains immutable
    geometry arrays; each call owns its working buffers.
    Host transfer buffers use exact-sized pinned allocations. Upload staging is
    reused after its copy completes; returned output slabs keep independent storage.
    Fractional detector rows are interpolated on CUDA from at most two measured
    source rows per requested slice; entire projection stacks are not uploaded.

    Parameters
    ----------
    slices_per_batch : int | None, optional
        Positive slab depth, or None to choose automatically as described above.
    filter : Filter or None, optional
        FBP detector response ("ramp", "hann" or "shepp-logan"), or None
        (default) for the native Fourier response with a detector-Nyquist
        disk cutoff. Without a filter, edges are sharper and high-frequency
        noise is higher. Passing "ramp" reproduces default FBP's detector
        resolution and noise at Fourier speed; "hann" and "shepp-logan" add
        the corresponding FBP windows.
    backend : Literal['auto', 'numpy', 'cupy'], optional
        Execution backend: "auto", "numpy" or "cupy". Explicit choices never change
        backends. Default is 'auto'.
    exterior : {"cylinder", "zero"}, optional
        Model beyond the detector: "cylinder" fits each row at its edges before
        transforming; "zero" assumes zero attenuation beyond the detector.
        Default is 'cylinder'.
    """

    method: ClassVar[Literal["fourier"]] = "fourier"

    slices_per_batch: int | None = None
    filter: _parameters.Filter | None = None
    backend: Literal["auto", "numpy", "cupy"] = "auto"

    exterior: _parameters.Exterior = _parameters.DEFAULT_EXTERIOR

    def __post_init__(self) -> None:
        _parameters.check_exterior(self.exterior)
        if self.filter is not None:
            object.__setattr__(self, "filter", _parameters.normalize_filter(self.filter))
        if self.backend not in ("auto", "numpy", "cupy"):
            raise InvalidParameter(
                "backend", "fourier_reconstruct: backend must be 'auto', 'numpy' or 'cupy'"
            )
        if self.slices_per_batch is not None and operator.index(self.slices_per_batch) < 1:
            raise InvalidParameter(
                "slices_per_batch", "fourier_reconstruct: slices_per_batch must be positive"
            )


@progress.task("reconstruct", "fourier", None, "slabs")
def fourier(
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
    A parallel detector roll (``detector_roll``) is resampled bilinearly onto the
    unrolled detector lattice. Axial pose rotations enter the view angles. Out-of-plane
    rotations do not preserve the independent axial slices required here.
    Detector offsets, anisotropic voxels, shifted/cropped grids and fractional
    detector-v rows retain their physical coordinates.

    This approximate inverse uses six-point Kaiser--Bessel radial interpolation,
    linear interpolation in measured angle and a padded Cartesian inverse FFT.
    With ``filter=None`` (default), the native response cuts off at detector
    Nyquist, preserving sharper edges while transmitting more high-frequency
    noise than FBP. Setting ``filter`` to ``"ramp"``, ``"hann"`` or
    ``"shepp-logan"`` applies FBP's linear detector-interpolation response,
    any chosen window and radial replicas inside the Cartesian frequency grid.
    DC uses periodic angular trapezoid weights. The default
    ``exterior="cylinder"`` continues each row as in :func:`tomojax.recon.fbp`,
    with each edge's amplitudes averaged over a sixteenth of the half turn; the
    Cartesian transform keeps the measured detector's field, where nothing
    within three field radii wraps into it. Set ``FourierConfig(exterior="zero")``
    to assume zero attenuation beyond the detector. Angular undersampling can
    cause artifacts.

    Input and output stay in NumPy arrays or memmaps, with float32 output in
    ``(nz, ny, nx)`` order. New outputs are C-contiguous for downstream volume
    operations, backed by owned demand-paged host storage; supplied output storage
    is preserved.
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
    InvalidParameter
        If CuPy is unavailable or sampled measurements are nonfinite in FP32.

    Examples
    --------
    >>> import numpy as np
    >>> import tomojax as tj
    >>> from tomojax.recon import FourierConfig, fourier
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = tj.project(geometry, np.ones((1, 4, 4), np.float32))
    >>> volume = fourier(geometry, geometry.grid, geometry.detector, np.asarray(data))
    >>> volume.shape
    (1, 4, 4)
    """
    cfg = FourierConfig() if config is None else config
    backend = resolve_fourier_backend(cfg.backend)
    shape = grid.shape
    nviews, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fourier_reconstruct projections"
    )
    sampling = _sampling_coordinates(geometry, grid, detector)
    depth = fourier_slab_depth(grid, sampling.detector, cfg.slices_per_batch, backend)
    output = validate_host_arrays(projections, out, shape, "fourier_reconstruct")
    host_bytes = 4 * (grid.nx * grid.ny * grid.nz + nviews * detector.nu * grid.nz)

    def prepare(start: int) -> np.ndarray | ProjectionRows:
        return sampling.sample(
            projections, slice(start, start + depth), slice(None), depth, backend
        )

    with checkpoint.slabs(output) as finished:

        def write(start: int, volume: np.ndarray) -> None:
            stop = min(start + depth, grid.nz)
            output[start:stop] = volume
            finished.add(start, stop)

        while True:
            progress.check_canceled()
            starts = [
                start
                for start in range(0, grid.nz, depth)
                if not finished.finished(start, min(start + depth, grid.nz))
            ]
            progress.remaining("slabs", len(starts))
            try:
                overlap = (
                    backend == "cupy" and grid.nz > depth and host_bytes >= _HOST_PIPELINE_MIN_BYTES
                )
                plan = FourierPlan(
                    grid,
                    sampling.detector,
                    nviews,
                    sampling.flipped,
                    sampling.angles,
                    depth,
                    backend,
                    async_transfers=overlap,
                    shifts=sampling.shifts,
                    filter=cfg.filter,
                    exterior=cfg.exterior,
                    order=sampling.order,
                )
                if overlap:
                    _run_host_pipeline(plan, grid.nz, depth, starts, prepare, write)
                else:
                    for start in starts:
                        progress.check_canceled()
                        count = min(depth, grid.nz - start)
                        with progress.preparing("dispatch fourier transform", start):
                            write(start, plan.reconstruct(prepare(start), count=count))
                        progress.step("slabs")
                return output
            except Exception as error:
                if backend != "cupy" or depth == 1 or not is_oom_error(error):
                    raise
                # The pipeline has drained its transfer events. Drop this attempt's
                # buffers and the exception traceback before allocating a smaller
                # slab, rewriting every slab not yet finished.
                plan = None
            progress.retried("slab depth")
            depth = max(1, depth // 2)


@dataclass(frozen=True)
class _Sampling:
    detector: Detector
    order: np.ndarray
    flipped: np.ndarray
    angles: tuple[float, ...]
    shifts: np.ndarray | None
    rows: np.ndarray
    vertical: np.ndarray | None
    rolled: tuple[float, float] | None

    def sample(
        self,
        projections: np.ndarray | jax.Array,
        slabs: slice,
        views: slice,
        depth: int,
        backend: Literal["numpy", "cupy"],
    ) -> np.ndarray | ProjectionRows:
        positions = self.rows[slabs]
        if self.vertical is not None:
            positions = positions[:, None] + self.vertical[None, views]
        return _prepare_slab(projections, self.order[views], positions, depth, backend, self.rolled)


def check_fourier_inputs(
    geometry: ScanGeometry,
    grid: Grid,
    projections: np.ndarray | jax.Array | None,
    *,
    check_samples: bool,
) -> None:
    """Validate exactly the contributing FP32 samples, in bounded groups of views and planes."""
    try:
        sampling = _sampling_coordinates(geometry, grid, geometry.detector)
    except ValueError as err:
        raise InvalidParameter("method", str(err)) from err
    if projections is not None and projections.dtype.kind not in "buif":
        raise InvalidParameter(
            "source", "fourier_reconstruct: projections must be a real NumPy array"
        )
    if projections is None or not check_samples:
        return
    for z in range(0, grid.nz, 8):
        progress.check_canceled()
        depth = min(8, grid.nz - z)
        for first in range(0, len(sampling.order), 32):
            _ = sampling.sample(
                projections, slice(z, z + depth), slice(first, first + 32), depth, "numpy"
            )
            del _


def _sampling_coordinates(geometry: ScanGeometry, grid: Grid, detector: Detector) -> _Sampling:
    """Resolve the physical detector rows, view order and translations shared by plan and run."""
    require_parallel_beam(geometry, "fourier_reconstruct")
    directions, translations, pose_angles = axial_trajectory(geometry)
    order, flipped, angles = sorted_half_turn(directions, fitted=pose_angles)
    plan_detector, shifts, vertical = detector, None, None
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
        vertical = translations[:, 1] / detector.dv
    if not np.isfinite(positions).all() or (
        vertical is not None and not np.isfinite(vertical).all()
    ):
        raise ValueError("fourier_reconstruct: detector-row coordinates must be finite")
    roll = float(getattr(geometry, "detector_roll", 0.0) or 0.0)
    rolled = (roll, detector.dv / detector.du) if roll else None
    return _Sampling(plan_detector, order, flipped, angles, shifts, positions, vertical, rolled)


def axial_trajectory(geometry: ScanGeometry) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Axial rotation and detector translation from the projector's world-from-object poses."""
    degrees = np.asarray(geometry.angles, dtype=np.float64)
    if type(geometry) is ParallelGeometry:
        return degrees, None, False
    poses = np.asarray([geometry.pose_matrix(i) for i in range(len(degrees))])
    if not np.allclose(poses[:, 2, :3], [0, 0, 1], atol=2e-7, rtol=0):
        raise ValueError("fourier_reconstruct: axial slices require rotations about z")
    # Keep the precision of recorded angles if the rotations are exactly those
    # angles rounded to the pose precision (e.g. a translation-only fit).
    radians = np.deg2rad(degrees)
    nominal = np.stack([np.cos(radians), np.sin(radians)], axis=1)
    measured = poses[:, :2, 0]
    fitted = not np.array_equal(measured.astype(np.float32), nominal.astype(np.float32))
    if fitted:
        degrees = np.rad2deg(np.arctan2(measured[:, 1], measured[:, 0]))
    translations = poses[:, [0, 2], 3]
    return degrees, translations if np.any(translations) else None, fitted


def _prepare_slab(
    projections: np.ndarray | jax.Array,
    order: np.ndarray,
    positions: np.ndarray,
    depth: int,
    backend: Literal["numpy", "cupy"],
    rolled: tuple[float, float] | None = None,
) -> np.ndarray | ProjectionRows:
    """Finite measured rows for a fixed-depth FFT, padded only after row sampling.

    ``rolled`` (roll in degrees, dv / du) resamples a rolled detector's rows.
    """
    if rolled is not None:
        data = sample_rolled_rows(projections, order, positions, *rolled)
    else:
        data = (prepare_projection_rows if backend == "cupy" else sample_projection_rows)(
            projections, order, positions
        )
    if isinstance(data, ProjectionRows):
        return data
    check_samples(data)
    if len(positions) < depth:
        data = np.pad(data, ((0, depth - len(positions)), (0, 0), (0, 0)))
    return data


def _run_host_pipeline(
    plan: FourierPlan,
    nz: int,
    depth: int,
    starts: Sequence[int],
    prepare: Callable[[int], np.ndarray | ProjectionRows],
    write: Callable[[int, np.ndarray], None],
) -> None:
    """Overlap staging and transfers of the slabs at ``starts``; CUDA stays on this thread."""
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
                progress.step("slabs")
                progress.check_canceled()
            written = writer.submit(write, start, volume)

        prepared = reader.submit(prepare, starts[0]) if starts else None
        try:
            for index, start in enumerate(starts):
                progress.check_canceled()
                try:
                    assert prepared is not None
                    data = prepared.result()
                    prepared = None
                except BaseException:
                    # A later invalid input must still commit completed slabs.
                    for first, job in pending:
                        finish(first, job)
                    pending.clear()
                    raise
                stop = min(start + depth, nz)
                if index + 1 < len(starts):
                    prepared = reader.submit(prepare, starts[index + 1])
                with progress.preparing("dispatch fourier transform", start):
                    pending.append((start, plan.enqueue(data, count=stop - start)))
                del data
                if len(pending) >= 2:
                    finish(*pending.pop(0))
            for first, job in pending:
                finish(first, job)
            pending.clear()
            if written is not None:
                written.result()
                progress.step("slabs")
        finally:
            # Drain GPU work even if an output write fails. Buffer owners stay
            # alive through their completion event; executors join host workers.
            for _, job in pending:
                job.synchronize()
