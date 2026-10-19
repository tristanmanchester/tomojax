"""FBP with host storage and bounded volume slabs on the accelerator."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, nullcontext
from dataclasses import dataclass, replace
import math
import operator
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends import device_free_memory_bytes
from tomojax.backends.api import is_oom_error
from tomojax.core.devices import as_devices
from tomojax.core.geometry.cone import require_parallel_beam
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import validate_detector, validate_projection_stack
from tomojax.geometry import Detector, Grid, ParallelGeometry, grid_volume_origin

from ._host_arrays import ProjectionRows, validate_host_arrays
from .fbp import (
    FBPConfig,
    _fbp_from_host,  # pyright: ignore[reportPrivateUsage]
    _fft_length,  # pyright: ignore[reportPrivateUsage]
    _filter_detector,  # pyright: ignore[reportPrivateUsage]
    _parallel_filter_detector,  # pyright: ignore[reportPrivateUsage]
    _view_weights,  # pyright: ignore[reportPrivateUsage]
    supports_parallel_fbp_z_integer,
    validate_fbp_geometry,
)
from .filters import rfft_filter_array
from .types import ReconstructionSlab

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


@dataclass(frozen=True, slots=True, kw_only=True)
class FBPHostConfig:
    """Control slab storage and filter batches for :func:`fbp_host`.

    The output remains in host storage, and so does the input unless it is a
    device array, which stays resident and is sliced there. For built-in parallel
    geometry, a slab is ``slices_per_batch`` axial (z) slices with only the
    detector rows they need (balanced slabs of at most 16 when ``None``). For
    laminography and other geometry, a slab is ``slices_per_batch`` x-slices;
    every view batch passes through the device once per slab. ``None`` sizes
    slabs to about half of the free device memory, so a volume that fits is
    reconstructed in one pass.
    Filtering uses ``views_per_batch`` views at a time. Axial CUDA slabs with
    many batches can stage up to 512 MiB of detector rows when those fit in a
    quarter of free device memory, avoiding per-batch host dispatch. Otherwise
    rows are read and transferred with at most two input batches on the device.
    Runtime/compiler caches add memory beyond these arrays. General slabs shrink
    their depth, then their view batch, after device allocation exhaustion.
    ``scale`` has the
    same meaning as in :class:`FBPConfig`.

    Parameters
    ----------
    slices_per_batch : int | None, optional
        Positive slab depth, or None to choose automatically as described above.
    views_per_batch : int, optional
        Number of views processed together to bound temporary storage. Default is
        32.
    filter : str, optional
        Frequency filter: "ramp", "shepp-logan" or "hann". Default is 'ramp'.
    scale : float | None, optional
        Constant angular weight per view, or None for geometry-derived weights.
    backprojector : Literal['auto', 'jax', 'cuda'], optional
        Backprojection backend: "auto", "jax" or "cuda". Auto uses CUDA
        acceleration when available. Default is 'auto'.
    """

    slices_per_batch: int | None = None
    views_per_batch: int = 32
    filter: str = "ramp"
    scale: float | None = None
    backprojector: Literal["auto", "jax", "cuda"] = "auto"


def _axial_depth(nz: int, requested: int | None) -> int:
    """Balance automatic slabs without exceeding 16 slices or adding a slab."""
    if requested is not None:
        return min(requested, nz)
    batches = (nz + 15) // 16
    return (nz + batches - 1) // batches


def _slab_layout(
    grid: Grid, detector: Detector, depth: int
) -> tuple[Grid, Detector, float, float, bool]:
    """Choose a fixed local coordinate system and interpolation support."""
    ox, oy, oz = grid_volume_origin(grid)
    integer = supports_parallel_fbp_z_integer(grid, detector)
    first = (oz - detector.center[1]) / detector.dv + (detector.nv - 1) / 2
    step = grid.vz / detector.dv
    if integer:
        first, step = round(first), round(step)
    integer &= first >= 0 and first + (grid.nz - 1) * step < detector.nv
    integer &= (depth - 1) * step + 1 <= detector.nv
    rows = (
        int((depth - 1) * step + 1)
        if integer
        else min(detector.nv, math.ceil((depth - 1) * step) + 2)
    )
    local_grid = replace(grid, nz=depth, vol_origin=(ox, oy, 0.0), vol_center=None)
    local_detector = replace(
        detector, nv=rows, center=(detector.center[0], (rows - 1) * detector.dv / 2)
    )
    local_detector = _parallel_filter_detector(local_grid, local_detector)
    return local_grid, local_detector, first, step, integer


def _device_rows(projections: jax.Array, start: int, rows: int) -> jax.Array:
    """Detector rows ``start:start + rows`` of a device stack, zero-extended."""
    lo, hi = max(0, start), min(projections.shape[1], start + rows)
    if hi <= lo:
        return jnp.zeros((projections.shape[0], rows, projections.shape[2]), jnp.float32)
    block = projections[:, lo:hi].astype(jnp.float32)
    return jnp.pad(block, ((0, 0), (lo - start, start + rows - hi), (0, 0)))


def _stage_rows(
    window: ProjectionRows, poses: jax.Array, views: int, backend: str
) -> ProjectionRows | jax.Array:
    """Stage a bounded axial window when it saves many CUDA host submissions.

    The original stack stays on the host. Only this slab's rows are uploaded;
    filtering and backprojection still use the requested view-batch size.
    Query after each preceding slab, since pending output and caches use memory.
    """
    n, nv, nu = window.shape
    size = 4 * n * nv * nu
    if backend != "cuda" or n <= 4 * views or size > 512 * 1024**2:
        return window
    devices = poses.devices()
    if len(devices) != 1:
        return window
    device = next(iter(devices))
    free = device_free_memory_bytes(device=device) or 0
    return jax.device_put(window[:], device) if size <= free // 4 else window


def _host_volume(volume: jax.Array, target: np.ndarray) -> np.ndarray:
    """Download in the caller's contiguous order without a CPU transpose copy."""
    if target.flags.f_contiguous:
        # A device-side transpose gives contiguous zyx host storage. Its xyz
        # view can then be copied directly into an axial Fortran-order slab.
        return np.asarray(jnp.transpose(volume, (2, 1, 0))).transpose(2, 1, 0)
    return np.asarray(volume)


class _VolumeWriter:
    """Keep at most one output transfer/write alive beside the current slab."""

    def __init__(self, *, enabled: bool) -> None:
        self.pool = ThreadPoolExecutor(max_workers=1) if enabled else None
        self.pending: Future[None] | None = None

    def __enter__(self) -> _VolumeWriter:
        return self

    def __call__(self, volume: jax.Array, target: np.ndarray, count: int) -> None:
        if self.pool is None:
            target[...] = _host_volume(volume, target)[:, :, :count]
            return
        if self.pending is not None:
            self.pending.result()
        packed = target.flags.f_contiguous
        value = jnp.transpose(volume, (2, 1, 0)) if packed else volume
        value.copy_to_host_async()

        def store() -> None:
            host = np.asarray(value)
            target[...] = (host.transpose(2, 1, 0) if packed else host)[:, :, :count]

        self.pending = self.pool.submit(store)

    def __exit__(self, *args: object) -> None:
        try:
            if self.pending is not None:
                self.pending.result()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)


def _output_memory(projections: np.ndarray | jax.Array, poses: jax.Array) -> int:
    """Budget the device that will actually produce a resident-input slab.

    Host input follows the pose/default device. Resident input is reconstructed
    where it already lives, which need not be the current default device.
    Sharded input does not identify one output-device budget.
    """
    devices = projections.devices() if isinstance(projections, jax.Array) else poses.devices()
    if len(devices) != 1:
        return 0
    device = next(iter(devices))
    return device_free_memory_bytes(device=device) or 0


def fbp_host(
    geometry: ParallelGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray | jax.Array,
    *,
    config: FBPHostConfig | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct without storing the full volume or projections on the device.

    Accept NumPy arrays, including memmaps, or device arrays and return a float32
    NumPy array. Automatically allocated axial output uses Fortran order for
    contiguous xy-planes; general geometry uses C order for contiguous x-slabs.
    ``out`` may be a writable memmap. Writes complete before returning, but
    a later failure can leave a partially written output. Input storage must
    remain unchanged until the call finishes. This host-returning routine is
    not differentiable. Use :func:`fbp` for device-resident output.

    Every geometry ``fbp`` accepts is supported, with the same exact weights:
    laminography and posed scans reconstruct in x-slabs, built-in parallel
    scans in axial slabs that read only the detector rows they need.
    Anisotropic spacing, detector offsets and shifted volume origins are
    supported. Filter tails use the same zero-extended measurement model as
    ``fbp``.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry, including view angles and any per-view poses.
    grid : Grid
        Reconstruction grid in ``(x, y, z)`` order, with physical voxel spacing.
    detector : Detector
        Detector dimensions and placement in the same length unit as the grid.
    projections : array-like
        Measured line integrals with shape ``(views, detector.nv, detector.nu)``.
    config : FBPHostConfig or None, optional
        Solver settings. None uses FBPHostConfig defaults.
    out : numpy.ndarray or None, optional
        Writable float32 host storage of shape ``(nx, ny, nz)``, including a memmap.
        Must not overlap input storage. None allocates a new output.

    Returns
    -------
    numpy.ndarray
        Float32 reconstruction of shape ``(nx, ny, nz)``. Returns out itself when supplied.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.
    """
    require_parallel_beam(geometry, "fbp_host")
    cfg = FBPHostConfig() if config is None else config
    slices = None if cfg.slices_per_batch is None else operator.index(cfg.slices_per_batch)
    views = operator.index(cfg.views_per_batch)
    if (slices is not None and slices < 1) or views < 1:
        raise ValueError("fbp_host: slices_per_batch and views_per_batch must be positive")
    if cfg.backprojector not in {"auto", "jax", "cuda"}:
        raise ValueError("fbp_host: backprojector must be 'auto', 'jax' or 'cuda'")
    shape = validate_fbp_geometry(geometry, grid, detector, cfg.scale, "fbp_host")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fbp_host projections"
    )

    poses = stack_view_poses(geometry, n_views)
    view_scale = _view_weights(poses, cfg.scale)[0]
    cuda = all(
        d.platform == "gpu" and d.client.platform_version.lower().startswith("cuda")
        for d in poses.devices()
    )
    if cfg.backprojector == "cuda" and not cuda:
        raise ValueError("fbp_host: CUDA requires CUDA")
    backend = "cuda" if cuda and cfg.backprojector != "jax" else "jax"
    if type(geometry) is not ParallelGeometry:
        return _general_slabs(poses, grid, detector, projections, cfg, out, backend, slices, views)
    depth = _axial_depth(grid.nz, slices)
    local_grid, local_detector, first, step, integer = _slab_layout(grid, detector, depth)
    rows = local_detector.nv
    ramp = rfft_filter_array(cfg.filter, local_detector.nu, detector.du, jnp.float32)
    output = validate_host_arrays(projections, out, shape, device_input=True, order="F")
    host_poses = np.asarray(poses, np.float32)
    params = np.zeros((n_views, 6), np.float32)
    slab_bytes = 4 * grid.nx * grid.ny * depth
    # Bound the previous result, its packed transfer and the current result;
    # leave the remaining memory for input/filter workspaces and runtime caches.
    enabled = (
        grid.nz > depth
        and (output.flags.f_contiguous or output.flags.c_contiguous)
        and slab_bytes <= _output_memory(projections, poses) // 4
    )
    with _VolumeWriter(enabled=enabled) as write:
        for start in range(0, grid.nz, depth):
            v = first + start * step
            v_start = math.floor(v)
            if not integer:
                v_start = max(0, min(v_start, detector.nv - rows))
            # Local grid/detector metadata stay identical for every slab. Only the
            # fractional row phase changes, passed as a dynamic pose translation.
            local_poses = host_poses.copy()
            if not integer:
                local_poses[:, 2, 3] = (v - v_start) * detector.dv
            volume = _fbp_from_host(
                local_poses,
                _device_rows(projections, v_start, rows)
                if isinstance(projections, jax.Array)
                else _stage_rows(ProjectionRows(projections, v_start, rows), poses, views, backend),
                view_scale,
                params,
                ramp,
                0.0,
                grid=local_grid,
                detector=local_detector,
                backend=backend,
                batch_size=min(views, n_views),
                z_mode="integer" if integer else "fixed",
                separable=True,
                check_finite=True,
            )
            count = min(depth, grid.nz - start)
            # Only one previous slab can remain pending while this slab computes.
            target = output[:, :, start : start + count]
            write(volume, target, count)
    return output


def _general_slabs(
    poses: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray | jax.Array,
    cfg: FBPHostConfig,
    out: np.ndarray | None,
    backend: str,
    slices: int | None,
    views: int,
) -> np.ndarray:
    """Reconstruct x-slabs, streaming every view batch once per slab."""
    output = validate_host_arrays(projections, out, (grid.nx, grid.ny, grid.nz), device_input=True)
    view_scale, params, arc_length, separable = _view_weights(poses, cfg.scale)
    host_poses = np.asarray(poses, np.float32)
    detector_f = _filter_detector(grid, detector, host_poses, pad_v=not separable)
    spectrum = rfft_filter_array(cfg.filter, detector_f.nu, detector.du, jnp.float32)
    batch = min(views, projections.shape[0])
    if slices is None:
        # Filtering workspace: padded real batch plus its complex spectrum.
        rows = 1 if separable else _fft_length(detector_f.nv)
        workspace = 16 * batch * max(rows, detector_f.nv) * (spectrum.shape[0] + detector_f.nu)
        free = _output_memory(projections, poses) or 2**30
        slices = max(1, (free // 2 - workspace) // (4 * grid.ny * grid.nz))
    depth = min(slices, grid.nx)
    ox, oy, oz = grid_volume_origin(grid)
    start = 0
    while start < grid.nx:
        count = min(depth, grid.nx - start)
        # Keep the slab shape fixed at this depth, including the padded tail.
        slab = replace(grid, nx=depth, vol_origin=(ox + start * grid.vx, oy, oz), vol_center=None)
        volume = None
        try:
            volume = _fbp_from_host(
                host_poses,
                projections,
                np.asarray(view_scale, np.float32),
                np.asarray(params, np.float32),
                spectrum,
                float(arc_length),
                grid=slab,
                detector=detector_f,
                backend=backend,
                batch_size=batch,
                separable=separable,
                check_finite=True,
            )
            target = output[start : start + count]
            target[...] = _host_volume(volume, target)[:count]
        except Exception as error:
            if not is_oom_error(error):
                raise
            volume = None
            if depth > 1:
                depth = max(1, depth // 2)
            elif batch > 1:
                batch = max(1, batch // 2)
            else:
                raise
            # Leave the exception scope before retrying: its traceback can own
            # device buffers. Completed slabs stay on the host and are not redone.
            continue
        # Do not keep the preceding device slab alive while constructing the next.
        del volume
        start += count
    return output


def fbp_slabs[T](
    geometry: ParallelGeometry,
    grid: Grid,
    detector: Detector,
    read: Callable[[int, int], tuple[np.ndarray | jax.Array, T]],
    *,
    config: FBPConfig,
) -> Generator[ReconstructionSlab[T], None, None]:
    """Yield independent axial FBP slabs without downloading corrected projections.

    ``read`` supplies all views of a measured detector-row interval and an opaque
    context retained until that output completes. The output and source arrays
    remain valid after advancing or closing the iterator. Inputs must not change.
    Slabs use at most eight output rows (four for aligned CUDA gathers), reducing
    depth on OOM. Aligned CUDA slabs retain filtered rows for one all-view gather;
    filtering itself still uses bounded view batches.
    The first configured device does the work, as for host-returning FBP.
    Closing the generator drains all pending downloads before releasing inputs.

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
    config : FBPConfig
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
    _ = validate_fbp_geometry(geometry, grid, detector, config.scale, "reconstruct_slabs")
    _ = validate_detector(detector, "reconstruct_slabs detector")
    if config.backprojector not in {"auto", "jax", "cuda"}:
        raise ValueError("FBP backprojector must be 'auto', 'jax' or 'cuda'")
    batches = (grid.nz + 7) // 8
    depth = (grid.nz + batches - 1) // batches
    devices = as_devices(config.devices)
    context = jax.default_device(devices[0]) if devices else nullcontext()
    with context:
        poses = stack_view_poses(geometry, len(geometry.angles))
        cuda = all(
            d.platform == "gpu" and d.client.platform_version.lower().startswith("cuda")
            for d in poses.devices()
        )
        if config.backprojector == "cuda" and not cuda:
            raise ValueError("FBP CUDA backprojection requires CUDA")
        backend = "cuda" if cuda and config.backprojector != "jax" else "jax"
        if backend == "cuda" and _slab_layout(grid, detector, depth)[-1]:
            depth = min(depth, 4)
        host_poses = np.asarray(poses, np.float32)
        scale = np.asarray(_view_weights(poses, config.scale)[0], np.float32)
        start = 0
        while start < grid.nz:
            try:
                with closing(
                    _run_slabs(
                        grid, detector, host_poses, scale, read, config, backend, depth, start
                    )
                ) as completed:
                    for slab in completed:
                        start = slab.stop
                        yield slab
            except Exception as error:
                if depth == 1 or not is_oom_error(error):
                    raise
                depth = max(1, depth // 2)


def _finish[T](item: tuple[ReconstructionSlab[T], jax.Array]) -> ReconstructionSlab[T]:
    slab, packed = item
    slab.volume = np.asarray(packed).transpose(2, 1, 0)
    if not np.isfinite(slab.volume).all():
        raise ValueError("FBP reconstruction must be finite in FP32")
    return slab


def _run_slabs[T](
    grid: Grid,
    detector: Detector,
    poses: np.ndarray,
    scale: np.ndarray,
    read: Callable[[int, int], tuple[np.ndarray | jax.Array, T]],
    config: FBPConfig,
    backend: str,
    depth: int,
    start: int,
) -> Generator[ReconstructionSlab[T], None, None]:
    local_grid, local_detector, first, step, integer = _slab_layout(grid, detector, depth)
    rows = local_detector.nv
    ramp = rfft_filter_array(config.filter, local_detector.nu, detector.du, jnp.float32)
    params = np.zeros((len(poses), 6), np.float32)
    pending: list[tuple[ReconstructionSlab[T], jax.Array]] = []
    try:
        for at in range(start, grid.nz, depth):
            stop = min(at + depth, grid.nz)
            v = first + at * step
            source_start = math.floor(v)
            if not integer:
                source_start = max(0, min(source_start, detector.nv - rows))
            source_stop = min(source_start + rows, detector.nv)
            data, context = read(source_start, source_stop)
            source = jnp.asarray(data, jnp.float32)
            if source_stop - source_start < rows:
                source = jnp.pad(source, ((0, 0), (0, rows - source.shape[1]), (0, 0)))
            local_poses = poses.copy()
            if not integer:
                local_poses[:, 2, 3] = (v - source_start) * detector.dv
            volume = _fbp_from_host(
                local_poses,
                source,
                scale,
                params,
                ramp,
                0.0,
                grid=local_grid,
                detector=local_detector,
                backend=backend,
                batch_size=min(32, len(poses)),
                z_mode="integer" if integer else "fixed",
                separable=True,
                check_finite=True,
                cache_filtered=backend == "cuda" and integer,
            )
            packed = jnp.transpose(volume[:, :, : stop - at], (2, 1, 0))
            packed.copy_to_host_async()
            slab = ReconstructionSlab(
                at, stop, slice(source_start, source_stop), data, np.empty(0), context
            )
            pending.append((slab, packed))
            if len(pending) >= 2:
                yield _finish(pending.pop(0))
        for item in pending:
            yield _finish(item)
    finally:
        jax.block_until_ready([item[1] for item in pending])
        pending.clear()
