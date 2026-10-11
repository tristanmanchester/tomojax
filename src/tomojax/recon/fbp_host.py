"""FBP with host storage and bounded volume slabs on the accelerator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends import device_free_memory_bytes
from tomojax.core.geometry.cone import require_parallel_beam
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import validate_projection_stack
from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry, grid_volume_origin

from ._host_arrays import (
    ProjectionRows,
    axial_output_axes,
    host_projection_input,
    validate_host_arrays,
    volume_writer,
)
from .fbp import (
    _fbp_from_host,  # pyright: ignore[reportPrivateUsage]
    _fft_length,  # pyright: ignore[reportPrivateUsage]
    _filter_detector,  # pyright: ignore[reportPrivateUsage]
    _parallel_filter_detector,  # pyright: ignore[reportPrivateUsage]
    _view_weights,  # pyright: ignore[reportPrivateUsage]
    supports_parallel_fbp_z_integer,
    validate_fbp_geometry,
)
from .filters import rfft_filter_array

_OUTPUT_OVERLAP_MIN_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True, kw_only=True)
class FBPHostConfig:
    """Control slab storage and filter batches for :func:`fbp_host`.

    Output remains in host storage. Native FP32 GPU rows can stay resident
    when the row gathers, output slabs and FFT workspace fit half of free memory.
    Other inputs use bounded host transfers. For built-in parallel
    geometry, a slab is ``slices_per_batch`` axial (z) slices with only the
    detector rows they need (balanced slabs of at most 16 when ``None``). For
    laminography and other geometry, a slab is ``slices_per_batch`` x-slices;
    every view batch passes through the device once per slab. ``None`` sizes
    slabs to about half of the free device memory, so a volume that fits is
    reconstructed in one pass.
    Projection rows are read and transferred only ``views_per_batch`` views at
    a time, keeping at most three input batches on CUDA when memory permits,
    or two otherwise, for either layout. The memory check also budgets for an
    extra estimated filtering workspace.
    Large CUDA axial volumes overlap output copies with the next slab, retaining at most
    two output slabs on the device, plus one staging slab for packed transfers.
    Runtime/compiler caches add memory beyond
    these arrays. ``scale`` has the
    same meaning as in :class:`FBPConfig`. So does ``padding``: ``'edge'``
    assumes constant exterior endpoint values instead of zero attenuation.
    """

    slices_per_batch: int | None = None
    views_per_batch: int = 32
    filter: str = "ramp"
    padding: Literal["zero", "edge"] = "zero"
    scale: float | None = None
    backprojector: Literal["auto", "jax", "pallas"] = "auto"


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


def _resident_axial_rows(
    projections: np.ndarray | jax.Array,
    start: int,
    detector: Detector,
    spectrum: jax.Array,
    grid: Grid,
    batch_size: int,
    *,
    native: bool,
) -> jax.Array | None:
    """Borrow native FP32 GPU rows only when bounded extra storage fits."""
    if not native or not isinstance(projections, jax.Array) or projections.dtype != np.float32:
        return None
    devices = projections.devices()
    if len(devices) != 1 or devices != spectrum.devices():
        return None
    device = next(iter(devices))
    stop = start + detector.nv
    if device.platform != "gpu" or start < 0 or stop > projections.shape[1]:
        return None
    with jax.default_device(device):
        free = device_free_memory_bytes()
    row_bytes = 4 * projections.shape[0] * detector.nv * projections.shape[2]
    slab_bytes = 4 * grid.nx * grid.ny * grid.nz
    workspace = 16 * batch_size * detector.nv * (spectrum.shape[0] + detector.nu)
    # Account for current/previous row gathers, accumulator, overlapped output
    # and its packed copy, plus FFT workspace; reserve half for caches/runtime.
    if free is None or 2 * row_bytes + 3 * slab_bytes + workspace > free // 2:
        return None
    return projections[:, start:stop]


def fbp_host(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray | jnp.ndarray,
    *,
    config: FBPHostConfig | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct bounded slabs into float32 NumPy storage (see FBPHostConfig).

    Accept NumPy/memmap or JAX projections. Accelerator inputs use bounded
    downloads or budgeted native resident slabs. Writable ``out`` is returned
    unchanged, with logical axes ``(x, y, z)``. Failures may leave partial output;
    input storage must remain unchanged. Not differentiable; use ``fbp`` for that.
    """
    require_parallel_beam(geometry, "fbp_host")
    projections = host_projection_input(projections)
    cfg = FBPHostConfig() if config is None else config
    if cfg.padding not in {"zero", "edge"}:
        raise ValueError("fbp_host: padding must be 'zero' or 'edge'")
    slices = None if cfg.slices_per_batch is None else operator.index(cfg.slices_per_batch)
    views = operator.index(cfg.views_per_batch)
    if (slices is not None and slices < 1) or views < 1:
        raise ValueError("fbp_host: slices_per_batch and views_per_batch must be positive")
    if cfg.backprojector not in {"auto", "jax", "pallas"}:
        raise ValueError("fbp_host: backprojector must be 'auto', 'jax' or 'pallas'")
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
    if cfg.backprojector == "pallas" and not cuda:
        raise ValueError("fbp_host: Pallas requires CUDA")
    backend = "pallas" if cuda and cfg.backprojector != "jax" else "jax"
    if type(geometry) is not ParallelGeometry:
        return _general_slabs(poses, grid, detector, projections, cfg, out, backend, slices, views)
    depth = _axial_depth(grid.nz, slices)
    local_grid, local_detector, first, step, integer = _slab_layout(grid, detector, depth)
    rows = local_detector.nv
    ramp = rfft_filter_array(cfg.filter, local_detector.nu, detector.du, jnp.float32)
    output = validate_host_arrays(projections, out, shape, device_input=True)
    host_poses = np.asarray(poses, np.float32)
    params = np.zeros((n_views, 6), np.float32)
    overlap = cuda and grid.nz > depth and output.nbytes >= _OUTPUT_OVERLAP_MIN_BYTES
    if overlap:
        free = device_free_memory_bytes()
        # Retain an extra output only when it takes a small part of the budget.
        extra = 2 if axial_output_axes(output) is not None else 1
        overlap = free is not None and extra * 4 * grid.nx * grid.ny * depth <= 0.2 * free
    with volume_writer(output, asynchronous=overlap) as write:
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
            resident = _resident_axial_rows(
                projections,
                v_start,
                local_detector,
                ramp,
                local_grid,
                min(views, n_views),
                native=integer and step == 1 and backend == "pallas",
            )
            volume = _fbp_from_host(
                local_poses,
                ProjectionRows(projections, v_start, rows) if resident is None else resident,
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
                padding=cfg.padding,
                resident_slab=resident is not None,
            )
            count = min(depth, grid.nz - start)
            write(volume, start, count)
            del volume  # Do not retain a downloaded slab during the next computation.
    return output


def _general_slabs(
    poses: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray | jnp.ndarray,
    cfg: FBPHostConfig,
    out: np.ndarray | None,
    backend: str,
    slices: int | None,
    views: int,
) -> np.ndarray:
    """Reconstruct x-slabs, streaming every view batch once per slab."""
    output = validate_host_arrays(projections, out, (grid.nx, grid.ny, grid.nz), device_input=True)
    view_scale, params, arc_length, separable = _view_weights(poses, cfg.scale)
    if cfg.padding == "edge" and not separable:
        raise ValueError("fbp_host: edge padding requires a separable detector-u filter")
    host_poses = np.asarray(poses, np.float32)
    detector_f = _filter_detector(grid, detector, host_poses, pad_v=not separable)
    spectrum = rfft_filter_array(cfg.filter, detector_f.nu, detector.du, jnp.float32)
    batch = min(views, projections.shape[0])
    if slices is None:
        # Filtering workspace: padded real batch plus its complex spectrum.
        rows = 1 if separable else _fft_length(detector_f.nv)
        workspace = 16 * batch * max(rows, detector_f.nv) * (spectrum.shape[0] + detector_f.nu)
        free = device_free_memory_bytes() or 2**30
        slices = max(1, (free // 2 - workspace) // (4 * grid.ny * grid.nz))
    depth = min(slices, grid.nx)
    ox, oy, oz = grid_volume_origin(grid)
    for start in range(0, grid.nx, depth):
        count = min(depth, grid.nx - start)
        # Every slab has the same shape, so one compiled step serves them all.
        slab = replace(grid, nx=depth, vol_origin=(ox + start * grid.vx, oy, oz), vol_center=None)
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
            padding=cfg.padding,
        )
        output[start : start + count] = np.asarray(volume)[:count]
    return output
