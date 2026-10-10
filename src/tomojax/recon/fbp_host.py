"""FBP with host storage and bounded volume slabs on the accelerator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
from typing import Literal

import jax.numpy as jnp
import numpy as np

from tomojax.backends import device_free_memory_bytes
from tomojax.core.geometry.cone import require_parallel_beam
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.geometry import Detector, Grid, ParallelGeometry, grid_volume_origin

from ._host_arrays import ProjectionRows, validate_host_arrays
from .fbp import (
    _fbp_from_host,  # pyright: ignore[reportPrivateUsage]
    _fft_length,  # pyright: ignore[reportPrivateUsage]
    _filter_detector,  # pyright: ignore[reportPrivateUsage]
    _parallel_filter_detector,  # pyright: ignore[reportPrivateUsage]
    _view_weights,  # pyright: ignore[reportPrivateUsage]
    supports_parallel_fbp_z_integer,
)
from .filters import rfft_filter_array


@dataclass(frozen=True, slots=True, kw_only=True)
class FBPHostConfig:
    """Control slab storage and filter batches for :func:`fbp_host`.

    The entire input and output remain in host storage. For built-in parallel
    geometry, a slab is ``slices_per_batch`` axial (z) slices with only the
    detector rows they need (16 when ``None``). For laminography and other
    geometry, a slab is ``slices_per_batch`` x-slices and every view batch passes
    through the device once per slab; ``None`` sizes slabs to about half of the
    free device memory, so a volume that fits is reconstructed in one pass.
    Projection rows are read and transferred only ``views_per_batch`` views at
    a time, keeping at most two input batches on the device for either layout.
    Runtime/compiler caches add memory beyond these arrays. ``scale`` has the
    same meaning as in :class:`FBPConfig`.
    """

    slices_per_batch: int | None = None
    views_per_batch: int = 32
    filter: str = "ramp"
    scale: float | None = None
    backprojector: Literal["auto", "jax", "pallas"] = "auto"


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


def fbp_host(
    geometry: ParallelGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    *,
    config: FBPHostConfig | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct without storing the full volume or projections on the device.

    Accept NumPy arrays, including memmaps, and return a float32 NumPy array.
    ``out`` may be a writable memmap. Completed slabs are written immediately;
    a later failure can leave a partially written output. Input storage must
    remain unchanged until the call finishes. This host-returning routine is
    not differentiable. Use :func:`fbp` for device-resident output.

    Every geometry ``fbp`` accepts is supported, with the same exact weights:
    laminography and posed scans reconstruct in x-slabs, built-in parallel
    scans in axial slabs that read only the detector rows they need.
    Anisotropic spacing, detector offsets and shifted volume origins are
    supported. Filter tails use the same zero-extended measurement model as
    ``fbp``.
    """
    require_parallel_beam(geometry, "fbp_host")
    cfg = FBPHostConfig() if config is None else config
    slices = None if cfg.slices_per_batch is None else operator.index(cfg.slices_per_batch)
    views = operator.index(cfg.views_per_batch)
    if (slices is not None and slices < 1) or views < 1:
        raise ValueError("fbp_host: slices_per_batch and views_per_batch must be positive")
    if cfg.backprojector not in {"auto", "jax", "pallas"}:
        raise ValueError("fbp_host: backprojector must be 'auto', 'jax' or 'pallas'")
    shape = validate_grid(grid, "fbp_host grid")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fbp_host projections"
    )
    ox, oy, oz = grid_volume_origin(grid)
    if not np.isfinite([ox, oy, oz, *detector.center]).all():
        raise ValueError("fbp_host: grid and detector placement must be finite")
    if cfg.scale is not None and not math.isfinite(cfg.scale):
        raise ValueError("fbp_host: scale must be finite")
    thetas = getattr(geometry, "angles", None)
    if thetas is not None and not np.isfinite(np.asarray(thetas)).all():
        raise ValueError("fbp_host: angles must be finite")

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
    depth = min(16 if slices is None else slices, grid.nz)
    local_grid, local_detector, first, step, integer = _slab_layout(grid, detector, depth)
    rows = local_detector.nv
    ramp = rfft_filter_array(cfg.filter, local_detector.nu, detector.du, jnp.float32)
    output = validate_host_arrays(projections, out, shape)
    host_poses = np.asarray(poses, np.float32)
    params = np.zeros((n_views, 6), np.float32)
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
            ProjectionRows(projections, v_start, rows),
            view_scale,
            params,
            ramp,
            0.0,
            grid=local_grid,
            detector=local_detector,
            backend=backend,
            batch_size=min(views, n_views),
            z_integer=integer,
            separable=True,
            check_finite=True,
        )
        count = min(depth, grid.nz - start)
        # The host copy synchronizes before the next slab reuses its buffers.
        output[:, :, start : start + count] = np.asarray(volume)[:, :, :count]
    return output


def _general_slabs(
    poses: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    cfg: FBPHostConfig,
    out: np.ndarray | None,
    backend: str,
    slices: int | None,
    views: int,
) -> np.ndarray:
    """Reconstruct x-slabs, streaming every view batch once per slab."""
    output = validate_host_arrays(projections, out, (grid.nx, grid.ny, grid.nz))
    view_scale, params, arc_length, separable = _view_weights(poses, cfg.scale)
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
        )
        output[start : start + count] = np.asarray(volume)[:count]
    return output
