"""Parallel FBP with host storage and bounded axial slabs on the accelerator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import validate_grid, validate_projection_stack
from tomojax.geometry import Detector, Grid, ParallelGeometry, grid_volume_origin

from .fbp import (
    _parallel_filter_detector,  # pyright: ignore[reportPrivateUsage]
    _rfft_filter_array,  # pyright: ignore[reportPrivateUsage]
    _run_parallel_fbp_streamed,  # pyright: ignore[reportPrivateUsage]
    default_fbp_scale,
    supports_parallel_fbp_z_integer,
)


@dataclass(frozen=True, slots=True)
class FBPHostConfig:
    """Control slab storage and filter batches for :func:`fbp_host`.

    Both batch sizes are explicit positive limits. The entire input and output
    remain in host storage; only a detector-row slab, output slab and bounded
    filtering workspace reside on the accelerator. Runtime/compiler caches add
    memory beyond these arrays. The default scale assumes a uniform half turn.
    """

    slices_per_batch: int = 16
    views_per_batch: int = 32
    filter_name: str = "ramp"
    scale: float | None = None
    backprojector: Literal["auto", "jax", "pallas"] = "auto"


def _mapped_storage(array: np.ndarray) -> np.memmap | None:
    current: object = array
    while isinstance(current, np.ndarray):
        if isinstance(current, np.memmap):
            return current
        current = current.base
    return None


def _validate_host_arrays(
    projections: np.ndarray,
    out: np.ndarray | None,
    shape: tuple[int, int, int],
    context: str = "fbp_host",
) -> np.ndarray:
    if not isinstance(projections, np.ndarray) or projections.dtype.kind not in "buif":
        raise TypeError(f"{context}: projections must be a real NumPy array or memmap")
    if out is None:
        return np.empty(shape, dtype=np.float32)
    if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != np.float32:
        raise ValueError(f"{context}: out must be a float32 NumPy array with the volume shape")
    if not out.flags.writeable:
        raise ValueError(f"{context}: out must be writable")
    if np.may_share_memory(projections, out):
        raise ValueError(f"{context}: input and output storage must not overlap")
    input_map, output_map = _mapped_storage(projections), _mapped_storage(out)
    if input_map is not None and output_map is not None:
        source, target = input_map.filename, output_map.filename
        if source is None or target is None or Path(source).samefile(target):
            raise ValueError(f"{context}: memory-mapped input and output require separate files")
    return out


def _slab_layout(
    grid: Grid, detector: Detector, depth: int
) -> tuple[Grid, Detector, float, float, bool]:
    """Choose a fixed local coordinate system and interpolation support."""
    ox, oy, oz = grid_volume_origin(grid)
    integer = supports_parallel_fbp_z_integer(grid, detector)
    first = (oz - detector.det_center[1]) / detector.dv + (detector.nv - 1) / 2
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
        detector, nv=rows, det_center=(detector.det_center[0], (rows - 1) * detector.dv / 2)
    )
    local_detector = _parallel_filter_detector(local_grid, local_detector)
    return local_grid, local_detector, first, step, integer


def _projection_slab(projections: np.ndarray, start: int, rows: int) -> np.ndarray:
    """Copy measured rows, allocating zero padding only when it is needed."""
    lo, hi = max(0, start), min(projections.shape[1], start + rows)
    if lo == start and hi == start + rows:
        return np.ascontiguousarray(projections[:, lo:hi], dtype=np.float32)
    data = np.zeros((projections.shape[0], rows, projections.shape[2]), dtype=np.float32)
    if hi > lo:
        data[:, lo - start : hi - start] = projections[:, lo:hi]
    return data


def fbp_host(
    geometry: ParallelGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    *,
    config: FBPHostConfig | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct built-in parallel geometry without full-volume device storage.

    Accept NumPy arrays, including memmaps, and return a float32 NumPy array.
    ``out`` may be a writable memmap. Completed slabs are written immediately;
    a later failure can leave a partially written output. Input storage must
    remain unchanged until the call finishes. This host-returning routine is
    not differentiable. Use :func:`fbp` for device-resident output.

    Anisotropic spacing, detector offsets and shifted volume origins are
    supported. Fractional detector-row interpolation retains both neighbors.
    Filter tails use the same zero-extended measurement model as ``fbp``.
    Laminography and custom geometry are rejected: axial separation requires
    the built-in z-axis parallel convention.
    """
    cfg = FBPHostConfig() if config is None else config
    slices = operator.index(cfg.slices_per_batch)
    views = operator.index(cfg.views_per_batch)
    if slices < 1 or views < 1:
        raise ValueError("fbp_host: slices_per_batch and views_per_batch must be positive")
    if type(geometry) is not ParallelGeometry:
        raise ValueError("fbp_host: requires built-in ParallelGeometry")
    if cfg.backprojector not in {"auto", "jax", "pallas"}:
        raise ValueError("fbp_host: backprojector must be 'auto', 'jax' or 'pallas'")
    shape = validate_grid(grid, "fbp_host grid")
    n_views, _, _ = validate_projection_stack(
        projections, detector, geometry=geometry, context="fbp_host projections"
    )
    ox, oy, oz = grid_volume_origin(grid)
    if not np.isfinite([ox, oy, oz, *detector.det_center]).all():
        raise ValueError("fbp_host: grid and detector placement must be finite")
    scale = default_fbp_scale(n_views) if cfg.scale is None else float(cfg.scale)
    if not math.isfinite(scale):
        raise ValueError("fbp_host: scale must be finite")
    if not np.isfinite(np.asarray(geometry.thetas_deg)).all():
        raise ValueError("fbp_host: angles must be finite")

    poses = stack_view_poses(geometry, n_views)
    cuda = all(
        d.platform == "gpu" and d.client.platform_version.lower().startswith("cuda")
        for d in poses.devices()
    )
    if cfg.backprojector == "pallas" and not cuda:
        raise ValueError("fbp_host: Pallas requires CUDA")
    backend = "pallas" if cuda and cfg.backprojector != "jax" else "jax"
    depth = min(slices, grid.nz)
    local_grid, local_detector, first, step, integer = _slab_layout(grid, detector, depth)
    rows = local_detector.nv
    ramp = _rfft_filter_array(cfg.filter_name, local_detector.nu, detector.du, jnp.float32)
    output = _validate_host_arrays(projections, out, shape)
    for start in range(0, grid.nz, depth):
        v = first + start * step
        v_start = math.floor(v)
        if not integer:
            v_start = max(0, min(v_start, detector.nv - rows))
        data = _projection_slab(projections, v_start, rows)
        if not np.isfinite(data).all():
            raise ValueError("fbp_host: sampled projection rows must be finite in FP32")
        # Local grid/detector metadata stay identical for every slab. Only the
        # fractional row phase changes, passed as a dynamic pose translation.
        local_poses = poses if integer else poses.at[:, 2, 3].set((v - v_start) * detector.dv)
        volume = _run_parallel_fbp_streamed(
            local_poses,
            jnp.asarray(data),
            ramp,
            grid=local_grid,
            detector=local_detector,
            backend=backend,
            batch_size=min(views, n_views),
            z_integer=integer,
        )
        count = min(depth, grid.nz - start)
        # The host copy synchronizes before the next slab reuses its buffers.
        np.multiply(
            np.asarray(volume)[:, :, :count], scale, out=output[:, :, start : start + count]
        )
    return output
