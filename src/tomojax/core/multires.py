"""Resolution-pyramid helpers for grids, detectors, projections, and volumes."""

from __future__ import annotations

from functools import cache
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .geometry.base import Detector, Grid, grid_volume_origin
from .validation import validate_detector, validate_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp


def validate_scale_factor(factor: object) -> int:
    """Return ``factor`` as an integer scale >= 1 or raise a clear ValueError."""
    try:
        value = float(factor)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Scale factor must be an integer >= 1, got {factor!r}") from exc
    if not math.isfinite(value) or value < 1 or int(value) != value:
        raise ValueError(f"Scale factor must be an integer >= 1, got {factor!r}")
    return int(value)


def scale_grid(grid: Grid, factor: int) -> Grid:
    """Coarsen the voxel count while preserving every physical volume face.

    Ceil division retains at least one voxel per axis. The spacing scales by
    the actual count ratio, including odd dimensions. Since ``vol_origin`` is
    a voxel centre, it must move by half the spacing change to retain the same
    lower face; keeping it unchanged would translate an explicitly placed grid.
    """
    f = validate_scale_factor(factor)
    validate_grid(grid, "scale_grid grid")
    if f == 1:
        return grid
    shape = (grid.nx, grid.ny, grid.nz)
    spacing = (grid.vx, grid.vy, grid.vz)
    coarse_shape = tuple(math.ceil(n / f) for n in shape)
    coarse_spacing = tuple(
        v * n / nc for v, n, nc in zip(spacing, shape, coarse_shape, strict=True)
    )
    origin = grid_volume_origin(grid)
    coarse_origin = tuple(
        o + (vc - v) / 2 for o, v, vc in zip(origin, spacing, coarse_spacing, strict=True)
    )
    return Grid(
        nx=coarse_shape[0],
        ny=coarse_shape[1],
        nz=coarse_shape[2],
        vx=coarse_spacing[0],
        vy=coarse_spacing[1],
        vz=coarse_spacing[2],
        vol_origin=coarse_origin,
        vol_center=grid.vol_center,
    )


def _decimated_axis(size: int, factor: int) -> tuple[int, int]:
    """Select a uniformly spaced, in-bounds subset without duplicated padding."""
    count = math.ceil(size / factor)
    offset = min(factor // 2, size - 1 - (count - 1) * factor)
    return count, offset


def scale_detector(det: Detector, factor: int) -> Detector:
    """Scale detector for a coarser multires level.

    Keep exactly the rays selected by ``bin_projections``. Spacing increases by
    ``factor`` and the centre follows the selected first/last pixels. No edge
    samples are duplicated or assigned fictitious uniformly spaced coordinates.
    """
    f = validate_scale_factor(factor)
    validate_detector(det, "scale_detector detector")
    if f == 1:
        return det
    nu, offset_u = _decimated_axis(det.nu, f)
    nv, offset_v = _decimated_axis(det.nv, f)

    def _scaled_center(n: int, d: float, center: float, n_coarse: int, offset: int) -> float:
        return center + (offset + (n_coarse - 1) * f / 2 - (n - 1) / 2) * d

    return Detector(
        nu=nu,
        nv=nv,
        du=det.du * f,
        dv=det.dv * f,
        center=(
            _scaled_center(det.nu, det.du, det.center[0], nu, offset_u),
            _scaled_center(det.nv, det.dv, det.center[1], nv, offset_v),
        ),
    )


def bin_projections[Array: jnp.ndarray | np.ndarray](proj: Array, factor: int) -> Array:
    """Decimate projections to uniformly spaced measured rays, without padding.

    This is point sampling, not detector-area averaging or an antialias filter.
    It preserves per-ray amplitude. The coarse solver is an initializer; final
    reconstruction must use the complete original data to recover fine detail.
    """
    f = validate_scale_factor(factor)
    if f == 1:
        return proj
    if proj.ndim != 3 or min(proj.shape) < 1:
        raise ValueError("bin_projections requires a nonempty (view, v, u) array")
    _, v0 = _decimated_axis(proj.shape[1], f)
    _, u0 = _decimated_axis(proj.shape[2], f)
    return cast("Array", proj[:, v0::f, u0::f])


def bin_volume(vol: jnp.ndarray, factor: int) -> jnp.ndarray:
    """Resample a volume onto ``scale_grid``'s grid with an antialias filter."""
    f = validate_scale_factor(factor)
    if f == 1:
        return vol
    if vol.ndim != 3 or min(vol.shape) < 1:
        raise ValueError("bin_volume requires a nonempty (x, y, z) array")
    shape = tuple(math.ceil(n / f) for n in vol.shape)
    import jax.image as jimage

    return jimage.resize(vol, shape, method="linear", antialias=True)


def _linear_resize[Array: jnp.ndarray | np.ndarray](
    vol: Array, shape: tuple[int, int, int], *, xp: Any
) -> Array:
    """Half-pixel linear interpolation without dense, mostly-zero weight matrices."""
    for axis, size in enumerate(shape):
        old = vol.shape[axis]
        if size == old:
            continue
        dtype = vol.dtype if xp is np else xp.result_type(0.0)
        coordinate = (xp.arange(size, dtype=dtype) + 0.5) * (1 / (size / old)) - 0.5
        coordinate = xp.clip(coordinate, 0, old - 1)
        lower = xp.floor(coordinate).astype(xp.int32)
        upper = xp.minimum(lower + 1, old - 1)
        broadcast = [1] * vol.ndim
        broadcast[axis] = size
        weight = (coordinate - lower).astype(vol.dtype).reshape(broadcast)
        left = xp.take(vol, lower, axis=axis, mode="clip")
        right = xp.take(vol, upper, axis=axis, mode="clip")
        vol = (1 - weight) * left + weight * right
    return vol


@cache
def _device_linear_resize() -> Callable[[jnp.ndarray, tuple[int, int, int]], jnp.ndarray]:
    import jax
    import jax.numpy as jnp

    return jax.jit(
        lambda vol, shape: _linear_resize(vol, shape, xp=jnp), static_argnames=("shape",)
    )


def upsample_volume[Array: jnp.ndarray | np.ndarray](
    vol: Array, factor: int, target_shape: tuple[int, int, int]
) -> Array:
    """Resize to `target_shape`, retaining host/device storage for floating arrays."""
    validate_scale_factor(factor)
    out_shape = tuple(int(s) for s in target_shape)
    if len(out_shape) != 3 or any(s < 1 for s in out_shape):
        raise ValueError(f"target_shape must contain positive dimensions, got {target_shape!r}")
    if tuple(int(s) for s in vol.shape) == out_shape:
        return vol
    if vol.dtype in (np.float32, np.float64):
        if isinstance(vol, np.ndarray):
            return cast("Array", _linear_resize(vol, out_shape, xp=np))
        return cast("Array", _device_linear_resize()(vol, out_shape))
    import jax.image as jimage

    v = jimage.resize(vol, out_shape, method="linear", antialias=False)
    converted = v.astype(vol.dtype)
    return cast("Array", np.asarray(converted) if isinstance(vol, np.ndarray) else converted)
