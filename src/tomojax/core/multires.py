"""Resolution-pyramid helpers for grids, detectors, projections, and volumes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax.image as jimage
import jax.numpy as jnp

from .geometry.base import Detector, Grid
from .validation import validate_detector, validate_grid, validate_projection_stack

if TYPE_CHECKING:
    from collections.abc import Iterable


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
    """Scale grid for a coarser multires level, tolerating non-divisible dims.

    - New dims use ceil division to retain coverage when dims aren't divisible.
    - Voxel sizes are multiplied by the factor to preserve physical extent.
    """
    f = validate_scale_factor(factor)
    validate_grid(grid, "scale_grid grid")
    nx = math.ceil(grid.nx / f)
    ny = math.ceil(grid.ny / f)
    nz = math.ceil(grid.nz / f)
    return Grid(
        nx=nx,
        ny=ny,
        nz=nz,
        vx=grid.vx * f,
        vy=grid.vy * f,
        vz=grid.vz * f,
        vol_origin=grid.vol_origin,
        vol_center=grid.vol_center,
    )


def scale_detector(det: Detector, factor: int) -> Detector:
    """Scale detector for a coarser multires level.

    Supports non-divisible sizes by using ceil(n/f) and increasing pixel size.
    The projector operates in world units; increasing du/dv by `factor` keeps
    per-ray spacing consistent with decimated projections. ``bin_projections``
    selects the center sample from each padded f x f block, so the coarse detector
    center must shift to keep those coarse rays aligned with the sampled pixels.
    """
    f = validate_scale_factor(factor)
    validate_detector(det, "scale_detector detector")
    nu = math.ceil(det.nu / f)
    nv = math.ceil(det.nv / f)

    def _scaled_center(n: int, d: float, center: float) -> float:
        pad = (f - (n % f)) % f
        left = pad // 2
        first_sample = ((f // 2) - left - (n / 2.0 - 0.5)) * d + center
        return first_sample + (math.ceil(n / f) / 2.0 - 0.5) * (d * f)

    return Detector(
        nu=nu,
        nv=nv,
        du=det.du * f,
        dv=det.dv * f,
        det_center=(
            _scaled_center(det.nu, det.du, det.det_center[0]),
            _scaled_center(det.nv, det.dv, det.det_center[1]),
        ),
    )


def _pad_to_multiple_jnp(arr: jnp.ndarray, m_v: int, m_u: int) -> jnp.ndarray:
    """Symmetrically pad last two dims to multiples of (m_v, m_u) using edge mode."""
    if m_v <= 1 and m_u <= 1:
        return arr
    nv = arr.shape[-2]
    nu = arr.shape[-1]
    pad_v = (m_v - (nv % m_v)) % m_v if m_v > 1 else 0
    pad_u = (m_u - (nu % m_u)) % m_u if m_u > 1 else 0
    if pad_v == 0 and pad_u == 0:
        return arr
    pv0 = pad_v // 2
    pv1 = pad_v - pv0
    pu0 = pad_u // 2
    pu1 = pad_u - pu0
    pad_width = ((0, 0), (pv0, pv1), (pu0, pu1))
    return jnp.pad(arr, pad_width, mode="edge")


def bin_projections(proj: jnp.ndarray, factor: int) -> jnp.ndarray:
    """Downsample projections by strided pick with symmetric edge padding.

    Pads to make dims divisible by `factor` (edge mode), then takes one pixel
    per f x f block using a centered offset (f//2). This preserves per-ray scale
    better than averaging while tolerating arbitrary input sizes.
    """
    f = validate_scale_factor(factor)
    if f == 1:
        return proj
    y = _pad_to_multiple_jnp(proj, f, f)
    v0 = f // 2
    u0 = f // 2
    return y[:, v0::f, u0::f]


def bin_volume(vol: jnp.ndarray, factor: int) -> jnp.ndarray:
    """Downsample a volume by block averaging with edge padding."""
    f = validate_scale_factor(factor)
    if f == 1:
        return vol
    nx, ny, nz = vol.shape
    px = (f - (nx % f)) % f
    py = (f - (ny % f)) % f
    pz = (f - (nz % f)) % f
    if px or py or pz:
        vol = jnp.pad(vol, ((0, px), (0, py), (0, pz)), mode="edge")
    nx, ny, nz = vol.shape
    v = vol.reshape(nx // f, f, ny // f, f, nz // f, f)
    return v.mean(axis=(1, 3, 5))


def upsample_volume(
    vol: jnp.ndarray, factor: int, target_shape: tuple[int, int, int]
) -> jnp.ndarray:
    """Resize `vol` to `target_shape`, regardless of the nominal scale factor."""
    validate_scale_factor(factor)
    out_shape = tuple(int(s) for s in target_shape)
    if len(out_shape) != 3 or any(s < 1 for s in out_shape):
        raise ValueError(f"target_shape must contain positive dimensions, got {target_shape!r}")
    if tuple(int(s) for s in vol.shape) == out_shape:
        return vol
    v = jimage.resize(vol, out_shape, method="linear", antialias=False)
    return v.astype(vol.dtype)


def create_resolution_pyramid(
    grid: Grid, detector: Detector, projections: jnp.ndarray, factors: Iterable[int]
) -> list[dict[str, Any]]:
    """Create coarser grid/detector/projection levels for each scale factor."""
    levels: list[dict[str, Any]] = []
    for f in factors:
        factor = validate_scale_factor(f)
        levels.append(
            {
                "factor": factor,
                "grid": scale_grid(grid, factor),
                "detector": scale_detector(detector, factor),
                "projections": bin_projections(projections, factor),
            }
        )
        validate_projection_stack(
            levels[-1]["projections"],
            levels[-1]["detector"],
            context=f"create_resolution_pyramid factor {factor} projections",
        )
    return levels
