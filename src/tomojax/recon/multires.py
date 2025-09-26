from __future__ import annotations

from typing import Iterable, List, Tuple
import math

import jax.numpy as jnp
import jax.image as jimage

from ..core.geometry import Grid, Detector, Geometry
from .fista_tv import fista_tv, grad_data_term
from ..utils.logging import progress_iter


def scale_grid(grid: Grid, factor: int) -> Grid:
    """Scale grid for a coarser multires level, tolerating non-divisible dims.

    - New dims use ceil division to retain coverage when dims aren't divisible.
    - Voxel sizes are multiplied by the factor to preserve physical extent.
    """
    assert factor >= 1 and int(factor) == factor
    f = int(factor)
    nx = int(math.ceil(grid.nx / f))
    ny = int(math.ceil(grid.ny / f))
    nz = int(math.ceil(grid.nz / f))
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
    per-ray spacing consistent with decimated projections.
    """
    assert factor >= 1 and int(factor) == factor
    f = int(factor)
    nu = int(math.ceil(det.nu / f))
    nv = int(math.ceil(det.nv / f))
    return Detector(
        nu=nu,
        nv=nv,
        du=det.du * f,
        dv=det.dv * f,
        det_center=det.det_center,
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

    - Pads to make dims divisible by `factor` (edge mode), then takes one pixel
      per f×f block using a centered offset (f//2). This preserves per-ray scale
      better than averaging while tolerating arbitrary input sizes.
    """
    if factor == 1:
        return proj
    f = int(factor)
    y = _pad_to_multiple_jnp(proj, f, f)
    v0 = f // 2
    u0 = f // 2
    return y[:, v0::f, u0::f]


def bin_volume(vol: jnp.ndarray, factor: int) -> jnp.ndarray:
    if factor == 1:
        return vol
    f = int(factor)
    # Pad to multiples on all three dims (edge) then average f^3 blocks
    nx, ny, nz = vol.shape
    px = (f - (nx % f)) % f
    py = (f - (ny % f)) % f
    pz = (f - (nz % f)) % f
    if px or py or pz:
        vol = jnp.pad(vol, ((0, px), (0, py), (0, pz)), mode="edge")
    nx, ny, nz = vol.shape
    v = vol.reshape(nx // f, f, ny // f, f, nz // f, f)
    return v.mean(axis=(1, 3, 5))


def upsample_volume(vol: jnp.ndarray, factor: int, target_shape: Tuple[int, int, int]) -> jnp.ndarray:
    if factor == 1:
        return vol
    # Trilinear resize to avoid nearest-neighbor blockiness when moving to finer levels
    out_shape = (int(target_shape[0]), int(target_shape[1]), int(target_shape[2]))
    v = jimage.resize(vol, out_shape, method="linear", antialias=False)
    return v


def create_resolution_pyramid(
    grid: Grid, detector: Detector, projections: jnp.ndarray, factors: Iterable[int]
):
    levels: List[dict] = []
    for f in factors:
        levels.append(
            {
                "factor": int(f),
                "grid": scale_grid(grid, f),
                "detector": scale_detector(detector, f),
                "projections": bin_projections(projections, f),
            }
        )
    return levels


def fista_multires(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    factors: Iterable[int] = (2, 1),
    iters_per_level: Iterable[int] = (10, 10),
    lambda_tv: float = 0.005,
) -> tuple[jnp.ndarray, dict]:
    """Coarse-to-fine FISTA reconstruction using simple binning/upsampling.

    Returns (x, info) where x is at finest resolution.
    """
    factors = tuple(factors)
    iters_per_level = tuple(iters_per_level)
    levels = create_resolution_pyramid(grid, detector, projections, factors)
    # Heuristic: if coarse levels get very few iterations, skip them and spend
    # the entire budget at the finest level (often better for tiny problems/tests).
    if len(levels) >= 2 and levels[-1]["factor"] == 1:
        coarse_iters = int(sum(iters_per_level[:-1]))
        if coarse_iters <= 3:
            total_iters = int(sum(iters_per_level))
            x_fine, info = fista_tv(geometry, grid, detector, projections, iters=total_iters, lambda_tv=lambda_tv)
            return x_fine, {"loss": info.get("loss", []), "factors": list(factors)}
    x_init = None
    loss_hist = []
    for lvl, iters in progress_iter(list(zip(levels, iters_per_level)), total=len(levels), desc="Multires: levels"):
        g = lvl["grid"]
        d = lvl["detector"]
        y = lvl["projections"]
        if x_init is not None:
            # Upsample previous x to current level as initialization
            f_up = prev_factor // lvl["factor"]
            x0 = upsample_volume(x_init, f_up, (g.nx, g.ny, g.nz))
        else:
            x0 = None
        x_lvl, info = fista_tv(geometry, g, d, y, iters=iters, lambda_tv=lambda_tv, init_x=x0)
        loss_hist.extend(info.get("loss", []))
        # Prepare initialization for next (finer) level
        prev_factor = lvl["factor"]
        x_init = x_lvl

    # Upsample to finest (factor=1) if last level not finest
    if levels and levels[-1]["factor"] != 1:
        # Find finest grid
        finest_grid = levels[-1]["grid"]
        x_final = upsample_volume(x_init, levels[-1]["factor"], (grid.nx, grid.ny, grid.nz))
    else:
        x_final = x_init

    info_all = {"loss": loss_hist, "factors": list(factors)}
    return x_final, info_all
