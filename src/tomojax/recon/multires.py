from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import jax.numpy as jnp

from ..core.geometry import Grid, Detector, Geometry
from .fista_tv import fista_tv, grad_data_term
from ..utils.logging import progress_iter


def scale_grid(grid: Grid, factor: int) -> Grid:
    assert factor >= 1 and int(factor) == factor
    f = int(factor)
    assert grid.nx % f == 0 and grid.ny % f == 0 and grid.nz % f == 0, "Grid dims must be divisible by factor"
    return Grid(
        nx=grid.nx // f,
        ny=grid.ny // f,
        nz=grid.nz // f,
        vx=grid.vx * f,
        vy=grid.vy * f,
        vz=grid.vz * f,
        vol_origin=grid.vol_origin,
        vol_center=grid.vol_center,
    )


def scale_detector(det: Detector, factor: int) -> Detector:
    assert factor >= 1 and int(factor) == factor
    f = int(factor)
    assert det.nu % f == 0 and det.nv % f == 0, "Detector dims must be divisible by factor"
    return Detector(
        nu=det.nu // f,
        nv=det.nv // f,
        du=det.du * f,
        dv=det.dv * f,
        det_center=det.det_center,
    )


def bin_projections(proj: jnp.ndarray, factor: int) -> jnp.ndarray:
    """Downsample projections by strided pick to preserve per-ray amplitude.

    Using mean pooling can reduce per-ray amplitude; for our forward model,
    each pixel is an independent ray integral, so stride-based decimation is
    a better coarse approximation.
    """
    if factor == 1:
        return proj
    n_views, nv, nu = proj.shape
    f = int(factor)
    assert nv % f == 0 and nu % f == 0
    v0 = f // 2
    u0 = f // 2
    return proj[:, v0::f, u0::f]


def bin_volume(vol: jnp.ndarray, factor: int) -> jnp.ndarray:
    if factor == 1:
        return vol
    nx, ny, nz = vol.shape
    f = int(factor)
    assert nx % f == 0 and ny % f == 0 and nz % f == 0
    v = vol.reshape(nx // f, f, ny // f, f, nz // f, f)
    # Average within f^3 blocks
    return v.mean(axis=(1, 3, 5))


def upsample_volume(vol: jnp.ndarray, factor: int, target_shape: Tuple[int, int, int]) -> jnp.ndarray:
    if factor == 1:
        return vol
    v = jnp.repeat(vol, factor, axis=0)
    v = jnp.repeat(v, factor, axis=1)
    v = jnp.repeat(v, factor, axis=2)
    # Crop in case of rounding (should match exactly when divisible)
    return v[: target_shape[0], : target_shape[1], : target_shape[2]]


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
