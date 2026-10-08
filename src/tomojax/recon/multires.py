"""Multiresolution reconstruction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomojax.core import progress_iter
from tomojax.core.multires import (
    bin_projections,
    create_resolution_pyramid,
    scale_detector,
    scale_grid,
    upsample_volume,
    validate_scale_factor,
)

from .fista_tv import FistaConfig, fista_tv

if TYPE_CHECKING:
    from collections.abc import Iterable

    import jax.numpy as jnp

    from tomojax.core.geometry.base import Detector, Geometry, Grid

_validated_scale_factor = validate_scale_factor

__all__ = [
    "bin_projections",
    "fista_multires",
    "scale_detector",
    "scale_grid",
    "upsample_volume",
]


def fista_multires(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    factors: Iterable[int] = (2, 1),
    iterations_per_level: Iterable[int] = (10, 10),
    tv_weight: float = 0.005,
) -> tuple[jnp.ndarray, dict]:
    """Coarse-to-fine FISTA reconstruction using simple binning/upsampling.

    Returns (x, info) where x is at finest resolution.
    """
    factors = tuple(validate_scale_factor(f) for f in factors)
    iterations_per_level = tuple(int(it) for it in iterations_per_level)
    if len(factors) != len(iterations_per_level):
        raise ValueError(
            "factors and iterations_per_level must have the same length; "
            f"got {len(factors)} and {len(iterations_per_level)}"
        )
    levels = create_resolution_pyramid(grid, detector, projections, factors)
    # Heuristic: if coarse levels get very few iterations, skip them and spend
    # the entire budget at the finest level (often better for tiny problems/tests).
    if len(levels) >= 2 and levels[-1]["factor"] == 1:
        coarse_iters = int(sum(iterations_per_level[:-1]))
        if coarse_iters <= 3:
            total_iters = int(sum(iterations_per_level))
            x_fine, info = fista_tv(
                geometry,
                grid,
                detector,
                projections,
                config=FistaConfig(iterations=total_iters, tv_weight=tv_weight),
            )
            return x_fine, {"loss": info.get("loss", []), "factors": list(factors)}
    x_init = None
    prev_factor: int | None = None
    loss_hist = []
    level_plan = list(zip(levels, iterations_per_level, strict=False))
    for lvl, iters in progress_iter(level_plan, total=len(level_plan), desc="Multires: levels"):
        g = lvl["grid"]
        d = lvl["detector"]
        y = lvl["projections"]
        if x_init is not None:
            # Upsample previous x to current level as initialization
            f_up = prev_factor // lvl["factor"]
            x0 = upsample_volume(x_init, f_up, (g.nx, g.ny, g.nz))
        else:
            x0 = None
        x_lvl, info = fista_tv(
            geometry,
            g,
            d,
            y,
            init_x=x0,
            config=FistaConfig(iterations=iters, tv_weight=tv_weight),
        )
        loss_hist.extend(info.get("loss", []))
        # Prepare initialization for next (finer) level
        prev_factor = lvl["factor"]
        x_init = x_lvl

    if x_init is None or prev_factor is None:
        raise ValueError("fista_multires requires at least one resolution level")

    # Upsample to finest resolution if the last processed level is still coarse.
    if prev_factor != 1:
        x_final = upsample_volume(x_init, prev_factor, (grid.nx, grid.ny, grid.nz))
    else:
        x_final = x_init

    info_all = {"loss": loss_hist, "factors": list(factors)}
    return x_final, info_all
