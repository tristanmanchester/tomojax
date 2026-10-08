"""Coarse-to-fine CGLS initialization followed by a full-data solve."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
import operator
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp

from tomojax.core.multires import (
    bin_projections,
    bin_volume,
    scale_detector,
    scale_grid,
    upsample_volume,
    validate_scale_factor,
)
from tomojax.core.validation import (
    validate_detector_grid,
    validate_grid,
    validate_projection_stack,
    validate_volume,
)

from .cgls import CGLSConfig, cgls

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tomojax.geometry import Detector, Geometry, Grid


def cgls_multires(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    factors: Iterable[int] = (2, 1),
    iterations_per_level: Iterable[int] = (32, 16),
    init_x: jnp.ndarray | None = None,
    config: CGLSConfig | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, dict[str, object]]:
    """Initialize CGLS on coarser grids, then refine against every measured ray.

    ``factors`` must decrease strictly to 1. Each level gets its explicit
    positive iteration budget from ``iterations_per_level``; ``config.iterations`` is
    overridden. Other CGLS settings apply unchanged at each level, including
    damping on that level's discrete problem. Tolerances may stop levels early.

    Volume faces, world poses and physical density are preserved. Coarse data
    select a uniformly strided subset of measured rays without averaging or
    amplitude scaling. Explicit detector coordinates use the identical subset.
    A supplied initial volume is at full resolution and is resampled onto the
    first level. The final solve uses the original grid, detector and all data.

    Coarsening is an optional initialization strategy, not a quality guarantee;
    fine detail and noisy data may require more fine-level iterations. Diagnostics
    contain every level's actual work and the final level's stopping criterion.
    Numerical breakdown at any level raises ``FloatingPointError``.
    """
    factors = tuple(validate_scale_factor(f) for f in factors)
    budgets = tuple(operator.index(it) for it in iterations_per_level)
    if not factors or len(factors) != len(budgets):
        raise ValueError("cgls_multires: factors, iterations_per_level need equal nonzero lengths")
    if factors[-1] != 1 or any(a <= b for a, b in pairwise(factors)):
        raise ValueError("cgls_multires: factors must decrease strictly and end at 1")
    if any(it < 1 for it in budgets):
        raise ValueError("cgls_multires: every level requires a positive iteration budget")
    cfg = CGLSConfig() if config is None else config
    validate_grid(grid, "cgls_multires grid")
    validate_projection_stack(projections, detector, geometry=geometry, context="cgls_multires")
    validate_detector_grid(det_grid, detector, context="cgls_multires")
    if init_x is not None:
        validate_volume(init_x, grid, context="cgls_multires", name="init_x")
    data = jnp.asarray(projections, dtype=jnp.float32)
    if not bool(jnp.all(jnp.isfinite(data))):
        raise ValueError("cgls_multires: projections must be finite")
    initial = None if init_x is None else jnp.asarray(init_x, dtype=jnp.float32)
    if initial is not None and not bool(jnp.all(jnp.isfinite(initial))):
        raise ValueError("cgls_multires: initial volume must be finite")
    if det_grid is not None:
        det_grid = (
            jnp.asarray(det_grid[0], dtype=jnp.float32),
            jnp.asarray(det_grid[1], dtype=jnp.float32),
        )
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in det_grid):
            raise ValueError("cgls_multires: detector coordinates must be finite")
    volume = None if initial is None else bin_volume(initial, factors[0])
    levels: list[dict[str, object]] = []
    info: dict[str, object] = {}
    for factor, budget in zip(factors, budgets, strict=True):
        coarse_grid = scale_grid(grid, factor)
        coarse_detector = scale_detector(detector, factor)
        if volume is not None:
            volume = upsample_volume(volume, 1, (coarse_grid.nx, coarse_grid.ny, coarse_grid.nz))
        coordinates = None
        if det_grid is not None:
            u, v = det_grid
            coordinates = (
                bin_projections(u.reshape(1, detector.nv, detector.nu), factor).ravel(),
                bin_projections(v.reshape(1, detector.nv, detector.nu), factor).ravel(),
            )
        volume, info = cgls(
            geometry,
            coarse_grid,
            coarse_detector,
            bin_projections(data, factor),
            init_x=volume,
            config=replace(cfg, iterations=budget),
            det_grid=coordinates,
        )
        levels.append(
            {
                **info,
                "factor": factor,
                "requested_iterations": budget,
                "grid": coarse_grid.to_dict(),
                "detector": coarse_detector.to_dict(),
            }
        )
        if info["termination"] == "numerical_breakdown":
            raise FloatingPointError(f"cgls_multires: numerical breakdown at factor {factor}")
    assert volume is not None
    return volume, {
        **info,
        "factors": list(factors),
        "requested_iterations": sum(budgets),
        "fine_effective_iterations": info["effective_iterations"],
        "effective_iterations": sum(cast("int", level["effective_iterations"]) for level in levels),
        "levels": levels,
    }
