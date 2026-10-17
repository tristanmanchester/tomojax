"""Row/column-normalized simultaneous and ordered-subset algebraic reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import math
import operator
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import (
    validate_detector_grid,
    validate_grid,
    validate_projection_stack,
    validate_volume,
)
from tomojax.recon._axial_slabs import axial_slabs, independent_planes
from tomojax.recon._projection import (
    ConeModel,
    ProjectorBackend,
    ProjectorModel,
    resolve_geometry_projector,
    subset_operators,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry


@dataclass(frozen=True, kw_only=True)
class SIRTConfig:
    """SIRT, or OS-SART with ``subsets > 1`` (one view per subset is SART).

    Each iteration visits all interleaved subsets once, in input view order.
    A subset updates ``x += relaxation * C * A.T * R * (data - A*x)``, where
    ``R = 1 / (A*support)`` and ``C = 1 / (A.T*ones)``; zero denominators give
    zero weights. The forward operator and transpose are exactly matched.
    ``relaxation`` must be in (0, 2); ordered subsets can cycle on inconsistent
    data, so use fewer subsets or lower relaxation for the final refinement.

    ``acceleration=True`` applies fixed FISTA momentum to full-data SIRT
    (``subsets=1``, ``relaxation <= 1``), reducing the iterations needed for
    the same weighted least-squares objective without changing the projector.

    ``lower_bound`` and ``upper_bound`` clip each update. ``support`` is a volume
    mask: zero excludes a voxel from reconstruction and fixes it to zero,
    including in the initial volume; nonzero includes it. Bounds apply only
    inside that support. Only nonnegative projector weights are used.
    """

    iterations: int = 50
    subsets: int = 1
    relaxation: float = 1.0
    acceleration: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    support: jax.Array | np.ndarray | None = None
    views_per_batch: int = 64
    projector_backend: ProjectorBackend = "auto"
    projector_model: ProjectorModel = "auto"


def _inverse(value: jax.Array) -> jax.Array:
    positive = value > 0
    return jnp.where(positive, 1.0 / jnp.where(positive, value, 1.0), 0.0)


def _accelerated_steps(
    update: Callable[[jax.Array, jax.Array], jax.Array],
    initial: jax.Array,
    iterations: jax.Array,
) -> jax.Array:
    # Row/column normalization bounds the spectrum by one in the C^-1 norm.
    # Box/support projection is coordinatewise in that norm, so the usual
    # FISTA recurrence applies with a relaxation of at most one.
    def step(
        index: jax.Array, state: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        previous, extrapolated, time = state
        current = update(index, extrapolated)
        next_time = (1.0 + jnp.sqrt(1.0 + 4.0 * time * time)) / 2.0
        momentum = (time - 1.0) / next_time
        return current, current + momentum * (current - previous), next_time

    return jax.lax.fori_loop(0, iterations, step, (initial, initial, jnp.float32(1.0)))[0]


@partial(
    jax.jit,
    static_argnames=(
        "grid",
        "detector",
        "backend",
        "batch",
        "model",
        "subsets",
        "cache_columns",
        "axial",
        "acceleration",
    ),
)
def _solve(
    poses: jax.Array,
    data: jax.Array,
    initial: jax.Array | None,
    support: jax.Array | None,
    det_grid: tuple[jax.Array, jax.Array] | None,
    iterations: jax.Array,
    relaxation: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch: int,
    model: str | ConeModel,
    subsets: int,
    cache_columns: bool,
    axial: bool,
    acceleration: bool,
) -> jax.Array:
    select = subset_operators(poses, grid, detector, det_grid, backend, batch, model, subsets)
    shape = (grid.nx, grid.ny, grid.nz)
    normalization = select
    normal_rows = detector.nv
    normal_shape = shape
    if axial:
        from tomojax.geometry import grid_volume_origin

        normal_grid = replace(grid, nz=1, vol_origin=(*grid_volume_origin(grid)[:2], 0.0))
        normal_detector = replace(detector, nv=1, center=(detector.center[0], 0.0))
        normalization = subset_operators(
            poses, normal_grid, normal_detector, None, backend, batch, model, subsets
        )
        normal_rows, normal_shape = 1, (grid.nx, grid.ny, 1)
    mask = jnp.ones(shape, jnp.float32) if support is None else (support != 0).astype(jnp.float32)

    def constrain(x: jax.Array) -> jax.Array:
        return jnp.where(mask != 0, jnp.clip(x, lower, upper), 0.0)

    initial = constrain(jnp.zeros(shape, jnp.float32) if initial is None else initial)

    def row_weights(index: jax.Array) -> jax.Array:
        forward, _, _ = (normalization if support is None else select)(index)
        return _inverse(forward(jnp.ones(normal_shape, jnp.float32) if support is None else mask))

    def column_weights(index: jax.Array) -> jax.Array:
        _, adjoint, _ = normalization(index)
        count = (data.shape[0] + subsets - 1) // subsets
        return _inverse(adjoint(jnp.ones((count, normal_rows, detector.nu), jnp.float32)))

    rows = jax.lax.map(row_weights, jnp.arange(subsets))
    # Avoid storing a volume for every subset on large scans. The exact subset
    # normalization is then recomputed at each update, not approximated by a
    # rescaled full-scan normalization. Cone updates share its ray geometry
    # with the residual transpose without sharing their accumulators.
    columns = jax.lax.map(column_weights, jnp.arange(subsets)) if cache_columns else None

    def update(step: jax.Array, x: jax.Array) -> jax.Array:
        index = step % subsets
        _, _, residual_adjoint = select(index)
        if columns is None and not axial:
            gradient, norm = residual_adjoint(x, data, rows[index], return_column=True)
            column = _inverse(norm)
        else:
            column = column_weights(index) if columns is None else columns[index]
            gradient = residual_adjoint(x, data, rows[index])
        return constrain(x + relaxation * column * gradient)

    if acceleration:
        return _accelerated_steps(update, initial, iterations)
    return jax.lax.fori_loop(0, iterations * subsets, update, initial)


def sirt(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    init_x: jax.Array | np.ndarray | None = None,
    config: SIRTConfig | None = None,
    det_grid: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array | np.ndarray, dict[str, object]]:
    """Reconstruct with SIRT/OS-SART; return the volume and full-sweep count.

    Supports parallel, tilted and cone-beam scans on the existing matched
    projectors. Inputs and updates are FP32. An iteration is a complete sweep,
    not one subset update. No residual-based stopping or quality claim is made.
    Large axial Joseph scans whose detector rows coincide with voxel planes
    run in independent slabs, returning a host array without changing the solve.
    """
    cfg = SIRTConfig() if config is None else config
    iterations = operator.index(cfg.iterations)
    subsets, batch = operator.index(cfg.subsets), operator.index(cfg.views_per_batch)
    validate_grid(grid, "sirt grid")
    n, _, _ = validate_projection_stack(projections, detector, geometry=geometry, context="sirt")
    validate_detector_grid(det_grid, detector, context="sirt")
    if iterations < 0 or batch < 1 or not 1 <= subsets <= n:
        raise ValueError(
            "sirt: iterations >= 0, views_per_batch >= 1 and 1 <= subsets <= views required"
        )
    if not math.isfinite(cfg.relaxation) or not 0 < cfg.relaxation < 2:
        raise ValueError("sirt: relaxation must be finite and in (0, 2)")
    if cfg.acceleration and (subsets != 1 or cfg.relaxation > 1):
        raise ValueError("sirt: acceleration requires subsets=1 and relaxation <= 1")
    lower = -np.inf if cfg.lower_bound is None else float(cfg.lower_bound)
    upper = np.inf if cfg.upper_bound is None else float(cfg.upper_bound)
    if lower > upper:
        raise ValueError("sirt: lower_bound must not exceed upper_bound")
    for name, value in (("init_x", init_x), ("support", cfg.support)):
        if value is not None:
            validate_volume(value, grid, context="sirt", name=name)
    model, backend = resolve_geometry_projector(
        geometry,
        cfg.projector_model,
        cfg.projector_backend,
        detector=detector,
        det_grid=det_grid,
        context="sirt",
    )
    axial = model == "joseph" and det_grid is None and independent_planes(geometry, grid, detector)
    if axial:

        def solve_slab(
            geometry: ParallelGeometry,
            data: jax.Array | np.ndarray,
            initial: jax.Array | np.ndarray | None,
            support: jax.Array | np.ndarray | None,
        ) -> tuple[jax.Array | np.ndarray, dict[str, object]]:
            return sirt(
                geometry,
                geometry.grid,
                geometry.detector,
                data,
                init_x=initial,
                config=replace(cfg, support=support),
            )

        result = axial_slabs(
            geometry,
            grid,
            detector,
            projections,
            initial=init_x,
            support=cfg.support,
            volume_buffers=3 + min(subsets, 16) + 2 * cfg.acceleration,
            projection_buffers=2 + 2 / subsets,
            solve=solve_slab,
        )
        if result is not None:
            return result
    from tomojax.backends import device_free_memory_bytes

    free = device_free_memory_bytes()
    column_budget = 256 * 1024**2 if free is None else free // 4
    poses = stack_view_poses(geometry, n)
    volume = _solve(
        poses, jnp.asarray(projections, jnp.float32),
        None if init_x is None else jnp.asarray(init_x, jnp.float32),
        None if cfg.support is None else jnp.asarray(cfg.support),
        det_grid, jnp.int32(iterations), jnp.float32(cfg.relaxation),
        jnp.float32(lower), jnp.float32(upper), grid=grid, detector=detector,
        backend=backend, batch=batch, model=model, subsets=subsets,
        cache_columns=subsets * grid.nx * grid.ny * (1 if axial else grid.nz) * 4 <= column_budget,
        axial=axial, acceleration=cfg.acceleration,
    )  # fmt: skip
    return volume, {
        "effective_iterations": iterations,
        "subset_updates": iterations * subsets,
        "termination": "iteration_limit",
        "projector_backend": backend,
        "projector_model": model,
    }
