"""Memory-bounded execution of resident nonnegative FISTA.

Residuals occupy detector storage; their scaled transpose updates an owned
extrapolate in place. Only owned work arrays are donated, never caller arrays.
The projector and optimizer are unchanged; scaling before the transpose changes
FP32 summation order. Small-volume problems retain the fused-gradient scan.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from tomojax.core.operator_norm import unit

from ._fista_initialization import ordered_data, power_start, subset_order
from ._projection import least_squares_operators, positive_normal_bound, projection_operators

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Grid

    from .fista_tv import FistaRuntime


class FistaScanState(NamedTuple):
    """State carried through the FISTA scan loop."""

    x: jnp.ndarray
    z: jnp.ndarray
    t: jnp.ndarray
    loss: jnp.ndarray
    prev_obj: jnp.ndarray
    streak: jnp.ndarray
    done: jnp.ndarray
    has_prev: jnp.ndarray
    last_obj: jnp.ndarray
    iters_done: jnp.ndarray


def _group(i: jax.Array, runtime: FistaRuntime):
    n, subsets = runtime.poses.shape[0], runtime.config.initial_subsets
    indices = i + subsets * jnp.arange((n + subsets - 1) // subsets)
    valid = (indices < n)[:, None, None]
    indices = jnp.minimum(indices, n - 1)
    return runtime.poses[indices], indices, valid


@partial(jax.jit, static_argnames=("grid", "detector"))
def _bounds(grid: Grid, detector: Detector, runtime: FistaRuntime) -> jax.Array:
    assert runtime.projector is not None
    _, backend, _ = runtime.projector

    def bound(i):
        poses, _, valid = _group(i, runtime)
        forward, adjoint = projection_operators(
            poses, grid, detector, None, backend, poses.shape[0], "joseph"
        )
        return positive_normal_bound(
            lambda x: forward(x) * valid,
            lambda y: adjoint(y * valid),
            (grid.nz, grid.ny, grid.nx),
            iterations=1,
        )

    return jax.lax.map(bound, jnp.arange(runtime.config.initial_subsets))


def _operators(grid: Grid, detector: Detector, runtime: FistaRuntime, k: jax.Array | None):
    assert runtime.projector is not None
    model, backend, batch = runtime.projector
    if k is None:
        poses, indices, valid = runtime.poses, None, None
    else:
        i = subset_order(runtime.config.initial_subsets)[k % runtime.config.initial_subsets]
        poses, indices, valid = _group(i, runtime)
        batch = poses.shape[0]
    forward, adjoint = projection_operators(poses, grid, detector, None, backend, batch, model)
    return forward, adjoint, indices, valid


@partial(jax.jit, static_argnames=("grid", "detector"))
def _residual(grid, detector, data, z, runtime, k):
    forward, _, indices, valid = _operators(grid, detector, runtime, k)
    images = data if indices is None else data[indices]
    residual = forward(z) - images
    return residual if valid is None else residual * valid


@partial(jax.jit, static_argnames=("grid", "detector"), donate_argnums=(2,))
def _apply(grid, detector, z, residual, limit, runtime, k):
    _, adjoint, _, _ = _operators(grid, detector, runtime, k)
    scaled = -(1 / limit) * residual if k is None else -residual / limit
    return jnp.maximum(adjoint(scaled, accumulate=z), 0)


@partial(jax.jit, static_argnames=("clipped",), donate_argnums=(0,))
def _momentum(old, new, t, *, clipped):
    tn = (1 + jnp.sqrt(1 + 4 * t * t)) / 2
    z = new + (t - 1) / tn * (new - old)
    return jnp.maximum(z, 0) if clipped else z, tn


def _initialize(grid, detector, data, runtime):
    poses, data = ordered_data(data, runtime)
    initial = runtime.x0
    runtime = replace(runtime, poses=poses, initial_order=None, x0=None)
    limits = _bounds(grid, detector, runtime)
    limits.block_until_ready()
    x = (
        jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32)
        if initial is None
        else jnp.maximum(initial, 0)
    )
    z, t = jnp.array(x, copy=True), jnp.float32(1)
    order = subset_order(runtime.config.initial_subsets)
    for step in range(runtime.config.initial_epochs * runtime.config.initial_subsets):
        k = jnp.int32(step)
        residual = _residual(grid, detector, data, z, runtime, k)
        residual.block_until_ready()
        limit = limits[order[step % runtime.config.initial_subsets]]
        new = _apply(grid, detector, z, residual, jnp.where(limit > 0, limit, 1), runtime, k)
        new.block_until_ready()
        z, t = _momentum(x, new, t, clipped=False)
        x = new
    jax.block_until_ready((x, z, t))
    z.delete()
    total = jnp.sum(limits)
    return x, jnp.where(total > 0, total, 1)


def use_owned_fista(grid: Grid, projections, runtime: FistaRuntime) -> bool:
    """Store residuals instead of another volume for the supported objective."""
    if grid.nx * grid.ny * grid.nz <= projections.size:
        return False
    cfg, constraints = runtime.config, runtime.constraints
    if cfg.initial_epochs:  # validation already established this objective/storage
        return True
    return (
        cfg.iterations > 0
        and runtime.x0 is None
        and cfg.nonnegative
        and not cfg.tv_weight
        and runtime.volume_mask is None
        and constraints.lower_bound in (None, 0)
        and constraints.upper_bound is None
        and not runtime.stream
        and runtime.split is None
        and runtime.projector is not None
        and runtime.projector[0] == "joseph"
    )


@partial(jax.jit, static_argnames=("grid", "detector"))
def _zero_gradient(grid: Grid, detector: Detector, data, runtime: FistaRuntime):
    assert runtime.projector is not None
    model, backend, batch = runtime.projector
    return least_squares_operators(runtime.poses, grid, detector, backend, batch, model)(None, data)


@partial(jax.jit, static_argnames=("grid", "detector"))
def _norm_forward(grid: Grid, detector: Detector, vector, runtime: FistaRuntime):
    assert runtime.projector is not None
    model, backend, batch = runtime.projector
    return projection_operators(runtime.poses, grid, detector, None, backend, batch, model)[0](
        vector
    )


@partial(jax.jit, static_argnames=("grid", "detector"))
def _norm_adjoint(grid: Grid, detector: Detector, data, runtime: FistaRuntime):
    assert runtime.projector is not None
    model, backend, batch = runtime.projector
    return projection_operators(runtime.poses, grid, detector, None, backend, batch, model)[1](data)


@jax.jit
def _norm_start(gradient):
    return unit(power_start(gradient))[0]


_normalize = jax.jit(unit, donate_argnums=(0,))


@partial(jax.jit, donate_argnums=(0,))
def _first_step(gradient, limit):
    return jnp.maximum(-(1 / limit) * gradient, 0)


def _zero_start(grid, detector, data, runtime):
    """Release each owned norm vector before allocating its transpose."""
    value, gradient = _zero_gradient(grid, detector, data, runtime)
    jax.block_until_ready((value, gradient))
    limit = runtime.lipschitz
    if limit is None:
        limit = jnp.float32(0)
        vector = _norm_start(gradient)
        vector.block_until_ready()
        for _ in range(max(1, runtime.config.power_iterations)):
            projected = _norm_forward(grid, detector, vector, runtime)
            projected.block_until_ready()
            vector.delete()
            back = _norm_adjoint(grid, detector, projected, runtime)
            back.block_until_ready()
            projected.delete()
            vector, limit = _normalize(back)
            jax.block_until_ready((vector, limit))
        vector.delete()
        limit = jnp.where(limit == 0, 1e-6, limit)
    x = _first_step(gradient, limit)
    x.block_until_ready()
    return x, value, limit


def _refine(grid, detector, data, runtime, x, limit, first_value=None):
    cfg = runtime.config
    z = jnp.array(x, copy=True)
    t = jnp.float32(1) if first_value is None else 0.5 * (1 + jnp.sqrt(jnp.float32(5)))
    losses = [] if first_value is None else [first_value]
    tolerance = float(cfg.recon_rel_tol) if cfg.recon_rel_tol is not None else 0.0
    early = tolerance > 0 and cfg.recon_patience > 0
    streak, done = 0, False
    previous = float(first_value) if early and first_value is not None else None
    value = jnp.float32(0) if first_value is None else first_value
    for _step in range(len(losses), cfg.iterations):
        residual = _residual(grid, detector, data, z, runtime, None)
        value = 0.5 * jnp.vdot(residual, residual).real
        residual.block_until_ready()
        new = _apply(grid, detector, z, residual, limit, runtime, None)
        new.block_until_ready()
        z, t = _momentum(x, new, t, clipped=True)
        x = new
        losses.append(value)
        if early:
            objective = float(value)
            small = (
                previous is not None
                and abs(objective - previous) / max(abs(previous), 1e-6) <= tolerance
            )
            streak = streak + 1 if small else 0
            previous = objective
            done = streak >= cfg.recon_patience
            if done:
                break
    jax.block_until_ready((x, z, t))
    count = len(losses)
    losses.extend([value] * (cfg.iterations - count))
    state = FistaScanState(
        x,
        z,
        t,
        jnp.stack(losses),
        value,
        jnp.int32(streak),
        jnp.bool_(done),
        jnp.bool_(True),
        value,
        jnp.int32(count),
    )
    return state, jnp.asarray(limit, jnp.float32)


def run_owned_fista(grid, detector, projections, runtime):
    """Run nonnegative least squares with owned, reusable volume buffers."""
    data = jnp.asarray(projections)
    if runtime.config.initial_epochs:
        x, bound = _initialize(grid, detector, data, runtime)
        limit = bound if runtime.lipschitz is None else runtime.lipschitz
        first_value = None
    else:
        x, first_value, limit = _zero_start(grid, detector, data, runtime)
    return _refine(grid, detector, data, replace(runtime, x0=None), x, limit, first_value)
