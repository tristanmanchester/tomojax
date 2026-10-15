"""Memory-bounded execution of initialized nonnegative FISTA.

Residuals occupy detector storage; their scaled transpose updates an owned
extrapolate in place. Only the two iterates are donated, never caller arrays.
The projector and optimizer are unchanged; scaling before the transpose changes
FP32 summation order. Small-volume problems retain the fused-gradient scan.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from ._fista_initialization import ordered_data, subset_order
from ._projection import positive_normal_bound, projection_operators

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
            (grid.nx, grid.ny, grid.nz),
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
        jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
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


def run_owned_fista(grid, detector, projections, runtime):
    """Run the validated resident initializer with owned, reusable volume buffers."""
    cfg = runtime.config
    data = jnp.asarray(projections)
    x, bound = _initialize(grid, detector, data, runtime)
    limit = bound if runtime.lipschitz is None else runtime.lipschitz
    runtime = replace(runtime, x0=None)
    z, t = jnp.array(x, copy=True), jnp.float32(1)
    losses, streak, previous, done = [], 0, None, False
    tolerance = float(cfg.recon_rel_tol) if cfg.recon_rel_tol is not None else 0.0
    early = tolerance > 0 and cfg.recon_patience > 0
    for _step in range(cfg.iterations):
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
