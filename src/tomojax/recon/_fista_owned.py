"""Memory-bounded execution of resident nonnegative FISTA.

Residuals occupy detector storage; their scaled transpose updates an owned
extrapolate in place. Only owned work arrays are donated, never caller arrays.
The projector and optimizer are unchanged; scaling before the transpose changes
FP32 summation order. Small-volume problems retain the fused-gradient scan.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import progress
from tomojax.core.operator_norm import bounds_settled, collatz_ratio, unit
from tomojax.recon._volume_storage import preview_planes
from tomojax.wire.metrics import Metric

from ._data_term import project_residual
from ._fista_initialization import power_start
from ._projection import least_squares_operators, projection_operators

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Grid

    from .fista_tv import FistaRuntime


class FistaResult(NamedTuple):
    """The final iterates and diagnostics that leave the compiled recurrence."""

    x: jax.Array
    # The extrapolated iterate leaves as its own buffer. Kept inside, it joins
    # the solve's one temporary block, and a larger block can take a larger
    # region of device memory although the bytes in use are the same.
    z: jax.Array
    loss: jax.Array
    iters_done: jax.Array
    done: jax.Array


class FistaScanState(NamedTuple):
    """State carried through the FISTA scan loop."""

    x: jnp.ndarray
    z: jnp.ndarray
    t: jnp.ndarray
    loss: jnp.ndarray
    prev_obj: jnp.ndarray
    streak: jnp.ndarray
    done: jnp.ndarray
    iters_done: jnp.ndarray

    @jax.jit
    def padded_loss(self) -> jax.Array:
        """Repeat the last observed loss through the unused iteration budget."""
        return jnp.where(jnp.arange(self.loss.size) < self.iters_done, self.loss, self.prev_obj)


@partial(jax.jit, static_argnames=("grid", "detector"))
def _residual(grid, detector, data, z, runtime):
    return project_residual(
        _norm_forward(grid, detector, z, runtime) - data, runtime.config.data_term
    )


@partial(jax.jit, static_argnames=("grid", "detector"), donate_argnums=(2,))
def _apply(grid, detector, z, residual, limit, runtime):
    model, backend, batch = runtime.projector
    adjoint = projection_operators(
        runtime.poses, grid, detector, runtime.detector_grid, backend, batch, model
    )[1]
    return jnp.maximum(adjoint(-(1 / limit) * residual, accumulate=z), 0)


@partial(jax.jit, static_argnames=("clipped",), donate_argnums=(0,))
def _momentum(old, new, t, *, clipped):
    tn = (1 + jnp.sqrt(1 + 4 * t * t)) / 2
    z = new + (t - 1) / tn * (new - old)
    return jnp.maximum(z, 0) if clipped else z, tn


def use_owned_fista(grid: Grid, projections, runtime: FistaRuntime) -> bool:
    """Store residuals instead of another volume for the supported objective."""
    if grid.nx * grid.ny * grid.nz <= projections.size:
        return False
    cfg, constraints = runtime.config, runtime.constraints
    return (
        cfg.iterations > 0
        and runtime.x0 is None
        and cfg.tv_weight == 0
        and runtime.volume_mask is None
        and constraints.lower_bound == 0.0
        and constraints.upper_bound is None
        and not runtime.stream
        and runtime.split is None
        and runtime.projector[0] == "joseph"
    )


@partial(jax.jit, static_argnames=("grid", "detector"))
def _zero_gradient(grid: Grid, detector: Detector, data, runtime: FistaRuntime):
    model, backend, batch = runtime.projector
    return least_squares_operators(
        runtime.poses,
        grid,
        detector,
        backend,
        batch,
        model,
        data_term=runtime.config.data_term,
        det_grid=runtime.detector_grid,
    )(None, data)


@partial(jax.jit, static_argnames=("grid", "detector"))
def _norm_forward(grid: Grid, detector: Detector, vector, runtime: FistaRuntime):
    model, backend, batch = runtime.projector
    return projection_operators(
        runtime.poses, grid, detector, runtime.detector_grid, backend, batch, model
    )[0](vector)


@partial(jax.jit, static_argnames=("grid", "detector"))
def _norm_adjoint(grid: Grid, detector: Detector, data, runtime: FistaRuntime):
    model, backend, batch = runtime.projector
    return projection_operators(
        runtime.poses, grid, detector, runtime.detector_grid, backend, batch, model
    )[1](data)


@jax.jit
def _norm_start(gradient):
    return unit(power_start(gradient))[0]


_normalize = jax.jit(unit, donate_argnums=(0,))
_collatz = jax.jit(collatz_ratio)


@partial(jax.jit, donate_argnums=(0,))
def _first_step(gradient, limit):
    return jnp.maximum(-(1 / limit) * gradient, 0)


def _zero_start(grid, detector, data, runtime):
    """Release each owned norm vector before allocating its transpose."""
    value, gradient = _zero_gradient(grid, detector, data, runtime)
    jax.block_until_ready((value, gradient))
    limit = runtime.lipschitz
    if limit is None:
        vector = _norm_start(gradient)
        vector.block_until_ready()
        iterations, k, power, upper = max(1, runtime.config.power_iterations), 0, 0.0, np.inf
        # The step-safe Collatz bound, once it meets the power estimate (see
        # tomojax.core.operator_norm.normal_upper_bound).
        while not bounds_settled(k, iterations, power, upper):
            progress.check_cancelled()
            projected = _norm_forward(grid, detector, vector, runtime)
            projected.block_until_ready()
            # Hold the iterate on the host while the transpose needs the device.
            held = np.asarray(vector)
            vector.delete()
            back = _norm_adjoint(grid, detector, projected, runtime)
            back.block_until_ready()
            projected.delete()
            upper = float(_collatz(back, jnp.asarray(held)))
            vector, norm = _normalize(back)
            power, k = float(norm), k + 1
        vector.delete()
        limit = jnp.float32(upper * (1 + 8 * np.finfo(np.float32).eps) or 1e-6)
    progress.check_cancelled()
    x = _first_step(gradient, limit)
    x.block_until_ready()
    return x, value, limit


def _refine(grid, detector, data, runtime, x, limit, first_value=None):
    cfg = runtime.config
    z = jnp.array(x, copy=True)
    t = jnp.float32(1) if first_value is None else 0.5 * (1 + jnp.sqrt(jnp.float32(5)))
    losses = [] if first_value is None else [first_value]
    tolerance = float(cfg.tolerance) if cfg.tolerance is not None else 0.0
    early = tolerance > 0 and cfg.patience > 0
    streak, done = 0, False
    previous = float(first_value) if early and first_value is not None else None
    value = jnp.float32(0) if first_value is None else first_value
    task = progress.for_unit("iterations")
    if task is not None and first_value is not None:
        task.advance(1, {Metric.loss.name: float(first_value)})
        progress.preview(lambda x=x: preview_planes(x, runtime.projector[0], grid))
    for _step in range(len(losses), cfg.iterations):
        progress.check_cancelled()
        with progress.preparing("first fista update", _step - (first_value is not None)):
            residual = _residual(grid, detector, data, z, runtime)
            value = 0.5 * jnp.vdot(residual, residual).real
            residual.block_until_ready()
            new = _apply(grid, detector, z, residual, limit, runtime)
            new.block_until_ready()
            z, t = _momentum(x, new, t, clipped=True)
        x = new
        losses.append(value)
        if task is not None:
            jax.block_until_ready((z, t))
            task.advance(len(losses), {Metric.loss.name: float(value)})
            progress.preview(lambda x=x: preview_planes(x, runtime.projector[0], grid))
        if early:
            objective = float(value)
            small = (
                previous is not None
                and abs(objective - previous) / max(abs(previous), 1e-6) <= tolerance
            )
            streak = streak + 1 if small else 0
            previous = objective
            done = streak >= cfg.patience
            if done:
                break
    jax.block_until_ready((x, z, t))
    count = len(losses)
    if count:
        progress.preview(lambda x=x: preview_planes(x, runtime.projector[0], grid), final=True)
    losses.extend([value] * (cfg.iterations - count))
    state = FistaResult(x, z, jnp.stack(losses), jnp.int32(count), jnp.bool_(done))
    return state, jnp.asarray(limit, jnp.float32)


def run_owned_fista(grid, detector, projections, runtime):
    """Run nonnegative least squares with owned, reusable volume buffers."""
    data = jnp.asarray(projections)
    with progress.preparing("initialise fista"):
        x, first_value, limit = _zero_start(grid, detector, data, runtime)
    state, limit = _refine(grid, detector, data, runtime, x, limit, first_value)
    return state, limit, runtime.metric
