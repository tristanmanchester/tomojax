"""Fixed-iteration Huber FISTA with host-owned image and momentum arrays."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ._tv_ops import grad3, huber_tv_grad, project_box
from .fista_tv_core import FistaCoreConfig, FistaCoreResult

if TYPE_CHECKING:
    from tomojax.core.host_planes import HostPlaneOperator


@partial(jax.jit, static_argnames=("offset", "length", "cfg"))
def _tile_value(halo: jax.Array, *, offset: int, length: int, cfg: FistaCoreConfig) -> jax.Array:
    gx, gy, gz = grad3(halo)
    squared = (gx * gx + gy * gy + gz * gz)[:, :, offset : offset + length]
    if cfg.regulariser == "huber_tv":
        norm = jnp.sqrt(jnp.maximum(squared, jnp.finfo(halo.dtype).tiny))
        value = jnp.where(
            norm <= cfg.huber_delta,
            0.5 * squared / cfg.huber_delta,
            norm - 0.5 * cfg.huber_delta,
        )
    else:
        value = jnp.sqrt(squared)
    return jnp.sum(value)


@partial(jax.jit, static_argnames=("offset", "length", "cfg"), donate_argnums=(0, 1))
def _update(
    gradient: jax.Array,
    previous: jax.Array,
    halo: jax.Array,
    support: jax.Array | None,
    limit: jax.Array,
    momentum: jax.Array,
    *,
    offset: int,
    length: int,
    cfg: FistaCoreConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    z = halo[:, :, offset : offset + length]
    if support is not None:
        gradient *= support
    if cfg.tv_weight:
        gradient += (
            cfg.tv_weight * huber_tv_grad(halo, cfg.huber_delta)[:, :, offset : offset + length]
        )
    project = partial(
        project_box,
        nonnegative=cfg.nonnegative,
        lower_bound=cfg.lower_bound,
        upper_bound=cfg.upper_bound,
    )
    x = project(z - gradient / limit)
    extrapolated = project(x + momentum * (x - previous))
    value = (
        _tile_value(halo, offset=offset, length=length, cfg=cfg)
        if cfg.compute_iteration_loss and cfg.tv_weight
        else jnp.float32(0)
    )
    return x, extrapolated, value


def _weighted_residual(
    prediction: np.ndarray,
    data: np.ndarray,
    weights: np.ndarray,
    *,
    compute_loss: bool,
) -> np.float32:
    """Replace prediction by the weighted residual, reducing the loss in blocks."""
    batch = max(1, (32 << 20) // (prediction.shape[1] * prediction.shape[2] * 4))
    loss = 0.0
    for first in range(0, len(prediction), batch):
        last = min(first + batch, len(prediction))
        residual = prediction[first:last]
        residual -= data[first:last]
        weight = weights[first:last, None, None]
        if compute_loss:
            loss += 0.5 * np.sum(weight * residual**2, dtype=np.float64)
        residual *= weight
    return np.float32(loss)


def _advance(
    operator: HostPlaneOperator,
    x: np.ndarray,
    z: np.ndarray,
    residual: np.ndarray,
    support: np.ndarray | None,
    *,
    cfg: FistaCoreConfig,
    momentum: np.float32,
) -> np.float32:
    regulariser = 0.0
    old_boundary = None
    for first, last, gradient in operator.transpose_slabs(residual):
        lo, hi = max(0, first - 1), min(z.shape[2], last + 1)
        halo = np.array(z[:, :, lo:hi], order="C")
        # z is overwritten in place. Preserve the old left halo for the next slab.
        if old_boundary is not None:
            halo[:, :, 0] = old_boundary
        old_boundary = np.array(z[:, :, last - 1], order="C")
        halo_device = operator.put(halo)
        support_device = None if support is None else operator.put(support[:, :, first:last])
        new_x, new_z, value = _update(
            gradient,
            operator.put(x[:, :, first:last]),
            halo_device,
            support_device,
            jnp.float32(cfg.lipschitz),
            jnp.float32(momentum),
            offset=first - lo,
            length=last - first,
            cfg=cfg,
        )
        x[:, :, first:last] = operator.get(new_x)
        z[:, :, first:last] = operator.get(new_z)
        regulariser += float(value)
        new_x.delete()
        new_z.delete()
        halo_device.delete()
        if support_device is not None:
            support_device.delete()
    return np.float32(regulariser)


def fista_tv_core_host(
    operator: HostPlaneOperator,
    initial: np.ndarray,
    *,
    projections: np.ndarray,
    cfg: FistaCoreConfig,
    view_weights: np.ndarray | None = None,
) -> FistaCoreResult[np.ndarray]:
    """Run the array-core FISTA recurrence using two private host images.

    The input image and projections are borrowed and never modified. The result
    image is a Fortran-contiguous NumPy array; diagnostics are small JAX arrays.
    Only volume slabs and bounded detector groups are placed on the device.
    This host-managed runner is not differentiable; the array core remains the
    differentiable interface.
    """
    if cfg.tv_weight != 0 and cfg.regulariser != "huber_tv":
        raise ValueError("fista_tv_core_host requires huber_tv or tv_weight=0")
    support = (
        None
        if cfg.support is None
        else np.broadcast_to(np.asarray(cfg.support, dtype=np.float32), operator.shape)
    )
    cfg = replace(cfg, support=None)
    weights = (
        np.ones(operator.projection_shape[0], np.float32)
        if view_weights is None
        else np.maximum(np.asarray(view_weights, np.float32), 0)
    )
    x = np.empty(operator.shape, np.float32, order="F")
    project = partial(
        project_box,
        nonnegative=cfg.nonnegative,
        lower_bound=cfg.lower_bound,
        upper_bound=cfg.upper_bound,
    )
    for first, last in operator.slabs():
        tile = operator.put(initial[:, :, first:last])
        x[:, :, first:last] = operator.get(project(tile))
        tile.delete()
    z = x.copy(order="F")
    prediction = np.empty(operator.projection_shape, np.float32)
    history = np.zeros(cfg.iterations, np.float32)
    last_data, t = np.float32(0), np.float32(1)
    for iteration in range(cfg.iterations):
        operator.project(z, out=prediction, support=support)
        last_data = _weighted_residual(
            prediction,
            projections,
            weights,
            compute_loss=cfg.compute_iteration_loss or cfg.compute_final_data_loss,
        )
        next_t = np.float32(0.5) * (np.float32(1) + np.sqrt(np.float32(1) + np.float32(4) * t * t))
        regulariser = _advance(
            operator,
            x,
            z,
            prediction,
            support,
            cfg=cfg,
            momentum=(t - np.float32(1)) / next_t,
        )
        if cfg.compute_iteration_loss:
            history[iteration] = last_data + np.float32(cfg.tv_weight) * regulariser
        t = next_t
    del z
    if cfg.compute_final_data_loss:
        operator.project(x, out=prediction, support=support)
        last_data = _weighted_residual(prediction, projections, weights, compute_loss=True)
    regulariser = 0.0
    if cfg.compute_final_regulariser_value:
        for first, last in operator.slabs():
            halo = operator.put(x[:, :, first : min(last + 1, x.shape[2])])
            regulariser += float(_tile_value(halo, offset=0, length=last - first, cfg=cfg))
            halo.delete()
    return FistaCoreResult(
        x=x,
        loss=jnp.asarray(history),
        data_loss=jnp.float32(last_data),
        regulariser_value=jnp.float32(regulariser),
        effective_iterations=jnp.int32(cfg.iterations),
        status="ok",
    )


__all__ = ["fista_tv_core_host"]
