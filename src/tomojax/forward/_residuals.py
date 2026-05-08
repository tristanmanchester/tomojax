"""Masked robust projection residuals."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ResidualResult:
    residual: jax.Array
    loss: jax.Array
    weights: jax.Array
    valid_count: jax.Array


ResidualLossMode = Literal["l2", "pseudo_huber"]


def masked_whitened_residual(
    predicted: jax.Array,
    observed: jax.Array,
    *,
    mask: jax.Array | None = None,
    sigma: float | jax.Array = 1.0,
) -> jax.Array:
    pred = jnp.asarray(predicted, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed projections must have matching shapes")
    residual = (pred - obs) / jnp.asarray(sigma, dtype=jnp.float32)
    if mask is None:
        return residual
    mask_arr = jnp.asarray(mask, dtype=jnp.float32)
    if mask_arr.shape != residual.shape:
        raise ValueError("mask must match projection shape")
    return residual * mask_arr


def pseudo_huber_loss(residual: jax.Array, *, delta: float = 1.0) -> jax.Array:
    r = jnp.asarray(residual, dtype=jnp.float32)
    d = jnp.asarray(delta, dtype=jnp.float32)
    return (d * d) * (jnp.sqrt(1.0 + (r / d) ** 2) - 1.0)


def pseudo_huber_weights(residual: jax.Array, *, delta: float = 1.0) -> jax.Array:
    r = jnp.asarray(residual, dtype=jnp.float32)
    d = jnp.asarray(delta, dtype=jnp.float32)
    return 1.0 / jnp.sqrt(1.0 + (r / d) ** 2)


def robust_residual_scale(residual: jax.Array, *, mask: jax.Array | None = None) -> jax.Array:
    """Estimate residual noise scale with the normal-consistent MAD."""
    values = jnp.asarray(residual, dtype=jnp.float32)
    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=bool)
        if mask_arr.shape != values.shape:
            raise ValueError("mask must match residual shape")
        values = values[mask_arr]
    if values.size == 0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    median = jnp.median(values)
    mad = jnp.median(jnp.abs(values - median))
    scale = jnp.asarray(1.4826, dtype=jnp.float32) * mad
    return jnp.where(
        jnp.isfinite(scale) & (scale > 0.0), scale, jnp.asarray(1.0, dtype=jnp.float32)
    )


def residual_loss(
    predicted: jax.Array,
    observed: jax.Array,
    *,
    mask: jax.Array | None = None,
    sigma: float | jax.Array = 1.0,
    delta: float = 1.0,
    mode: ResidualLossMode = "pseudo_huber",
) -> ResidualResult:
    residual = masked_whitened_residual(predicted, observed, mask=mask, sigma=sigma)
    if mode == "pseudo_huber":
        loss_map = pseudo_huber_loss(residual, delta=delta)
        weights = pseudo_huber_weights(residual, delta=delta)
    elif mode == "l2":
        loss_map = jnp.asarray(0.5, dtype=jnp.float32) * residual * residual
        weights = jnp.ones_like(residual, dtype=jnp.float32)
    else:
        raise ValueError(f"unknown residual loss mode {mode!r}")
    if mask is None:
        valid_count = jnp.asarray(residual.size, dtype=jnp.float32)
    else:
        valid_count = jnp.sum(jnp.asarray(mask, dtype=jnp.float32))
    loss = jnp.sum(loss_map) / jnp.maximum(valid_count, 1.0)
    return ResidualResult(residual=residual, loss=loss, weights=weights, valid_count=valid_count)
