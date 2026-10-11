"""Masked robust projection residuals."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
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
    """Whiten differences, excluding zero-mask entries before unsafe arithmetic."""
    pred = jnp.asarray(predicted, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed projections must have matching shapes")
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float32)
    if mask is None:
        return (pred - obs) / sigma_arr
    mask_arr = jnp.asarray(mask, dtype=jnp.float32)
    if mask_arr.shape != pred.shape:
        raise ValueError("mask must match projection shape")
    active = mask_arr != 0
    pred = jnp.where(active, pred, 0.0)
    obs = jnp.where(active, obs, 0.0)
    sigma_arr = jnp.where(active, sigma_arr, 1.0)
    return ((pred - obs) / sigma_arr) * mask_arr


def pseudo_huber_loss(residual: jax.Array, *, delta: float = 1.0) -> jax.Array:
    """Elementwise pseudo-Huber loss, without cancellation for small residuals."""
    r = jnp.asarray(residual, dtype=jnp.float32)
    d = jnp.asarray(delta, dtype=jnp.float32)
    # d^2 (sqrt(1 + (r/d)^2) - 1), rewritten so that |r| << d does not give 0.
    return r * r / (1.0 + jnp.sqrt(1.0 + (r / d) ** 2))


def pseudo_huber_weights(residual: jax.Array, *, delta: float = 1.0) -> jax.Array:
    r = jnp.asarray(residual, dtype=jnp.float32)
    d = jnp.asarray(delta, dtype=jnp.float32)
    return 1.0 / jnp.sqrt(1.0 + (r / d) ** 2)


def robust_residual_scale(residual: jax.Array, *, mask: jax.Array | None = None) -> jax.Array:
    """Estimate normal-consistent MAD, including dynamic masks under JAX transforms.

    Empty, constant, or nonfinite-MAD samples retain the fallback scale of one.
    Nonzero masks select samples; their magnitudes do not weight the medians.
    """
    values = jnp.asarray(residual, dtype=jnp.float32)
    valid = jnp.asarray(True)
    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=bool)
        if mask_arr.shape != values.shape:
            raise ValueError("mask must match residual shape")
    if values.size == 0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    if mask is None:
        median = jnp.median(values)
        mad = jnp.median(jnp.abs(values - median))
    else:
        # Missing NaN/Inf samples must be removed before centering, not merely
        # hidden after an unsafe subtraction. Sorting sentinels keeps the shape
        # fixed while the selected count and median indices stay dynamic.
        values = jnp.where(mask_arr, values, 0.0)
        valid = ~jnp.any(jnp.isnan(values))
        count = jnp.sum(mask_arr)
        median = _selected_median(values, mask_arr, count)
        mad = _selected_median(jnp.abs(values - median), mask_arr, count)
    scale = jnp.asarray(1.4826, dtype=jnp.float32) * mad
    return jnp.where(
        valid & jnp.isfinite(scale) & (scale > 0.0), scale, jnp.asarray(1.0, dtype=jnp.float32)
    )


def _selected_median(values: jax.Array, mask: jax.Array, count: jax.Array) -> jax.Array:
    ordered = jnp.sort(jnp.where(mask, values, jnp.inf).ravel())
    lower = jnp.where(count > 0, ordered[jnp.maximum((count - 1) // 2, 0)], 0.0)
    upper = jnp.where(count > 0, ordered[count // 2], 0.0)
    return lower * 0.5 + upper * 0.5


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
