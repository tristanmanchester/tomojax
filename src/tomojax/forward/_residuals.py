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
    """Elementwise pseudo-Huber loss, without cancellation or squared-ratio overflow."""
    r = jnp.asarray(residual, dtype=jnp.float32)
    d = jnp.abs(jnp.asarray(delta, dtype=jnp.float32))
    return _pseudo_huber_value(r, d)


def pseudo_huber_weights(residual: jax.Array, *, delta: float = 1.0) -> jax.Array:
    """Elementwise IRLS weights for pseudo-Huber, with a bounded norm ratio."""
    r = jnp.asarray(residual, dtype=jnp.float32)
    d = jnp.abs(jnp.asarray(delta, dtype=jnp.float32))
    small, _, ratio, norm = _pseudo_huber_parts(r, d)
    return jnp.where(small, 1.0, ratio) / norm


@jax.custom_jvp
def _bounded_ratio(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    return numerator / denominator


@_bounded_ratio.defjvp
def _bounded_ratio_jvp(primals, tangents):
    numerator, denominator = primals
    dn, dd = tangents
    ratio = _bounded_ratio(numerator, denominator)
    # Factoring the quotient rule avoids squaring a physical scale, including
    # when the loss's analytic first derivative is differentiated again.
    return ratio, (dn - ratio * dd) / denominator


def _pseudo_huber_parts(r: jax.Array, d: jax.Array) -> tuple[jax.Array, ...]:
    magnitude = jnp.abs(r)
    small = magnitude <= d
    scale = jnp.where(small, d, magnitude)
    scale = jnp.where(scale == 0, 1.0, scale)
    ratio = _bounded_ratio(jnp.where(small, r, d), scale)
    return small, magnitude, ratio, jnp.sqrt(1.0 + ratio * ratio)


@jax.custom_jvp
def _pseudo_huber_value(r: jax.Array, d: jax.Array) -> jax.Array:
    small, magnitude, ratio, norm = _pseudo_huber_parts(r, d)
    quadratic_r = jnp.where(small, r, 0.0)
    linear_r = jnp.where(small, 0.0, magnitude)
    return quadratic_r * (quadratic_r / (norm + 1.0)) + d * (linear_r / (norm + ratio))


@_pseudo_huber_value.defjvp
def _pseudo_huber_jvp(primals, tangents):
    r, d = primals
    dr, dd = tangents
    small, magnitude, ratio, norm = _pseudo_huber_parts(r, d)
    # Analytic derivatives avoid a squared denominator and retain the right
    # tangent even when the primal loss or its IRLS weight underflows.
    slope_r = jnp.where(small, r, jnp.sign(r) * d) / norm
    quadratic_delta = jnp.where(small, d, 0.0) * (ratio * ratio / (norm + 1.0)) ** 2 / norm
    linear_delta = jnp.where(small, 0.0, magnitude) / (norm * (norm + ratio) ** 2)
    return _pseudo_huber_value(r, d), slope_r * dr + (quadratic_delta + linear_delta) * dd


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
