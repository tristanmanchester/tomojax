"""Per-view gain/offset nuisance model."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GainOffsetModel:
    """Per-view affine intensity nuisance state."""

    gain: jax.Array
    offset: jax.Array

    @classmethod
    def identity(cls, n_views: int) -> GainOffsetModel:
        """Return a neutral model for ``n_views`` projections."""
        shape = (int(n_views),)
        return cls(
            gain=jnp.ones(shape, dtype=jnp.float32),
            offset=jnp.zeros(shape, dtype=jnp.float32),
        )

    def apply(self, projections: jax.Array) -> jax.Array:
        """Apply per-view gain and offset to a projection stack."""
        values = jnp.asarray(projections, dtype=jnp.float32)
        view_shape = (int(values.shape[0]),) + (1,) * (values.ndim - 1)
        gain = jnp.asarray(self.gain, dtype=jnp.float32).reshape(view_shape)
        offset = jnp.asarray(self.offset, dtype=jnp.float32).reshape(view_shape)
        return gain * values + offset

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable nuisance payload."""
        return {
            "schema": "tomojax.gain_offset_model",
            "gain": [float(value) for value in self.gain],
            "offset": [float(value) for value in self.offset],
        }


def estimate_gain_offset(
    predicted: jax.Array,
    observed: jax.Array,
    *,
    mask: jax.Array | None = None,
    ridge: float = 1.0e-6,
) -> GainOffsetModel:
    """Fit per-view ``observed ~= gain * predicted + offset`` by weighted least squares."""
    pred = jnp.asarray(predicted, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed projections must have the same shape")
    if pred.ndim < 2:
        raise ValueError("projection stacks must include a view axis and detector axes")
    weights = _mask_weights(mask, pred)
    pred_flat = pred.reshape((pred.shape[0], -1))
    obs_flat = obs.reshape((obs.shape[0], -1))
    weights_flat = weights.reshape((weights.shape[0], -1))
    sum_w = jnp.sum(weights_flat, axis=1)
    sum_x = jnp.sum(weights_flat * pred_flat, axis=1)
    sum_y = jnp.sum(weights_flat * obs_flat, axis=1)
    sum_xx = jnp.sum(weights_flat * pred_flat * pred_flat, axis=1)
    sum_xy = jnp.sum(weights_flat * pred_flat * obs_flat, axis=1)
    safe_sum_w = jnp.maximum(sum_w, jnp.asarray(1.0, dtype=jnp.float32))
    x_mean = sum_x / safe_sum_w
    y_mean = sum_y / safe_sum_w
    covariance = sum_xy - sum_x * y_mean
    variance = sum_xx - sum_x * x_mean
    ridge_value = jnp.asarray(max(float(ridge), 0.0), dtype=jnp.float32)
    gain = covariance / jnp.maximum(variance + ridge_value, ridge_value)
    offset = y_mean - gain * x_mean
    valid = sum_w > 0.0
    return GainOffsetModel(
        gain=jnp.where(valid, gain, 1.0),
        offset=jnp.where(valid, offset, 0.0),
    )


def _mask_weights(mask: jax.Array | None, predicted: jax.Array) -> jax.Array:
    if mask is None:
        return jnp.ones_like(predicted, dtype=jnp.float32)
    weights = jnp.asarray(mask, dtype=jnp.float32)
    if weights.shape == predicted.shape:
        return weights
    if weights.ndim == predicted.ndim - 1:
        return jnp.broadcast_to(weights, predicted.shape)
    raise ValueError("mask must match projection shape or detector shape")


__all__ = ["GainOffsetModel", "estimate_gain_offset"]
