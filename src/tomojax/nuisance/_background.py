"""Low-frequency background nuisance model."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BackgroundOffsetModel:
    """Per-view constant plus vertical-gradient background offsets."""

    constant: jax.Array
    vertical_gradient: jax.Array

    @classmethod
    def zeros(cls, n_views: int) -> BackgroundOffsetModel:
        """Return a neutral background model for ``n_views`` projections."""
        shape = (int(n_views),)
        return cls(
            constant=jnp.zeros(shape, dtype=jnp.float32),
            vertical_gradient=jnp.zeros(shape, dtype=jnp.float32),
        )

    def background(self, shape: tuple[int, int, int]) -> jax.Array:
        """Return the additive background stack for ``(views, rows, cols)``."""
        n_views, rows, _cols = shape
        ramp = _vertical_ramp(rows)
        constant = jnp.asarray(self.constant, dtype=jnp.float32).reshape((n_views, 1, 1))
        gradient = jnp.asarray(self.vertical_gradient, dtype=jnp.float32).reshape((n_views, 1, 1))
        return constant + gradient * ramp.reshape((1, rows, 1))

    def apply(self, projections: jax.Array) -> jax.Array:
        """Add the background model to a projection stack."""
        values = jnp.asarray(projections, dtype=jnp.float32)
        if values.ndim != 3:
            raise ValueError("background model expects projections shaped (views, rows, cols)")
        shape = (int(values.shape[0]), int(values.shape[1]), int(values.shape[2]))
        return values + self.background(shape)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable background payload."""
        return {
            "schema": "tomojax.background_offset_model.v1",
            "basis": "constant_plus_vertical_gradient",
            "constant": [float(value) for value in self.constant],
            "vertical_gradient": [float(value) for value in self.vertical_gradient],
        }


def estimate_background_offset(
    predicted: jax.Array,
    observed: jax.Array,
    *,
    mask: jax.Array | None = None,
    ridge: float = 1.0e-6,
) -> BackgroundOffsetModel:
    """Fit low-frequency background to ``observed - predicted`` per view."""
    pred = jnp.asarray(predicted, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed projections must have the same shape")
    if pred.ndim != 3:
        raise ValueError("background fitting expects projections shaped (views, rows, cols)")
    residual = obs - pred
    weights = _mask_weights(mask, pred)
    ramp = _vertical_ramp(int(pred.shape[1])).reshape((1, pred.shape[1], 1))
    ones = jnp.ones_like(residual, dtype=jnp.float32)
    a00 = jnp.sum(weights * ones, axis=(1, 2))
    a01 = jnp.sum(weights * ramp, axis=(1, 2))
    a11 = jnp.sum(weights * ramp * ramp, axis=(1, 2))
    b0 = jnp.sum(weights * residual, axis=(1, 2))
    b1 = jnp.sum(weights * residual * ramp, axis=(1, 2))
    ridge_value = jnp.asarray(max(float(ridge), 0.0), dtype=jnp.float32)
    a00_r = a00 + ridge_value
    a11_r = a11 + ridge_value
    determinant = a00_r * a11_r - a01 * a01
    safe_determinant = jnp.where(
        jnp.abs(determinant) > ridge_value,
        determinant,
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    constant = (b0 * a11_r - b1 * a01) / safe_determinant
    vertical_gradient = (a00_r * b1 - a01 * b0) / safe_determinant
    valid = a00 > 0.0
    return BackgroundOffsetModel(
        constant=jnp.where(valid, constant, 0.0),
        vertical_gradient=jnp.where(valid, vertical_gradient, 0.0),
    )


def _vertical_ramp(rows: int) -> jax.Array:
    if int(rows) <= 1:
        return jnp.zeros((int(rows),), dtype=jnp.float32)
    return jnp.linspace(-1.0, 1.0, int(rows), dtype=jnp.float32)


def _mask_weights(mask: jax.Array | None, predicted: jax.Array) -> jax.Array:
    if mask is None:
        return jnp.ones_like(predicted, dtype=jnp.float32)
    weights = jnp.asarray(mask, dtype=jnp.float32)
    if weights.shape == predicted.shape:
        return weights
    if weights.ndim == predicted.ndim - 1:
        return jnp.broadcast_to(weights, predicted.shape)
    raise ValueError("mask must match projection shape or detector shape")


__all__ = ["BackgroundOffsetModel", "estimate_background_offset"]
