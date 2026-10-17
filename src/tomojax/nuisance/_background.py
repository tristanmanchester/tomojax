"""Low-frequency background nuisance model."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._weights import mask_weights


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
            "schema": "tomojax.background_offset_model",
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
    """Fit low-frequency background to ``observed - predicted`` per view.

    ``ridge`` penalizes both squared coefficients. Without ridge, a view with
    only one observed detector row uses the minimum-norm constant/gradient
    pair. Unobserved views have zero background, and zero-weight pixels are
    excluded even if their values are nonfinite.
    """
    pred = jnp.asarray(predicted, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed projections must have the same shape")
    if pred.ndim != 3:
        raise ValueError("background fitting expects projections shaped (views, rows, cols)")
    weights = mask_weights(mask, pred)
    included = weights != 0.0
    residual = jnp.where(included, obs, 0.0) - jnp.where(included, pred, 0.0)
    ramp = _vertical_ramp(int(pred.shape[1])).reshape((1, pred.shape[1], 1))
    sum_w = jnp.sum(weights, axis=(1, 2))
    safe_sum_w = jnp.where(sum_w > 0.0, sum_w, 1.0)
    ramp_mean = jnp.sum(weights * ramp, axis=(1, 2)) / safe_sum_w
    residual_mean = jnp.sum(weights * residual, axis=(1, 2)) / safe_sum_w
    centred_ramp = ramp - ramp_mean[:, None, None]
    centred_residual = jnp.where(included, residual - residual_mean[:, None, None], 0.0)
    variance = jnp.sum(weights * centred_ramp**2, axis=(1, 2))
    covariance = jnp.sum(weights * centred_ramp * centred_residual, axis=(1, 2))
    ridge_value = jnp.asarray(max(float(ridge), 0.0), dtype=jnp.float32)
    shrink = sum_w / jnp.where(sum_w + ridge_value > 0.0, sum_w + ridge_value, 1.0)
    # Eliminate the intercept using centered moments. Unlike a 2x2 determinant,
    # this denominator does not subtract nearly equal raw products or compare
    # quantities with incompatible units to a fixed numerical floor.
    denominator = variance + ridge_value * (1.0 + ramp_mean**2 * shrink)
    numerator = covariance + ridge_value * ramp_mean * shrink * residual_mean
    vertical_gradient = numerator / jnp.where(denominator > 0.0, denominator, 1.0)
    observed_rows = jnp.sum(jnp.any(included, axis=2), axis=1)
    identifiable = (observed_rows > 1) | (ridge_value > 0.0)
    minimum_norm = ramp_mean * residual_mean / (1.0 + ramp_mean**2)
    vertical_gradient = jnp.where(identifiable, vertical_gradient, minimum_norm)
    constant = shrink * (residual_mean - ramp_mean * vertical_gradient)
    valid = sum_w > 0.0
    return BackgroundOffsetModel(
        constant=jnp.where(valid, constant, 0.0),
        vertical_gradient=jnp.where(valid, vertical_gradient, 0.0),
    )


def _vertical_ramp(rows: int) -> jax.Array:
    if int(rows) <= 1:
        return jnp.zeros((int(rows),), dtype=jnp.float32)
    return jnp.linspace(-1.0, 1.0, int(rows), dtype=jnp.float32)


__all__ = ["BackgroundOffsetModel", "estimate_background_offset"]
