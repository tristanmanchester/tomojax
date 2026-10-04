"""Compact interpolation weights for the voxel-plane model."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def validate_interpolation(interpolation: str) -> None:
    if interpolation not in {"linear", "cubic"}:
        raise ValueError("Plane interpolation must be 'linear' or 'cubic'")


def cubic_weight(distance: jax.Array) -> jax.Array:
    """Keys cubic convolution with a=-1/2, including its negative lobes."""
    r = jnp.abs(distance)
    return jnp.where(
        r < 1,
        1 + r * r * (1.5 * r - 2.5),
        # The expanded polynomial cancels near r=2, destroying its tiny
        # negative lobe. Factor the two roots before FP32 evaluation.
        jnp.where(r < 2, -0.5 * (r - 1) * (r - 2) ** 2, 0.0),
    )


def cubic_weights_and_derivatives(fraction: jax.Array) -> tuple[tuple, tuple]:
    """Weights and coordinate derivatives for floor(q)+[-1, 0, 1, 2]."""
    t = fraction
    weights = (
        -0.5 * t * (1 - t) ** 2,
        1 + t * t * (1.5 * t - 2.5),
        t * (0.5 + t * (2 - 1.5 * t)),
        -0.5 * t * t * (1 - t),
    )
    derivatives = (
        -0.5 + t * (2 - 1.5 * t),
        t * (4.5 * t - 5),
        0.5 + t * (4 - 4.5 * t),
        t * (1.5 * t - 1),
    )
    return weights, derivatives
