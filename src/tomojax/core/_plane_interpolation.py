"""Compact interpolation weights for the voxel-plane model."""

from __future__ import annotations

import jax


def validate_interpolation(interpolation: str) -> None:
    if interpolation not in {"linear", "cubic"}:
        raise ValueError("Plane interpolation must be 'linear' or 'cubic'")


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
