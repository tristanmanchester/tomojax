"""Separable Poisson likelihood update for a known additive background."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax


def background_multiplier(
    score: jax.Array,
    sensitivity: jax.Array,
    curvature: jax.Array,
    where: Callable[[jax.Array, jax.Array | float, jax.Array | float], jax.Array],
) -> jax.Array:
    """Advance toward the minimum of each voxel's convex Jensen surrogate.

    With interior signal s, mean p = s + b and counts y, the score is A.T(y/p)
    and the curvature is A.T((y/p)(s/p)). At multiplier z, the surrogate's
    negative derivative plus the sensitivity sums A_ij (y_i/p_i) /
    (1 + (z-1) s_i/p_i). Jensen's inequality bounds that sum below for z >= 1.
    The secant between interior fractions zero and one bounds it above for
    z <= 1. Moving to the corresponding root decreases the surrogate on either
    side and recovers ordinary EM when the background vanishes.
    """
    gradient = score - sensitivity
    increase = 1 + gradient / where(sensitivity > 0, sensitivity, 1) * (
        score / where(curvature > 0, curvature, 1)
    )
    denominator = curvature - gradient
    decrease = curvature / where(denominator > 0, denominator, 1)
    return where(gradient >= 0, increase, decrease)
