"""Physical first-difference quadratic penalty with free volume boundaries."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def regularization_normal(
    volume: jax.Array,
    spacing: tuple[float, float, float],
    damping: jax.Array,
    gradient_damping: jax.Array | None = None,
) -> jax.Array:
    """Gradient of half the combined squared-volume and squared-gradient penalties."""
    result = damping**2 * volume
    if gradient_damping is not None:
        result = result + gradient_damping**2 * gradient_normal(volume, spacing)
    return result


def gradient_energy(volume: jax.Array, spacing: tuple[float, float, float]) -> jax.Array:
    """Return ||D x||²; D contains adjacent differences divided by voxel spacing."""
    return sum(
        jnp.sum((jnp.diff(volume, axis=axis) / step) ** 2) for axis, step in enumerate(spacing)
    )


def gradient_normal(
    volume: jax.Array,
    spacing: tuple[float, float, float],
    *,
    absolute_weights: bool = False,
) -> jax.Array:
    """Apply D.T D, or abs(D.T D) for a componentwise roundoff estimate.

    Only edges within the volume contribute. Constant volumes therefore have
    zero energy even at the boundary, and singleton axes contribute nothing.
    """
    result = jnp.zeros_like(volume)
    for axis, step in enumerate(spacing):
        if volume.shape[axis] == 1:
            continue
        if absolute_weights:
            left = [slice(None)] * 3
            right = [slice(None)] * 3
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            edge = (volume[tuple(left)] + volume[tuple(right)]) / step**2
        else:
            edge = jnp.diff(volume, axis=axis) / step**2
        lower, upper = [(0, 0)] * 3, [(0, 0)] * 3
        lower[axis], upper[axis] = (1, 0), (0, 1)
        from_left, from_right = jnp.pad(edge, lower), jnp.pad(edge, upper)
        result = result + (from_left + from_right if absolute_weights else from_left - from_right)
    return result
