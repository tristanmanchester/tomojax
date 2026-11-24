"""Interpolate missing transmission from measured neighbours on physical detector rows."""

import jax
import jax.numpy as jnp


def measured_neighbours(usable: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Nearest usable column on each side, with -1 and width at unbracketed edges."""
    cols = usable.shape[-1]
    column = jnp.arange(cols)
    left = jax.lax.associative_scan(jnp.maximum, jnp.where(usable, column, -1), axis=1)
    right = jax.lax.associative_scan(
        jnp.minimum, jnp.where(usable, column, cols), axis=1, reverse=True
    )
    return left, right


def interpolate_unmeasured(
    transmission: jax.Array, usable: jax.Array, columns: jax.Array | None
) -> jax.Array:
    """Fill bracketed calibration gaps from the same view's measured neighbours.

    Only columns between two measured pixels are interpolated. A row with no
    measured pixels, and the unilluminated margins of any row, retain the neutral
    transmission supplied by calibration. Measured pixels, including low-gain
    ones, retain their calibrated transmission.
    """
    cols = transmission.shape[-1]
    left, right = measured_neighbours(usable)
    lower = jnp.take_along_axis(transmission, jnp.clip(left, 0, cols - 1)[None], axis=2)
    upper = jnp.take_along_axis(transmission, jnp.clip(right, 0, cols - 1)[None], axis=2)
    interior = ~usable & (left >= 0) & (right < cols)
    if columns is None:
        column = jnp.arange(cols)
    else:
        column, left, right = (
            columns,
            columns[jnp.clip(left, 0, cols - 1)],
            columns[jnp.clip(right, 0, cols - 1)],
        )
    weight = (column - left) / jnp.maximum(right - left, 1)
    filled = lower + weight[None] * (upper - lower)
    return jnp.where(interior[None], filled, transmission)
