"""Estimate full-turn detector response from matched angular arcs."""

from math import ceil, sqrt

import jax
import jax.numpy as jnp
import numpy as np

from ._ring_offsets import correlate, quadratic_taps, window_sum
from ._ring_parity import opposed_contrast


def _arc_means(values: jax.Array, count: int) -> jax.Array:
    """Integrate equal angular intervals, splitting the boundary views by overlap."""
    views = len(values)
    edges = np.arange(count + 1) * views / count
    start = np.floor(edges[:-1]).astype(int)[:, None]
    indices = start + np.arange(ceil(views / count) + 1)
    width = np.minimum(indices + 1, edges[1:, None]) - np.maximum(indices, edges[:-1, None])
    weights = (np.maximum(width, 0) * (count / views)).astype(values.dtype)
    return (values[np.minimum(indices, views - 1)] * weights[:, :, None]).sum(1)


def turn_response(
    q: jax.Array,
    ref: jax.Array,
    transmission: jax.Array,
    level: jax.Array,
    beta: jax.Array,
    intercept: jax.Array,
    axis: float,
) -> tuple[jax.Array, jax.Array]:
    """Combine the spatial estimate with transmission and mean contrasts across arcs.

    Opposed rays cancel the sample. Half the sum and half the difference of
    adjacent opposed contrasts have equal independent noise power. Subtracting
    those powers estimates the detector signal that persists between arcs.
    Fit the remaining response after subtracting the initial level and slope.
    This keeps well-measured narrow defects out of the broad power estimate.
    """
    views, columns = q.shape
    count = 2 * min(views // 2, max(1, round(sqrt(views) / 2)))
    measured = _arc_means(q, count)
    predictor = _arc_means(transmission, count)
    centered = predictor - predictor.mean(0)
    radius = max(3, round(2 * sqrt(columns)))
    calibrated = measured - (level + beta * centered)
    residual = calibrated - correlate(calibrated, quadratic_taps(radius, 1))
    broad = residual.mean(0)
    variation = residual - broad
    predictor_variance = (centered**2).mean(0) + 1e-4
    slope = (centered * variation).mean(0) / predictor_variance

    opposed = opposed_contrast(residual, axis)
    adjacent = jnp.roll(opposed, 1, axis=0)
    common, difference = (opposed + adjacent) / 2, (opposed - adjacent) / 2
    u = np.arange(columns)
    lo = np.maximum(u - radius, 0)
    hi = np.minimum(u + radius + 1, columns)
    length = hi - lo

    def power(values: jax.Array) -> jax.Array:
        return window_sum((values**2).mean(0), 2 * radius + 1, "SAME") / length

    noise = power(difference)
    available = jnp.maximum(power(common) - noise, 0)
    dynamic = jnp.maximum(power(common - common.mean(0)) - noise, 0)
    remaining = variation - slope * centered
    level_noise = power(remaining) / count
    slope_noise = power(remaining * centered / jnp.sqrt(predictor_variance)) / count
    # Overlapping reflected neighborhoods sample the same detector columns.
    # Their odd power misses a broad even offset. Weight its mean by the
    # uncertainty across arcs in the shared part of the two neighborhoods.
    mirror = 2 * axis - u
    overlap = np.maximum(np.minimum(hi, mirror + radius + 1) - np.maximum(lo, mirror - radius), 0)
    shared = overlap / length
    measured_gain = jnp.minimum(available / (power(broad[None]) + level_noise + 1e-15), 1)
    mean_gain = available / (available + level_noise + 1e-15)
    level += ((1 - shared) * measured_gain + shared * mean_gain) * broad
    slope *= jnp.minimum(dynamic / (power(slope * centered) + slope_noise + 1e-15), 1)

    # The median spatial estimate can be much more precise on smooth rays.
    # Compare angular scatter at its resolution with the arc fit's error.
    spatial = q - ref
    scatter = jnp.median(jnp.abs(spatial - jnp.median(spatial, axis=0)), axis=0)
    scatter = jnp.sqrt(power(scatter[None]))
    contrast = remaining - correlate(remaining, quadratic_taps(6, 0))
    uncertainty = (contrast**2).mean(0) / count
    weight = scatter**2 / (scatter**2 + uncertainty + 1e-18)
    mean_transmission = transmission.mean(0)
    fine = intercept + beta * mean_transmission
    beta += weight * slope
    return fine + weight * (level - fine) - beta * mean_transmission, beta
