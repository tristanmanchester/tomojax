"""Exact view medians without sorting megapixel detector frames."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def view_medians(batch: jax.Array) -> jax.Array:
    """One exact median per detector frame, using bounded-workspace selection."""
    values = batch.reshape(batch.shape[0], math.prod(batch.shape[1:]))
    # Sorting wins for small images. Large FP32 views instead use bounded
    # integer selection: at most 32 refinements, with no sorted image workspace.
    if values.dtype == jnp.float32 and values.shape[1] >= 2**20:
        return _radix_median(values)
    return jnp.median(values, axis=1)


@jax.custom_jvp
def _radix_median(values: jax.Array) -> jax.Array:
    # CUDA amortizes loop synchronization across three simultaneous pivots.
    # Select by the compilation target; a CPU input can be used even when
    # CUDA is the default backend.
    return jax.lax.platform_dependent(
        values,
        cuda=lambda x: _key_median(x, radix=4),
        default=lambda x: _key_median(x, radix=2),
    )


def _key_median(values: jax.Array, *, radix: int) -> jax.Array:
    bits = jax.lax.bitcast_convert_type(values, jnp.uint32)
    sign = jnp.uint32(0x80000000)
    # Unsigned keys preserve numerical order, including infinities and zeros.
    keys = jnp.where((bits & sign) != 0, ~bits, bits ^ sign)
    lower, upper = jnp.min(keys, axis=1), jnp.max(keys, axis=1)
    rank = values.shape[1] // 2
    if radix == 4:
        # Sampling chooses pivots, never the median. Counts over every key
        # keep the exact rank bracket even when the sample is badly biased.
        size = 512
        stride, remainder = divmod(values.shape[1] - 1, size - 1)
        at = jnp.arange(size, dtype=jnp.int32)
        indices = at * stride + at * remainder // (size - 1)
        sample = jnp.sort(keys[:, indices], axis=1)
        pivots = sample[:, jnp.array([size * 3 // 8, size // 2, size * 5 // 8])]
        lower, upper = _pivot_bounds(keys, lower, upper, pivots, rank)

    def narrow(bounds: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        lo, hi = bounds
        if radix == 2:
            pivot = lo + ((hi - lo) >> 1)
            count = jnp.sum(keys <= pivot[:, None], axis=1)
            return jnp.where(count <= rank, pivot + 1, lo), jnp.where(count > rank, pivot, hi)
        # Three cumulative counts locate one of four inclusive key ranges.
        # Equal pivots in tiny ranges still advance the lower bound.
        step = (hi - lo) >> 2
        pivots = lo[:, None] + step[:, None] * jnp.arange(1, 4, dtype=jnp.uint32)
        return _pivot_bounds(keys, lo, hi, pivots, rank)

    middle, _ = jax.lax.while_loop(
        lambda bounds: jnp.any(bounds[0] < bounds[1]), narrow, (lower, upper)
    )
    median_bits = jnp.where((middle & sign) != 0, middle ^ sign, ~middle)
    median = jax.lax.bitcast_convert_type(median_bits, jnp.float32)
    if values.shape[1] % 2 == 0:
        # If exactly half the values precede the upper middle, the lower
        # middle is their maximum. Otherwise both central values are equal.
        below = keys < middle[:, None]
        count = jnp.sum(below, axis=1)
        preceding = jnp.max(jnp.where(below, keys, jnp.uint32(0)), axis=1)
        preceding_bits = jnp.where((preceding & sign) != 0, preceding ^ sign, ~preceding)
        previous = jax.lax.bitcast_convert_type(preceding_bits, jnp.float32)
        median = 0.5 * (median + jnp.where(count == rank, previous, median))
    # NumPy/JAX median propagates even a single NaN, rather than ignoring it.
    return jnp.where(jnp.any(jnp.isnan(values), axis=1), jnp.nan, median)


def _pivot_bounds(
    keys: jax.Array, lo: jax.Array, hi: jax.Array, pivots: jax.Array, rank: int
) -> tuple[jax.Array, jax.Array]:
    count = jnp.sum(keys[:, :, None] <= pivots[:, None, :], axis=1)
    index = jnp.sum(count <= rank, axis=1)[:, None]
    lows = jnp.concatenate([lo[:, None], pivots + jnp.uint32(1)], axis=1)
    highs = jnp.concatenate([pivots, hi[:, None]], axis=1)
    return (
        jnp.take_along_axis(lows, index, axis=1)[:, 0],
        jnp.take_along_axis(highs, index, axis=1)[:, 0],
    )


@_radix_median.defjvp
def _radix_median_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (values,), (direction,) = primals, tangents
    # Selection is discrete; preserve JAX median's tangent and tie convention
    # for callers differentiating a statistic. Ordinary correction does not
    # pay for this sorting path.
    _, tangent = jax.jvp(lambda x: jnp.median(x, axis=1), (values,), (direction,))
    return _radix_median(values), tangent
