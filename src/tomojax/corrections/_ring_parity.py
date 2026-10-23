"""Stationary detector-field priors from full-turn opposed-ray parity."""

from functools import lru_cache
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=16)
def _plan(size: int) -> tuple[np.ndarray, ...]:
    """An undecimated partition and its reflection-aware, band-shared variances.

    The odd field estimates one real Gaussian mode per frequency. For a filter
    with covariance C, its even coefficient at i has variance
    (C(0) + C(2*i - size + 1))/2, including correlated mirror neighbours.
    DC has no odd observation: extending the first band's PSD to it is a prior.
    """
    from scipy.stats import f as fisher

    frequency = np.fft.rfftfreq(size)
    low = np.ones_like(frequency)
    filters = []
    for level in range(int(np.ceil(np.log2(size)))):
        smooth = low * np.cos(np.pi * frequency * 2**level) ** 4
        filters.append(low - smooth)
        low = smooth
    filters.append(low)
    edges = np.unique(np.r_[1, np.geomspace(2, len(frequency), 24).astype(int)])
    masks = np.zeros((len(edges) - 1, len(frequency)))
    for k, (lo, hi) in enumerate(pairwise(edges)):
        masks[k, lo:hi] = 1
    masks[0, 0] = 1
    lag = (2 * np.arange(size) - (size - 1)) % size
    weights = []
    for filt in filters:
        covariance = np.fft.irfft(masks * filt[None, :] ** 2, n=size, axis=1)
        weights.append(np.maximum(0.5 * (covariance[:, :1] + covariance[:, lag]), 0))
    degrees = np.geomspace(1, size // 2 + 1, 512)
    # A pointwise test across thousands of coefficients invents residual stripes.
    # This is a per-scale simultaneous threshold, not a calibrated scan-wide bound:
    # the stationary PSD and Satterthwaite degrees are themselves estimated.
    critical = fisher.isf(0.05 / size, 1, degrees)
    return (
        edges,
        np.diff(edges).astype(float),
        np.stack(filters),
        np.stack(weights),
        np.log(degrees),
        np.log(critical),
    )


def regularize(profile: jax.Array, axis: float, keep_even: str) -> jax.Array:
    """Retain odd detector response and locally supported even response.

    A complete paired interval avoids a half-profile FFT boundary at the axis.
    The undecimated filters have no privileged spatial origin and sum to one.
    Fine corrections use simultaneous significance; broad corrections use a
    Wiener gain. Even scene/response separation remains a statistical prior.
    """
    n = profile.size
    lo, hi = max(0, int(2 * axis - n + 1)), min(n, int(2 * axis) + 1)
    size = hi - lo
    if size < 16:
        return profile
    paired = profile[lo:hi]
    even, odd = 0.5 * (paired + paired[::-1]), 0.5 * (paired - paired[::-1])
    fe, fo = jnp.fft.rfft(even), jnp.fft.rfft(odd)
    power = 2 * jnp.abs(fo) ** 2 / size
    if size % 2 == 0:
        power = power.at[-1].multiply(0.5)
    edges, counts, filters, weights, log_degrees, log_critical = _plan(size)
    spectra = jnp.stack([power[a:b].mean() for a, b in pairwise(edges)])
    estimate = jnp.zeros_like(even)
    for raw_filter, raw_weights in zip(filters, weights, strict=True):
        coefficient = jnp.fft.irfft(fe * jnp.asarray(raw_filter, profile.dtype), n=size)
        terms = jnp.asarray(raw_weights, profile.dtype) * spectra[:, None]
        variance = terms.sum(0)
        scale: float | jax.Array = 1.0
        if keep_even != "wiener":
            # All frequencies in a band share one estimated PSD. Collapse its
            # location-dependent weights before computing effective degrees.
            fractions = terms / jnp.where(variance > 0, variance, 1)[None, :]
            inverse_df = (fractions**2 / jnp.asarray(counts[:, None], profile.dtype)).sum(0)
            degrees = jnp.clip(1 / jnp.where(inverse_df > 0, inverse_df, 1), 1, size // 2 + 1)
            scale = jnp.exp(
                jnp.interp(
                    jnp.log(degrees),
                    jnp.asarray(log_degrees, profile.dtype),
                    jnp.asarray(log_critical, profile.dtype),
                )
            )
        squared = coefficient * coefficient
        gain = jnp.minimum(scale * variance / jnp.where(squared > 0, squared, 1), 1)
        estimate = estimate + gain * coefficient
    return profile.at[lo:hi].set(odd + estimate)
