"""Measurement-side calibration of static and transient additive detector-column errors."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

from tomojax import progress
from tomojax.core.robust import MAD_TO_SD

if TYPE_CHECKING:
    from collections.abc import Sequence


type Float32Array = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type Float64Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


def _air_noise(y: Float32Array, w: Float32Array | None, sigma: float) -> float:
    """Noise in low-attenuation rays from spacings of lower normal order statistics."""
    step = (
        slice(None, None, max(1, len(y) // 64)),
        slice(None),
        slice(None, None, max(1, y.shape[2] // 64)),
    )
    samples = y[step]
    if w is not None:
        samples = np.where(w[step] > 0, samples, np.inf)
    ranked = np.sort(samples, axis=1)
    count = np.asanyarray(np.isfinite(ranked).sum(axis=1), dtype=np.intp)
    views, cols = np.nonzero(count >= 2)
    if not len(views):
        return sigma
    size = count[views, cols]
    k = np.minimum(size - 1, np.rint(np.sqrt(size)).astype(int))
    # Blom's normal scores for the first and (k+1)th order statistics. Curved
    # specimen signal can increase this spacing, hence the independent-pixel bound.
    span = ranked[views, k, cols] - ranked[views, np.zeros_like(k), cols]
    normal_score = NormalDist().inv_cdf
    spacings = np.ones(y.shape[1] + 1)
    for rows in range(2, len(spacings)):
        rank = min(rows - 1, round(math.sqrt(rows)))
        spacings[rows] = normal_score((rank + 0.625) / (rows + 0.25)) - normal_score(
            0.625 / (rows + 0.25)
        )
    expected = spacings[size]
    return min(sigma, float(np.median(span / expected)))


def _observations(
    y: Float32Array, w: Float32Array | None, sd: float
) -> tuple[Float64Array, BoolArray, Float64Array]:
    """Per-view column floors, their observation mask, and column-invariant scene features."""
    n, v, u = y.shape
    side = max(2, round(2 * math.sqrt(math.sqrt(min(n, v * u)))))
    rows = np.linspace(0, v, min(v, side) + 1, dtype=int)
    cols = np.linspace(0, u, min(u, side) + 1, dtype=int)
    features = np.empty((n, len(rows) - 1, len(cols) - 1))
    floor = np.empty((n, u))
    observed = np.ones((n, u), dtype=bool)
    for i in range(0, n, 16):
        progress.check_canceled()
        slab = slice(i, i + 16)
        data = y[slab]
        weight = None if w is None else w[slab]
        if weight is None:
            bands = (
                np.add.reduceat(data, rows[:-1], axis=1, dtype=float) / np.diff(rows)[None, :, None]
            )
            blocks = np.add.reduceat(bands, cols[:-1], axis=2) / np.diff(cols)
            features[slab] = blocks - np.average(blocks, axis=1, weights=np.diff(rows))[:, None, :]
            low = np.asanyarray(data.min(axis=1), dtype=np.float32)
            if sd:
                low = low - sd * np.log(
                    np.mean(np.exp(-(data - low[:, None, :]) / sd), axis=1, dtype=np.float32)
                )
        else:
            counts = np.add.reduceat(weight, rows[:-1], axis=1, dtype=float)
            sums = np.add.reduceat(weight * data, rows[:-1], axis=1, dtype=float)
            total = np.asanyarray(counts.sum(axis=1), dtype=np.float64)
            observed[slab] = total > 0
            mean = np.divide(
                np.asanyarray(sums.sum(axis=1), dtype=np.float64),
                total,
                out=np.zeros_like(total),
                where=total > 0,
            )
            contrast = np.add.reduceat(sums - counts * mean[:, None, :], cols[:-1], axis=2)
            mass = np.add.reduceat(counts, cols[:-1], axis=2)
            features[slab] = np.divide(contrast, mass, out=np.zeros_like(contrast), where=mass > 0)
            low = np.asanyarray(
                data.min(axis=1, where=weight > 0, initial=np.inf), dtype=np.float32
            )
            low = np.where(total > 0, low, 0)
            if sd:
                gap = np.where(weight > 0, data - low[:, None, :], 0)
                moment = np.asanyarray((weight * np.exp(-gap / sd)).sum(axis=1), dtype=np.float32)
                low -= sd * np.log(
                    np.divide(moment, total, out=np.ones_like(total), where=total > 0)
                )
        floor[slab] = low
    return floor, observed, features.reshape(n, -1)


def _observed_median(values: Float64Array, observed: BoolArray) -> Float64Array:
    present = np.asanyarray(observed.any(axis=0), dtype=bool)
    location = np.zeros(values.shape[1])
    location[present] = np.nanmedian(
        np.where(observed[:, present], values[:, present], np.nan), axis=0
    )
    return location


def _innovations(
    floor: Float64Array, observed: BoolArray, features: Float64Array, sd: float, v: int
) -> Float64Array:
    """Estimate exposure-wise excess variance conditional on the measured row contrasts."""
    n, u = floor.shape
    features -= np.mean(features, axis=0, dtype=np.float64)
    left, singular, _ = np.linalg.svd(features, full_matrices=False)
    rank = min(len(singular), math.isqrt(n))
    basis = np.column_stack([np.ones(n), left[:, :rank] * singular[:rank]])
    # Expected Gram contribution from independent errors in block means.
    ridge = np.diag(np.full(rank + 1, sd**2 * n * features.shape[1] / (v * u)))
    ridge[0, 0] = 0
    weights = np.ones(n)
    groups, inverse = np.unique(observed.T, axis=0, return_inverse=True)
    residual = np.zeros_like(floor)
    for _ in range(6):
        for group in range(len(groups)):
            available = groups[group, :]
            columns = np.flatnonzero(np.equal(inverse, group))
            x = basis[available]
            xtw = x.T * weights[available]
            selection = np.ix_(available, columns)
            beta = np.linalg.pinv(xtw @ x + ridge, hermitian=True) @ (xtw @ floor[selection])
            residual[selection] = floor[selection] - x @ beta
        scatter = _observed_median(abs(residual), observed)
        scatter = np.maximum(scatter * MAD_TO_SD, sd / math.sqrt(v))
        scores = np.divide(residual, scatter, out=np.zeros_like(residual), where=scatter > 0)
        power = np.mean(scores**2, axis=1, dtype=np.float64) / np.maximum(
            1 / u, np.mean(observed, axis=1, dtype=np.float64)
        )
        weights = 1 / np.maximum(1, power)
    return residual * (1 - weights[:, None])


def _fixed(samples: Float64Array, observed: BoolArray) -> Float64Array:
    """Wiener estimate of a fixed white column offset against a spatially smooth pedestal."""
    profile = _observed_median(samples, observed)
    profile -= np.mean(profile, dtype=np.float64)
    spectrum = np.fft.rfft(np.concatenate([profile, profile[::-1]]))[:-1]
    # Interleaved views measure the uncertainty of the recurrent high frequencies.
    half_diff = (
        _observed_median(samples[:-1:2], observed[:-1:2])
        - _observed_median(samples[1::2], observed[1::2])
    ) / 2
    high = slice(len(profile) // 2, None)
    power = np.abs(spectrum) ** 2
    error = np.fft.rfft(np.concatenate([half_diff, half_diff[::-1]]))[:-1]
    uncertainty = np.asanyarray(np.median(np.abs(error[high]) ** 2), dtype=np.float64).item()
    measured = np.asanyarray(np.median(power[high]), dtype=np.float64).item()
    variance = max(0.0, measured - uncertainty) * MAD_TO_SD**2
    width = max(1, round(math.sqrt(len(profile))))
    padded = np.pad(power, (width // 2, (width - 1) // 2), mode="symmetric")
    total = np.convolve(padded, np.full(width, 1 / width), mode="valid")
    gain = np.divide(variance, total, out=np.zeros_like(total), where=total > 0)
    return np.fft.irfft(np.append(spectrum * np.minimum(1, gain), 0))[: len(profile)]


def calibrate_columns(
    levels: Sequence[tuple[Float32Array, Float32Array | None]], bins: Sequence[int], noise: float
) -> None:
    """Subtract the estimated field at each resolution, averaging over its observed rays."""
    fine = min(bins)
    y, w = levels[bins.index(fine)]
    n, v, u = y.shape
    if min(n, v, u) < 2:
        # A common component requires replicated rows and views and a horizontal contrast.
        return
    sd = _air_noise(y, w, noise / fine)
    floor, observed, features = _observations(y, w, sd)
    field = _innovations(floor, observed, features, sd, v)
    field += _fixed(floor - field, observed)
    field -= np.mean(field, axis=1, keepdims=True, dtype=np.float64)  # retain the fitted flux gauge
    field = field.astype(np.float32)
    for (data, _), b in zip(levels, bins, strict=True):
        ratio = b // fine
        columns = field.reshape(n, data.shape[2], ratio)
        if w is None:
            mean = np.asanyarray(columns.mean(axis=2, dtype=np.float64), dtype=np.float32)
            data[:] -= mean[:, None, :]
        else:
            for first in range(0, n, 16):
                sl = slice(first, first + 16)
                count = w[sl].reshape(-1, data.shape[1], ratio, data.shape[2], ratio)
                summed = np.asanyarray(count.sum(axis=2, dtype=np.float64), dtype=np.float64)
                weight = np.asanyarray(summed.sum(axis=-1), dtype=np.float64)
                correction = np.asanyarray(
                    (summed * columns[sl, None]).sum(axis=-1), dtype=np.float64
                )
                data[sl] -= np.divide(
                    correction, weight, out=np.zeros_like(weight), where=weight > 0
                )
