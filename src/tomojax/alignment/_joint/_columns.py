"""Measurement-side calibration of static and transient additive detector-column errors."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from tomojax import progress
from tomojax.core.robust import MAD_TO_SD

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.alignment._config import DetectorColumns


type Float32Array = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type Float64Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


class ColumnNoise(NamedTuple):
    """Attenuation of the column contrasts and independent faint-ray noise at this scale."""

    attenuation: float = 0.0
    faint_sd: float = 0.0


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


def _offset_variance(
    samples: Float64Array, observed: BoolArray
) -> tuple[float, Float64Array, Float64Array]:
    """Recurrent high-frequency power of the faint rays in an orthonormal cosine basis."""
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
    fixed = np.fft.irfft(np.append(spectrum * np.minimum(1, gain), 0))[: len(profile)]
    return variance / (2 * len(profile)), profile, fixed


def _sine(values: Float64Array) -> Float64Array:
    """Orthonormal type-I sine transform along the last axis (odd extension at each edge)."""
    zero = np.zeros_like(values[..., :1])
    odd = np.concatenate((zero, values, zero, -values[..., ::-1]), axis=-1)
    return -np.fft.rfft(odd).imag[..., 1:-1] / math.sqrt(2 * (values.shape[-1] + 1))


def _fixed(
    y: Float32Array,
    w: Float32Array | None,
    variance: float,
    floor: Float64Array,
    reference: Float64Array,
) -> Float64Array:
    """Estimate recurrent column curvature across views and rows; its affine part is unobserved."""
    n, v, u = y.shape
    if not variance or u < 3:
        return reference
    step = (slice(None, None, max(1, n // 64)), slice(None, None, max(1, v // 64)))
    samples = y[step].reshape(-1, u).astype(float)
    curvature = np.diff(samples, n=2)
    # All three pixels must be measured. Isolated missing columns supply no curvature.
    seen = np.ones_like(curvature, dtype=bool)
    if w is not None:
        valid = w[step].reshape(-1, u) > 0
        seen = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    location = _observed_median(curvature, seen)
    groups = np.array_split(np.arange(len(curvature)), min(len(curvature), 16))
    locations = np.stack([_observed_median(curvature[g], seen[g]) for g in groups])
    error = np.mean(_sine(locations - location) ** 2, axis=0, dtype=np.float64) / len(groups)
    # A stationary specimen edge also recurs in every stratum. Comparing its
    # curvature across all rows and faint rays cancels the fixed detector error.
    disagreement = _sine(location - np.diff(floor, n=2)) ** 2
    width = max(1, round(math.sqrt(u)))
    padded = np.pad(disagreement, (width // 2, (width - 1) // 2), mode="symmetric")
    error = np.maximum(error, np.convolve(padded, np.full(width, 1 / width), mode="valid"))
    eig = -4 * np.sin(np.pi * np.arange(1, u - 1) / (2 * (u - 1))) ** 2
    total = variance * eig**2 + error
    remainder = location - np.diff(reference, n=2)
    field = np.pad(_sine(variance * eig * _sine(remainder) / total), 1)
    axis = np.arange(u) - (u - 1) / 2
    return (
        reference + field - np.mean(field, dtype=np.float64) - axis * (axis @ field) / (axis @ axis)
    )


def transient_profile(
    y: Float32Array, w: Float32Array | None, sigma: float
) -> tuple[Float64Array, BoolArray, Float64Array, float]:
    """Conditional faint-ray innovations, retaining the measured mean of each exposure."""
    sd = _air_noise(y, w, sigma)
    floor, observed, features = _observations(y, w, sd)
    residual = _innovations(floor, observed, features, sd, y.shape[1])
    mass = np.maximum(np.asanyarray(observed.sum(axis=1, keepdims=True), dtype=np.intp), 1)
    mean = np.asanyarray(residual.sum(axis=1, keepdims=True), dtype=np.float64) / mass
    return np.where(observed, residual - mean, 0), observed, floor, sd


def _blocked(
    field: Float64Array, observed: BoolArray, ratio: int
) -> tuple[Float64Array, BoolArray]:
    """Average a column statistic over the observed columns of each coarser block."""
    n = len(field)
    count = np.asanyarray(observed.reshape(n, -1, ratio).sum(axis=2), dtype=np.intp)
    summed = np.asanyarray(field.reshape(n, -1, ratio).sum(axis=2), dtype=np.float64)
    return summed / np.maximum(count, 1), count > 0


def calibrate_columns(
    levels: Sequence[tuple[Float32Array, Float32Array | None]],
    bins: Sequence[int],
    noise: float,
    *,
    model: DetectorColumns,
    coverage: bool,
) -> dict[int, ColumnNoise]:
    """Remove recurrent gain errors and estimate each level's column-contrast attenuation."""
    fine = min(bins)
    y, w = levels[bins.index(fine)]
    n, v, u = y.shape
    attenuation = {b: ColumnNoise(0.0, noise / b) for b in bins}
    if min(n, v, u) < 2:
        return attenuation
    if model == "none":
        sd = _air_noise(y, w, noise / fine) if coverage else noise / fine
        return {b: ColumnNoise(0.0, sd * fine / b) for b in bins}
    dynamic, observed, floor, sd = transient_profile(y, w, noise / fine)
    fixed = (
        _fixed(y, w, *_offset_variance(floor - dynamic, observed))
        if model in ("both", "fixed")
        else 0.0
    )
    ends: list[tuple[Float64Array, BoolArray]] = []
    gain = 0.0
    if model in ("both", "transient"):
        # A random detector gain is common even to distant rows. Independent
        # estimates in the two outer quartiles distinguish it from variance
        # confined to the intervening specimen. Their cross-product is unbiased
        # for the common variance under independent errors in the two estimates.
        band = max(1, v // 4)
        for rows in (slice(0, band), slice(-band, None)):
            edge, seen, _, _ = transient_profile(
                y[:, rows], None if w is None else w[:, rows], noise / fine
            )
            ends.append((edge, seen))
        paired_native = ends[0][1] & ends[1][1]
        covariance_native = np.sum(ends[0][0] * ends[1][0], where=paired_native, dtype=np.float64)
        power = np.sum(dynamic**2, where=paired_native, dtype=np.float64)
        gain = (
            np.asanyarray(np.clip(covariance_native / power, 0, 1), dtype=np.float64).item()
            if power > 0
            else 0.0
        )
    corrected_field = fixed + gain * dynamic
    for (data, _), b in zip(levels, bins, strict=True):
        ratio = b // fine
        variance = 0.0
        if ends:
            left, left_seen = _blocked(*ends[0], ratio)
            right, right_seen = _blocked(*ends[1], ratio)
            paired = left_seen & right_seen
            covariance = np.sum(left * right, where=paired, dtype=np.float64) / max(
                1, np.count_nonzero(paired)
            )
            # Conditioning on the full-height estimate gives posterior variance
            # tau² (1 - gain), as well as the shrunk correction gain * dynamic.
            variance = (1 - gain) * max(0.0, np.asanyarray(covariance, dtype=np.float64).item())
        columns = corrected_field.reshape(n, data.shape[2], ratio)
        independent = (noise / b) ** 2
        total = independent + data.shape[1] * variance
        attenuation[b] = ColumnNoise(
            1 - math.sqrt(independent / total) if total else 0.0, sd / ratio
        )
        if w is None:
            mean = np.asanyarray(columns.mean(axis=2, dtype=np.float64), dtype=np.float32)
            data[:] -= mean[:, None, :]
        else:
            for first in range(0, n, 16):
                sl = slice(first, first + 16)
                weights = w[sl].reshape(-1, data.shape[1], ratio, data.shape[2], ratio)
                summed = np.asanyarray(weights.sum(axis=2, dtype=np.float64), dtype=np.float64)
                weight = np.asanyarray(summed.sum(axis=-1), dtype=np.float64)
                correction = np.asanyarray(
                    (summed * columns[sl, None]).sum(axis=-1), dtype=np.float64
                )
                data[sl] -= np.divide(
                    correction, weight, out=np.zeros_like(weight), where=weight > 0
                )
    return attenuation
