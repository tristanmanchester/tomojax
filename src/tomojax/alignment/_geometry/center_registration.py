"""Native opposing-ray registration with detector-fixed gain and flat-field drift."""

from __future__ import annotations

from itertools import combinations
import math
from typing import TYPE_CHECKING

import numpy as np

from .center_statistics import filter_columns

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray

    from .center_focus import FocusScores
    from .detector_center import CenterSelection

type Profile = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def _detector_gradients(
    projections: np.ndarray | jax.Array,
    rows: NDArray[np.intp],
    positions: NDArray[np.float64],
    needed: set[int],
) -> dict[int, Profile]:
    """Estimate the persistent gradient and its drift separately in each acquisition."""
    templates: dict[int, Profile] = {}
    breaks = np.flatnonzero(np.diff(positions) <= 0) + 1
    for segment in np.split(np.arange(len(positions)), breaks):
        start, stop = int(segment.item(0)), int(segment.item(-1))
        views = [i for i in needed if start <= i <= stop]
        indices = np.linspace(start, stop, min(96, len(segment))).round().astype(int)
        # Each time block needs a median, and the drift needs independent slopes.
        count = min(6, len(indices) // 3)
        if not views or count < 3:
            continue
        times: list[float] = []
        medians: list[Profile] = []
        for block in np.array_split(indices, count):
            times.append(np.mean(positions[block], dtype=np.float64).item() - positions.item(start))
            data = np.asarray(projections[block[:, None], rows[None, :], :], dtype=np.float64)
            gradients = filter_columns(data, 1, order=1)
            medians.append(np.median(gradients, axis=0))
        slopes = [
            (medians[j] - medians[i]) / (times[j] - times[i])
            for i, j in combinations(range(count), 2)
        ]
        slope = np.median(slopes, axis=0)
        intercept = np.median([p - t * slope for t, p in zip(times, medians, strict=True)], axis=0)
        for i in views:
            templates[i] = intercept + (positions.item(i) - positions.item(start)) * slope
    return templates


def _match(a: Profile, b: Profile, stride: int, lags: NDArray[np.float64]) -> NDArray[np.float64]:
    """Profile out relative gain and variance for two gradient images at every lag."""
    width = a.shape[1]
    n = 1 << (2 * width - 2).bit_length()
    power = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if power == 0:
        return np.zeros_like(lags)
    spectrum = np.asarray(
        np.sum(np.fft.rfft(a, n=n) * np.conj(np.fft.rfft(b, n=n)), axis=0), np.complex128
    )
    spectrum[-1] *= 0.5
    corr = np.fft.fftshift(np.fft.irfft(spectrum, n=16 * n)) * (16 / power)
    rho = np.interp(lags, (np.arange(16 * n) / 16 - n // 2) * stride, corr, left=0, right=0)
    return -np.log(np.maximum(1 - np.maximum(rho, 0) ** 2, np.finfo(float).eps))


def _raw_match(
    a: Profile, b: Profile, lags: NDArray[np.float64], sigma: float, log_scale: float
) -> NDArray[np.float64]:
    """Profile native residual energy on the detector pixels measured at both angles."""
    width = a.shape[1]
    n = 1 << (2 * width - 2).bit_length()
    u = np.arange(width)
    # Fraction of each Gaussian derivative lobe inside the detector.
    distance = np.minimum(u, width - 1 - u)
    window = -np.expm1(-0.5 * (distance / sigma) ** 2)
    transform = np.fft.rfft(window, n=n)
    x, y = (np.fft.rfft(p * window, n=n) for p in (a, b))
    ax, by = (np.fft.rfft(np.sum(p * p, axis=0, dtype=np.float64) * window, n=n) for p in (a, b))
    cross = np.asarray(np.sum(x * y.conj(), axis=0), np.complex128)
    energy = ax * transform.conj() + transform * by.conj()
    # Zero-padding the cross spectra interpolates both the residual and its
    # normalization to fractional lags before taking the log likelihood.
    spectra = np.stack((cross, energy))
    spectra[:, -1] *= 0.5
    indices = np.rint(lags * 16).astype(int) % (16 * n)
    values = np.fft.irfft(spectra, n=16 * n)[:, indices] * 16
    aligned = np.asarray(values[0], np.float64)
    power = np.asarray(values[1], np.float64)
    total = float(ax.item(0).real + by.item(0).real)
    fraction: NDArray[np.float64] = np.maximum(power, 0) / total if total else np.zeros_like(power)
    rho = np.divide(2 * aligned, power, out=np.zeros_like(power), where=power > 0).clip(0, 1)
    return fraction * (log_scale - np.log(np.maximum(1 - rho**2, np.finfo(float).eps)))


def _template_score(
    a: Profile, b: Profile, x: Profile, y: Profile, sigma: float, lags: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Marginalize native and detector-corrected matches on their illuminated support."""
    # The participation ratio counts illuminated columns. Smooth specimen
    # edges can correlate more of those columns than the gradient filter alone.
    energy = np.asarray(np.sum(x**2 + y**2, axis=0, dtype=np.float64), np.float64)
    squared = float(energy @ energy)
    if not squared:
        return np.zeros_like(lags)
    support = np.sum(energy, dtype=np.float64).item() ** 2 / squared
    size = 2 * x.shape[1]
    density = np.asarray(
        np.sum(abs(np.fft.rfft(x, n=size)) ** 2 + abs(np.fft.rfft(y, n=size)) ** 2, axis=0),
        dtype=np.float64,
    )
    autocorrelation = np.fft.irfft(density, n=size)
    normalized = autocorrelation / autocorrelation.item(0)
    # Integral of the squared normalized autocorrelation, with the Gaussian
    # derivative's own correlation width as the noise floor.
    correlation_width = max(0.75 * math.sqrt(2 * math.pi) * sigma, float(normalized @ normalized))
    independent = support / correlation_width
    corrected = _match(x, -y[:, ::-1], 1, lags)
    # Compare both hypotheses in the same gradient-variance units.
    # A temporal median can contain stationary parts of the specimen.
    raw_power = float(np.linalg.norm(a) * np.linalg.norm(b))
    moving_power = float(np.linalg.norm(x) * np.linalg.norm(y))
    score = independent * corrected
    if moving_power and raw_power:
        log_scale = math.log(min(1.0, moving_power / raw_power))
        raw = _raw_match(a, -b[:, ::-1], lags, sigma, log_scale)
        score = np.logaddexp(score, independent * raw)
    return score


def _shape_score(a: Profile, b: Profile, lags: NDArray[np.float64]) -> NDArray[np.float64]:
    """Match each spatial octave, retaining shapes with little angular variation."""
    score = np.zeros_like(lags)
    b = -b[:, ::-1]
    stride = 1
    while True:
        score += _match(a, b, stride, lags)
        if a.shape[1] < 32:  # two full Gaussian supports in the next octave
            return score
        a, b = (filter_columns(p, 1.5)[:, ::2] for p in (a, b))
        stride *= 2


def paired_center(
    selection: CenterSelection,
    projections: np.ndarray | jax.Array,
    positions: np.ndarray | None,
    *,
    focus: FocusScores | None = None,
) -> float:
    """Combine opposing-ray registration and the optional angular score over every lag."""
    d = selection.geometry.detector
    rows = np.linspace(0, d.nv - 1, min(64, d.nv)).round().astype(int)
    at = (
        np.arange(len(selection.geometry.angles), dtype=float)
        if positions is None
        else np.asarray(positions, dtype=float)
    )
    needed = {i for left, right, _ in selection.samples for i, _ in left + right}
    templates = _detector_gradients(projections, rows, at, needed)
    lags = np.arange(16 * (1 - d.nu), 16 * (d.nu - 1) + 1) / 16
    angular = np.zeros_like(lags) if focus is None else np.interp(-lags / 2, *focus)
    offsets: list[float] = []
    for (left, right, mismatch), first, second in selection.images(projections, rows):
        valid = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
        if not valid.any():
            continue
        sigma = max(1.0, d.nu * math.radians(mismatch) / 4)
        a = filter_columns(first[valid], sigma, order=1)
        b = filter_columns(second[valid], sigma, order=1)
        score = np.zeros_like(lags)
        if all(i in templates for i, _ in left + right):
            expected = [
                np.asarray(sum(w * templates[i][valid] for i, w in side), dtype=np.float64)
                for side in (left, right)
            ]
            # Match the one-pixel template to the interpolation bandwidth.
            if sigma > 1:
                expected = [filter_columns(p, math.sqrt(sigma**2 - 1)) for p in expected]
            x, y = a - expected[0], b - expected[1]
            good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
            score += _template_score(a[good], b[good], x[good], y[good], sigma, lags)
        score += _shape_score(a, b, lags)
        score += angular
        if np.ptp(score) > 0:
            i = int(np.argmax(score))
            delta = 0.0
            if 0 < i < len(score) - 1:
                lo, mid, hi = (score.item(k) for k in (i - 1, i, i + 1))
                delta = 0.5 * (lo - hi) / (lo - 2 * mid + hi)
            offsets.append(-0.5 * (lags.item(i) + delta / 16))
    if offsets:
        return float(np.median(offsets))
    if focus is not None:
        centers, evidence = focus
        if np.ptp(evidence) == 0:
            raise ValueError(
                "center estimation needs angular structure or nonconstant opposing views"
            )
        return centers.item(int(np.argmax(evidence)))
    return d.center[0] / d.du
