"""Full-turn axis registration on the detector columns shared by opposing rays."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from tomojax import progress
from tomojax._typed_arrays import linspace_indices

from .center_statistics import filter_columns

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray

    from tomojax._typed_arrays import Float64Array


def _angular_samples(
    angles: Float64Array, width: int
) -> tuple[NDArray[np.intp], NDArray[np.intp], Float64Array]:
    """Bracket both halves of the same angular grid, retaining acquisition indices."""
    nodes, order = np.unique(np.remainder(angles, 360.0), return_index=True)
    count = min(width, len(nodes) // 2)
    targets = np.remainder(
        nodes.item(0) + np.arange(2 * count, dtype=np.float64) * (180 / count), 360.0
    )
    hi = np.searchsorted(nodes, targets) % len(nodes)
    lo = (hi - 1) % len(nodes)
    weight = np.remainder(nodes[hi] - targets, 360.0) / np.remainder(nodes[hi] - nodes[lo], 360.0)
    return order[lo], order[hi], weight[:, None]


def _covariance(a: Float64Array, b: Float64Array, lags: NDArray[np.intp]) -> Float64Array:
    """Cross-products and energies restricted to the actual overlap at each lag."""
    width = len(a.T)
    nfft = 1 << (2 * width - 2).bit_length()
    cross = np.zeros(nfft // 2 + 1, dtype=np.complex128)
    for i in range(0, len(a), 64):
        x, y = (np.fft.rfft(v[i : i + 64], nfft) for v in (a, b))
        cross += (x * y.conj()).sum(axis=0, dtype=np.complex128)
    xx, yy = (
        np.append(0.0, np.cumsum(np.einsum("ij,ij->j", v, v), dtype=np.float64)) for v in (a, b)
    )
    energy = xx[np.clip(width + lags, 0, width)] - xx[np.maximum(lags, 0)]
    energy += yy[np.clip(width - lags, 0, width)] - yy[np.maximum(-lags, 0)]
    return np.stack((2 * np.fft.irfft(cross, nfft)[lags % nfft], energy))


def turn_center(
    projections: np.ndarray | jax.Array, angles: np.ndarray
) -> tuple[float, dict[str, object]] | None:
    """Match angular contrast across a full turn, marginalizing fixed gain and linear drift.

    At the correct center, the angular pattern of a detector column repeats
    half a turn later at its conjugate. Normalize on measured common support
    at every lag so distant, unpaired structures cannot dilute a narrow match.
    Native contrast selects the global mode. Resolved gradients at one, two
    and four pixels locate it more precisely in the presence of slow shading.
    """
    _, height, width = map(int, projections.shape)
    lags = np.arange(1 - width, width, dtype=np.intp)
    lo, hi, weight = _angular_samples(np.asarray(angles, np.float64), width)
    count = len(lo) // 2
    if count < 3:  # Removing a mean and slope leaves no angular contrast.
        return None
    t = np.linspace(-1.0, 1.0, count)
    bands = (0, 1, 2, 4)
    cov = np.zeros((len(bands), 2, len(lags)), np.float64)
    rows = linspace_indices(0, height - 1, min(8, height))
    for row in rows:
        progress.check_canceled()
        a = np.asarray(projections[hi, row, :], dtype=np.float64)
        samples = a + weight * (np.asarray(projections[lo, row, :]) - a)
        a, b = np.split(samples, 2)
        b = b[:, ::-1]
        for v in (a, b):
            v[:] -= v.mean(axis=0)
            v[:] -= t[:, None] * ((t @ v) / (t @ t))
        for i, sigma in enumerate(bands):
            x, y = (filter_columns(v, sigma, order=1) for v in (a, b)) if sigma else (a, b)
            cov[i] += _covariance(x, y, lags)
    numerator, energy = cov[:, 0], cov[:, 1]
    rho = np.divide(numerator, energy, out=np.zeros_like(energy), where=energy > 0).clip(0, 1)
    pairs = count * (width - abs(lags))
    # A correlation in a few shared pixels is more likely to occur by chance.
    lower: Float64Array = rho[0] - np.sqrt(2 * math.log(len(lags)) / pairs)
    if lower.max() <= 0:
        return None
    score = lower.copy()
    for index, scale in enumerate(bands[1:], start=1):
        # Excess explained gradient variance, in sampling-error units.
        score += np.sqrt(pairs / scale) * rho[index] ** 2
    best = int(np.argmax(score))
    delta = 0.0
    if 0 < best < len(score) - 1:
        a0, mid, b0 = (score.item(i) for i in range(best - 1, best + 2))
        delta = 0.5 * (a0 - b0) / (a0 - 2 * mid + b0)
    center = -0.5 * (lags.item(best) + delta)
    return float(center), {
        "pairs": count,
        "rows": len(rows),
        "overlap_px": float(width - 2 * abs(center)),
    }
