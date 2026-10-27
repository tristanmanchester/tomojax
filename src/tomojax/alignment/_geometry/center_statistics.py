"""Conditional translation statistics for opposing detector rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift
from scipy.stats import t


@dataclass(frozen=True)
class PairStatistics:
    """Location and detector-band uncertainty conditional on one matching mode."""

    offset_px: float
    sigma_px: float
    interval90_px: tuple[float, float]


def _overlap(columns: np.ndarray, *, lag: float, bandwidth: float) -> np.ndarray:
    margin = 4 * bandwidth
    return (
        (columns - lag > margin)
        & (columns - lag < len(columns) - 1 - margin)
        & (columns > margin)
        & (columns < len(columns) - 1 - margin)
    )


def refine_pair(a: np.ndarray, b: np.ndarray, *, lag: float, bandwidth: float) -> PairStatistics:
    """Fit derivatives on common support, not their zero-padded boundary energy.

    The symmetric score treats both images as noisy. Detector-column blocks
    keep correlations within a feature or a stripe together. The cross-Hessian
    excludes the template-noise contribution to the squared-gradient Hessian;
    block leverage and effective degrees of freedom account for sparse signal.
    Intervals describe a selected rigid-registration mode, not mode selection,
    unmodelled motion, or an absolute physical rotation centre.
    """
    width = a.shape[-1]
    columns = np.arange(width)
    derivative_a = np.gradient(a, axis=-1)
    for _ in range(8):
        moved = shift(b, (0, lag), order=3, mode="constant", cval=0)
        derivative_b = np.gradient(moved, axis=-1)
        gradient = 0.5 * (derivative_a + derivative_b)
        residual = a - moved
        valid = _overlap(columns, lag=lag, bandwidth=bandwidth)
        curvature = float(np.sum(gradient[:, valid] ** 2))
        if curvature == 0:
            return PairStatistics(-lag / 2, float("inf"), (-float("inf"), float("inf")))
        step = float(np.sum(residual[:, valid] * gradient[:, valid])) / curvature
        # A trust region keeps refinement in the selected correlation mode.
        lag -= float(np.clip(step, -bandwidth, bandwidth))
        if abs(step) < 1e-5:
            break
    # Evaluate uncertainty at the fitted location, not the previous iterate.
    moved = shift(b, (0, lag), order=3, mode="constant", cval=0)
    derivative_b = np.gradient(moved, axis=-1)
    gradient = 0.5 * (derivative_a + derivative_b)
    residual = a - moved
    valid = _overlap(columns, lag=lag, bandwidth=bandwidth)
    information = float(np.sum(derivative_a[:, valid] * derivative_b[:, valid]))
    scores, leverages = [], []
    for block in np.array_split(np.flatnonzero(valid), 8):
        scores.append(float(np.sum(residual[:, block] * gradient[:, block])))
        leverages.append(float(np.sum(gradient[:, block] ** 2)))
    leverage = np.asarray(leverages) / np.sum(leverages)
    score = np.asarray(scores) - leverage * np.sum(scores)
    degrees = float(1 / np.sum(leverage**2) - 1)
    if information <= 0 or degrees <= 0 or np.max(leverage) >= 1:
        return PairStatistics(-lag / 2, float("inf"), (-float("inf"), float("inf")))
    sigma = float(0.5 * np.linalg.norm(score / (1 - leverage)) / information)
    radius = float(t.ppf(0.95, degrees)) * sigma
    center = -lag / 2
    return PairStatistics(center, sigma, (center - radius, center + radius))
