"""Translation-invariant rotation-axis initialization from Fourier common lines.

A Fourier slice pair agrees along its intersection. Using magnitudes removes
per-view translations, so a wrong detector centre cannot imitate an axis tilt.
The sampler owns image preparation and physical frequency coordinates; this
module fits only the requested x/y axis rotations, in the calibration convention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class AxisDirectionSeed:
    """Absolute setup rotations in degrees and their common-line agreement."""

    rotations_deg: tuple[float, float]
    correlation_before: float
    correlation_after: float


def _axis_in_detector(
    nominal: np.ndarray, rotations: np.ndarray, detector_roll_deg: float
) -> np.ndarray:
    """Calibration ``Ry @ Rx`` axis, expressed in the rolled detector frame."""
    x, y = np.deg2rad(rotations)
    cx, sx, cy, sy = np.cos(x), np.sin(x), np.cos(y), np.sin(y)
    nx, ny, nz = nominal
    yy, zz = cx * ny + sx * nz, -sx * ny + cx * nz
    xx, zz = cy * nx + sy * zz, -sy * nx + cy * zz
    r = np.deg2rad(detector_roll_deg)
    return np.array([np.cos(r) * xx + np.sin(r) * zz, yy, -np.sin(r) * xx + np.cos(r) * zz])


def _magnitude_agreement(first: np.ndarray, second: np.ndarray) -> float:
    """Weighted pair correlation, not a fit to the strongest Fourier peak alone."""
    a, b = np.abs(first), np.abs(second)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = norm > 0
    if not np.any(valid):
        return 0.0
    correlation = np.sum(a[valid] * b[valid], axis=1) / norm[valid]
    return float(np.average(correlation, weights=np.sqrt(norm[valid])))


def fit_axis_direction(
    sample: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    nominal_axis: np.ndarray,
    *,
    rotations_deg: tuple[float, float] = (0.0, 0.0),
    active: tuple[bool, bool] = (True, True),
    bounds_deg: tuple[tuple[float, float], tuple[float, float]] = ((-60.0, 60.0), (-60.0, 60.0)),
    detector_roll_deg: float = 0.0,
) -> AxisDirectionSeed:
    """Initialize active axis rotations within a local angular capture range.

    ``sample(axis)`` returns complex spectra of corresponding common lines,
    shaped ``(pairs, frequencies)``, in a physical detector coordinate system.
    The coarse search spans fifteen degrees in x and four in y, followed by
    local refinement. Inactive rotations and explicit bounds remain fixed.
    This is an initializer, not a reconstruction or pose-accuracy certificate.
    """
    initial = np.asarray(rotations_deg, dtype=np.float64)
    nominal = np.asarray(nominal_axis, dtype=np.float64)
    nominal = nominal / np.linalg.norm(nominal)
    indices = np.flatnonzero(active)
    bounds = np.asarray(bounds_deg, dtype=np.float64)

    def score(x: np.ndarray) -> float:
        axis = _axis_in_detector(nominal, x, detector_roll_deg)
        return -_magnitude_agreement(*sample(axis))

    before = score(initial)
    x = initial.copy()
    # Keep equal-score trials closest to the current estimate: featureless
    # projections must not turn an arbitrary grid endpoint into a calibration.
    for dim, span, step in ((0, 15.0, 1.0), (1, 4.0, 0.25), (0, 2.0, 0.2), (1, 1.0, 0.1)):
        if not active[dim]:
            continue
        candidates = [(score(x), 0.0, x)]
        for offset in np.arange(-span, span + step / 2, step):
            trial = x.copy()
            trial[dim] = np.clip(trial[dim] + offset, *bounds[dim])
            candidates.append((score(trial), abs(float(offset)), trial))
        x = min(candidates, key=lambda item: item[:2])[2]

    if indices.size:

        def objective(values: np.ndarray) -> float:
            trial = x.copy()
            trial[indices] = values
            return score(trial)

        result = minimize(
            objective,
            x[indices],
            method="Nelder-Mead",
            bounds=bounds[indices],
            options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 120},
        )
        if result.fun < score(x):
            x = x.copy()
            x[indices] = result.x
    after = score(x)
    if after > before:
        x, after = initial, before
    return AxisDirectionSeed((float(x[0]), float(x[1])), -before, -after)
