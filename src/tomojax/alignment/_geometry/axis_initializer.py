"""Translation-invariant rotation-axis initialization from Fourier common lines.

A Fourier slice pair agrees along its intersection. Using magnitudes removes
per-view translations, so a wrong detector centre cannot imitate an axis tilt.
The sampler owns image preparation and physical frequency coordinates; this
module fits only the requested x/y axis rotations, in the calibration convention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class AxisDirectionSeed:
    """Absolute setup rotations in degrees and their common-line agreement."""

    rotations_deg: tuple[float, float]
    correlation_before: float
    correlation_after: float
    # Boundary contact is diagnostic; explicit bounded fits retain their solution.
    on_search_boundary: bool = False


def axis_in_detector(
    nominal: np.ndarray, rotations: np.ndarray, *, detector_roll_deg: float
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
    bounds_deg: tuple[tuple[float, float], tuple[float, float]] = ((-180.0, 180.0), (-60.0, 60.0)),
    detector_roll_deg: float = 0.0,
) -> AxisDirectionSeed:
    """Fit local axis candidates on both beam sides, retaining recorded angles.

    A file's nominal tilt need not specify which way the rotation axis leans
    along the beam. Rank the current and reflected candidates by their measured
    common-line agreement. Explicit bounds and frozen rotations remain binding;
    equal evidence retains the candidate near the supplied axis.
    Explicit bounded fits retain their solution; ``on_search_boundary`` records
    whether an active, nonperiodic bound may limit its interpretation.
    """
    fit_local = partial(
        _fit_local_axis_direction,
        sample,
        nominal_axis,
        active=active,
        bounds_deg=bounds_deg,
        detector_roll_deg=detector_roll_deg,
    )
    nominal = np.asarray(nominal_axis, np.float64)
    start = rotations_deg
    # An axial starting geometry carries no beam-side or laminography tilt.
    # Search the allowed hemisphere before the local fit, rather than silently
    # imposing the local capture range as an absent metadata prior.
    if all(active) and np.allclose(nominal, (0, 0, 1)) and np.allclose(start, (0, 0)):
        start = _global_axis_start(sample, nominal, bounds_deg, detector_roll_deg)
    fitted = fit_local(rotations_deg=start)
    if start != rotations_deg:
        before = _magnitude_agreement(
            *sample(
                axis_in_detector(
                    nominal, np.asarray(rotations_deg), detector_roll_deg=detector_roll_deg
                )
            )
        )
        fitted = replace(fitted, correlation_before=before)
    reflected_x = -2 * np.rad2deg(np.arctan2(nominal[1], nominal[2])) - rotations_deg[0]
    reflected_x = float((reflected_x + 180) % 360 - 180)
    if (
        not active[0]
        or np.isclose(reflected_x, rotations_deg[0])
        or not bounds_deg[0][0] <= reflected_x <= bounds_deg[0][1]
    ):
        return fitted
    reflected = fit_local(rotations_deg=(reflected_x, rotations_deg[1]))
    if reflected.correlation_after <= fitted.correlation_after:
        return fitted
    return replace(reflected, correlation_before=fitted.correlation_before)


def _global_axis_start(sample, nominal, bounds, roll):
    """Coarse signed axis search, with equal evidence retaining the axial prior."""
    candidates = [(0.0, 0.0)]
    candidates.extend(
        (float(x), float(y))
        for x in np.arange(-80.0, 81.0, 5.0)
        for y in np.arange(-60.0, 61.0, 10.0)
        if bounds[0][0] <= x <= bounds[0][1] and bounds[1][0] <= y <= bounds[1][1]
    )

    def rank(rotations):
        axis = axis_in_detector(nominal, np.asarray(rotations), detector_roll_deg=roll)
        return -_magnitude_agreement(*sample(axis)), sum(x * x for x in rotations)

    return min(candidates, key=rank)


def _fit_local_axis_direction(
    sample: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    nominal_axis: np.ndarray,
    *,
    rotations_deg: tuple[float, float],
    active: tuple[bool, bool],
    bounds_deg: tuple[tuple[float, float], tuple[float, float]],
    detector_roll_deg: float,
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
        axis = axis_in_detector(nominal, x, detector_roll_deg=detector_roll_deg)
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
    # A full turn ends at a periodic coordinate seam, not an inference limit.
    limited = indices[np.diff(bounds[indices], axis=1).ravel() < 360 - 1e-3]
    on_boundary = bool(np.any(np.isclose(x[limited, None], bounds[limited], rtol=0, atol=1e-3)))
    return AxisDirectionSeed((float(x[0]), float(x[1])), -before, -after, on_boundary)
