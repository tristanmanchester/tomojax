"""Exact filtered-backprojection weights for circular parallel-beam scans.

View ``i`` measures the volume's Fourier transform on the plane perpendicular
to its ray ``r_i``. For rays rotating about a fixed axis ``a``, a frequency ``k``
lies in the planes of the views where ``k . r(theta) = 0``. Changing variables
from angle to frequency contributes ``|d(k . r)/d theta| = |k . (a x r)|``: a ramp
along the detector direction of ``a x r``. Dividing by the number ``N(k)`` of
scanned views that measure ``k`` (one or two) gives

    x = sum_i dtheta_i * R_i^T F^-1[ |k . (a x r_i)| / N_i(k) * F p_i ],

exact on every measured frequency for any axis tilt, arc length or angular
spacing. An untilted half turn reduces to standard FBP and a full turn to a ramp
with ``N = 2``. Unmeasured frequencies, such as laminography's missing cone,
stay zero; no prior can be inferred from the data alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TWO_PI = 2.0 * np.pi
# Relative tolerance for treating a scan as separable or a turn as complete.
_EXACT = 1e-6


@dataclass(frozen=True, slots=True)
class FBPWeights:
    """Per-view filter parameters for one circular scan.

    ``params`` columns are ``t_u, t_v, c_u, c_v, phase, weight``: the detector
    components of ``a x r``, the detector components of ``a`` scaled by ``a . r``,
    the view angle from the start of the scanned arc and its quadrature weight.
    ``separable`` scans need only a per-view scalar times a 1D ramp along u;
    ``view_scale`` holds those scalars.
    """

    params: np.ndarray
    arc_length: float
    full_turn: bool
    separable: bool
    view_scale: np.ndarray


def fbp_weights(poses: np.ndarray) -> FBPWeights:
    """Fit the scan's rotation axis and derive exact per-view FBP weights.

    Poses are world-from-object transforms with rays along world +y and the
    detector u/v axes along world x/z. The axis is the normal of the cone traced
    by the object-frame ray directions, so built-in and calibrated geometries
    are handled alike. Raises ``ValueError`` when the views do not determine a
    rotation (fewer than three distinct directions).
    """
    rotation = np.asarray(poses, dtype=np.float64)[:, :3, :3]
    if rotation.shape[0] < 3:
        raise ValueError("FBP weights need at least three views")
    detector_u, ray, detector_v = rotation[:, 0, :], rotation[:, 1, :], rotation[:, 2, :]
    centred = ray - ray.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred)
    if eigenvalues[1] <= 1e-12 * max(eigenvalues[2], 1e-300):
        raise ValueError("FBP weights need views that rotate about an axis")
    axis = eigenvectors[:, 0]
    # Orient the axis so that the views' angular order is a right-hand rotation.
    reference = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    reference /= np.linalg.norm(reference)
    other = np.cross(axis, reference)
    angles = np.arctan2(ray @ other, ray @ reference)

    order = np.argsort(angles)
    unwrapped = np.unwrap(angles[order])
    gaps = np.diff(unwrapped)
    # The largest gap between sorted views marks where the scanned arc starts.
    wrap_gap = _TWO_PI - (unwrapped[-1] - unwrapped[0])
    largest = int(np.argmax(np.append(gaps, wrap_gap)))
    if largest < len(gaps):
        order = np.roll(order, -(largest + 1))
        unwrapped = np.unwrap(angles[order])
        gaps = np.diff(unwrapped)
        wrap_gap = _TWO_PI - (unwrapped[-1] - unwrapped[0])
    span = unwrapped[-1] - unwrapped[0]
    full_turn = wrap_gap <= float(np.median(gaps)) * (1 + 1e-3)
    if full_turn:
        # Midpoint quadrature around the closed circle.
        previous = np.append(wrap_gap, gaps)
        following = np.append(gaps, wrap_gap)
        start, arc_length = unwrapped[0] - 0.5 * wrap_gap, _TWO_PI
    else:
        # Each end view covers half a gap beyond itself.
        previous = np.append(gaps[0], gaps)
        following = np.append(gaps, gaps[-1])
        start, arc_length = unwrapped[0] - 0.5 * gaps[0], span + 0.5 * (gaps[0] + gaps[-1])
    sorted_weights = 0.5 * (previous + following)
    weights = np.empty_like(sorted_weights)
    weights[order] = sorted_weights
    phase = np.empty_like(unwrapped)
    phase[order] = unwrapped - start

    tangent = np.cross(axis, ray)
    axial = ray @ axis
    t_u, t_v = np.sum(tangent * detector_u, 1), np.sum(tangent * detector_v, 1)
    c_u, c_v = (detector_u @ axis) * axial, (detector_v @ axis) * axial
    params = np.stack([t_u, t_v, c_u, c_v, phase, weights], axis=1)

    untilted = bool(np.all(np.abs(axial) <= _EXACT))
    row_ramp = bool(np.all(np.abs(t_v) <= _EXACT * np.abs(t_u)))
    separable = row_ramp and (full_turn or untilted)
    if full_turn:
        count = np.full(len(weights), 2.0)
    else:
        # Untilted views share frequencies only with the opposite view.
        count = 1.0 + (np.mod(phase + np.pi, _TWO_PI) < arc_length)
    view_scale = np.abs(t_u) * weights / count
    return FBPWeights(
        params=params.astype(np.float32),
        arc_length=float(arc_length),
        full_turn=bool(full_turn),
        separable=separable,
        view_scale=view_scale.astype(np.float32),
    )
