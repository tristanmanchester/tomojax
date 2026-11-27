"""Angular coverage and conjugate rays of a circular acquisition."""

from __future__ import annotations

import numpy as np

_TURN = 2 * np.pi
# Views within a millionth of a turn are one direction measured again.
REPEAT_TOLERANCE = _TURN * 1e-6


def group_repeats(
    angles: np.ndarray, *, tolerance: float = REPEAT_TOLERANCE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct directions in ``[0, 2 pi)``, each view's group, and group sizes.

    Angles in radians within a millionth of a turn of a neighbor share a
    group by default, whichever whole turn they are labeled with and whatever
    their rounding, so a direction measured twice keeps one angular cell.
    ``tolerance`` is in radians; literal high-precision acquisition angles can
    request only roundoff-scale grouping instead of fitted-pose tolerance.
    """
    canonical = np.remainder(np.asarray(angles, np.float64), _TURN)
    order = np.argsort(canonical)
    gaps = np.diff(np.append(canonical[order], canonical[order[0]] + _TURN))
    # Start after the largest gap so that no group straddles the cut.
    order = np.roll(order, -(int(np.argmax(gaps)) + 1))
    unwrapped = canonical[order[0]] + np.append(
        0.0, np.cumsum(np.remainder(np.diff(canonical[order]), _TURN))
    )
    sorted_groups = np.cumsum(np.append(0, np.diff(unwrapped) > tolerance))
    repeats = np.bincount(sorted_groups)
    representatives = np.bincount(sorted_groups, weights=unwrapped) / repeats
    groups = np.empty_like(sorted_groups)
    groups[order] = sorted_groups
    return np.remainder(representatives, _TURN), groups, repeats


def ordered_arc(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Order distinct angles from the largest circular gap and unwrap the arc.

    A closing gap no larger than 1.5 typical view steps counts as a full turn,
    matching FDK's full-turn tolerance without depending on angle labels.
    """
    if len(angles) < 2:
        raise ValueError("at least two distinct rotation angles are required")
    order = np.argsort(angles)
    sorted_angles = angles[order]
    circular_gaps = np.diff(np.append(sorted_angles, sorted_angles[0] + _TURN))
    largest = int(np.argmax(circular_gaps))
    order = np.roll(order, -(largest + 1))
    gaps = np.remainder(np.diff(angles[order]), _TURN)
    sorted_angles = angles[order[0]] + np.append(0.0, np.cumsum(gaps))
    full_turn = circular_gaps[largest] <= 1.5 * float(np.median(gaps))
    return order, sorted_angles, bool(full_turn)


def full_rotation(angles: np.ndarray) -> bool:
    """Whether the angular cells of the supplied degree angles close a full turn."""
    if len(angles) < 3:
        return False
    directions = group_repeats(np.deg2rad(angles))[0]
    return len(directions) >= 3 and ordered_arc(directions)[2]


def conjugate_weights(nu: int, center_px: float) -> np.ndarray:
    """Relative full-turn weights at detector pixels, with unit mean over conjugate rays.

    Ray s at angle theta and ray -s at theta + pi measure the same line.
    Their shares follow their distances from the panel edges. A ray seen
    only once carries twice its ordinary full-turn weight.
    """

    s = np.arange(nu) - (nu - 1) / 2 + center_px
    lo, hi = center_px - nu / 2, center_px + nu / 2
    parameters = np.array([lo, hi, 1.0])
    own = reliability(s, parameters)
    opposite = reliability(-s, parameters)
    return np.asarray(2 * own / (own + opposite), dtype=np.float32)


def reliability(ray: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Smooth edge reliability odds, recovering Wang's sin-squared partition."""
    lo, hi, pixel = parameters[..., 0], parameters[..., 1], parameters[..., 2]
    width = np.maximum(np.minimum(-lo, hi), pixel)
    distance = np.minimum(ray - lo, hi - ray)
    return np.tan(np.pi / 4 * np.clip(distance / width, 0, 1)) ** 2


def parallel_field(geometry: object) -> tuple[float, float] | None:
    """Distances from the axis to the near and far horizontal edges in pixels.

    These define the doubly measured and complete field radii of a stationary
    axial parallel full turn. A negative near distance means a central gap.
    """
    from .parallel import ParallelGeometry

    if (
        type(geometry) is not ParallelGeometry
        or geometry.detector_roll
        or not full_rotation(np.asarray(geometry.angles))
    ):
        return None
    detector = geometry.detector
    offset = abs(detector.center[0]) / detector.du
    return detector.nu / 2 - offset, detector.nu / 2 + offset
