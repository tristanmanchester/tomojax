"""Angular coverage and conjugate rays of a circular acquisition."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tomojax._typed_arrays import Float64Array
    from tomojax.geometry import Detector, Geometry

_TURN = 2 * np.pi
# Views within a millionth of a turn are one direction measured again.
REPEAT_TOLERANCE = _TURN * 1e-6


def group_repeats(angles: np.ndarray) -> tuple[Float64Array, NDArray[np.intp], NDArray[np.intp]]:
    """Distinct directions in ``[0, 2 pi)``, each view's group, and group sizes.

    Angles in radians within a millionth of a turn of a neighbor share a
    group across rounding and whole turns, so a direction measured twice
    keeps one angular cell.
    """
    canonical = np.remainder(np.asarray(angles, np.float64), _TURN)
    order = np.argsort(canonical)
    gaps = np.diff(np.append(canonical[order], canonical[order[0]] + _TURN))
    # Start after the largest gap so that no group straddles the cut.
    order = np.roll(order, -(int(np.argmax(gaps)) + 1))
    unwrapped = canonical[order[0]] + np.append(
        0.0, np.cumsum(np.remainder(np.diff(canonical[order]), _TURN))
    )
    sorted_groups = np.cumsum(np.append(0, np.diff(unwrapped) > REPEAT_TOLERANCE))
    repeats = np.bincount(sorted_groups)
    representatives = np.bincount(sorted_groups, weights=unwrapped) / repeats
    groups = np.empty_like(sorted_groups)
    groups[order] = sorted_groups
    return np.remainder(representatives, _TURN), groups, repeats


def ordered_arc(angles: Float64Array) -> tuple[NDArray[np.intp], Float64Array, bool]:
    """Order distinct angles from the largest circular gap and unwrap the arc.

    A closing gap below 1.5 typical view steps counts as a full turn,
    matching FDK's full-turn tolerance without depending on angle labels.
    """
    if len(angles) < 2:
        raise ValueError("at least two distinct rotation angles are required")
    order = np.argsort(angles)
    sorted_angles = angles[order]
    circular_gaps = np.diff(np.append(sorted_angles, sorted_angles.item(0) + _TURN))
    largest = int(np.argmax(circular_gaps))
    order = np.roll(order, -(largest + 1))
    gaps = np.remainder(np.diff(angles[order]), _TURN)
    sorted_angles = angles[order[0]] + np.append(0.0, np.cumsum(gaps))
    # At the half-step boundary a missing angular cell is as plausible as jitter.
    # Treat the boundary consistently across sorting, angle units and wraparound.
    full_turn = float(circular_gaps.item(largest)) < 1.5 * float(np.median(gaps)) - 32 * math.ulp(
        _TURN
    )
    return order, sorted_angles, bool(full_turn)


def full_rotation(angles: np.ndarray) -> bool:
    """Whether the angular cells of the supplied degree angles close a full turn."""
    if len(angles) < 3:
        return False
    directions = group_repeats(np.deg2rad(angles))[0]
    return len(directions) >= 3 and ordered_arc(directions)[2]


def conjugate_weights(nu: int, center_px: float) -> NDArray[np.float32]:
    """Relative full-turn weights at detector pixels, with unit mean over conjugate rays.

    Ray s at angle theta and ray -s at theta + pi measure the same line.
    Their shares follow their distances from the panel edges. A ray seen
    only once carries twice its ordinary full-turn weight.
    """
    s = np.arange(nu, dtype=np.float64) - (nu - 1) / 2 + center_px
    lo, hi = center_px - nu / 2, center_px + nu / 2
    parameters = np.array([lo, hi, 1.0])
    return np.asarray(2 * conjugate_share(s, parameters), dtype=np.float32)


def reliability(ray: Float64Array, parameters: Float64Array) -> Float64Array:
    """Smooth edge reliability odds, recovering Wang's sin-squared partition."""
    lo, hi, pixel = parameters[..., 0], parameters[..., 1], parameters[..., 2]
    width = np.maximum(np.minimum(-lo, hi), pixel)
    distance = np.minimum(ray - lo, hi - ray)
    return np.tan(np.pi / 4 * np.clip(distance / width, 0, 1)) ** 2


def conjugate_share(
    ray: Float64Array, panel: Float64Array, total: Float64Array | None = None
) -> Float64Array:
    """Share of a conjugate ray on this panel, including a supplied multi-panel total."""
    own = reliability(ray, panel)
    if total is None:
        total = own + reliability(-ray, panel)
    return np.divide(own, total, out=np.zeros_like(total), where=total > 0)


def axial_trajectory(
    geometry: Geometry, angles: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """Axial rotation and detector translation from the projector's world-from-object poses."""
    from tomojax.geometry import ParallelGeometry

    degrees = np.asarray(angles, dtype=np.float64)
    if type(geometry) is ParallelGeometry:
        return degrees, None
    poses = np.asarray([geometry.pose_matrix(i) for i in range(len(degrees))], np.float64)
    measured = poses[:, :2, 0]
    angle = np.arctan2(measured[:, 1], measured[:, 0])
    axial = np.zeros((len(poses), 3, 3))
    axial[:, 0, 0] = axial[:, 1, 1] = np.cos(angle)
    axial[:, 1, 0], axial[:, 0, 1] = np.sin(angle), -np.sin(angle)
    axial[:, 2, 2] = 1
    if not np.allclose(poses[:, :3, :3], axial, atol=2e-7, rtol=0):
        raise ValueError("fourier_reconstruct: axial slices require rotations about z")
    # Keep the precision of recorded angles if the rotations are exactly those
    # angles rounded to the pose precision (e.g. a translation-only fit).
    radians = np.deg2rad(degrees)
    nominal = np.stack([np.cos(radians), np.sin(radians)], axis=1)
    fitted = not np.array_equal(measured.astype(np.float32), nominal.astype(np.float32))
    if fitted:
        degrees = np.rad2deg(angle)
    translations = poses[:, [0, 2], 3]
    return degrees, translations if np.any(translations) else None


def stationary_full_turn(angles: np.ndarray, shifts: np.ndarray | None) -> bool:
    """Whether opposite axial rays measure the same line over the directed circle."""
    return shifts is None and full_rotation(angles)


def full_parallel_turn(geometry: Geometry) -> bool:
    """Whether the measured orbit is axial and stationary over a full rotation."""
    from tomojax.geometry import ParallelGeometry

    base = getattr(geometry, "base", geometry)
    if not isinstance(base, ParallelGeometry):
        return False
    try:
        angles, shifts = axial_trajectory(
            geometry, np.asarray(getattr(geometry, "angles", base.angles))
        )
    except ValueError:  # An out-of-plane pose does not have axial conjugates.
        return False
    return stationary_full_turn(angles, shifts)


def outer_edge(center: float, *, full_turn: bool) -> int | None:
    """The only truncated end of an offset full-turn chord, or None for both ends.

    Zero denotes the low-coordinate end and one the high-coordinate end.
    Opposing axial rays cover the near side of an offset detector.
    """
    return int(center > 0) if full_turn and center != 0 else None


def scan_outer_edge(geometry: Geometry, detector: Detector) -> int | None:
    """Outer edge of an axis-aligned detector on the measured trajectory."""
    center = detector.center[0]
    return outer_edge(center, full_turn=bool(center) and full_parallel_turn(geometry))


def panel_field(width: int, center_px: float) -> tuple[float, float]:
    """Near and far edge distances for a horizontal detector chord, in pixels."""
    return width / 2 - abs(center_px), width / 2 + abs(center_px)


def detector_axes(roll: float) -> tuple[float, float]:
    """Cosine and sine of the detector roll with exact signed quarter turns."""
    turns, residual = divmod(roll, 90.0)
    c, s = math.cos(math.radians(residual)), math.sin(math.radians(residual))
    return ((c, s), (-s, c), (-c, -s), (s, -c))[int(turns) % 4]


def horizontal_detector(detector: Detector, roll: float) -> Detector:
    """Axis-aligned sampling lattice enclosing the detector footprint."""
    if not roll:
        return detector
    c, s = detector_axes(roll)
    du = 1 / math.hypot(c / detector.du, s / detector.dv)
    dv = 1 / math.hypot(s / detector.du, c / detector.dv)
    width, height = detector.nu * detector.du, detector.nv * detector.dv
    nu = max(1, math.ceil((abs(c) * width + abs(s) * height) / du - 1e-10))
    nv = max(1, math.ceil((abs(s) * width + abs(c) * height) / dv - 1e-10))
    if min(detector.nu, detector.nv) == 1 and roll % 90:
        # Anchor the lattice on the measured center of a single-pixel strip.
        nu, nv = nu | 1, nv | 1
    return replace(detector, nu=nu, nv=nv, du=du, dv=dv)
