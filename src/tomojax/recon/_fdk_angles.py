"""Circular view ordering shared by FBP, FDK and virtual detector columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomojax.core.geometry.cone import cone_parts
from tomojax.core.geometry.views import host_view_poses

from ._detector_rows import DetectorRows
from ._directions import group_repeats
from ._fdk_coverage import full_turn_weights, reliability

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry
    from tomojax.core.geometry.cone import ConeBeam

_TURN = 2 * np.pi


def grouped_angles(geometry: Geometry, n_views: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct angles modulo a turn, each view's group, and group sizes.

    Angles within a millionth of a turn share their angular measure, including
    views on opposite sides of the 0/360-degree boundary.
    """
    thetas = getattr(geometry, "angles", None)
    if thetas is None:
        raise ValueError("FDK needs a geometry with view angles")
    degrees = np.asarray(thetas, dtype=np.float64)
    if degrees.ndim != 1 or len(degrees) < n_views:
        raise ValueError("FDK needs one finite rotation angle per view")
    degrees = degrees[:n_views]
    if not np.isfinite(degrees).all():
        raise ValueError("FDK needs one finite rotation angle per view")
    # The angle of the source around object z defines the quadrature. Inverting
    # world-from-object gives R.T @ (source - translation) in the object.
    parts = cone_parts(geometry)
    if parts is None:
        raise ValueError("FDK needs a cone-beam trajectory")
    poses = host_view_poses(geometry, n_views, dtype=np.float64)
    if poses is None:
        poses = np.asarray([geometry.pose_matrix(i) for i in range(n_views)], np.float64)
    azimuth = np.empty(n_views)
    start = 0
    for count, beam, _ in parts:
        stop = n_views if count is None else min(start + count, n_views)
        pose = poses[start:stop]
        source = np.einsum("nij,ni->nj", pose[:, :3, :3], beam.source() - pose[:, :3, 3])
        azimuth[start:stop] = np.arctan2(-source[:, 0], -source[:, 1])
        start = stop
    return group_repeats(azimuth)


def ordered_arc(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Order distinct angles from the largest circular gap and unwrap the arc.

    A closing gap no larger than 1.5 typical view steps counts as a full turn,
    matching FDK's full-turn tolerance without depending on angle labels.
    """
    if len(angles) < 2:
        raise ValueError("FDK needs at least two distinct rotation angles")
    order = np.argsort(angles)
    sorted_angles = angles[order]
    circular_gaps = np.diff(np.append(sorted_angles, sorted_angles[0] + _TURN))
    largest = int(np.argmax(circular_gaps))
    order = np.roll(order, -(largest + 1))
    gaps = np.remainder(np.diff(angles[order]), _TURN)
    sorted_angles = angles[order[0]] + np.append(0.0, np.cumsum(gaps))
    full_turn = circular_gaps[largest] <= 1.5 * float(np.median(gaps))
    return order, sorted_angles, bool(full_turn)


def fan_angles(beam: ConeBeam, detector: Detector) -> np.ndarray:
    """Fan angle of each detector column from the ray through the rotation axis."""
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.center[0]
    return np.arctan(u / float(beam.source_to_detector)) - np.arctan(
        float(beam.axis_offset) / float(beam.source_to_axis)
    )


def column_weights(
    parts: tuple[tuple[int | None, ConeBeam, Detector], ...],
    grouped: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Weights on rectangular horizontal detector rows for a complete trajectory."""
    angles, view_angle, repeats = grouped
    n_views = len(view_angle)
    ordering = ordered_arc(angles)
    if ordering[2]:
        fans, pixels, counts = [], [], []
        for count, beam, own in parts:
            size = n_views if count is None else count
            counts.append(size)
            fan, pixel = full_turn_fan(beam, own)
            fans.append(np.broadcast_to(fan, (size, own.nu)))
            pixels.append(np.full(size, pixel))
        return full_turn_weights(
            angles, view_angle, repeats, np.concatenate(fans), np.concatenate(pixels), tuple(counts)
        )
    weights, first = [], 0
    for count, beam, own in parts:
        stop = n_views if count is None else first + count
        selected = view_angle[first:stop]
        weights.append(_distinct_view_weights(beam, own, selected, ordering))
        first = stop
    return np.concatenate(weights) / repeats[view_angle][:, None]


def check_angular_coverage(geometry: Geometry, detector: Detector, n_views: int) -> None:
    """Check the arc of each calibrated fan before allocating weights or reading frames."""
    _, angles, full = ordered_arc(grouped_angles(geometry, n_views)[0])
    for _, beam, own in cone_parts(geometry, detector) or ():
        panel = DetectorRows(own, beam.detector_roll).horizontal if beam.detector_roll else own
        _ = full_turn_fan(beam, panel) if full else short_scan_fan(beam, panel, angles)


def full_turn_fan(beam: ConeBeam, detector: Detector) -> tuple[np.ndarray, float]:
    """Check that this panel measures the rotation axis before forming conjugate weights."""
    fan = fan_angles(beam, detector)
    pixel = float(detector.du) / float(beam.source_to_detector)
    if fan.min() > pixel or fan.max() < -pixel:
        raise ValueError("FDK needs the rotation axis to project onto the detector")
    return fan, pixel


def short_scan_fan(
    beam: ConeBeam, detector: Detector, angles: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """Resolve the virtual fan used for Parker weighting and validate the acquired arc."""
    gaps = np.diff(angles)
    step = float(np.median(gaps))
    arc = float(angles[-1] - angles[0]) + step
    gamma = fan_angles(beam, detector)
    delta = float(np.max(np.abs(gamma)))
    if arc < np.pi + 2 * delta - 1e-6:
        raise ValueError(
            f"FDK short scans need at least 180 degrees plus the fan angle "
            f"({np.rad2deg(np.pi + 2 * delta):.1f} degrees); got {np.rad2deg(arc):.1f}"
        )
    # Treat the extra acquisition angle as a virtual fan. Otherwise the
    # minimum-scan window assigns zero to valid later views of an overscan.
    return step, gamma, max(delta, (arc - np.pi) / 2)


def _distinct_view_weights(
    beam: ConeBeam,
    detector: Detector,
    selected: np.ndarray,
    ordering: tuple[np.ndarray, np.ndarray, bool],
) -> np.ndarray:
    """:func:`view_weights` for views at distinct ``angles`` (radians)."""
    order, sorted_angles, _ = ordering
    step, gamma, delta = short_scan_fan(beam, detector, sorted_angles)
    pixel = float(detector.du) / float(beam.source_to_detector)
    panel = np.array([gamma.min() - pixel / 2, gamma.max() + pixel / 2, pixel])
    return parker_weights(order, sorted_angles, selected, step, gamma, delta, panel)


def parker_weights(
    order: np.ndarray,
    sorted_angles: np.ndarray,
    selected: np.ndarray,
    step: float,
    gamma: np.ndarray,
    delta: float,
    panel: np.ndarray,
) -> np.ndarray:
    """Short-scan quadrature for the measured fan interval of a physical detector row."""
    n_views = len(order)
    # Angular measure of each view: half the gap to each neighbor.
    measure = np.empty(n_views)
    padded = np.concatenate([[sorted_angles[0] - step], sorted_angles, [sorted_angles[-1] + step]])
    measure[order] = 0.5 * (padded[2:] - padded[:-2])
    end = np.pi + 2 * delta

    def parker(beta: np.ndarray, g: np.ndarray) -> np.ndarray:
        rise = np.sin(np.pi / 4 * beta / np.maximum(delta - g, 1e-12)) ** 2
        fall = np.sin(np.pi / 4 * (end - beta) / np.maximum(delta + g, 1e-12)) ** 2
        weight = np.where(beta < 2 * (delta - g), rise, 1.0)
        weight = np.where(beta > np.pi - 2 * g, fall, weight)
        return np.where((beta < 0) | (beta > end), 0.0, weight)

    beta = np.empty((n_views, 1))
    beta[order, 0] = sorted_angles - sorted_angles[0]
    beta = beta[selected]
    g = gamma[None, :]
    # Parker's pair sums to one only where the conjugate ray was measured. An
    # offset panel misses some conjugates: share each line's weight between the
    # rays that measured it, by panel-edge reliability as in a full turn.
    own = parker(beta, g) * reliability(g, panel)
    conjugates = parker(beta + np.pi + 2 * g, -g) + parker(beta - np.pi + 2 * g, -g)
    total = own + conjugates * reliability(-g, panel)
    weight = np.divide(own, total, out=np.ones_like(own), where=total > 0)
    return weight * measure[selected, None]
