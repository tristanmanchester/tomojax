"""Circular view ordering shared by FBP, FDK and virtual detector columns."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from tomojax.core.geometry.cone import ConeGeometry, cone_parts
from tomojax.geometry.coverage import group_repeats, ordered_arc, reliability

from ._detector_rows import DetectorRows
from ._fdk_coverage import full_turn_weights, pair_reliability

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tomojax._typed_arrays import Float64Array
    from tomojax.core.geometry.base import Detector, Geometry
    from tomojax.core.geometry.cone import ConeBeam

    type Grouped = tuple[Float64Array, NDArray[np.int64], NDArray[np.int64]]

_TURN = 2 * np.pi


def _require_source_orbit(radius: Float64Array | float, distance: Float64Array | float) -> None:
    # Aligning the spin axis can leave a roundoff-sized transverse component.
    if np.any(radius <= 1e-10 * np.asarray(distance)):
        raise ValueError("FDK needs a source orbit around the rotation axis")


def grouped_angles(geometry: Geometry, n_views: int) -> Grouped:
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
    if type(geometry) is ConeGeometry:
        # With a fixed axis alignment S, the object-frame source is
        # Rz(-theta) @ S.T @ (source - translation). Its azimuth is theta
        # plus a constant phase, which cancels from every angular difference.
        if geometry.axis_unit is not None or geometry.tilt:
            lab_source = geometry.beam.source()
            lab_source[0] -= geometry.beam.axis_offset
            radius = float(np.linalg.norm(np.cross(geometry.rotation_axis(), lab_source)))
            _require_source_orbit(radius, float(np.linalg.norm(lab_source)))
        return group_repeats(np.deg2rad(degrees))
    # The angle of the source around object z defines the quadrature. Inverting
    # world-from-object gives R.T @ (source - translation) in the object.
    parts = cone_parts(geometry)
    if parts is None:
        raise ValueError("FDK needs a cone-beam trajectory")
    poses = np.asarray([geometry.pose_matrix(i) for i in range(n_views)], np.float64)
    azimuth = np.empty(n_views)
    start = 0
    for count, beam, _ in parts:
        stop = n_views if count is None else min(start + count, n_views)
        pose = poses[start:stop]
        source = np.asarray(
            np.einsum("nij,ni->nj", pose[:, :3, :3], beam.source() - pose[:, :3, 3]), np.float64
        )
        radii = np.hypot(source[:, 0], source[:, 1])
        _require_source_orbit(radii, np.hypot(radii, source[:, 2]))
        azimuth[start:stop] = np.arctan2(-source[:, 0], -source[:, 1])
        start = stop
    return group_repeats(azimuth)


def fan_angles(beam: ConeBeam, detector: Detector) -> Float64Array:
    """Fan angle of each detector column from the ray through the rotation axis."""
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.center[0]
    return np.arctan(u / float(beam.source_to_detector)) - np.arctan(
        float(beam.axis_offset) / float(beam.source_to_axis)
    )


def column_weights(
    parts: tuple[tuple[int | None, ConeBeam, Detector], ...],
    grouped: Grouped,
) -> np.ndarray:
    """Weights on rectangular horizontal detector rows for a complete trajectory."""
    angles, view_angle, repeats = grouped
    n_views = len(view_angle)
    ordering = ordered_arc(angles)
    if ordering[2]:
        fans: list[np.ndarray] = []
        pixels: list[np.ndarray] = []
        counts: list[int] = []
        for count, beam, own in parts:
            size = n_views if count is None else count
            counts.append(size)
            fan, pixel = full_turn_fan(beam, own)
            fans.append(np.broadcast_to(fan, (size, own.nu)))
            pixels.append(np.full(size, pixel))
        return full_turn_weights(
            angles, view_angle, repeats, np.concatenate(fans), np.concatenate(pixels), tuple(counts)
        )
    order, arc, _ = ordering
    weights: list[np.ndarray] = []
    short_fans = [short_scan_fan(beam, own, arc) for _, beam, own in parts]
    pitches = [own.du / beam.source_to_detector for _, beam, own in parts]
    counts = [count or n_views for count, _, _ in parts]
    panels: Float64Array = np.asarray(
        [
            [g.min().item() - p / 2, g.max().item() + p / 2, p]
            for (_, g, _), p in zip(short_fans, pitches, strict=True)
        ]
    )
    parameters = np.repeat(panels, counts, axis=0)
    first = 0
    for (step, gamma, delta), panel, count in zip(short_fans, panels, counts, strict=True):
        selected = view_angle[first : first + count]
        pair = None
        if len(parts) > 1:
            pair = pair_reliability(
                angles, view_angle, repeats, gamma[None, :], parameters, selected
            )
        weights.append(parker_weights(order, arc, selected, step, gamma, delta, panel, pair))
        first += count
    return np.concatenate(weights) / repeats[view_angle][:, None]


def check_angular_coverage(geometry: Geometry, detector: Detector, n_views: int) -> None:
    """Check the arc of each calibrated fan before allocating weights or reading frames."""
    _, angles, full = ordered_arc(grouped_angles(geometry, n_views)[0])
    for _, beam, own in cone_parts(geometry, detector) or ():
        panel = own
        if beam.detector_roll:
            layout = DetectorRows(own, beam.detector_roll)
            panel = layout.horizontal
            if not layout.quarter_turn:
                lo, hi = layout.limits(panel, np.arange(panel.nv))
                used = hi > lo
                left, right = lo[used].min().item(), hi[used].max().item()
                shift = (left + right - panel.nu) * panel.du / 2
                panel = replace(
                    panel, nu=right - left, center=(panel.center[0] + shift, panel.center[1])
                )
        _ = full_turn_fan(beam, panel) if full else short_scan_fan(beam, panel, angles)


def full_turn_fan(beam: ConeBeam, detector: Detector) -> tuple[np.ndarray, float]:
    """Check that this panel measures the rotation axis before forming conjugate weights."""
    fan = fan_angles(beam, detector)
    pixel = float(detector.du) / float(beam.source_to_detector)
    if fan.min() > pixel or fan.max() < -pixel:
        raise ValueError("FDK needs the rotation axis to project onto the detector")
    return fan, pixel


def short_scan_fan(
    beam: ConeBeam, detector: Detector, angles: Float64Array
) -> tuple[float, Float64Array, float]:
    """Resolve the virtual fan used for Parker weighting and validate the acquired arc."""
    gaps = np.diff(angles)
    step = float(np.median(gaps))
    arc = (angles.item(-1) - angles.item(0)) + step
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


def parker_weights(
    order: np.ndarray,
    sorted_angles: np.ndarray,
    selected: np.ndarray,
    step: float,
    gamma: np.ndarray,
    delta: float,
    panel: np.ndarray,
    pair: tuple[Float64Array, Float64Array] | None = None,
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
    window = parker(beta, g)
    own = window * reliability(g, panel)
    conjugates = parker(beta + np.pi + 2 * g, -g) + parker(beta - np.pi + 2 * g, -g)
    total = (
        own + conjugates * reliability(-g, panel)
        if pair is None
        else window * pair[0] + conjugates * pair[1]
    )
    weight = np.divide(own, total, out=np.ones_like(own), where=total > 0)
    return weight * measure[selected, None]
