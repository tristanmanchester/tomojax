"""Publish joint-fit calibration as ordinary, serializable scan geometry."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from tomojax._data.geometry_meta import nominal_geometry
from tomojax.geometry import (
    ConeGeometry,
    ConeSegments,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax._scan import Scan
    from tomojax.geometry import ScanGeometry

type _SingleGeometry = ConeGeometry | ParallelGeometry | LaminographyGeometry | RotationAxisGeometry
type _CalibrationValue = float | tuple[float, float]
type _SegmentValue = _CalibrationValue | list[_SegmentValue]
type _PublishedValues = dict[str, _SegmentValue]


def _calibrated_geometry(
    geometry: ScanGeometry,
    corrections: Mapping[str, _CalibrationValue],
    native: tuple[float, float],
) -> ScanGeometry:
    if isinstance(geometry, ConeSegments):
        return replace(
            geometry,
            segments=tuple(
                _calibrated_geometry(part, corrections, native) for part in geometry.segments
            ),
        )
    if TYPE_CHECKING:
        assert isinstance(
            geometry, ConeGeometry | ParallelGeometry | LaminographyGeometry | RotationAxisGeometry
        )
    cone = isinstance(geometry, ConeGeometry)
    center = np.atleast_1d(corrections.get("center", (0.0, 0.0) if cone else 0.0))
    shift = (float(center[0]) * native[0], float(center[1]) * native[1] if cone else 0.0)
    detector = replace(
        geometry.detector,
        center=tuple(
            float(value + delta)
            for value, delta in zip(geometry.detector.center, shift, strict=True)
        ),
    )
    geometry = replace(geometry, detector=detector)
    roll = corrections.get("roll", 0.0)
    if TYPE_CHECKING:
        assert isinstance(roll, float)
    roll = float(roll)
    if cone:
        if TYPE_CHECKING:
            assert isinstance(geometry, ConeGeometry)
        beam = geometry.beam
        geometry = replace(geometry, beam=replace(beam, detector_roll=beam.detector_roll + roll))
    else:
        if TYPE_CHECKING:
            assert not isinstance(geometry, ConeGeometry)
        geometry = replace(geometry, detector_roll=geometry.detector_roll + roll)
    tilt = corrections.get("tilt", 0.0)
    if TYPE_CHECKING:
        assert isinstance(tilt, float)
    tilt = float(tilt)
    if tilt:
        # A lab tilt of an arbitrary axis changes its minimum-alignment frame
        # by a common encoder phase. Preserve that phase in the declared angles;
        # it is global geometry, not a per-view rotation correction.
        zero = replace(geometry, angles=[0.0])
        before = np.asarray(zero.pose_matrix(0), np.float64)[:3, :3]
        tilted = Rotation.from_rotvec([-np.deg2rad(tilt), 0, 0]).as_matrix() @ before
        axis = tuple(float(v) for v in tilted[:, 2])
        if TYPE_CHECKING:
            assert len(axis) == 3
        updated = _about_axis(geometry, axis)
        zero = replace(updated, angles=[0.0])
        after = np.asarray(zero.pose_matrix(0), np.float64)[:3, :3]
        phase = after.T @ tilted
        phase_deg = np.rad2deg(np.arctan2(phase[1, 0], phase[0, 0]))
        geometry = replace(updated, angles=np.asarray(geometry.angles) + phase_deg)
    return geometry


def _about_axis(geometry: _SingleGeometry, axis: tuple[float, float, float]) -> _SingleGeometry:
    """``geometry`` rotating about the lab ``axis``, as the public class that holds it."""
    if isinstance(geometry, ConeGeometry):
        return replace(geometry, axis_unit=axis)
    grid, detector, angles = geometry.grid, geometry.detector, geometry.angles
    roll = geometry.detector_roll
    if isinstance(geometry, ParallelGeometry) or (
        isinstance(geometry, LaminographyGeometry) and geometry.tilt_about == "x"
    ):
        # A tilt about x keeps the axis in the y-z plane: laminography with that tilt.
        tilt = float(np.rad2deg(np.arctan2(axis[1], axis[2])))
        return LaminographyGeometry(grid, detector, angles, tilt, detector_roll=roll)
    return RotationAxisGeometry(grid, detector, angles, axis, detector_roll=roll)


def _calibration_values(geometry: ScanGeometry, names: tuple[str, ...]) -> _PublishedValues:
    if isinstance(geometry, ConeSegments):
        parts = [_calibration_values(part, names) for part in geometry.segments]
        return {
            name: parts[0][name]
            if all(v[name] == parts[0][name] for v in parts)
            else [v[name] for v in parts]
            for name in names
        }
    if TYPE_CHECKING:
        assert isinstance(
            geometry, ConeGeometry | ParallelGeometry | LaminographyGeometry | RotationAxisGeometry
        )
    cone = isinstance(geometry, ConeGeometry)
    axis = np.asarray(geometry.pose_matrix(0), np.float64)[:3, 2]
    center = (
        tuple(map(float, geometry.detector.center)) if cone else float(geometry.detector.center[0])
    )
    if TYPE_CHECKING and isinstance(center, tuple):
        assert len(center) == 2
    values: _PublishedValues = {
        "center": center,
        "tilt": float(np.rad2deg(np.arctan2(axis[1], axis[2]))),
        "roll": float(geometry.beam.detector_roll if cone else geometry.detector_roll),
    }
    return {name: values[name] for name in names}


def publish_calibration(
    scan: Scan,
    fitted: np.ndarray | None,
    corrections: Mapping[str, _CalibrationValue],
    *,
    native_pixel_size: tuple[float, float],
) -> tuple[Scan, _PublishedValues]:
    """Calibrate the declared geometry and compose the previous and fitted local poses once.

    ``corrections`` contains native-pixel centre corrections and degree tilt/roll
    corrections from the joint fit. The returned calibration has the same names:
    centre is the final physical detector centre (u, or u/v for cone); tilt is
    the signed lab inclination ``atan2(axis_y, axis_z)`` in degrees; roll is the
    final detector angle from u towards v, also in degrees. ``fitted`` is None
    when the fit selected no per-view parameters. Different cone
    arrangements report their final values in segment order when those differ.
    The returned scan retains its input projections and existing lab translations.
    """
    geometry = _calibrated_geometry(nominal_geometry(scan.geometry), corrections, native_pixel_size)
    calibration = _calibration_values(geometry, tuple(corrections))
    previous = scan.poses  # degrees, as the published table
    if fitted is None:  # no per-view parameters were fitted: the scan keeps its own
        return replace(scan, geometry=geometry).with_poses(previous), calibration
    table = np.concatenate([np.degrees(fitted[:, :3]), fitted[:, 3:]], axis=1)
    if previous is not None:
        table[:, 3:] += previous[:, 3:]
        if np.any(table[:, :3]):
            before = Rotation.from_euler("YXZ", previous[:, [1, 0, 2]], degrees=True)
            update = Rotation.from_euler("YXZ", table[:, [1, 0, 2]], degrees=True)
            table[:, :3] = (before * update).as_euler("YXZ", degrees=True)[:, [1, 0, 2]]
        else:
            table[:, :3] = previous[:, :3]
    return replace(scan, geometry=geometry).with_poses(table), calibration
