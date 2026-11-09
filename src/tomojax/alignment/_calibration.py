"""Publish joint-fit calibration as ordinary, serializable scan geometry."""

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from tomojax._data.geometry_meta import DetectorRollGeometry, nominal_geometry
from tomojax.geometry import ConeGeometry, ConeSegments, RotationAxisGeometry


def _calibrated_geometry(geometry, corrections, native):
    if isinstance(geometry, ConeSegments):
        return replace(
            geometry,
            segments=tuple(
                _calibrated_geometry(part, corrections, native) for part in geometry.segments
            ),
        )
    roll = 0.0
    if isinstance(geometry, DetectorRollGeometry):
        # The outer wrapper owns the effective detector roll, as in describe().
        roll = float(geometry.detector_roll_deg)
        while isinstance(geometry, DetectorRollGeometry):
            geometry = geometry.base
    cone = isinstance(geometry, ConeGeometry)
    centre = np.atleast_1d(corrections.get("centre", (0.0, 0.0) if cone else 0.0))
    shift = (float(centre[0]) * native[0], float(centre[1]) * native[1] if cone else 0.0)
    detector = replace(
        geometry.detector,
        center=tuple(
            float(value + delta)
            for value, delta in zip(geometry.detector.center, shift, strict=True)
        ),
    )
    geometry = replace(geometry, detector=detector)
    if cone:
        geometry = replace(
            geometry,
            beam=replace(
                geometry.beam,
                detector_roll_deg=geometry.beam.detector_roll_deg
                - float(corrections.get("roll", 0.0)),
            ),
        )
    else:
        roll += float(corrections.get("roll", 0.0))
    tilt = float(corrections.get("tilt", 0.0))
    if tilt:
        # A lab tilt of an arbitrary axis changes its minimum-alignment frame
        # by a common encoder phase. Preserve that phase in the declared angles;
        # it is global geometry, not a per-view rotation correction.
        zero = replace(geometry, angles=[0.0])
        before = np.asarray(zero.pose_for_view(0), np.float64)[:3, :3]
        tilted = Rotation.from_rotvec([-np.deg2rad(tilt), 0, 0]).as_matrix() @ before
        axis = tuple(float(v) for v in tilted[:, 2])
        updated = (
            replace(geometry, axis_unit=axis)
            if cone
            else RotationAxisGeometry(geometry.grid, detector, geometry.angles, axis)
        )
        zero = replace(updated, angles=[0.0])
        after = np.asarray(zero.pose_for_view(0), np.float64)[:3, :3]
        phase = after.T @ tilted
        phase_deg = np.rad2deg(np.arctan2(phase[1, 0], phase[0, 0]))
        geometry = replace(updated, angles=np.asarray(geometry.angles) + phase_deg)
    return DetectorRollGeometry(geometry, roll) if not cone and roll else geometry


def _calibration_values(geometry, names):
    if isinstance(geometry, ConeSegments):
        values = [_calibration_values(part, names) for part in geometry.segments]
        return {
            name: values[0][name]
            if all(v[name] == values[0][name] for v in values)
            else [v[name] for v in values]
            for name in names
        }
    roll = 0.0
    if isinstance(geometry, DetectorRollGeometry):
        roll = float(geometry.detector_roll_deg)
        geometry = geometry.base
    cone = isinstance(geometry, ConeGeometry)
    axis = np.asarray(geometry.pose_for_view(0), np.float64)[:3, 2]
    values = {
        "centre": tuple(map(float, geometry.detector.center))
        if cone
        else float(geometry.detector.center[0]),
        "tilt": float(np.rad2deg(np.arctan2(axis[1], axis[2]))),
        "roll": -float(geometry.beam.detector_roll_deg) if cone else roll,
    }
    return {name: values[name] for name in names}


def publish_calibration(scan, fitted, corrections, *, native_pixel_size):
    """Calibrate the declared geometry and compose the previous and fitted local poses once.

    ``corrections`` contains native-pixel centre corrections and degree tilt/roll
    corrections from the joint fit. The returned calibration has the same names:
    centre is the final physical detector centre (u, or u/v for cone); tilt is
    the signed lab inclination ``atan2(axis_y, axis_z)`` in degrees; roll is the
    final detector angle from u towards v, also in degrees. Different cone
    arrangements report their final values in segment order when those differ.
    The returned scan retains its input projections and existing lab translations.
    """
    geometry = _calibrated_geometry(nominal_geometry(scan.geometry), corrections, native_pixel_size)
    calibration = _calibration_values(geometry, tuple(corrections))
    table = np.array(fitted, np.float64, copy=True)
    previous = scan.poses
    if previous is not None:
        table[:, 3:] += previous[:, 3:]
        if np.any(table[:, :3]):
            before = Rotation.from_euler("YXZ", previous[:, [1, 0, 2]])
            update = Rotation.from_euler("YXZ", table[:, [1, 0, 2]])
            table[:, :3] = (before * update).as_euler("YXZ")[:, [1, 0, 2]]
        else:
            table[:, :3] = previous[:, :3]
    return replace(scan, geometry=geometry).with_poses(table), calibration
