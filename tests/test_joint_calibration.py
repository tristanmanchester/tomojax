"""Publishing calibration preserves physical geometry and saved local corrections."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tomojax import Scan

# check-public-imports: allow-private
from tomojax._data.geometry_meta import DetectorRollGeometry
from tomojax._scan import describe, record_of, scan_from_record

# check-public-imports: allow-private
from tomojax.alignment._calibration import publish_calibration
from tomojax.core.geometry.cone import cone_parts  # check-public-imports: allow-private
from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    ConeSegments,
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
    stack_view_poses,
)


@pytest.mark.parametrize("kind", ["parallel", "lamino", "axis", "cone", "segments"])
def test_calibration_and_local_poses_replay_the_fitted_physical_geometry(kind):
    n = 60
    grid = Grid(16, 16, 12, 0.7, 0.7, 0.9)
    detector = Detector(48, 48, 0.7, 0.9, center=(0.2, -0.3))
    angles = np.linspace(0.3, 360.3, n, endpoint=False)
    axis = np.array([0.2, 0.3, 1.0])
    axis /= np.linalg.norm(axis)
    cone = kind in {"cone", "segments"}
    if cone:
        beam = ConeBeam(40, 65, detector_roll_deg=4, detector_pitch_deg=-3, source_height=0.4)
        geometry = ConeGeometry(grid, detector, angles, beam, axis_unit=tuple(axis))
        if kind == "segments":
            geometry = ConeSegments(
                (
                    replace(geometry, angles=angles[:30]),
                    replace(
                        geometry,
                        angles=angles[30:],
                        detector=replace(detector, center=(0.7, -0.1)),
                        beam=replace(beam, source_height=1.3, detector_roll_deg=-2),
                    ),
                )
            )
    else:
        geometry = (
            ParallelGeometry(grid, detector, angles)
            if kind == "parallel"
            else LaminographyGeometry(grid, detector, angles, tilt_deg=31)
            if kind == "lamino"
            else RotationAxisGeometry(grid, detector, angles, axis)
        )
        geometry = DetectorRollGeometry(geometry, 3.2)
    rng = np.random.default_rng(1152)
    previous = rng.normal(size=(n, 6)) * [0.002, 0.001, 0.003, 0.03, 0.04, 0.02]
    fitted = rng.normal(size=(n, 6)) * [0.001, 0.002, 0.001, 0.02, 0.03, 0.01]
    scan = Scan(np.zeros((n, 48, 48), np.float32), geometry).with_poses(previous)
    corrections = {"centre": (0.6, -0.4) if cone else 0.6, "tilt": 0.4, "roll": 0.75}
    corrected, calibration = publish_calibration(
        scan, fitted, corrections, native_pixel_size=(0.35, 0.45)
    )
    before = np.asarray(stack_view_poses(scan.geometry, n), np.float64)
    expected = before.copy()
    lab_tilt = Rotation.from_rotvec([-np.deg2rad(0.4), 0, 0]).as_matrix()
    delta = Rotation.from_euler("YXZ", fitted[:, [1, 0, 2]]).as_matrix()
    expected[:, :3, :3] = lab_tilt @ before[:, :3, :3] @ delta
    expected[:, :3, 3] += fitted[:, [3, 5, 4]]
    np.testing.assert_allclose(stack_view_poses(corrected.geometry, n), expected, atol=6e-7)
    assert corrected.projections is scan.projections
    assert set(calibration) == set(corrections)
    expected_centre = np.array(detector.center) + np.array([0.6 * 0.35, -0.4 * 0.45 if cone else 0])
    np.testing.assert_allclose(corrected.detector.center, expected_centre, atol=1e-14)
    if cone:
        for (_, old_beam, _), (_, new_beam, _) in zip(
            cone_parts(scan.geometry), cone_parts(corrected.geometry), strict=True
        ):
            np.testing.assert_array_equal(old_beam.source(), new_beam.source())
            assert old_beam.source_to_detector == new_beam.source_to_detector
            assert new_beam.detector_roll_deg == old_beam.detector_roll_deg - 0.75
    else:
        assert describe(corrected.geometry)[1]["detector_roll_deg"] == 3.95
        assert calibration["centre"] == pytest.approx(expected_centre[0])
        assert calibration["roll"] == 3.95
    replay = scan_from_record(record_of(corrected), poses=True)
    np.testing.assert_allclose(stack_view_poses(replay.geometry, n), expected, atol=8e-7)
    np.testing.assert_allclose(replay.detector.center, corrected.detector.center, atol=1e-14)


def test_global_calibration_does_not_leak_into_held_local_rotations():
    grid = Grid(16, 16, 12, 1, 1, 1)
    detector = Detector(48, 48, 1, 1)
    geometry = RotationAxisGeometry(grid, detector, np.linspace(0, 360, 60), (0.2, 0.3, 1))
    previous = np.tile([0.001, -0.002, 0.003, 0.1, 0.2, 0], (60, 1)).astype(np.float32)
    scan = Scan(np.zeros((60, 48, 48), np.float32), geometry).with_poses(previous)
    fitted = np.zeros((60, 6))
    fitted[:, 3] = np.linspace(-0.1, 0.1, 60)
    corrected, _ = publish_calibration(
        scan, fitted, {"tilt": 0.7, "roll": 0.75}, native_pixel_size=(1, 1)
    )
    np.testing.assert_array_equal(corrected.poses[:, :3], previous[:, :3])
    np.testing.assert_allclose(corrected.poses[:, 3:], previous[:, 3:] + fitted[:, 3:], atol=1e-8)
