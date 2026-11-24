"""fit="rotations" is the axis's wobble: each view's angle about the axis stays as recorded."""

import numpy as np
import pytest

import tomojax as tj


@pytest.mark.slow
def test_rotations_fit_the_wobble_and_leave_every_view_angle_as_recorded():
    grid, detector = tj.Grid(16, 16, 12, 1, 1, 1), tj.Detector(24, 16, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 24, endpoint=False))
    z, y, x = np.mgrid[:12, :16, :16]
    volume = (((x - 7.5) ** 2 + (y - 6) ** 2 + (z - 5.5) ** 2) < 16).astype(np.float32)
    volume += (((x - 4) ** 2 + (y - 10) ** 2 + (z - 7) ** 2) < 4).astype(np.float32)
    rng = np.random.default_rng(5)
    motion = np.zeros((24, 6))
    motion[:, :3] = np.deg2rad(rng.normal(0, 0.5, (24, 3)))  # wobble and angle jitter
    projections = np.array(
        tj.project(
            tj.Scan(np.zeros((24, 16, 24), np.float32), geometry).with_poses(motion).geometry,
            volume,
        )
    )
    result = tj.align(tj.Scan(projections, geometry), fit=("translations", "rotations"))
    assert np.all(result.poses[:, 2] == 0.0)  # the recorded angle is never corrected
    assert np.any(result.poses[:, :2] != 0.0)  # the wobble is
    assert result.info["pose_columns"] == ["alpha", "beta", "x", "z"]
