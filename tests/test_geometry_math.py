from __future__ import annotations

import numpy as np
import pytest

from tomojax.core.geometry import Detector, Grid
from tomojax.core.geometry.transforms import align_u_to_v
from tomojax.geometry import cylindrical_mask_xy, grid_from_detector_fov_slices

# check-public-imports: allow-private
from tomojax.geometry._axis_geometry import _align_ez_to_axis


def test_align_u_to_v_handles_exact_antiparallel_vectors() -> None:
    u = np.asarray([0.0, 0.0, 1.0])
    v = np.asarray([0.0, 0.0, -1.0])

    rotation = align_u_to_v(u, v)

    np.testing.assert_allclose(rotation @ u, v, atol=1e-9)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_align_ez_to_axis_preserves_near_minus_z_tilt() -> None:
    tilt_rad = np.deg2rad(0.05)
    axis = np.asarray([np.sin(tilt_rad), 0.0, -np.cos(tilt_rad)], dtype=np.float32)

    rotation = np.asarray(_align_ez_to_axis(axis))

    np.testing.assert_allclose(rotation @ np.asarray([0.0, 0.0, 1.0]), axis, atol=1e-4)


def test_grid_from_detector_fov_slices_preserves_grid_when_no_crop_needed() -> None:
    grid = Grid(nx=32, ny=32, nz=16, vx=1.0, vy=1.0, vz=1.0, vol_origin=(10.0, 20.0, 30.0))
    detector = Detector(nu=64, nv=64, du=1.0, dv=1.0)

    cropped = grid_from_detector_fov_slices(grid, detector)

    assert cropped is grid


def test_cylindrical_mask_xy_uses_grid_world_coordinates() -> None:
    grid = Grid(nx=11, ny=11, nz=3, vx=1.0, vy=1.0, vz=1.0, vol_center=(5.0, 0.0, 0.0))
    detector = Detector(nu=7, nv=11, du=1.0, dv=1.0)

    shifted_mask = cylindrical_mask_xy(grid, detector)
    centered_mask = cylindrical_mask_xy(
        Grid(nx=11, ny=11, nz=3, vx=1.0, vy=1.0, vz=1.0),
        detector,
    )

    assert not np.array_equal(shifted_mask, centered_mask)
    assert shifted_mask[0, 5]
    assert not shifted_mask[5, 5]
