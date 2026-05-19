from __future__ import annotations

import numpy as np
import pytest

from tomojax.core.geometry import Detector, Grid
from tomojax.core.geometry.transforms import align_u_to_v
from tomojax.geometry import cylindrical_mask_xy, grid_from_detector_fov_slices


def test_align_u_to_v_handles_exact_antiparallel_vectors() -> None:
    u = np.asarray([0.0, 0.0, 1.0])
    v = np.asarray([0.0, 0.0, -1.0])

    rotation = align_u_to_v(u, v)

    np.testing.assert_allclose(rotation @ u, v, atol=1e-9)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


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
