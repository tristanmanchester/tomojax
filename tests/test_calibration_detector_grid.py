from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from tomojax.calibration import detector_grid_from_center_offset, zero_center_detector_grid
from tomojax.calibration.detector_grid import (
    detector_grid_from_calibration,
    detector_grid_from_detector_roll,
    offset_detector_grid,
    transform_detector_grid,
)
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import backproject_view_T, forward_project_view_T


def test_dynamic_detector_grid_matches_static_detector_center_forward_projection():
    grid = Grid(nx=8, ny=8, nz=6, vx=1.0, vy=1.0, vz=1.0)
    det_zero = Detector(nu=8, nv=6, du=0.5, dv=0.25, det_center=(0.0, 0.0))
    det_static = Detector(nu=8, nv=6, du=0.5, dv=0.25, det_center=(2.0, -1.5))
    geom = ParallelGeometry(grid=grid, detector=det_zero, thetas_deg=[0.0])
    pose = jnp.asarray(geom.pose_for_view(0), dtype=jnp.float32)
    volume = jax.random.normal(jax.random.key(0), (grid.nx, grid.ny, grid.nz))

    det_grid = detector_grid_from_center_offset(det_zero, det_u_px=4.0, det_v_px=-6.0)

    dynamic = forward_project_view_T(pose, grid, det_zero, volume, det_grid=det_grid)
    static = forward_project_view_T(pose, grid, det_static, volume)

    np.testing.assert_allclose(np.asarray(dynamic), np.asarray(static), atol=1e-5, rtol=1e-5)


def test_dynamic_detector_grid_matches_static_detector_center_backprojection():
    grid = Grid(nx=8, ny=8, nz=6, vx=1.0, vy=1.0, vz=1.0)
    det_zero = Detector(nu=8, nv=6, du=0.5, dv=0.25, det_center=(0.0, 0.0))
    det_static = Detector(nu=8, nv=6, du=0.5, dv=0.25, det_center=(2.0, -1.5))
    geom = ParallelGeometry(grid=grid, detector=det_zero, thetas_deg=[0.0])
    poses = stack_view_poses(geom, 1)
    image = jax.random.normal(jax.random.key(1), (det_zero.nv, det_zero.nu))

    det_grid = detector_grid_from_center_offset(det_zero, det_u_px=4.0, det_v_px=-6.0)

    dynamic = backproject_view_T(poses[0], grid, det_zero, image, det_grid=det_grid)
    static = backproject_view_T(poses[0], grid, det_static, image)

    np.testing.assert_allclose(np.asarray(dynamic), np.asarray(static), atol=1e-5, rtol=1e-5)


def test_detector_grid_offsets_accept_jax_scalars_for_future_optimisation():
    detector = Detector(nu=4, nv=3, du=0.5, dv=0.25, det_center=(0.0, 0.0))
    base = zero_center_detector_grid(detector)

    shifted = offset_detector_grid(
        base,
        det_u_px=jnp.asarray(2.0, dtype=jnp.float32),
        det_v_px=jnp.asarray(-4.0, dtype=jnp.float32),
        native_du=detector.du,
        native_dv=detector.dv,
    )

    np.testing.assert_allclose(np.asarray(shifted[0] - base[0]), 1.0)
    np.testing.assert_allclose(np.asarray(shifted[1] - base[1]), -1.0)


def test_detector_grid_roll_rotates_about_zero_centre_before_offset():
    detector = Detector(nu=3, nv=3, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    base = zero_center_detector_grid(detector)

    rolled = transform_detector_grid(
        base,
        detector_roll_deg=90.0,
        det_u_px=2.0,
        det_v_px=-1.0,
        native_du=detector.du,
        native_dv=detector.dv,
    )

    expected_u = -base[1] + 2.0
    expected_v = base[0] - 1.0
    np.testing.assert_allclose(np.asarray(rolled[0]), np.asarray(expected_u), atol=1e-6)
    np.testing.assert_allclose(np.asarray(rolled[1]), np.asarray(expected_v), atol=1e-6)


def test_detector_grid_from_calibration_accepts_jax_roll_scalar():
    detector = Detector(nu=3, nv=3, du=1.0, dv=2.0, det_center=(0.0, 0.0))
    direct = detector_grid_from_calibration(
        detector,
        detector_roll_deg=jnp.asarray(15.0, dtype=jnp.float32),
    )

    assert direct[0].shape == (detector.nu * detector.nv,)
    assert direct[1].shape == (detector.nu * detector.nv,)


def test_detector_roll_grid_preserves_detector_physical_center():
    detector = Detector(nu=3, nv=3, du=0.5, dv=0.25, det_center=(1.0, -0.5))
    unrolled = detector_grid_from_detector_roll(detector, detector_roll_deg=0.0)
    static = detector_grid_from_center_offset(detector, det_u_px=2.0, det_v_px=-2.0)

    np.testing.assert_allclose(np.asarray(unrolled[0]), np.asarray(static[0]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(unrolled[1]), np.asarray(static[1]), atol=1e-6)
