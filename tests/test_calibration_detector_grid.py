from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from tomojax.calibration import detector_grid_from_center_offset, zero_center_detector_grid
from tomojax.calibration.detector_grid import offset_detector_grid
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
