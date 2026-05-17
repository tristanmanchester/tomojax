from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align.geometry.geometry_applier import (
    BaseGeometryArrays,
    apply_alignment_state,
    apply_setup_to_detector_grid,
    materialize_setup_geometry,
    pose_stack_for_setup,
)
from tomojax.align.model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.core.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.multires import scale_detector
from tomojax.geometry import detector_grid_from_calibration


def test_zero_state_matches_nominal_parallel_pose_and_detector_grid():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=6, du=0.5, dv=0.25)
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=[0.0, 45.0, 90.0])
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(3))

    effective = apply_alignment_state(base, state)
    nominal = stack_view_poses(geometry, 3)
    nominal_grid = detector_grid_from_calibration(detector)

    np.testing.assert_allclose(np.asarray(effective.pose_stack), np.asarray(nominal), atol=1e-7)
    np.testing.assert_allclose(np.asarray(effective.det_grid[0]), np.asarray(nominal_grid[0]))
    np.testing.assert_allclose(np.asarray(effective.det_grid[1]), np.asarray(nominal_grid[1]))


def test_detector_center_state_is_native_pixels_scaled_only_in_geometry_applier():
    native = Detector(nu=128, nv=96, du=0.5, dv=0.25)
    setup = SetupGeometryState.from_degrees(det_u_px=-4.0, det_v_px=8.0)

    for factor, expected_u_px, expected_v_px in ((1, -4.0, 8.0), (2, -2.0, 4.0), (4, -1.0, 2.0)):
        detector = scale_detector(native, factor)
        shifted = apply_setup_to_detector_grid(detector, setup, level_factor=factor)
        baseline = apply_setup_to_detector_grid(detector, SetupGeometryState(), level_factor=factor)
        u_shift_level_px = np.mean(np.asarray((shifted[0] - baseline[0]) / detector.du))
        v_shift_level_px = np.mean(np.asarray((shifted[1] - baseline[1]) / detector.dv))

        assert u_shift_level_px == np.float32(expected_u_px)
        assert v_shift_level_px == np.float32(expected_v_px)
        assert float(setup.det_u_px) == -4.0


def test_detector_roll_rotates_about_detector_center_without_changing_mean_offset():
    detector = Detector(nu=5, nv=5, du=1.0, dv=1.0)
    setup = SetupGeometryState.from_degrees(det_u_px=2.0, det_v_px=-1.0)
    rolled = setup.replace(detector_roll_rad=jnp.deg2rad(jnp.asarray(30.0, dtype=jnp.float32)))

    unrolled_grid = apply_setup_to_detector_grid(detector, setup)
    rolled_grid = apply_setup_to_detector_grid(detector, rolled)

    np.testing.assert_allclose(
        np.mean(np.asarray(rolled_grid[0])), np.mean(np.asarray(unrolled_grid[0]))
    )
    np.testing.assert_allclose(
        np.mean(np.asarray(rolled_grid[1])), np.mean(np.asarray(unrolled_grid[1]))
    )


def test_axis_pitch_yaw_have_nonzero_local_sensitivity():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=[0.0, 45.0, 90.0])
    base = BaseGeometryArrays.from_geometry(geometry, detector)

    def flattened(values: jnp.ndarray) -> jnp.ndarray:
        setup = SetupGeometryState(
            axis_rot_x_rad=values[0],
            axis_rot_y_rad=values[1],
        )
        return pose_stack_for_setup(base, setup).reshape(-1)

    jac = jax.jacfwd(flattened)(jnp.asarray([0.0, 0.0], dtype=jnp.float32))

    assert float(jnp.linalg.norm(jac)) > 1e-4


def test_laminography_nominal_axis_matches_lamino_pose_stack():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    geometry = LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=[0.0, 45.0],
        tilt_deg=30.0,
        tilt_about="x",
    )
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    setup = SetupGeometryState(nominal_axis_unit=base.nominal_axis_unit)

    np.testing.assert_allclose(
        np.asarray(pose_stack_for_setup(base, setup)),
        np.asarray(stack_view_poses(geometry, 2)),
        atol=1e-6,
    )


def test_laminography_tilt_state_materializes_effective_tilt():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    geometry = LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=[0.0, 45.0],
        tilt_deg=30.0,
        tilt_about="x",
    )
    setup = SetupGeometryState.from_degrees(tilt_deg=5.0)

    materialized = materialize_setup_geometry(geometry, grid, detector, setup)

    assert isinstance(materialized, LaminographyGeometry)
    np.testing.assert_allclose(materialized.tilt_deg, 35.0)
    np.testing.assert_allclose(
        np.asarray(stack_view_poses(materialized, 2)),
        np.asarray(
            stack_view_poses(
                LaminographyGeometry(
                    grid=grid,
                    detector=detector,
                    thetas_deg=[0.0, 45.0],
                    tilt_deg=35.0,
                    tilt_about="x",
                ),
                2,
            )
        ),
        atol=1e-6,
    )


def test_laminography_tilt_state_changes_array_pose_stack():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    geometry = LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=[0.0, 45.0],
        tilt_deg=30.0,
        tilt_about="z",
    )
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState(
        setup=SetupGeometryState.from_degrees(tilt_deg=-5.0),
        pose=PoseState.zeros(2),
    )

    effective = apply_alignment_state(base, state)

    np.testing.assert_allclose(
        np.asarray(effective.pose_stack),
        np.asarray(
            stack_view_poses(
                LaminographyGeometry(
                    grid=grid,
                    detector=detector,
                    thetas_deg=[0.0, 45.0],
                    tilt_deg=25.0,
                    tilt_about="z",
                ),
                2,
            )
        ),
        atol=1e-6,
    )


def test_geometry_applier_jits_for_dynamic_setup_values():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=[0.0, 45.0])
    base = BaseGeometryArrays.from_geometry(geometry, detector)

    @jax.jit
    def summed(det_u: jnp.ndarray) -> jnp.ndarray:
        state = AlignmentState(
            setup=SetupGeometryState(det_u_px=det_u),
            pose=PoseState.zeros(2),
        )
        effective = apply_alignment_state(base, state)
        return jnp.sum(effective.pose_stack) + jnp.sum(effective.det_grid[0])

    assert jnp.isfinite(summed(jnp.asarray(1.0, dtype=jnp.float32)))
    assert jnp.isfinite(summed(jnp.asarray(2.0, dtype=jnp.float32)))
