from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.core.geometry.transforms import align_u_to_v
from tomojax.geometry import (
    Detector,
    Grid,
    RotationAxisGeometry,
)

# check-public-imports: allow-private
from tomojax.geometry._axis_geometry import _align_ez_to_axis
from tomojax.geometry.api import axis_pose_stack


def test_align_u_to_v_handles_exact_antiparallel_vectors() -> None:
    u = np.asarray([0.0, 0.0, 1.0])
    v = np.asarray([0.0, 0.0, -1.0])

    rotation = align_u_to_v(u, v)

    np.testing.assert_allclose(rotation @ u, v, atol=1e-9)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize("tilt", [0.05, 0.09, 1.0, 5.0])
def test_align_ez_to_axis_preserves_near_minus_z_tilt(tilt: float) -> None:
    tilt_rad = np.deg2rad(tilt)
    axis = np.asarray([np.sin(tilt_rad), 0.0, -np.cos(tilt_rad)], dtype=np.float32)

    rotation = np.asarray(_align_ez_to_axis(axis))

    np.testing.assert_allclose(rotation @ np.asarray([0.0, 0.0, 1.0]), axis, atol=1e-4)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("azimuth", [-70, 210])
@pytest.mark.parametrize("tilt", [0, 90, 171.88, 171.9, 179.9])
def test_axis_poses_retain_the_minimum_rotation_frame(tilt, azimuth):
    # Specifying the full independent rotation catches a changed in-plane
    # frame even when the third column still matches the requested axis.
    phi = np.deg2rad(azimuth)
    alignment = Rotation.from_rotvec(
        np.deg2rad(tilt) * np.asarray([-np.sin(phi), np.cos(phi), 0])
    ).as_matrix()
    angles = np.asarray([0, 17, 90, 137])
    expected = alignment @ Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    actual = np.asarray(jax.jit(axis_pose_stack)(angles, alignment[:, 2]))[:, :3, :3]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_exact_reversed_axis_uses_the_same_half_turn_as_host_geometry():
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(7, 5, 1, 1)
    angles = np.asarray([0, 17, 90], dtype=np.float32)
    geometry = RotationAxisGeometry(grid, detector, angles, (0, 0, -1))
    expected = np.asarray([geometry.pose_matrix(i) for i in range(len(angles))])
    listed = RotationAxisGeometry(grid, detector, angles.tolist(), (0, 0, -1))
    np.testing.assert_array_equal(expected, [listed.pose_matrix(i) for i in range(len(angles))])
    actual = jax.jit(axis_pose_stack)(angles, np.asarray([0, 0, -1]))
    np.testing.assert_allclose(actual, expected, atol=2e-7)


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_axis_alignment_has_finite_forward_and_reverse_derivatives(sign):
    axis = jnp.asarray([0.0, 0.0, sign])

    def poses(a):
        return axis_pose_stack(jnp.asarray([0.0, 17.0]), a)

    for derivative in (jax.jacfwd, jax.jacrev):
        assert np.isfinite(jax.jit(derivative(poses))(axis)).all()


@pytest.mark.parametrize("tilt", [0, 30, 90, 171.9, 179.9])
def test_axis_pose_derivative_matches_the_physical_rotation(tilt):
    azimuth = np.deg2rad(30)
    direction = np.asarray([-np.sin(azimuth), np.cos(azimuth), 0])

    def poses(theta):
        axis = jnp.asarray(
            [jnp.sin(theta) * np.cos(azimuth), jnp.sin(theta) * np.sin(azimuth), jnp.cos(theta)]
        )
        return axis_pose_stack(jnp.asarray([0.0]), axis)[0, :3, :3]

    theta, step = np.deg2rad(tilt), 1e-5
    expected = (
        Rotation.from_rotvec((theta + step) * direction).as_matrix()
        - Rotation.from_rotvec((theta - step) * direction).as_matrix()
    ) / (2 * step)
    for derivative in (jax.jacfwd, jax.jacrev):  # including exactly at +z (tilt 0)
        actual = jax.jit(derivative(poses))(theta)
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("as_array", [list, jnp.asarray])
def test_axis_pose_stack_accepts_array_like_inputs(as_array):
    angles, axis = as_array([0.0, 17.0, 90.0]), as_array([0.0, 0.0, 1.0])
    expected = Rotation.from_euler("z", np.asarray(angles)[:, None], degrees=True).as_matrix()
    np.testing.assert_allclose(axis_pose_stack(angles, axis)[:, :3, :3], expected, atol=2e-7)


@pytest.mark.parametrize(
    "placement",
    [{}, {"vol_center": (3, -2, 1)}, {"vol_origin": (2, 4, -3), "vol_center": (9, 9, 9)}],
)
def test_grid_resampling_preserves_cell_edges_and_placement(placement):
    from tomojax.geometry import grid_volume_origin

    grid = Grid(11, 8, 5, 0.7, 1.3, 2.1, **placement)
    resampled = grid.resampled((4, 13, 3))
    old_pitch = np.array([grid.vx, grid.vy, grid.vz])
    new_pitch = np.array([resampled.vx, resampled.vy, resampled.vz])
    np.testing.assert_allclose(new_pitch * [4, 13, 3], old_pitch * [11, 8, 5])
    np.testing.assert_allclose(
        np.array(grid_volume_origin(resampled)) - new_pitch / 2,
        np.array(grid_volume_origin(grid)) - old_pitch / 2,
    )
    assert grid.resampled((11, 8, 5)) is grid


def test_detector_grid_is_flattened_row_by_row_about_the_centre() -> None:
    from tomojax.geometry.detector_grid import zero_center_detector_grid

    detector = Detector(nu=3, nv=2, du=2.0, dv=4.0, center=(10.0, -5.0))

    u, v = zero_center_detector_grid(detector)

    np.testing.assert_allclose(u, [-2.0, 0.0, 2.0, -2.0, 0.0, 2.0])
    np.testing.assert_allclose(v, [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0])
