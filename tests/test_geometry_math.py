from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tomojax.alignment.api import (
    AlignmentState,
    BaseGeometryArrays,
    SetupGeometryState,
    apply_alignment_state,
)

# check-public-imports: allow-private
from tomojax.core.geometry.transforms import align_u_to_v
from tomojax.geometry import (
    Detector,
    Grid,
    ParallelGeometry,
    RotationAxisGeometry,
    cylindrical_mask_xy,
    grid_from_detector_fov_slices,
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


@pytest.mark.parametrize("tilt_deg", [0.05, 0.09, 1.0, 5.0])
def test_align_ez_to_axis_preserves_near_minus_z_tilt(tilt_deg: float) -> None:
    tilt_rad = np.deg2rad(tilt_deg)
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
    expected = np.asarray([geometry.pose_for_view(i) for i in range(len(angles))])
    listed = RotationAxisGeometry(grid, detector, angles.tolist(), (0, 0, -1))
    np.testing.assert_array_equal(expected, [listed.pose_for_view(i) for i in range(len(angles))])
    actual = jax.jit(axis_pose_stack)(angles, np.asarray([0, 0, -1]))
    np.testing.assert_allclose(actual, expected, atol=2e-7)


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_axis_alignment_has_finite_forward_and_reverse_derivatives(sign):
    axis = jnp.asarray([0.0, 0.0, sign])

    def poses(a):
        return axis_pose_stack(jnp.asarray([0.0, 17.0]), a)

    for derivative in (jax.jacfwd, jax.jacrev):
        assert np.isfinite(jax.jit(derivative(poses))(axis)).all()


@pytest.mark.parametrize("tilt", [30, 90, 171.9, 179.9])
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
    actual = jax.jit(jax.jacfwd(poses))(theta)
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)


def test_small_axis_setup_change_does_not_flip_the_object_frame():
    grid = Grid(13, 12, 11, 0.8, 1.1, 1.3)
    detector = Detector(21, 19, 0.7, 0.9, (0.13, -0.21))
    angles = np.asarray([0, 17, 90, 137])
    nominal_tilt, change = 171.88, 0.04
    nominal = Rotation.from_euler("x", -nominal_tilt, degrees=True).as_matrix()
    adjusted = Rotation.from_euler("x", -nominal_tilt - change, degrees=True).as_matrix()
    geometry = RotationAxisGeometry(grid, detector, angles, nominal[:, 2])
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState.zeros(n_views=len(angles)).replace(
        setup=SetupGeometryState.from_degrees(
            axis_rot_x_deg=change, nominal_axis_unit=nominal[:, 2]
        )
    )
    actual = jax.jit(lambda s: apply_alignment_state(base, s).pose_stack)(state)
    expected = np.broadcast_to(np.eye(4), (len(angles), 4, 4)).copy()
    expected[:, :3, :3] = (
        adjusted @ Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    )
    # The full pose, not only its rotation axis, catches a half-turn frame error.
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("views", [7, 720])
@pytest.mark.parametrize("kind", ["parallel", "axis"])
@pytest.mark.parametrize("precision", [None, "bfloat16"])
def test_axis_setup_retains_arbitrary_nominal_poses_and_their_derivatives(views, kind, precision):
    angles = np.linspace(0, 180, views, endpoint=False)
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(9, 7, 1, 1)
    if kind == "parallel":
        nominal = np.eye(3)
        geometry = ParallelGeometry(grid, detector, angles)
    else:
        nominal = Rotation.from_rotvec(np.deg2rad(175) * np.asarray([0.6, 0.8, 0])).as_matrix()
        geometry = RotationAxisGeometry(grid, detector, angles, nominal[:, 2])
    rng = np.random.default_rng(61)
    extra_rotations = Rotation.from_euler("xyz", rng.normal(size=(views, 3)) * 0.007).as_matrix()
    translations = rng.normal(size=(views, 3)) * 0.25
    view_rotations = Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    nominal_poses = np.broadcast_to(np.eye(4), (views, 4, 4)).copy()
    nominal_poses[:, :3, :3] = nominal @ view_rotations @ extra_rotations
    nominal_poses[:, :3, 3] = translations
    base = replace(
        BaseGeometryArrays.from_geometry(geometry, detector),
        nominal_pose_stack=jnp.asarray(nominal_poses, jnp.float32),
    )

    def expected(parameters):
        calibration = Rotation.from_euler("y", parameters[1], degrees=True) * Rotation.from_euler(
            "x", -parameters[0], degrees=True
        )
        axis = calibration.apply(nominal[:, 2])
        adjusted, _ = Rotation.align_vectors(axis[None], np.asarray([[0, 0, 1]]))
        adjusted = adjusted.as_matrix()
        poses = np.broadcast_to(np.eye(4), (views, 4, 4)).copy()
        poses[:, :3, :3] = adjusted @ view_rotations @ extra_rotations
        # Nominal translations are already in the lab frame, not co-rotating
        # object offsets. Only the change in the axis basis acts on them.
        poses[:, :3, 3] = np.einsum("ij,nj->ni", adjusted @ nominal.T, translations)
        return poses

    def poses(parameters):
        setup = SetupGeometryState.from_degrees(
            axis_rot_x_deg=parameters[0],
            axis_rot_y_deg=parameters[1],
            nominal_axis_unit=nominal[:, 2],
        )
        state = AlignmentState.zeros(n_views=views).replace(setup=setup)
        return apply_alignment_state(base, state).pose_stack

    parameters = np.asarray([0.05, -0.07])
    step = 1e-3
    derivative = np.stack(
        [
            (expected(parameters + delta) - expected(parameters - delta)) / (2 * step)
            for delta in np.eye(2) * step
        ],
        axis=-1,
    )
    direction = jnp.asarray([0.7, -0.4], jnp.float32)
    forward = jax.jit(lambda p: jax.jvp(poses, (p,), (direction,)))
    with jax.default_matmul_precision(precision):
        actual, tangent = forward(jnp.asarray(parameters, jnp.float32))
    np.testing.assert_allclose(actual, expected(parameters), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(tangent, derivative @ np.asarray(direction), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("as_array", [list, jnp.asarray])
def test_axis_pose_stack_accepts_array_like_inputs(as_array):
    angles, axis = as_array([0.0, 17.0, 90.0]), as_array([0.0, 0.0, 1.0])
    expected = Rotation.from_euler("z", np.asarray(angles)[:, None], degrees=True).as_matrix()
    np.testing.assert_allclose(axis_pose_stack(angles, axis)[:, :3, :3], expected, atol=2e-7)


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


def test_disk_axis_inference_preserves_ambiguity_and_missing_metadata():
    from tomojax.geometry.api import infer_disk_axes

    grid = {"nx": 3, "ny": 5, "nz": 7}
    assert infer_disk_axes((3, 5, 7), grid) == "xyz"
    assert infer_disk_axes((7, 5, 3), grid) == "zyx"
    assert infer_disk_axes((3, 5, 3), {**grid, "nz": 3}) is None
    assert infer_disk_axes((3, 5), grid) is None
    assert infer_disk_axes((3, 5, 7), None) is None
