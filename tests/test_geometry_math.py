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
from tomojax.forward import project_joseph
from tomojax.geometry import (
    Detector,
    Grid,
    LaminographyGeometry,
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


@pytest.mark.parametrize("azimuth", [-70, 0, 30, 90, 210])
@pytest.mark.parametrize("tilt", [0, 30, 90, 171.88, 171.9, 175, 179.9])
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
    np.testing.assert_allclose(
        actual[:, :, 2], np.broadcast_to(alignment[:, 2], (len(angles), 3)), atol=3e-7
    )
    np.testing.assert_allclose(
        actual @ actual.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), actual.shape),
        atol=1e-6,
    )


def test_exact_reversed_axis_uses_the_same_half_turn_as_host_geometry():
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(7, 5, 1, 1)
    angles = np.asarray([0, 17, 90])
    geometry = RotationAxisGeometry(grid, detector, angles, (0, 0, -1))
    expected = np.asarray([geometry.pose_for_view(i) for i in range(len(angles))])
    actual = jax.jit(axis_pose_stack)(angles, np.asarray([0, 0, -1]))
    np.testing.assert_allclose(actual, expected, atol=2e-7)


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_axis_alignment_has_finite_forward_and_reverse_derivatives(sign):
    axis = jnp.asarray([0.0, 0.0, sign])

    def poses(a):
        return axis_pose_stack(jnp.asarray([0.0, 17.0]), a)

    for derivative in (jax.jacfwd, jax.jacrev):
        assert np.isfinite(jax.jit(derivative(poses))(axis)).all()


@pytest.mark.parametrize("tilt", [0, 30, 89.99, 90, 90.01, 171.88, 171.9, 179.9])
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


@pytest.mark.numerical
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_small_axis_setup_change_does_not_flip_the_object_frame(backend):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
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
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    # An asymmetric image prevents a half-turn frame error hiding behind a
    # correct rotation axis or a rotationally symmetric phantom.
    index = np.indices((13, 12, 11), dtype=np.float32)
    volume = np.exp(-np.sum(((index - np.asarray([4, 7, 5])[:, None, None, None]) / 2) ** 2, 0))
    volume = jnp.asarray(volume, jnp.float32)
    reference = project_joseph(
        volume, jnp.asarray(expected, jnp.float32), grid, detector, backend=backend
    )
    projected = project_joseph(volume, actual, grid, detector, backend=backend)
    assert float(jnp.linalg.norm(projected - reference) / jnp.linalg.norm(reference)) < 5e-5


@pytest.mark.parametrize("views", [64, 720])
@pytest.mark.parametrize("kind", ["parallel", "lamino", "axis"])
@pytest.mark.parametrize("precision", [None, "bfloat16"])
@pytest.mark.parametrize("calibrated", [False, True])
def test_batched_axis_setup_keeps_rigid_physical_poses(views, kind, precision, calibrated):
    angles = np.linspace(0, 180, views, endpoint=False)
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(9, 7, 1, 1)
    if kind == "parallel":
        nominal = np.eye(3)
        geometry = ParallelGeometry(grid, detector, angles)
    elif kind == "lamino":
        nominal = Rotation.from_euler("x", -30, degrees=True).as_matrix()
        geometry = LaminographyGeometry(grid, detector, angles, tilt_deg=30)
    else:
        nominal = Rotation.from_rotvec(np.deg2rad(175) * np.asarray([0.6, 0.8, 0])).as_matrix()
        geometry = RotationAxisGeometry(grid, detector, angles, nominal[:, 2])
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    # Match the public setup convention, independently of its matrix code:
    # axis x calibration is clockwise, then axis y calibration is counterclockwise.
    dx, dy = (0.05, -0.07) if calibrated else (0.0, 0.0)
    calibration = Rotation.from_euler("y", dy, degrees=True) * Rotation.from_euler(
        "x", -dx, degrees=True
    )
    adjusted_axis = calibration.apply(nominal[:, 2])
    adjusted, _ = Rotation.align_vectors(adjusted_axis[None], np.asarray([[0, 0, 1]]))
    expected = (
        adjusted.as_matrix() @ Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    )
    state = AlignmentState.zeros(n_views=views).replace(
        setup=SetupGeometryState.from_degrees(
            axis_rot_x_deg=dx, axis_rot_y_deg=dy, nominal_axis_unit=nominal[:, 2]
        )
    )
    with jax.default_matmul_precision(precision):
        poses = jax.jit(lambda s: apply_alignment_state(base, s).pose_stack)(state)
    rotations = np.asarray(poses, dtype=np.float64)[:, :3, :3]
    np.testing.assert_allclose(rotations, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        rotations @ rotations.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), rotations.shape),
        atol=1e-6,
    )


@pytest.mark.parametrize("views", [7, 720])
@pytest.mark.parametrize("kind", ["parallel", "lamino", "axis"])
@pytest.mark.parametrize("precision", [None, "bfloat16"])
@pytest.mark.parametrize("execution", ["eager", "jit"])
def test_axis_setup_retains_arbitrary_nominal_poses_and_their_derivatives(
    views, kind, precision, execution
):
    angles = np.linspace(0, 180, views, endpoint=False)
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(9, 7, 1, 1)
    if kind == "parallel":
        nominal = np.eye(3)
        geometry = ParallelGeometry(grid, detector, angles)
    elif kind == "lamino":
        nominal = Rotation.from_euler("x", -30, degrees=True).as_matrix()
        geometry = LaminographyGeometry(grid, detector, angles, tilt_deg=30)
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
    direction = np.asarray([0.7, -0.4], np.float32)
    weights = rng.normal(size=(views, 4, 4)) / np.sqrt(views * 16)

    def forward(p):
        return jax.jvp(poses, (p,), (jnp.asarray(direction),))

    reverse = jax.grad(lambda p: jnp.sum(poses(p) * jnp.asarray(weights, jnp.float32)))
    if execution == "jit":
        forward, reverse = jax.jit(forward), jax.jit(reverse)
    with jax.default_matmul_precision(precision):
        actual, tangent = forward(jnp.asarray(parameters, jnp.float32))
        gradient = reverse(jnp.asarray(parameters, jnp.float32))
    np.testing.assert_allclose(actual, expected(parameters), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(tangent, derivative @ direction, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        gradient, np.einsum("nijk,nij->k", derivative, weights), atol=2e-6, rtol=2e-6
    )


@pytest.mark.parametrize("input_kind", ["list", "numpy", "jax"])
def test_axis_pose_stack_accepts_array_like_inputs(input_kind):
    angles, axis = [0.0, 17.0, 90.0], [0.0, 0.0, 1.0]
    if input_kind == "numpy":
        angles, axis = np.asarray(angles), np.asarray(axis)
    elif input_kind == "jax":
        angles, axis = jnp.asarray(angles), jnp.asarray(axis)
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
