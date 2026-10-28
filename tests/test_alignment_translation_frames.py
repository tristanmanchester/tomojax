"""Physical translation observability and consistent public alignment frames."""

from __future__ import annotations

import csv
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.alignment import AlignConfig, _prealign, align_multires

# check-public-imports: allow-private
from tomojax.alignment._geometry.initializers import reprojection_det_u_seed

# check-public-imports: allow-private
from tomojax.alignment._objectives import fold_recon

# check-public-imports: allow-private
from tomojax.alignment._objectives.recon_layer import PoseAdjustedGeometry

# check-public-imports: allow-private
from tomojax.alignment._prealign import estimate_view_shifts, translation_params_from_shifts
from tomojax.alignment.api import (
    AlignmentSchedule,
    AlignmentStage,
    AlignmentState,
    AlignMultiresResumeState,
    AlignResumeState,
    BaseGeometryArrays,
    L2LossSpec,
    PoseState,
    SetupGeometryState,
    align,
    apply_alignment_state,
    apply_pose_update,
    apply_pose_updates,
    least_motion_estimate,
    save_alignment_params_csv,
    save_alignment_params_json,
    se3_from_pose_params,
)
from tomojax.core.cone import cone_model, cone_project, frame_coefficients

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.forward import project_joseph
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
from tomojax.io import build_geometry_from_dataset_metadata
from tomojax.recon import FistaConfig


def _geometry(size, kind, angles):
    if kind == "anisotropic":
        grid = Grid(size, size - 3, size // 2, 0.8, 1.2, 1.4)
        detector = Detector(size + 5, size // 2 + 3, 0.8, 1.4, (0.27, -0.31))
    else:
        grid = Grid(size, size, size, 1.0, 1.0, 1.0)
        detector = Detector(size, size, 1.0, 1.0)
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if kind == "lamino"
        else ParallelGeometry(grid, detector, angles)
    )
    return geometry, grid, detector


@pytest.mark.parametrize("size", [64, 128, 256])
@pytest.mark.parametrize("kind", ["parallel", "anisotropic", "lamino"])
def test_detector_translation_is_observable_in_every_scheduled_geometry(size, kind):
    angles = np.array([0.0, 13.7, 89.97, 90.0, 90.03, 167.2], np.float32)
    geometry, _, detector = _geometry(size, kind, angles)
    nominal = stack_view_poses(geometry, len(angles))
    pitch = jnp.array([detector.du, detector.dv])

    def detector_position(uv, pose):
        parameters = jnp.zeros(5).at[3:].set(uv * pitch)
        moved = apply_pose_update(pose, parameters, translation_frame="detector")
        return moved[jnp.array([0, 2]), 3] / pitch

    jacobian = jax.vmap(jax.jacfwd(detector_position), in_axes=(None, 0))(jnp.zeros(2), nominal)
    np.testing.assert_allclose(jacobian, np.broadcast_to(np.eye(2), jacobian.shape), atol=1e-7)

    # Preserve the old, explicitly object-frame convention for existing poses.
    parameters = jnp.tile(jnp.array([0.02, -0.03, 0.01, 0.4, -0.7]), (len(angles), 1))
    expected = np.asarray(nominal, dtype=np.float64) @ np.asarray(
        jax.vmap(se3_from_pose_params)(parameters), dtype=np.float64
    )
    np.testing.assert_allclose(apply_pose_updates(nominal, parameters), expected, atol=2e-7)


def test_detector_updates_preserve_nominal_beam_translation_and_object_rotation():
    nominal = jnp.array([[0.0, -1, 0, 4], [1, 0, 0, 7], [0, 0, 1, -2], [0, 0, 0, 1]])
    parameters = jnp.array([0.03, -0.01, 0.05, 0.2, -0.4])
    updated = apply_pose_update(nominal, parameters, translation_frame="detector")
    np.testing.assert_allclose(updated[:3, 3], [4.2, 7, -2.4])
    np.testing.assert_allclose(
        updated[:3, :3], (nominal @ se3_from_pose_params(parameters))[:3, :3]
    )


@pytest.mark.gpu
def test_rigid_geometry_retains_precision_with_reduced_matmul_defaults():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA matrix multiplication precision modes")
    geometry, _, _ = _geometry(12, "lamino", np.array([17.0, 90.0, 143.0], np.float32))
    nominal = stack_view_poses(geometry, 3)
    params = np.array([[0.006, -0.004, 0.005, 0.12, -0.07]] * 3, np.float32)
    expected = np.asarray(nominal, dtype=np.float64).copy()
    rotation = Rotation.from_euler("YXZ", params[:, [1, 0, 2]]).as_matrix()
    expected[:, :3, :3] = expected[:, :3, :3] @ rotation
    expected[:, [0, 2], 3] += params[:, 3:]
    with jax.default_matmul_precision("bfloat16"):
        actual = jax.jit(lambda n, p: apply_pose_updates(n, p, translation_frame="detector"))(
            nominal, jnp.asarray(params)
        )
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    rotations = np.asarray(actual)[:, :3, :3]
    np.testing.assert_allclose(
        rotations @ rotations.transpose(0, 2, 1), np.broadcast_to(np.eye(3), (3, 3, 3)), atol=3e-7
    )


@pytest.mark.parametrize("kind", ["parallel", "anisotropic", "lamino"])
def test_state_and_reconstruction_use_the_same_detector_frame(kind):
    geometry, _, detector = _geometry(16, kind, np.array([11.0, 90.0, 137.0], np.float32))
    params = jnp.array(
        [[0.02, -0.01, 0.03, 0.2, -0.7], [0, 0, 0, -0.3, 0.8], [0.01, 0, -0.03, 0.1, 0.4]]
    )
    state = AlignmentState.zeros(n_views=3)
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = state.replace(
        setup=state.setup.replace(nominal_axis_unit=base.nominal_axis_unit),
        pose=PoseState(params, translation_frame="detector"),
    )
    effective_poses = jax.jit(lambda value: apply_alignment_state(base, value).pose_stack)(state)
    adapter = PoseAdjustedGeometry(geometry, params, translation_frame="detector")
    expected = stack_view_poses(adapter, 3)
    np.testing.assert_allclose(effective_poses, expected, atol=2e-7)
    leaves, structure = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert restored.pose.translation_frame == "detector"
    assert restored.pose.replace(pose_params=params / 2).translation_frame == "detector"


def test_translation_frames_are_named():
    cfg = AlignConfig(pose_translation_frame="detector")
    assert cfg.pose_translation_frame == "detector"
    with pytest.raises(ValueError, match="pose_translation_frame"):
        AlignConfig(pose_translation_frame="pixels")


@pytest.mark.parametrize("step", [0.0, -1e-3, float("nan"), float("inf")])
def test_finite_difference_step_must_be_finite_and_positive(step):
    with pytest.raises(ValueError, match="gn_difference_step"):
        AlignConfig(gn_difference_step=step)


def test_gauss_newton_jacobian_is_explicit():
    assert AlignConfig().gn_jacobian == "autodiff"
    with pytest.raises(ValueError, match="gn_jacobian"):
        AlignConfig(gn_jacobian="unknown")


@pytest.mark.parametrize("multires", [False, True])
def test_resume_cannot_silently_change_translation_frame(multires):
    geometry, grid, detector = _geometry(8, "parallel", np.array([0.0, 90.0], np.float32))
    data = jnp.zeros((2, detector.nv, detector.nu))
    volume, params = jnp.zeros((8, 8, 8)), jnp.zeros((2, 5))
    cfg = AlignConfig(pose_translation_frame="detector", backend="jax")
    if multires:
        resume = AlignMultiresResumeState(volume, params)
        with pytest.raises(ValueError, match="pose_translation_frame differs"):
            align_multires(geometry, grid, detector, data, config=cfg, resume_state=resume)
    else:
        resume = AlignResumeState(volume, params)
        with pytest.raises(ValueError, match="pose_translation_frame differs"):
            align(geometry, grid, detector, data, config=cfg, resume_state=resume)


@pytest.mark.numerical
@pytest.mark.parametrize("kind", ["parallel", "anisotropic", "lamino"])
@pytest.mark.parametrize("multires", [False, True])
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("motion", ["translation", "rigid"])
def test_public_alignment_and_checkpoint_keep_detector_frame_at_ninety_degrees(
    kind, multires, backend, motion
):
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("requires the CUDA CUDA reconstruction path")
    geometry, grid, detector = _geometry(12, kind, np.array([17.0, 90.0, 143.0], np.float32))
    volume = jnp.zeros((grid.nx, grid.ny, grid.nz)).at[3:8, 3:7, 1:4].set(1)
    volume = volume.at[7:9, 5:7, 3:5].set(0.6)
    nominal = stack_view_poses(geometry, 3)
    truth = np.array(
        [[0, 0, 0, 0.12, -0.07], [0, 0, 0, -0.09, 0.13], [0, 0, 0, 0.1, 0.08]], np.float32
    )
    if motion == "rigid":
        truth[:, :3] = [[0.006, -0.004, 0.005], [-0.004, 0.007, -0.006], [0.003, 0.004, -0.007]]
    # Construct world translations and rotations independently of the update.
    poses = np.asarray(nominal).copy()
    for i, parameters in enumerate(truth):
        rotation = Rotation.from_euler("YXZ", parameters[[1, 0, 2]]).as_matrix()
        poses[i, :3, :3] = np.asarray(nominal)[i, :3, :3] @ rotation
    poses[:, 0, 3] += truth[:, 3]
    poses[:, 2, 3] += truth[:, 4]
    data = jax.vmap(lambda t: forward_project_view_T(t, grid, detector, volume))(jnp.asarray(poses))
    saved = []
    cfg = AlignConfig(
        pose_translation_frame="detector",
        backend=backend,
        optimise_dofs=("dx", "dz") if motion == "translation" else None,
        outer_iterations=8,
        iterations=1,
        # This is a fixed-volume API/derivative regression, not joint-recovery evidence.
        lipschitz=1e12,
        tv_weight=0,
        loss=L2LossSpec(),
        early_stop=False,
        gather_dtype="fp32",
        gn_jacobian="central",
    )
    if multires:
        initial = AlignMultiresResumeState(
            volume, jnp.zeros((3, 5)), pose_translation_frame="detector"
        )
        _, params, info = align_multires(
            geometry,
            grid,
            detector,
            data,
            config=cfg,
            factors=(1,),
            resume_state=initial,
            checkpoint_callback=saved.append,
        )
    else:
        _, params, info = align(
            geometry,
            grid,
            detector,
            data,
            config=cfg,
            init_x=volume,
            checkpoint_callback=saved.append,
        )
    # Alignment reports the least-motion estimate of the truth.
    expected = least_motion_estimate(
        np.asarray(volume),
        np.pad(truth, ((0, 0), (0, 1))),
        nominal=np.asarray(nominal),
        grid=grid,
        translation_frame="detector",
        active=("dx", "dz") if motion == "translation" else ("alpha", "beta", "phi", "dx", "dz"),
        cone_beam=False,
    )[1]
    np.testing.assert_allclose(params[:, 3:5], expected[:, 3:5], atol=3e-3)
    np.testing.assert_allclose(params[:, :3], expected[:, :3], atol=5e-5)
    assert info["pose_translation_frame"] == "detector"
    assert saved and all(item.pose_translation_frame == "detector" for item in saved)
    if multires:
        _, resumed, resumed_info = align_multires(
            geometry, grid, detector, data, config=cfg, factors=(1,), resume_state=saved[-1]
        )
    else:
        _, resumed, resumed_info = align(
            geometry, grid, detector, data, config=cfg, resume_state=saved[-1]
        )
    np.testing.assert_array_equal(resumed, params)
    assert resumed_info["pose_translation_frame"] == "detector"


@pytest.mark.parametrize("frame", ["object", "detector"])
@pytest.mark.parametrize("roll", [0.0, 0.2])
def test_train_fold_preserves_pose_frame_and_excludes_padding(monkeypatch, frame, roll):
    geometry, grid, detector = _geometry(8, "lamino", np.array([17.0, 90.0, 143.0], np.float32))
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState.zeros(n_views=3)
    params = jnp.array([[0.02, 0, 0, 0.3, -0.7], [0, 0.01, 0, 0.2, -0.4], [0, 0, -0.03, -0.2, 0.6]])
    state = state.replace(
        setup=state.setup.replace(nominal_axis_unit=base.nominal_axis_unit, detector_roll_rad=roll),
        pose=PoseState(params, translation_frame=frame),
    )
    expected = apply_alignment_state(base, state).pose_stack[jnp.array([2, 0])]
    observed = []
    configurations = []

    def fake_reconstruct(actual_geometry, actual_grid, actual_detector, data, **kwargs):
        observed.append(stack_view_poses(actual_geometry, data.shape[0]))
        configurations.append(kwargs["config"])
        return jnp.zeros((8, 8, 8)), {}

    monkeypatch.setattr(fold_recon, "fista_tv", fake_reconstruct)
    config = FistaConfig(
        iterations=0,
        ray_integrator="exact",
        projector_unroll=3,
        checkpoint_projector=False,
        gather_dtype="fp16",
        tv_weight=0,
        regulariser="huber_tv",
        huber_delta=0.01,
        tv_prox_iterations=1,
        nonnegative=False,
    )
    _, info = fold_recon.reconstruct_train_fold_nograd(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=jnp.zeros((3, 8, 8)),
        state=state,
        train_idx=jnp.array([2, 0, 2]),
        train_mask=jnp.array([1, 1, 0]),
        init_x=None,
        level_factor=1,
        cfg=config,
    )
    np.testing.assert_allclose(observed[0], expected, atol=2e-7)
    assert info["train_indices"] == [2, 0]
    expected_cfg = replace(config, iterations=1)
    if roll:
        expected_cfg = replace(
            expected_cfg,
            projector_model="ray",
            backend="jax",
            views_per_batch=1,
            grad_mode="stream",
        )
    else:
        expected_cfg = replace(
            expected_cfg, projector_unroll=1, checkpoint_projector=True, gather_dtype="fp32"
        )
    assert configurations == [expected_cfg]


def test_train_fold_preserves_represented_degree_metadata(monkeypatch):
    # float64 degrees that float32 cannot represent must reach the setup geometry.
    angles = np.asarray([17.123456789, 63.375, 137.625])
    grid, detector = Grid(5, 4, 3, 0.8, 1.1, 1.3), Detector(9, 7, 0.7, 0.9)
    geometry = RotationAxisGeometry(grid, detector, angles, (0.6, 0.8, 0.1))
    observed = []

    def fake_reconstruct(actual_geometry, actual_grid, actual_detector, data, **kwargs):
        observed.append(np.asarray(actual_geometry.geometry.angles))
        return jnp.zeros((5, 4, 3)), {}

    monkeypatch.setattr(fold_recon, "fista_tv", fake_reconstruct)
    state = AlignmentState.zeros(n_views=3).replace(
        setup=SetupGeometryState.from_degrees(axis_rot_x_deg=0.05, axis_rot_y_deg=-0.07)
    )
    fold_recon.reconstruct_train_fold_nograd(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=jnp.zeros((3, 7, 9)),
        state=state,
        train_idx=jnp.asarray([2, 0, 2]),
        train_mask=jnp.asarray([1, 1, 0]),
        init_x=None,
        level_factor=1,
        cfg=FistaConfig(
            iterations=1,
            tv_weight=0,
            regulariser="huber_tv",
            huber_delta=0.01,
            tv_prox_iterations=1,
            nonnegative=False,
        ),
    )
    np.testing.assert_array_equal(observed[0], angles[[2, 0]])


def test_detector_pose_exports_carry_their_frame(tmp_path):
    params = np.array([[0.0, 0, 0, 0.4, -0.7]], np.float32)
    json_path, csv_path = tmp_path / "pose.json", tmp_path / "pose.csv"
    save_alignment_params_json(json_path, params, du=0.8, dv=1.4, translation_frame="detector")
    payload = json.loads(json_path.read_text())
    assert payload["pose_translation_frame"] == "detector"
    assert payload["views"][0]["dx_px"] == pytest.approx(0.5)
    save_alignment_params_csv(csv_path, params, du=0.8, dv=1.4, translation_frame="detector")
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["pose_translation_frame"] == "detector"


@pytest.mark.parametrize("frame", ["object", "detector"])
@pytest.mark.parametrize("active", [(True, True), (False, True)])
@pytest.mark.parametrize("kind", ["parallel", "anisotropic", "lamino"])
def test_translation_seed_converts_image_motion_without_moving_frozen_dofs(frame, active, kind):
    geometry, _, detector = _geometry(12, kind, np.array([17.0, 90.0, 143.0], np.float32))
    shifts = np.array([[1, -2], [-1, 1], [2, 0]], np.float64) * [detector.du, detector.dv]
    params = np.zeros((3, 5), np.float32)
    params[:, 3] = 0.4
    poses = np.asarray(stack_view_poses(geometry, 3), np.float64)
    mapping = (
        np.broadcast_to(np.eye(3)[[0, 2]], (len(poses), 2, 3))
        if frame == "detector"
        else poses[:, [0, 2], :3]
    )
    actual = translation_params_from_shifts(shifts, mapping, params, active=active)
    for i in range(3):
        basis = np.eye(2) if frame == "detector" else poses[i][[0, 2], :3][:, [0, 2]]
        fixed = np.where(active, 0, params[i, 3:])
        target = shifts[i] - basis @ fixed
        expected = np.linalg.lstsq(basis * np.array(active)[None, :], target, rcond=1e-3)[0]
        np.testing.assert_allclose(actual[i, 3:], np.where(active, expected, fixed), atol=2e-6)
        # Each seeded translation reproduces its view's detector shift where observable.
        if all(active) and frame == "detector":
            moved = apply_pose_update(
                jnp.asarray(poses[i], jnp.float32),
                jnp.asarray(actual[i]),
                translation_frame=frame,
            )
            np.testing.assert_allclose(
                np.asarray(moved)[[0, 2], 3] - poses[i][[0, 2], 3], shifts[i], atol=1e-5
            )


@pytest.mark.parametrize("frame", ["object", "detector"])
@pytest.mark.parametrize("active", [(True, True), (False, True)])
def test_translation_seed_subtracts_frozen_depth_motion(frame, active):
    geometry, _, _ = _geometry(8, "lamino", np.array([11.0, 47.0, 137.0]))
    poses = np.asarray(stack_view_poses(geometry, 3), np.float64)
    # A calibrated tilted detector sees the frozen depth translation too.
    detector_axes = Rotation.from_euler("XYZ", [0.25, -0.13, 0.42]).as_matrix()[[0, 2]]
    mapping = np.broadcast_to(detector_axes, (3, 2, 3))
    if frame == "object":
        mapping = mapping @ poses[:, :3, :3]
    expected = np.array(
        [
            [0.01, -0.02, 0.03, 0.4, -0.2, 0.6],
            [-0.02, 0.01, -0.01, -0.1, 0.3, -0.5],
            [0.03, -0.01, 0.02, 0.2, 0.1, 0.4],
        ],
        dtype=np.float32,
    )
    shifts = np.einsum("nij,nj->ni", mapping, expected[:, [3, 5, 4]])
    initial = expected.copy()
    initial[:, 3:5] *= np.logical_not(active)
    actual = translation_params_from_shifts(shifts, mapping, initial, active=active)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    np.testing.assert_array_equal(actual[:, [0, 1, 2, 5]], initial[:, [0, 1, 2, 5]])


@pytest.mark.parametrize("kind", ["parallel", "lamino"])
def test_shift_search_recovers_large_view_shifts(kind):
    angles = np.linspace(0.0, 180.0 if kind == "parallel" else 360.0, 40, endpoint=False)
    geometry, grid, detector = _geometry(32, kind, angles.astype(np.float32))
    c = (np.arange(32) - 15.5) / 32
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    volume = np.zeros((32, 32, 32), np.float32)
    for (cx, cy, cz), r in [((0.1, -0.1, 0.05), 0.12), ((-0.12, 0.08, -0.1), 0.08)]:
        volume += np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * r * r))
    poses = stack_view_poses(geometry, len(angles))
    clean = project_joseph(jnp.asarray(volume), poses, grid, detector)
    rng = np.random.default_rng(4)
    truth = rng.uniform(-5, 5, (len(angles), 2))
    data = _prealign.shift_views(clean, jnp.asarray(truth, jnp.float32))
    found = estimate_view_shifts(geometry, grid, detector, data) / [detector.du, detector.dv]
    rotations = np.asarray(poses, np.float64)[:, :3, :3]
    spacing = (detector.du, detector.dv)
    expected = _prealign._remove_object_translation(truth, rotations[:, [0, 2], :], spacing)
    actual = _prealign._remove_object_translation(found, rotations[:, [0, 2], :], spacing)
    # +/-5 px shifts come within the local solver's sub-pixel reach.
    assert np.sqrt(np.mean((actual - expected) ** 2)) < 0.4


@pytest.mark.parametrize("sigma", [5.0, 12.0])
@pytest.mark.parametrize("cropped", [False, True])
def test_shift_correlation_recovers_finite_support_translations(sigma, cropped):
    # Evaluate the continuous field at shifted detector coordinates, without
    # reusing the implementation's interpolation, FFTs, or zero-fill rule. A
    # large or cropped object overlaps itself less at big lags, which an
    # unnormalised correlation prefers over the true shift.
    nv, nu = 32, 64
    v, u = np.meshgrid(np.arange(nv) - (nv - 1) / 2, np.arange(nu) - (nu - 1) / 2, indexing="ij")

    def field(du, dv):
        blobs = [
            (-0.42 * nu if cropped else 0.0, 0.0, 1.0, sigma),
            (0.1 * nu, 0.1 * nv, 0.7, 0.6 * sigma),
            (-0.18 * nu, -0.15 * nv, 0.4, 0.9 * sigma),
        ]
        image = np.full((nv, nu), 0.5)
        for x, y, amplitude, radius in blobs:
            image += amplitude * np.exp(-0.5 * ((u - du - x) ** 2 + (v - dv - y) ** 2) / radius**2)
        return image.astype(np.float32)

    shifts = np.array([[3.3, -2.2], [-9.4, 3.1], [0.4, -0.3]])
    measured = jnp.asarray(1.4 * np.stack([field(*shift) for shift in shifts]) + 0.3)
    model = jnp.asarray(np.repeat(field(0.0, 0.0)[None], len(shifts), axis=0))

    actual = _prealign.correlation_peaks(measured, model, max_u=12, max_v=8)

    np.testing.assert_allclose(actual, shifts, atol=0.15, rtol=0.0)


@pytest.mark.parametrize("widths", [(6.0, 1.4), (12.0, 3.0)])
def test_shift_correlation_refines_coupled_detector_axes(widths):
    # An elongated feature at 45 degrees couples the u and v peak curvature.
    v, u = np.meshgrid(np.arange(48) - 23.5, np.arange(64) - 31.5, indexing="ij")
    cs = sn = np.sqrt(0.5)

    def field(du, dv):
        x, y = cs * (u - du) + sn * (v - dv), -sn * (u - du) + cs * (v - dv)
        return (0.3 + np.exp(-0.5 * (x / widths[0]) ** 2 - 0.5 * (y / widths[1]) ** 2)).astype(
            np.float32
        )

    truth = np.array([[0.4, -0.3], [3.3, -2.2], [-5.4, 3.1]])
    measured = jnp.asarray(np.stack([field(*shift) for shift in truth]))
    model = jnp.asarray(np.repeat(field(0.0, 0.0)[None], len(truth), axis=0))

    actual = _prealign.correlation_peaks(measured, model, max_u=8, max_v=8)

    np.testing.assert_allclose(actual, truth, atol=0.15, rtol=0.0)


@pytest.mark.parametrize("axis", ["u", "v"])
def test_shift_correlation_keeps_an_unobservable_axis_stationary(axis):
    # A projection of an extruded object has features in only one direction.
    v, u = np.meshgrid(np.arange(32) - 15.5, np.arange(64) - 31.5, indexing="ij")

    def field(shift):
        coordinate = (u if axis == "u" else v) - shift
        return (0.3 + np.exp(-0.5 * (coordinate / 3) ** 2)).astype(np.float32)

    shifts = np.array([0.4, -3.3, 2.2])
    truth = np.zeros((len(shifts), 2))
    truth[:, 0 if axis == "u" else 1] = shifts
    measured = jnp.asarray(np.stack([field(shift) for shift in shifts]))
    model = jnp.asarray(np.repeat(field(0.0)[None], len(shifts), axis=0))

    actual = _prealign.correlation_peaks(measured, model, max_u=5, max_v=5)

    np.testing.assert_allclose(actual, truth, atol=0.15, rtol=0.0)
    np.testing.assert_array_equal(actual[:, 1 if axis == "u" else 0], 0.0)


@pytest.mark.parametrize(("sign_u", "sign_v"), [(-1, -1), (1, 1), (-1, 1)])
def test_shift_correlation_has_no_wrapped_overlap_at_search_bounds(sign_u, sign_v):
    nv, nu, max_u, max_v = 13, 16, 5, 4
    du, dv = sign_u * max_u, sign_v * max_v
    model = np.random.default_rng(67).uniform(0.1, 1.0, (nv, nu)).astype(np.float32)
    # Literal integer translation with zero fill: an exact match at the bound.
    measured = np.zeros_like(model)
    target = (slice(max(dv, 0), min(nv + dv, nv)), slice(max(du, 0), min(nu + du, nu)))
    source = (slice(max(-dv, 0), min(nv - dv, nv)), slice(max(-du, 0), min(nu - du, nu)))
    measured[target] = model[source]

    actual = _prealign.correlation_peaks(
        jnp.asarray(measured[None]), jnp.asarray(model[None]), max_u=max_u, max_v=max_v
    )

    np.testing.assert_array_equal(actual, [[du, dv]])


def test_shift_correlation_keeps_constant_views_stationary():
    images = jnp.full((2, 13, 16), 1.2, jnp.float32)
    actual = _prealign.correlation_peaks(images, images, max_u=4, max_v=3)
    np.testing.assert_array_equal(actual, np.zeros((2, 2)))


@pytest.mark.parametrize("factors", [None, (2, 1)])
def test_translation_seed_runs_once_at_the_first_level(monkeypatch, factors):
    # check-public-imports: allow-private
    from tomojax.alignment._pose import _pose_loop

    # check-public-imports: allow-private
    from tomojax.alignment._stages import _stage_runners

    geometry, grid, detector = _geometry(8, "parallel", np.linspace(0, 180, 6, endpoint=False))
    data = jnp.zeros((6, detector.nv, detector.nu), jnp.float32)
    calls = []

    def record(geometry, grid, detector, projections, cfg):
        calls.append((projections.shape, cfg.seed_translations))

    monkeypatch.setattr(_pose_loop, "seeded_translation_params", record)
    monkeypatch.setattr(_stage_runners, "seeded_translation_params", record)
    cfg = AlignConfig(outer_iterations=1, iterations=1, seed_translations=True)
    if factors is None:
        align(geometry, grid, detector, data, config=cfg)
        assert calls == [((6, 8, 8), True)]
    else:
        align_multires(geometry, grid, detector, data, factors=factors, config=cfg)
        assert calls == [((6, 4, 4), True)]


@pytest.mark.parametrize("kind", ["parallel", "lamino"])
def test_reprojection_seed_recovers_a_detector_centre_offset(kind):
    angles = np.linspace(0.0, 180.0 if kind == "parallel" else 360.0, 48, endpoint=False)
    geometry, grid, detector = _geometry(32, kind, angles.astype(np.float32))
    c = (np.arange(32) - 15.5) / 32
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    volume = np.zeros((32, 32, 32), np.float32)
    for (cx, cy, cz), r in [((0.12, -0.08, 0.05), 0.1), ((-0.1, 0.1, -0.08), 0.07)]:
        volume += np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * r * r))
    poses = stack_view_poses(geometry, len(angles))
    clean = project_joseph(jnp.asarray(volume), poses, grid, detector)
    shift = jnp.tile(jnp.asarray([[2.3, 0.0]], jnp.float32), (len(angles), 1))
    data = _prealign.shift_views(clean, shift)
    seed = reprojection_det_u_seed(data, geometry, grid, detector)
    assert seed.status == "ok_reprojection"
    assert abs(seed.det_u_px + 2.3) < 0.1
    # Shifts that clip more of the views never score better by discarding signal.
    score = _prealign.reprojection_residual(geometry, grid, detector, data)
    clipped = [score(shift) for shift in (-3.0, -6.0, -10.0, -14.0)]
    assert clipped == sorted(clipped)


def test_reprojection_seed_recovers_a_detector_centre_offset_in_a_segmented_cone_scan():
    grid, detector = Grid(16, 16, 16, 1.0, 1.0, 1.0), Detector(24, 24, 1.0, 1.0)
    beam = ConeBeam(40, 80)
    geometry = ConeSegments(
        (
            ConeGeometry(grid, detector, np.linspace(0, 360, 16, endpoint=False), beam),
            ConeGeometry(grid, detector, np.linspace(7, 367, 16, endpoint=False), beam),
        )
    )
    c = (np.arange(16) - 7.5) / 16
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    volume = np.zeros((16, 16, 16), np.float32)
    for (cx, cy, cz), r in [((0.12, -0.08, 0.05), 0.1), ((-0.1, 0.1, -0.08), 0.07)]:
        volume += np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * r * r))
    coefficients = cone_model(geometry, detector).coefficients(stack_view_poses(geometry, 32), grid)
    clean = cone_project(jnp.asarray(volume), coefficients, grid, detector, backend="jax")
    shift = jnp.tile(jnp.asarray([[2.3, 0.0]], jnp.float32), (32, 1))

    seed = reprojection_det_u_seed(_prealign.shift_views(clean, shift), geometry, grid, detector)

    assert abs(seed.det_u_px + 2.3) < 0.15


@pytest.mark.parametrize(
    ("turn", "wide"), [(180.0, False), (360.0, False), (180.0, True)], ids=["half", "full", "wide"]
)
def test_cor_alignment_finds_the_axis_of_a_parallel_scan(turn, wide):
    import tomojax as tj

    n, rows, views, centre = 64, 20, 120, 4.6
    angles = np.linspace(0.0, turn, views, endpoint=False)
    # A wide object fills more than the field of view, as a sample in a
    # capillary does; an FBP-consistency search is misled by such scans.
    size = 96 if wide else n
    grid = Grid(size, size, rows, 1.0, 1.0, 1.0)
    c = np.arange(size) - (size - 1) / 2
    x, y = np.meshgrid(c, c, indexing="ij")
    rng = np.random.default_rng(4)
    plane = np.zeros((size, size), np.float32)
    for cx, cy, r in zip(
        *rng.uniform(-0.4 * size, 0.4 * size, (2, 12)), rng.uniform(2, 6, 12), strict=False
    ):
        plane += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r)).astype(np.float32)
    volume = np.repeat(plane[:, :, None], rows, axis=2)
    true = ParallelGeometry(grid, Detector(n, rows, 1.0, 1.0, (centre, 0.0)), angles)
    data = np.asarray(tj.project(true, volume))
    scan = tj.Scan(data, ParallelGeometry(grid, Detector(n, rows, 1.0, 1.0), angles))
    found = tj.align(scan, mode="cor").scan.detector.center[0]
    assert abs(found - centre) <= 0.3


@pytest.mark.parametrize("kind", ["parallel", "lamino", "anisotropic"])
def test_least_motion_moves_a_constant_u_shift_into_the_detector_centre(kind):
    angles = np.linspace(0.0, 360.0, 24, endpoint=False)
    geometry, grid, detector = _geometry(24, kind, angles)
    nominal = np.asarray(stack_view_poses(geometry, angles.size), np.float64)
    rng = np.random.default_rng(3)
    params = np.zeros((angles.size, 6), np.float32)
    params[:, :3] = rng.normal(0.0, 1e-2, (angles.size, 3))
    params[:, 3] = -1.5 + nominal[:, 0, :3] @ np.asarray([0.7, -0.4, 0.3])
    params[:, 4] = rng.normal(0.0, 0.2, angles.size)
    # A smooth object well inside the grid, so moving it loses nothing.
    index = np.indices((grid.nx, grid.ny, grid.nz), np.float64)
    shape = np.array([grid.nx, grid.ny, grid.nz])[:, None, None, None]
    volume = np.exp(-0.5 * np.sum(((index - (shape - 1) / 2) / (shape / 8)) ** 2, axis=0))
    volume = volume.astype(np.float32)

    moved_volume, moved, gauge = least_motion_estimate(
        volume,
        params,
        nominal=nominal,
        grid=grid,
        translation_frame="detector",
        active=("alpha", "beta", "phi", "dx", "dz"),
        cone_beam=False,
        detector_offset=True,
    )

    assert gauge is not None
    assert gauge.detector_offset == pytest.approx(1.5, abs=0.02)
    shifted = replace(
        detector,
        center=(detector.center[0] + gauge.detector_offset, detector.center[1]),
    )
    for view in (0, 6, 13):
        pose = jnp.asarray(nominal[view], jnp.float32)
        before = apply_pose_update(pose, jnp.asarray(params[view]), translation_frame="detector")
        after = apply_pose_update(pose, jnp.asarray(moved[view]), translation_frame="detector")
        expected = forward_project_view_T(before, grid, detector, jnp.asarray(volume))
        actual = forward_project_view_T(after, grid, shifted, jnp.asarray(moved_volume))
        assert float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)) < 0.05


@pytest.mark.parametrize("frame", ["object", "detector"])
def test_saved_alignment_is_reapplied_in_its_translation_frame(frame):
    angles = np.linspace(0.0, 360.0, 6, endpoint=False)
    geometry, grid, detector = _geometry(8, "lamino", angles)
    params = np.random.default_rng(5).normal(0.0, 0.3, (angles.size, 5)).astype(np.float32)
    meta = {
        "detector": detector.to_dict(),
        "grid": grid.to_dict(),
        "angles": angles.astype(np.float32),
        "geometry_type": "lamino",
        "tilt_deg": 30.0,
        "align_params": params,
        "align_gauge": {"pose_translation_frame": frame},
    }
    _, _, saved = build_geometry_from_dataset_metadata(meta, poses=True)
    expected = apply_pose_updates(
        stack_view_poses(geometry, angles.size), jnp.asarray(params), translation_frame=frame
    )
    for view in range(angles.size):
        np.testing.assert_allclose(saved.pose_for_view(view), expected[view], atol=1e-5)


def test_cor_then_pose_reports_the_constant_detector_shift_as_the_centre():
    angles = np.linspace(0.0, 360.0, 8, endpoint=False).astype(np.float32)
    geometry, grid, detector = _geometry(12, "parallel", angles)
    volume = jnp.zeros((grid.nx, grid.ny, grid.nz)).at[3:8, 3:7, 2:9].set(1)
    volume = volume.at[7:9, 5:7, 4:6].set(0.6)
    offset, motion = 0.3, 0.1 * np.sin(np.deg2rad(3 * angles))
    poses = np.asarray(stack_view_poses(geometry, angles.size)).copy()
    poses[:, 0, 3] += motion - offset  # a detector offset c shifts images by -c
    data = jax.vmap(lambda t: forward_project_view_T(t, grid, detector, volume))(jnp.asarray(poses))
    cfg = AlignConfig(
        schedule="cor_then_pose",
        freeze=("alpha", "beta", "phi"),  # isolate the translation split
        pose_translation_frame="detector",
        backend="jax",
        outer_iterations=8,
        iterations=1,
        lipschitz=1e12,  # keeps the known volume fixed
        tv_weight=0,
        loss=L2LossSpec(),
        early_stop=False,
        gather_dtype="fp32",
        gn_jacobian="central",
    )
    initial = AlignMultiresResumeState(
        volume, jnp.zeros((angles.size, 5)), pose_translation_frame="detector"
    )
    _, params, info = align_multires(
        geometry, grid, detector, data, config=cfg, factors=(1,), resume_state=initial
    )

    variables = {v["name"]: v for v in info["geometry_calibration_state"]["detector"]}
    assert variables["det_u_px"]["value"] == pytest.approx(offset / detector.du, abs=0.01)
    assert variables["det_u_px"]["status"] == "estimated"
    np.testing.assert_allclose(params[:, 3], motion, atol=5e-3)

    with pytest.raises(ValueError, match="cor_then_pose"):
        AlignConfig(schedule="cor_then_pose")


@pytest.fixture(params=["tilted", "segments"])
def cone_proposal_case(request):
    grid = Grid(9, 8, 7, 0.8, 1.1, 1.3)
    detector = Detector(13, 11, 0.7, 0.9, (0.13, -0.21))
    first = ConeGeometry(
        grid, detector, [11.0, 59.0], ConeBeam(23, 41, axis_offset=0.6), tilt_deg=23
    )
    second = ConeGeometry(
        grid,
        detector,
        [127.0, 181.0, 237.0],
        ConeBeam(29, 47, detector_roll_deg=7, detector_pitch_deg=-3, axis_offset=-0.3),
        tilt_deg=31,
    )
    geometry = first if request.param == "tilted" else ConeSegments((first, second))
    poses = stack_view_poses(geometry, len(geometry.angles))
    volume = jnp.asarray(np.random.default_rng(92).uniform(0.1, 1, (9, 8, 7)), jnp.float32)
    model = cone_model(geometry, detector)
    assert model is not None
    frames = jnp.asarray(model.frames(len(poses)))
    targets = cone_project(
        volume, frame_coefficients(poses, frames, grid, detector), grid, detector, backend="jax"
    )
    return geometry, grid, detector, poses, frames, volume, targets


def _cone_detector_coordinates(frames, points):
    centre, u, v, source = (np.asarray(frames, np.float64)[:, i] for i in range(4))
    normal = np.cross(u, v)
    ray = points - source
    length = np.sum(normal * (centre - source), axis=1) / np.sum(normal * ray, axis=1)
    hit = source + length[:, None] * ray - centre
    return np.stack([np.sum(hit * u, axis=1), np.sum(hit * v, axis=1)], axis=1)


def _skip_cuda_without_gpu(backend):
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("CUDA required")


def _cone_proposal_config(backend, *, dof="dx"):
    schedule = AlignmentSchedule(
        "cone_proposal",
        (
            AlignmentStage(
                "pose",
                (dof,),
                "fixed_volume",
                "gn",
                stage_role="proposal",
                differentiability="performance_only",
                maxiter=1,
            ),
        ),
    )
    return AlignConfig(
        schedule=schedule,
        ray_integrator="joseph",
        backend=backend,
        pose_translation_frame="detector",
        seed_translations=False,
        tv_weight=0,
        views_per_batch=2,
    )


@pytest.mark.numerical
@pytest.mark.parametrize("perturbation", [("dx", 3, 1.0), ("phi", 2, 0.02)])
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_public_cone_proposal_stage_recovers_known_pose(cone_proposal_case, backend, perturbation):
    _skip_cuda_without_gpu(backend)
    geometry, grid, detector, poses, frames, volume, _ = cone_proposal_case
    dof, column, step = perturbation
    truth_params = jnp.zeros((len(poses), 6), jnp.float32).at[:, column].set(step)
    truth = apply_pose_updates(poses, truth_params, translation_frame="detector")
    targets = cone_project(
        volume, frame_coefficients(truth, frames, grid, detector), grid, detector, backend="jax"
    )
    resume = AlignMultiresResumeState(
        x=volume,
        pose_params=jnp.zeros((len(poses), 6)),
        pose_translation_frame="detector",
        ray_integrator="joseph",
    )
    cfg = _cone_proposal_config(backend, dof=dof)
    result_volume, params, info = align_multires(
        geometry, grid, detector, targets, factors=(1,), config=cfg, resume_state=resume
    )
    # Only physical cone rays, not parallel-ray scoring, recover the pose exactly.
    stat = info["outer_stats"][0]
    assert stat["proposal_best_name"] == f"{dof}_plus" and stat["loss_after"] < 1e-7
    assert stat["proposal_backend_provenance"]["actual_backend"] == backend
    actual = apply_pose_updates(poses, params, translation_frame="detector")
    np.testing.assert_allclose(actual, truth, atol=2e-6)
    np.testing.assert_array_equal(result_volume, volume)


@pytest.mark.numerical
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_zero_volume_proposal_retains_poses_without_scoring(cone_proposal_case, backend):
    _skip_cuda_without_gpu(backend)
    geometry, grid, detector, poses, _, volume, targets = cone_proposal_case
    volume = jnp.zeros_like(volume)
    initial = jnp.zeros((len(poses), 6)).at[:, 3].set(jnp.arange(len(poses)) * 0.1)
    resume = AlignMultiresResumeState(
        x=volume,
        pose_params=initial,
        pose_translation_frame="detector",
        ray_integrator="joseph",
    )
    result_volume, params, info = align_multires(
        geometry,
        grid,
        detector,
        targets,
        factors=(1,),
        config=_cone_proposal_config(backend),
        resume_state=resume,
    )
    np.testing.assert_array_equal(params, initial)
    np.testing.assert_array_equal(result_volume, volume)
    stat = info["outer_stats"][0]
    assert stat["proposal_status"] == "skipped"
    assert stat["proposal_reason"] == "zero volume has no pose evidence"
    assert stat["proposal_candidate_count"] == 0
    assert stat["proposal_backend_provenance"] == {}


@pytest.mark.numerical
@pytest.mark.parametrize("frame", ["object", "detector"])
def test_cone_translation_seed_uses_each_physical_detector_frame(
    monkeypatch, cone_proposal_case, frame
):
    geometry, grid, detector, _, _, _, targets = cone_proposal_case
    # Rotated detectors separate detector u/v from lab x/z; segments also
    # differ in magnification.
    segments = geometry.segments if isinstance(geometry, ConeSegments) else (geometry,)
    segments = tuple(
        replace(
            segment,
            beam=replace(
                segment.beam,
                detector_roll_deg=25 - 10 * i,
                detector_pitch_deg=7 - 4 * i,
                detector_yaw_deg=-9 + 5 * i,
            ),
        )
        for i, segment in enumerate(segments)
    )
    geometry = ConeSegments(segments) if len(segments) > 1 else segments[0]
    poses = np.asarray(stack_view_poses(geometry, len(geometry.angles)), np.float64)
    model = cone_model(geometry, detector)
    assert model is not None
    frames = model.frames(len(poses))
    true = np.zeros((len(poses), 6), np.float32)
    true[:, 3] = np.linspace(-0.001, 0.001, len(poses))
    true[:, 4] = np.linspace(0.0007, -0.0003, len(poses))
    translations = true[:, [3, 5, 4]].astype(np.float64)
    if frame == "object":
        translations = np.einsum("nij,nj->ni", poses[:, :3, :3], translations)
    before = poses[:, :3, 3]
    shifts = _cone_detector_coordinates(frames, before + translations)
    shifts -= _cone_detector_coordinates(frames, before)
    monkeypatch.setattr(_prealign, "estimate_view_shifts", lambda *args, **kwargs: shifts)
    cfg = AlignConfig(
        optimise_dofs=("dx", "dz"), seed_translations=True, pose_translation_frame=frame
    )
    actual = _prealign.seeded_translation_params(geometry, grid, detector, targets, cfg)
    assert actual is not None
    # Linearised at the object origin: the finite-motion perspective error is
    # second order, so this is close but not exact.
    np.testing.assert_allclose(actual, true, atol=2e-7, rtol=2e-4)
    np.testing.assert_array_equal(actual[:, [0, 1, 2, 5]], true[:, [0, 1, 2, 5]])


@pytest.mark.numerical
def test_cone_shift_search_removes_a_rigid_shift_in_each_arrangement(
    monkeypatch, cone_proposal_case
):
    geometry, grid, detector, poses, frames, _, targets = cone_proposal_case
    nominal = np.asarray(poses, np.float64)
    # A common object translation is absorbed by the reconstructed image,
    # not reported as different physical shifts between magnified orbits.
    origin = nominal[:, :3, 3]
    shifted = origin + nominal[:, :3, :3] @ np.array([0.001, -0.0007, 0.0004])
    shifts = _cone_detector_coordinates(frames, shifted) - _cone_detector_coordinates(
        frames, origin
    )
    shifts /= [detector.du, detector.dv]
    monkeypatch.setattr(
        _prealign, "_quick_reconstruction", lambda *args: jnp.zeros((grid.nx, grid.ny, grid.nz))
    )
    monkeypatch.setattr(
        _prealign, "reprojector", lambda *args: lambda views, volume: targets[views]
    )
    monkeypatch.setattr(_prealign, "correlation_peaks", lambda *args, **kwargs: shifts)
    actual = estimate_view_shifts(geometry, grid, detector, targets, iters=1)
    np.testing.assert_allclose(actual, 0, atol=2e-7)


def test_shift_search_declines_unsettled_proposals(monkeypatch):
    geometry, grid, detector = _geometry(8, "parallel", np.array([0, 90, 180, 270]))
    data = jnp.zeros((4, detector.nv, detector.nu))
    shifts = np.array([[2, 0], [-2, 0], [2, 0], [-2, 0]], np.float32)
    steps = iter((shifts, -shifts))
    monkeypatch.setattr(_prealign, "_quick_reconstruction", lambda *a: None)
    monkeypatch.setattr(_prealign, "reprojector", lambda *a: lambda c, x: data[c])
    monkeypatch.setattr(_prealign, "correlation_peaks", lambda *a, **k: next(steps))
    actual = estimate_view_shifts(geometry, grid, detector, data, iters=2)
    np.testing.assert_array_equal(actual, np.zeros_like(shifts))


@pytest.mark.parametrize("status", ["accepted", "declined", "unavailable"])
def test_unsettled_search_uses_physical_phase_seed_in_acquisition_order(monkeypatch, status):
    from types import SimpleNamespace

    angles = np.array([0, 60, 120, 180, 120, 60, 0])
    geometry, grid, detector = _geometry(8, "lamino", angles)
    detector = replace(detector, du=0.7, dv=1.3, center=(2.1, -0.4))
    geometry = replace(geometry, detector=detector)
    data = jnp.zeros((len(angles), detector.nv, detector.nu))
    nominal = np.asarray(stack_view_poses(geometry, len(angles)))
    basis = nominal[:, [0, 2], :3].reshape(-1, 3)
    selected = np.array([6, 0, 3])
    physical = np.array([[0.7, 0.2], [-0.2, 0.6], [0.5, -0.8]])

    def motion(axis, *, center_u):
        np.testing.assert_allclose(axis, geometry.axis_unit_lab)
        assert center_u == detector.center[0]
        return SimpleNamespace(
            shifts=physical if status == "accepted" else None,
            status=status,
            coherence_before=0.2,
            coherence_after=0.8,
        )

    lines = SimpleNamespace(view_indices=selected, motion=motion)
    monkeypatch.setattr(
        _prealign, "prepare_common_lines", lambda *a: None if status == "unavailable" else lines
    )
    peaks = np.array([[2, 0], [-2, 0], [2, 0], [-2, 0], [2, 0], [-2, 0], [2, 0]])
    steps = iter((peaks, -peaks))
    monkeypatch.setattr(_prealign, "_quick_reconstruction", lambda *a: None)
    monkeypatch.setattr(_prealign, "reprojector", lambda *a: lambda c, x: data[c])
    monkeypatch.setattr(_prealign, "correlation_peaks", lambda *a, **k: next(steps))
    actual = estimate_view_shifts(geometry, grid, detector, data, iters=2)
    expected = np.column_stack(
        [np.interp(np.arange(7), [0, 3, 6], physical[[1, 2, 0], i]) for i in range(2)]
    ).ravel()
    expected -= basis @ np.linalg.lstsq(basis, expected, rcond=1e-8)[0]
    np.testing.assert_allclose(
        actual, expected.reshape(-1, 2) if status == "accepted" else 0, atol=1e-7
    )
