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
from tomojax.alignment._objectives import fold_recon

# check-public-imports: allow-private
from tomojax.alignment._objectives.recon_layer import PoseAdjustedGeometry

# check-public-imports: allow-private
from tomojax.alignment._prealign import estimate_view_shifts, translation_params_from_shifts
from tomojax.alignment.api import (
    AlignmentState,
    AlignMultiresResumeState,
    AlignResumeState,
    BaseGeometryArrays,
    L2LossSpec,
    PoseState,
    align,
    apply_alignment_state,
    apply_pose_update,
    apply_pose_updates,
    least_motion_estimate,
    save_alignment_params_csv,
    save_alignment_params_json,
    se3_from_pose_params,
)

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.forward import project_joseph
from tomojax.geometry import (
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    stack_view_poses,
)
from tomojax.io import build_geometry_from_dataset_metadata


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
    cfg = AlignConfig(pose_translation_frame="detector", projector_backend="jax")
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
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("motion", ["translation", "rigid"])
def test_public_alignment_and_checkpoint_keep_detector_frame_at_ninety_degrees(
    kind, multires, backend, motion
):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires the CUDA Pallas reconstruction path")
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
        projector_backend=backend,
        optimise_dofs=("dx", "dz") if motion == "translation" else None,
        outer_iters=8,
        recon_iters=1,
        # This is a fixed-volume API/derivative regression, not joint-recovery evidence.
        recon_L=1e12,
        lambda_tv=0,
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
def test_train_fold_preserves_pose_frame_and_excludes_padding(monkeypatch, frame):
    geometry, grid, detector = _geometry(8, "lamino", np.array([17.0, 90.0, 143.0], np.float32))
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    state = AlignmentState.zeros(n_views=3)
    params = jnp.array([[0.02, 0, 0, 0.3, -0.7], [0, 0.01, 0, 0.2, -0.4], [0, 0, -0.03, -0.2, 0.6]])
    state = state.replace(
        setup=state.setup.replace(nominal_axis_unit=base.nominal_axis_unit),
        pose=PoseState(params, translation_frame=frame),
    )
    expected = apply_alignment_state(base, state).pose_stack[jnp.array([2, 0])]
    observed = []

    def fake_reconstruct(actual_geometry, actual_grid, actual_detector, data, **kwargs):
        observed.append(stack_view_poses(actual_geometry, data.shape[0]))
        return jnp.zeros((8, 8, 8)), {}

    monkeypatch.setattr(fold_recon, "fista_tv", fake_reconstruct)
    config = fold_recon.FoldReconstructionConfig(
        iters=1,
        lambda_tv=0,
        regulariser="huber_tv",
        huber_delta=0.01,
        tv_prox_iters=1,
        positivity=False,
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
    actual = translation_params_from_shifts(shifts, poses, params, frame=frame, active=active)
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
    data = _prealign._shift_views(clean, jnp.asarray(truth, jnp.float32))
    found = estimate_view_shifts(geometry, grid, detector, data) / [detector.du, detector.dv]
    rotations = np.asarray(poses, np.float64)[:, :3, :3]
    spacing = (detector.du, detector.dv)
    expected = _prealign._remove_object_translation(truth, rotations, spacing)
    actual = _prealign._remove_object_translation(found, rotations, spacing)
    # +/-5 px shifts come within the local solver's sub-pixel reach.
    assert np.sqrt(np.mean((actual - expected) ** 2)) < 0.4


@pytest.mark.parametrize("factors", [None, (2, 1)])
def test_translation_seed_runs_once_at_the_first_level(monkeypatch, factors):
    # check-public-imports: allow-private
    from tomojax.alignment._pose import _pose_loop

    # check-public-imports: allow-private
    from tomojax.alignment._stages import _stage_multires

    geometry, grid, detector = _geometry(8, "parallel", np.linspace(0, 180, 6, endpoint=False))
    data = jnp.zeros((6, detector.nv, detector.nu), jnp.float32)
    calls = []

    def record(geometry, grid, detector, projections, cfg):
        calls.append((projections.shape, cfg.seed_translations))

    monkeypatch.setattr(_pose_loop, "seeded_translation_params", record)
    monkeypatch.setattr(_stage_multires, "seeded_translation_params", record)
    cfg = AlignConfig(outer_iters=1, recon_iters=1, seed_translations=True)
    if factors is None:
        align(geometry, grid, detector, data, config=cfg)
        assert calls == [((6, 8, 8), True)]
    else:
        align_multires(geometry, grid, detector, data, factors=factors, config=cfg)
        assert calls == [((6, 4, 4), True)]


@pytest.mark.parametrize("kind", ["parallel", "lamino"])
def test_reprojection_seed_recovers_a_detector_centre_offset(kind):
    # check-public-imports: allow-private
    from tomojax.alignment._geometry.initializers import reprojection_det_u_seed

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
    data = _prealign._shift_views(clean, shift)
    seed = reprojection_det_u_seed(data, geometry, grid, detector)
    assert seed.status == "ok_reprojection"
    assert abs(seed.det_u_px + 2.3) < 0.1


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
        det_center=(detector.det_center[0] + gauge.detector_offset, detector.det_center[1]),
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
        "thetas_deg": angles.astype(np.float32),
        "geometry_type": "lamino",
        "tilt_deg": 30.0,
        "align_params": params,
        "align_gauge": {"pose_translation_frame": frame},
    }
    _, _, saved = build_geometry_from_dataset_metadata(meta, apply_saved_alignment=True)
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
        freeze_dofs=("alpha", "beta", "phi"),  # isolate the translation split
        pose_translation_frame="detector",
        projector_backend="jax",
        outer_iters=8,
        recon_iters=1,
        recon_L=1e12,  # keeps the known volume fixed
        lambda_tv=0,
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
