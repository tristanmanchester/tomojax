"""Independent dense Schur references for the coupled free-voxel update."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._pose import (
    _coupled_host,
    _coupled_objective,
    _coupled_program,
    _coupled_vectors,
    _matching_model,
    _pose_objective,
    _pose_steps,
)

# check-public-imports: allow-private
from tomojax.alignment._pose._coupled_linear import solve_coupled_normal

# check-public-imports: allow-private
from tomojax.alignment._pose._pose_context import _pose_objective_context

# check-public-imports: allow-private
from tomojax.alignment._pose._pose_loop import _build_alignment_runtime_context


@pytest.fixture(params=[None, 2], ids=["one_batch", "batches_of_two"])
def views_per_batch(request, monkeypatch):
    """The program's batch size, or two views a batch: three views then end in a shifted batch."""
    if request.param is None:
        yield
        return
    monkeypatch.setattr(_coupled_program, "_VIEWS_PER_BATCH", request.param)
    # The batch size is read while tracing: drop programs traced with another.
    _coupled_program.run_update.clear_cache()
    _coupled_program.run_loss.clear_cache()
    yield
    _coupled_program.run_update.clear_cache()
    _coupled_program.run_loss.clear_cache()


def test_joint_coupling_selects_the_solver_for_every_pose_stage():
    from tomojax.alignment import AlignConfig

    # check-public-imports: allow-private
    from tomojax.alignment._config import _resolved_schedule_for_cfg

    implicit = _resolved_schedule_for_cfg(AlignConfig(gn_coupling="joint"))
    named = _resolved_schedule_for_cfg(AlignConfig(gn_coupling="joint", schedule="lightning_pose"))
    alternating = _resolved_schedule_for_cfg(AlignConfig(schedule="pose_only"))
    assert implicit.stages[0].objective_kind == "joint_volume_pose"
    assert {stage.objective_kind for stage in named.stages} == {"joint_volume_pose"}
    assert alternating.stages[0].objective_kind == "fixed_volume"
    # Setup stages keep their own objectives.
    cor = _resolved_schedule_for_cfg(AlignConfig(gn_coupling="joint", schedule="setup_safe"))
    assert cor.stages[0].objective_kind != "joint_volume_pose"


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["stacked", "pose_eliminated"])
def test_frozen_columns_use_no_cache_space_and_preserve_the_update(monkeypatch, solver):
    from dataclasses import replace

    import tomojax as tj
    from tomojax.alignment import AlignConfig

    grid, detector = tj.Grid(4, 3, 5, 0.7, 1.1, 0.8), tj.Detector(5, 6, 0.8, 0.9)
    geometry = tj.LaminographyGeometry(grid, detector, [13, 47, 103], tilt_deg=23)
    x = jnp.asarray(np.random.default_rng(9).uniform(0.2, 1.0, (4, 3, 5)), jnp.float32)
    cfg = AlignConfig(
        gn_coupling="joint",
        gn_joint_solver=solver,
        ray_integrator="joseph",
        gather_dtype="fp32",
    )
    common = dict(
        detector=detector,
        projections=tj.project(geometry, x),
        cfg=cfg,
        n_views=3,
        active_mask=jnp.array([0, 0, 0, 1, 1, 0], jnp.float32),
    )
    runtime = _build_alignment_runtime_context(geometry=geometry, det_grid_override=None, **common)
    ctx = _pose_objective_context(runtime=runtime, grid=grid, **common)
    captured = []
    monkeypatch.setattr(_coupled_objective, "_pose_cache_limit", lambda: 3 * 5 * 6 * (1 + 2) * 4)
    monkeypatch.setattr(
        _coupled_objective,
        "_check_device_memory",
        lambda arrays, spec: captured.append((arrays, spec)),
    )
    objective = _coupled_objective.build_coupled_objective(ctx)
    assert objective.pose_columns_cached
    arrays, spec = captured[0]
    p = jnp.zeros((3, 6), jnp.float32).at[:, 3].set(jnp.array([0.03, -0.01, 0.02]))
    compact = objective.update(p, 0.9 * x)
    full = _coupled_objective.run_update(
        arrays, p, 0.9 * x, spec=replace(spec, active_columns=tuple(range(6)))
    )
    assert compact.finite and full.finite
    for actual, expected in zip(compact.increment, full.increment, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=3e-6)
    np.testing.assert_array_equal(compact.increment[1][:, [0, 1, 2, 5]], 0)


@pytest.mark.numerical
@pytest.mark.usefixtures("views_per_batch")
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_fp32_cone_joint_update_is_unchanged_with_x64_enabled(backend):
    import tomojax as tj
    from tomojax.alignment import AlignConfig

    grid = tj.Grid(3, 4, 3, 0.7, 0.9, 1.1)
    detector = tj.Detector(5, 4, 0.6, 0.8)
    geometry = tj.ConeGeometry(grid, detector, [0, 60, 120], tj.ConeBeam(20, 35), tilt_deg=13)
    truth = jnp.asarray(np.random.default_rng(3).uniform(0.2, 1, (3, 4, 3)), jnp.float32)
    projections = jnp.asarray(tj.project(geometry, truth), jnp.float32)
    p = jnp.zeros((3, 6), jnp.float32)
    cfg = AlignConfig(
        gn_coupling="joint",
        gn_joint_solver="pose_eliminated",
        gn_joint_iterations=3,
        gn_damping=0.2,
        gn_volume_damping=0.2,
        gather_dtype="fp32",
        backend=backend,
        w_rot=0.03,
        w_trans=0.02,
    )
    common = dict(
        detector=detector,
        projections=projections,
        cfg=cfg,
        n_views=3,
        active_mask=jnp.ones(6, jnp.float32),
    )

    def step():
        runtime = _build_alignment_runtime_context(
            geometry=geometry, det_grid_override=None, **common
        )
        ctx = _pose_objective_context(runtime=runtime, grid=grid, **common)
        return _coupled_objective.build_coupled_objective(ctx).update(p, 0.9 * truth)

    with jax.enable_x64(False):
        expected = step()
    # A default-typed count mask used to promote the pose block to FP64,
    # making the batched update fail despite explicitly FP32 inputs.
    with jax.enable_x64():
        actual = step()
    assert actual.finite and expected.finite
    for a, e in zip(actual.increment, expected.increment, strict=True):
        assert a.dtype == e.dtype == jnp.float32
        np.testing.assert_allclose(a, e, rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(actual.relative_residual, expected.relative_residual, rtol=3e-4)


def _force_uncached_pose_columns(monkeypatch):
    """Stream pose columns on both the device and host paths."""
    monkeypatch.setattr(_coupled_objective, "_POSE_CACHE_BYTES", 0)
    monkeypatch.setattr(_coupled_host, "host_column_capacity", lambda spec, views: 0)


@pytest.mark.numerical
@pytest.mark.usefixtures("views_per_batch")
@pytest.mark.parametrize("scan_variant", [0, 1])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("solver", ["stacked", "pose_eliminated"])
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("integrator", ["exact", "joseph"])
def test_physical_coupled_step_matches_independent_dense_model(  # noqa: PLR0915
    monkeypatch, stream, backend, solver, scan_variant, integrator
):
    from dataclasses import replace
    from pathlib import Path

    from tomojax.alignment import AlignConfig
    from tomojax.alignment.api import L2OtsuLossSpec
    from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses

    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("CUDA required")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    from public_alignment_benchmark import physical_poses
    from voxel_truth import project_voxel_truth as project_exact

    from tomojax.forward import project_joseph

    def project_voxel_truth(volume, poses, grid, det):
        """Independent reference for the selected integrator."""
        if integrator == "exact":
            return project_exact(volume, poses, grid, det)
        projected = project_joseph(
            jnp.asarray(volume, jnp.float32), jnp.asarray(poses, jnp.float32), grid, det
        )
        return np.asarray(projected, np.float64)

    if stream:
        _force_uncached_pose_columns(monkeypatch)
    grid = Grid(2, 3, 2, 0.7, 1.1, 1.4, vol_origin=(-0.32, -0.71, -0.53))
    det = Detector(4, 3, 0.65, 0.91, (0.23, -0.17))
    # Identical shapes/configuration must accept new scan arrays without
    # reusing an earlier scan's embedded measurements, poses or weights.
    angles = np.array([7.2, 49.3, 113.7]) + 0.37 * scan_variant
    geometry = LaminographyGeometry(grid, det, angles, tilt_deg=23)
    nominal = np.asarray(stack_view_poses(geometry, 3), np.float64)
    rng = np.random.default_rng(34 + scan_variant)
    x = rng.uniform(0.2, 1.3, (2, 3, 2)).astype(np.float32)
    support = np.ones_like(x)
    support[0, 0, 0] = 1 - scan_variant
    x *= support
    # The sixth column, dy along the beam, is inactive for parallel rays.
    p = np.concatenate([rng.normal(0, 0.01, (3, 5)), np.zeros((3, 1))], axis=1).astype(np.float32)
    pose = physical_poses(nominal, p[:, :5])
    predicted = project_voxel_truth(x, pose, grid, det)
    target = predicted + rng.normal(0, 0.03, predicted.shape)
    target = jnp.asarray(target, jnp.float32)
    active = np.array([1 - scan_variant, scan_variant, 1, 1, 0, 0], np.float32)
    cfg = AlignConfig(
        gn_coupling="joint",
        gn_joint_solver="pose_eliminated",
        gn_joint_iterations=100,
        gn_joint_rtol=1e-6,
        gn_damping=0.2,
        # Accepted scalar-like options must remain usable in a hashable
        # program specification, including zero-dimensional NumPy arrays.
        gn_volume_damping=0.2 if scan_variant == 0 else np.asarray(0.2),
        gn_jacobian="central",
        gn_difference_step=0.003 if scan_variant == 0 else np.asarray(0.003),
        ray_integrator=integrator,
        backend=backend,
        gather_dtype="fp32",
        pose_translation_frame="detector",
        tv_weight=0,
        loss=L2OtsuLossSpec(),
        w_rot=0.03 + 0.005 * scan_variant,
        w_trans=0.05 + 0.003 * scan_variant,
    )
    common = dict(
        detector=det,
        projections=target,
        cfg=cfg,
        n_views=3,
        active_mask=jnp.asarray(active),
    )
    runtime = _build_alignment_runtime_context(geometry=geometry, det_grid_override=None, **common)
    ctx = _pose_objective_context(runtime=runtime, grid=grid, **common)
    ctx = replace(ctx, volume_mask=jnp.asarray(support))
    objective = _coupled_objective.build_coupled_objective(ctx)
    result = objective.update(jnp.asarray(p), jnp.asarray(x))
    assert result.finite
    assert objective.pose_columns_cached != stream
    weight = (1 / (1 + np.exp(-(np.asarray(ctx.loss_mask) - 0.5) / 0.5))).ravel()
    a = (
        np.column_stack(
            [
                project_voxel_truth(v.reshape(x.shape) * support, pose, grid, det).ravel()
                for v in np.eye(x.size)
            ]
        )
        * weight[:, None]
    )
    displacement = cfg.gn_difference_step * min(grid.vx, grid.vy, grid.vz)
    radius = 0.5 * np.linalg.norm(np.array(x.shape) * [grid.vx, grid.vy, grid.vz])
    steps = np.array([displacement / radius] * 3 + [displacement] * 3)
    j = np.zeros((target.size, p.size))
    for i in range(p.size):
        view, dof = divmod(i, 6)
        plus, minus = p.astype(float), p.astype(float)
        plus[view, dof] += steps[dof]
        minus[view, dof] -= steps[dof]
        j[:, i] = (
            weight
            * active[dof]
            * (
                project_voxel_truth(x, physical_poses(nominal, plus[:, :5]), grid, det).ravel()
                - project_voxel_truth(x, physical_poses(nominal, minus[:, :5]), grid, det).ravel()
            )
            / (2 * steps[dof])
        )
    smooth = np.kron(
        np.array([[1, -2, 1]]), np.diag(np.array([cfg.w_rot] * 3 + [cfg.w_trans] * 3) * active)
    )
    h_smooth = 2 * smooth.T @ smooth
    design = np.column_stack([a, j])
    residual = weight * (predicted - np.asarray(target)).ravel()
    hessian = design.T @ design + 0.2 * np.eye(x.size + p.size)
    hessian[x.size :, x.size :] += h_smooth
    gradient = design.T @ residual
    gradient[x.size :] += h_smooth @ p.ravel()
    expected = np.linalg.solve(hessian, -gradient)
    actual = np.concatenate([np.asarray(v).ravel() for v in result.increment])
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-5)
    np.testing.assert_array_equal(np.asarray(result.increment[1])[:, active == 0], 0)
    expected_loss = 0.5 * residual @ residual + np.sum((smooth @ p.ravel()) ** 2)
    np.testing.assert_allclose(
        objective.loss(jnp.asarray(p), jnp.asarray(x)), expected_loss, rtol=1e-4
    )


@pytest.mark.numerical
def test_matrix_free_coupled_solve_matches_dense_schur_complement():
    rng = np.random.default_rng(761)
    a = rng.normal(size=(23, 7))
    j = a[:, :3] + 0.1 * rng.normal(size=(23, 3))
    residual = rng.normal(size=23)
    mu, damping = 0.2, 0.07
    aa = a.T @ a + mu * np.eye(7)
    cross = a.T @ j
    jj = j.T @ j + damping * np.eye(3)
    gv, gp = a.T @ residual, j.T @ residual
    schur = jj - cross.T @ np.linalg.solve(aa, cross)
    dp = np.linalg.solve(schur, -gp + cross.T @ np.linalg.solve(aa, gv))
    dv = np.linalg.solve(aa, -gv - cross @ dp)
    matrix = jnp.asarray(np.block([[aa, cross], [cross.T, jj]]), jnp.float32)

    def normal(blocks):
        product = jnp.matmul(matrix, jnp.concatenate(blocks), precision=jax.lax.Precision.HIGHEST)
        return product[:7], product[7:]

    rhs = (jnp.asarray(-gv, jnp.float32), jnp.asarray(-gp, jnp.float32))
    inverse = (1 / jnp.diag(matrix)[:7], 1 / jnp.diag(matrix)[7:])
    result = jax.jit(
        lambda rhs: solve_coupled_normal(
            normal,
            rhs,
            inverse,
            max_iters=80,
            rtol=2e-6,
        )
    )(rhs)
    assert result.finite
    np.testing.assert_allclose(result.increment[0], dv, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(result.increment[1], dp, rtol=2e-4, atol=2e-5)
    true_residual = np.concatenate(rhs) - np.asarray(matrix) @ np.concatenate(result.increment)
    np.testing.assert_allclose(
        result.relative_residual,
        np.linalg.norm(true_residual) / np.linalg.norm(np.concatenate(rhs)),
        rtol=0.25,
        atol=1e-7,
    )


@pytest.mark.numerical
def test_coupled_proposal_must_be_scored_with_its_volume_update():
    # Nearly compensating object and pose directions. The joint step improves
    # the data fit, but keeping only its pose component would look much worse.
    a = np.array([[1.0], [0.0]])
    j = np.array([[1.0], [0.1]])
    matrix = np.column_stack([a, j])
    residual = np.array([0.0, -1.0])
    delta = np.linalg.solve(matrix.T @ matrix + 1e-3 * np.eye(2), -matrix.T @ residual)
    assert np.linalg.norm(residual + matrix @ delta) < 0.2 * np.linalg.norm(residual)
    assert np.linalg.norm(residual + j[:, 0] * delta[1]) > 5 * np.linalg.norm(residual)
    assert 10 + delta[0] > 0  # joint trial is feasible for a nonnegative volume


@pytest.mark.parametrize("nonnegative", [False, True])
def test_joint_trial_preserves_the_image_and_direction_for_backtracking(nonnegative):
    x, dx = jnp.arange(12).reshape(2, 2, 3).astype(jnp.float32), jnp.full((2, 2, 3), -3.0)
    for scale in (1.0, 0.5):
        expected = np.asarray(x) - 3 * scale
        actual = _pose_steps._joint_trial_volume(x, dx, scale, nonnegative=nonnegative)
        np.testing.assert_array_equal(actual, np.maximum(expected, 0) if nonnegative else expected)
    np.testing.assert_array_equal(x, np.arange(12).reshape(2, 2, 3))
    np.testing.assert_array_equal(dx, -3.0)


@pytest.mark.numerical
@pytest.mark.parametrize("placement", [_coupled_vectors.HOST, _coupled_vectors.DEVICE])
def test_coupled_solve_reports_actual_residual_at_budget_and_handles_zero_rhs(placement):
    xp = placement.array
    matrix = xp.diag(xp.array([1.0, 3.0, 8.0, 15.0], dtype=xp.float32))

    def normal(blocks):
        result = matrix @ xp.concatenate(blocks)
        return result[:2], result[2:]

    rhs = (xp.ones(2, xp.float32), xp.ones(2, xp.float32))
    result = solve_coupled_normal(normal, rhs, rhs, max_iters=1, rtol=1e-6, placement=placement)
    assert int(result.iterations) == 1 and result.relative_residual > 1e-2
    zero = (xp.zeros(2, xp.float32), xp.zeros(2, xp.float32))
    result = solve_coupled_normal(normal, zero, rhs, max_iters=10, rtol=1e-6, placement=placement)
    assert result.finite and result.iterations == 0 and result.relative_residual == 0
    failed = solve_coupled_normal(
        lambda block: tuple(-a for a in normal(block)),
        rhs,
        rhs,
        max_iters=2,
        rtol=1e-6,
        placement=placement,
    )
    assert not failed.finite and failed.iterations == 1
    np.testing.assert_allclose(failed.relative_residual, 1)


@pytest.mark.numerical
@pytest.mark.usefixtures("views_per_batch")
@pytest.mark.parametrize("integrator", ["sampled", "exact"])
@pytest.mark.parametrize("solver", ["stacked", "pose_eliminated"])
def test_public_joint_huber_constraints_and_resume(integrator, solver):
    # Resuming must reproduce the uninterrupted run, which needs reproducible
    # arithmetic: on a GPU the sampled transpose adds with atomics in varying
    # order, and this tiny ill-conditioned solve amplifies that past any
    # tolerance by the second iteration. The bookkeeping is the same on any device.
    with jax.default_device(jax.devices("cpu")[0]):
        _joint_huber_constraints_and_resume(integrator, solver)


def _joint_huber_constraints_and_resume(integrator, solver):
    from tomojax.alignment import AlignConfig
    from tomojax.alignment.api import L2LossSpec, align
    from tomojax.core.projector import forward_project_view_T
    from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses

    grid, det = Grid(4, 3, 4, 0.8, 1.1, 1.4), Detector(5, 4, 0.8, 1.4, (0.17, -0.2))
    geometry = LaminographyGeometry(grid, det, [7, 51, 107], tilt_deg=27)
    truth = jnp.asarray(np.random.default_rng(49).uniform(0.2, 1, (4, 3, 4)), jnp.float32)
    target = jax.lax.map(
        lambda t: forward_project_view_T(t, grid, det, truth, ray_integrator=integrator),
        stack_view_poses(geometry, 3),
    )
    cfg = AlignConfig(
        gn_coupling="joint",
        gn_joint_solver="pose_eliminated",
        gn_jacobian="central",
        gather_dtype="fp32",
        ray_integrator=integrator,
        backend="jax",
        tv_weight=0.002,
        outer_iterations=2,
        iterations=2,
        lipschitz=100,
        loss=L2LossSpec(),
        early_stop=False,
        freeze=("beta",),
        bounds=(("dx", -0.01, 0.01),),
    )
    init = jnp.zeros_like(truth).at[1:3, 1, 1:3].set(0.5)
    checkpoints = []
    x, p, info = align(geometry, grid, det, target, config=cfg, init_x=init)
    assert np.isfinite(x).all() and np.all(x >= 0)
    assert all(s["joint_linear_finite"] for s in info["outer_stats"])
    assert all(s["loss_after"] <= s["loss_before"] for s in info["outer_stats"])
    assert any(s["joint_step_scale"] > 0 for s in info["outer_stats"])
    np.testing.assert_array_equal(p[:, 1], 0)
    assert np.max(np.abs(p[:, 3])) <= 0.010001
    align(
        geometry,
        grid,
        det,
        target,
        config=cfg,
        init_x=init,
        observer=lambda *_: "stop_run",
        checkpoint_callback=checkpoints.append,
    )
    resumed_x, resumed_p, resumed_info = align(
        geometry,
        grid,
        det,
        target,
        config=cfg,
        resume_state=checkpoints[-1],
    )
    np.testing.assert_allclose(resumed_x, x, atol=1e-6)
    np.testing.assert_allclose(resumed_p, p, atol=1e-6)
    np.testing.assert_allclose(resumed_info["loss"], info["loss"], atol=1e-6)


def _small_coupled_context():
    from dataclasses import replace

    from tomojax.alignment import AlignConfig

    # check-public-imports: allow-private
    from tomojax.alignment._pose._pose_context import _pose_objective_context

    # check-public-imports: allow-private
    from tomojax.alignment._pose._pose_loop import _build_alignment_runtime_context
    from tomojax.geometry import Detector, Grid, ParallelGeometry

    grid, det = Grid(4, 4, 4, 1.0, 1.0, 1.0), Detector(5, 4, 1.0, 1.0)
    geometry = ParallelGeometry(grid, det, [0.0, 60.0, 120.0])
    projections = jnp.asarray(np.random.default_rng(3).uniform(0, 1, (3, 4, 5)), jnp.float32)
    cfg = AlignConfig(gn_coupling="joint", gather_dtype="fp32", backend="jax")
    # Compare placements after an accurate solve, rather than at the loose default CG stop.
    cfg = replace(cfg, gn_damping=0.2, gn_volume_damping=0.2, gn_joint_rtol=1e-6)
    common = dict(
        detector=det,
        projections=projections,
        cfg=cfg,
        n_views=3,
        active_mask=jnp.ones(6, jnp.float32),
    )
    runtime = _build_alignment_runtime_context(geometry=geometry, det_grid_override=None, **common)
    return _pose_objective_context(runtime=runtime, grid=grid, **common)


def test_memory_check_compiles_the_update_unless_far_below_the_limit(monkeypatch):
    # check-public-imports: allow-private
    from tomojax.alignment._pose import _coupled_objective as coupled

    stats = dict(bytes_limit=1000, bytes_in_use=100, largest_free_block_bytes=0)
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: stats)
    assert coupled._free_device_memory(SimpleNamespace(devices=lambda: (device,))) == 900
    with monkeypatch.context() as direct:
        direct.setattr(device, "memory_stats", lambda: None)
        direct.setattr("tomojax.backends.device_free_memory_bytes", lambda **_: 800)
        assert coupled._free_device_memory(SimpleNamespace(devices=lambda: (device,))) == 800
    ctx = _small_coupled_context()
    assert coupled._pose_cache_limit() > 0  # the columns are cached
    volume, sinogram = 4 * 4**3, 4 * 3 * 4 * 5
    # A dozen-odd volumes, a few sinograms and the six cached columns.
    estimate = 13 * volume + 9 * sinogram
    p, x = jnp.zeros((3, 6), jnp.float32), jnp.ones((4, 4, 4), jnp.float32)
    expected = coupled.build_coupled_objective(ctx).update(p, x)
    lowered = []
    lower = coupled.run_update.lower

    def counting_lower(*args, **kwargs):
        lowered.append((args[0], kwargs["spec"]))
        return lower(*args, **kwargs)

    monkeypatch.setattr(coupled.run_update, "lower", counting_lower)

    def free(nbytes):
        lowered.clear()
        monkeypatch.setattr(coupled, "_free_device_memory", lambda _array: nbytes)

    def assert_update_matches(objective):
        result = objective.update(p, x)
        for a, b in zip(result.increment, expected.increment, strict=True):
            # Within rounding of the whole increment: GPU atomics sum in no
            # fixed order and the joint solve amplifies it, so near-zero
            # elements carry the rounding of the largest.
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-4 * float(np.abs(b).max()))

    # Far below the limit, the estimate is trusted.
    free(4 * estimate)
    assert_update_matches(coupled.build_coupled_objective(ctx))
    assert not lowered
    # Near it, XLA's own figure decides. Here the update needs more than the
    # estimate, which a shortcut without the columns or a margin would miss.
    free(int(1.5 * estimate))
    assert 13 * volume + 3 * sinogram < 1.5 * estimate
    # The memory guard still refuses the oversized compiled update; the
    # objective can now stage the same equations on the host instead.
    assert_update_matches(coupled.build_coupled_objective(ctx))
    assert lowered
    arrays, spec = lowered[-1]
    with pytest.raises(coupled.AlignmentMemoryError, match="GiB is free") as error:
        coupled._check_device_memory(arrays, spec)
    assert error.value.available == int(1.5 * estimate) and error.value.needed > 1.5 * estimate
    # When it fits, the update compiled for the check is the one used.
    monkeypatch.setattr(coupled, "_MEMORY_MARGIN", 1000)
    free(error.value.needed)
    objective = coupled.build_coupled_objective(ctx)
    assert_update_matches(objective)
    assert len(lowered) == 1
    # Reconstruction can consume memory after admission, before this update.
    monkeypatch.setattr(coupled, "_free_device_memory", lambda _array: 0)
    with pytest.raises(coupled.AlignmentMemoryError) as late_error:
        objective.update(p, x)
    assert late_error.value.available == 0 < late_error.value.needed
    assert len(lowered) == 1  # no recompilation, nor dispatch of the oversized update


@pytest.mark.parametrize(
    ("available", "views", "expected"),
    [
        (None, 3, 0),
        (255, 3, 0),
        (831, 3, 0),
        (832, 3, 3),
        (6399, 67, 0),
        (6400, 67, 32),
        (13119, 67, 64),
        (13120, 67, 67),
    ],
)
@pytest.mark.parametrize("refinement", [None, object()])
def test_host_columns_reserve_krylov_workspace(monkeypatch, available, views, expected, refinement):
    spec = SimpleNamespace(
        grid=SimpleNamespace(nx=2, ny=2, nz=2),
        detector=SimpleNamespace(nv=3, nu=4),
        active_columns=(0, 3),
        refinement=refinement,
    )
    if available is not None and refinement is not None:
        available += 4 * 32
    monkeypatch.setattr(_coupled_host, "device_free_memory_bytes", lambda **_: available)
    assert _coupled_host.host_column_capacity(spec, views) == expected


def test_a_cached_update_that_does_not_fit_retries_without_columns(monkeypatch):
    checks = []

    def check(_arrays, spec):
        checks.append(spec.cache_columns)
        if spec.cache_columns:
            raise _coupled_objective.AlignmentMemoryError(2**30, 2**29)

    monkeypatch.setattr(_coupled_objective, "_check_device_memory", check)
    objective = _coupled_objective.build_coupled_objective(_small_coupled_context())
    assert checks == [True, False] and not objective.pose_columns_cached


@pytest.fixture(params=["device", "host", "shared"])
def joint_placement(request, monkeypatch):
    """Run one constrained model through each supported placement."""
    if request.param == "shared":
        if len(jax.devices()) < 2:
            pytest.skip("multiple devices required")
        monkeypatch.setattr(_coupled_objective, "shared_devices", jax.devices)
    if request.param == "host":
        monkeypatch.setattr(
            _coupled_objective,
            "_check_device_memory",
            lambda *_: (_ for _ in ()).throw(_coupled_objective.AlignmentMemoryError(2, 1)),
        )


def _affine_residual_reference(ramp, rng, weighted):
    weights = rng.uniform(0.5, 1.5, (3, 3, 7)) if weighted else np.ones((3, 3, 7))
    if weighted:
        weights[:, :, 2:4] = 0
        weights[0, 0] = 0
    basis = np.column_stack([np.repeat(np.eye(3), 7, axis=0), ramp.ravel()])
    whitened = weights.reshape(3, -1, 1) * basis
    residualizers = np.eye(21) - whitened @ np.linalg.pinv(whitened)

    def filtered(y):
        return np.einsum("nij,nj->ni", residualizers, (weights * y).reshape(3, -1)).reshape(y.shape)

    return weights, filtered


@pytest.mark.parametrize("solver", ["stacked", "pose_eliminated"])
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.usefixtures("joint_placement")
def test_affine_joint_increment_and_loss_match_dense_physical_model(
    monkeypatch, solver, backend, stream, weighted
):
    from dataclasses import replace

    from scipy import signal

    import tomojax as tj
    from tomojax.alignment import AlignConfig
    from tomojax.alignment.api import L2LossSpec
    from tomojax.geometry import stack_view_poses

    if stream:
        _force_uncached_pose_columns(monkeypatch)
    rng = np.random.default_rng(2)
    grid, detector = tj.Grid(3, 4, 2, 1, 1, 1), tj.Detector(7, 3, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, [12, 57, 103])
    volume = rng.uniform(0.2, 1, (3, 4, 2)).astype(np.float32)
    support = np.ones_like(volume)
    support[0, 0, 0] = not stream
    poses = np.zeros((3, 6), np.float32)
    poses[:, 3] = [0.02, -0.01, 0.03]

    def project(x, p):
        shifted = tj.Scan(np.zeros((3, 3, 7), np.float32), geometry).with_poses(p)
        frames = stack_view_poses(shifted.geometry, 3)
        model = _matching_model.projectors(grid, detector, backend="jax", masked=lambda v: v)
        return np.asarray(model.forward(frames, jnp.asarray(x) * support), np.float64)

    native = np.arange(2 * detector.nu, dtype=np.float64) - (2 * detector.nu - 1) / 2
    spectrum = np.fft.fft(native) * np.exp(1j * np.pi * np.fft.fftfreq(native.size))
    ramp = np.broadcast_to(
        signal.resample(spectrum, detector.nu, domain="freq").real, (detector.nv, detector.nu)
    )

    weights, filtered = _affine_residual_reference(ramp, rng, weighted)

    measured = project(volume, poses) + rng.normal(0, 0.03, (3, 3, 7))
    cfg = AlignConfig(
        loss=L2LossSpec(),
        gn_coupling="joint",
        gn_joint_solver=solver,
        gn_joint_iterations=100,
        gn_joint_rtol=1e-6,
        gn_damping=0.2,
        gn_volume_damping=0.2,
        gn_jacobian="central",
        gn_difference_step=0.003,
        ray_integrator="joseph",
        backend=backend,
        gather_dtype="fp32",
        pose_translation_frame="detector",
        nonnegative=False,
        tv_weight=0,
        w_rot=0,
        w_trans=0,
    )
    common = dict(
        detector=detector,
        projections=jnp.asarray(measured, jnp.float32),
        cfg=cfg,
        n_views=3,
        active_mask=jnp.array([0, 0, 0, 1, 0, 0], jnp.float32),
    )
    runtime = _build_alignment_runtime_context(geometry=geometry, det_grid_override=None, **common)
    ctx = replace(
        _pose_objective_context(runtime=runtime, grid=grid, **common), volume_mask=support
    )
    ctx = replace(
        ctx,
        has_loss_mask=weighted,
        loss_adapter=replace(
            ctx.loss_adapter, gauss_newton_weights=lambda *_: jnp.asarray(weights, jnp.float32)
        ),
    )
    objective = _coupled_objective.build_coupled_objective(
        ctx,
        refinement=_matching_model.JointRefinement(native_width=2 * detector.nu),
    )
    assert objective.pose_columns_cached != stream
    result = objective.update(jnp.asarray(poses), jnp.asarray(volume))
    design = np.column_stack(
        [
            filtered(project(v.reshape(volume.shape), poses)).ravel()
            for v in np.eye(volume.size, dtype=np.float32)
        ]
    )
    columns = []
    for view in range(3):
        plus, minus = poses.copy(), poses.copy()
        plus[view, 3] += 0.003
        minus[view, 3] -= 0.003
        columns.append(filtered((project(volume, plus) - project(volume, minus)) / 0.006).ravel())
    design = np.column_stack([design, *columns])
    residual = filtered(project(volume, poses) - measured).ravel()
    normal = design.T @ design + 0.2 * np.eye(design.shape[1])
    gauge = np.zeros((design.shape[1], 2))
    gauge[-3:] = np.asarray(stack_view_poses(geometry, 3))[:, 0, :2]
    kkt = np.block([[normal, gauge], [gauge.T, np.zeros((2, 2))]])
    expected = np.linalg.solve(kkt, np.r_[-design.T @ residual, [0, 0]])[:-2]
    actual = np.r_[np.asarray(result.increment[0]).ravel(), np.asarray(result.increment[1])[:, 3]]
    np.testing.assert_allclose(actual, expected, rtol=0.003, atol=3e-5)
    np.testing.assert_allclose(
        objective.loss(jnp.asarray(poses), jnp.asarray(volume)),
        0.5 * residual @ residual,
        rtol=2e-5,
    )
    assert result.finite and result.relative_residual < 2e-5


@pytest.mark.parametrize("solver", ["stacked", "pose_eliminated"])
@pytest.mark.parametrize("smooth", [0.0, 0.03])
@pytest.mark.parametrize("integrator", ["joseph", "joseph_cubic"])
def test_staged_joint_update_matches_device_model(monkeypatch, solver, smooth, integrator):
    from dataclasses import replace
    from functools import partial

    import tomojax as tj
    from tomojax.alignment import AlignConfig
    from tomojax.alignment.api import L2OtsuLossSpec

    grid, det = tj.Grid(3, 2, 3, 0.8, 1.1, 0.7), tj.Detector(5, 4, 0.7, 0.9, (0.13, -0.21))
    geometry = tj.LaminographyGeometry(grid, det, [13, 47, 103], tilt_deg=23)
    x = (
        jnp.asarray(np.random.default_rng(9).uniform(0.2, 1, (3, 2, 3)), jnp.float32)
        .at[0, 0, 0]
        .set(0)
    )
    cfg = AlignConfig(
        gn_coupling="joint",
        gn_joint_solver=solver,
        ray_integrator=integrator,
        backend="cuda" if jax.default_backend() == "gpu" else "jax",
        gather_dtype="fp32",
        gn_joint_iterations=5 if integrator == "joseph_cubic" else 100,
        gn_joint_rtol=1e-6,
        gn_damping=0.2,
        gn_volume_damping=0.2,
        loss=L2OtsuLossSpec(),
        w_rot=smooth,
        w_trans=smooth,
    )
    common = dict(
        detector=det,
        projections=tj.project(geometry, x) * 0.9,
        cfg=cfg,
        n_views=3,
        active_mask=jnp.asarray([1, 0, 1, 1, 1, 0], jnp.float32),
    )
    runtime = _build_alignment_runtime_context(geometry=geometry, det_grid_override=None, **common)
    ctx = _pose_objective_context(runtime=runtime, grid=grid, **common)
    if smooth:
        ctx = replace(ctx, volume_mask=jnp.ones_like(x).at[0, 0, 0].set(0))
    else:
        # Exercise implicit unit-support row sums even on this tiny CUDA grid.
        monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    monkeypatch.setattr(_coupled_host, "_pass", partial(_coupled_host._pass, batch_size=2))
    captured = []
    monkeypatch.setattr(
        _coupled_objective, "_check_device_memory", lambda a, s: captured.append((a, s))
    )
    objective = _coupled_objective.build_coupled_objective(ctx)
    arrays, spec = captured[0]
    gradient = jax.grad(partial(_coupled_program.volume_regularisation, spec=spec))
    expected_gradient, hessian = jax.linearize(gradient, x)
    direction = jnp.linspace(-1, 1, x.size).reshape(x.shape)
    np.testing.assert_allclose(
        _coupled_host._regularisation_derivative(x, None, spec=spec), expected_gradient, atol=2e-8
    )
    np.testing.assert_allclose(
        _coupled_host._regularisation_derivative(x, direction, spec=spec),
        hessian(direction),
        rtol=2e-6,
        atol=2e-8,
    )
    monkeypatch.setattr(_coupled_host, "_REGULARISATION_TILE_BYTES", 384)
    _coupled_host._finish_regularised_product.clear_cache()
    probe = jnp.tile(x, (3, 1, 1))
    probe_spec = replace(spec, grid=replace(grid, nx=probe.shape[0]))
    derivative = jax.grad(partial(_coupled_program.volume_regularisation, spec=probe_spec))
    tangent = jnp.linspace(-1, 1, probe.size).reshape(probe.shape)
    free = tangent > -0.5
    curvature = jax.jvp(derivative, (probe,), (tangent,))[1]
    product = _coupled_host._finish_regularised_product(
        jnp.ones_like(probe), tangent, probe, free, 0.1, spec=probe_spec
    )
    np.testing.assert_allclose(
        product, free * (1 + 0.1 * tangent + curvature), rtol=2e-6, atol=2e-8
    )
    initial = _coupled_host._finish_regularised_product(
        jnp.zeros_like(probe), None, probe, jnp.float32(1), 0.0, spec=probe_spec
    )
    np.testing.assert_allclose(initial, derivative(probe), rtol=2e-6, atol=2e-8)
    p = jnp.zeros((3, 6), jnp.float32).at[:, 3].set(jnp.asarray([0.03, -0.01, 0.02]))
    expected = objective.update(p, x)
    streamed = None
    for cached_views in (0, 2, 3):
        update, loss = _coupled_host.build_host_objective(
            arrays, replace(spec, cache_columns=cached_views > 0), cached_views=cached_views
        )
        actual = update(p, x)
        assert actual.finite and expected.finite
        if streamed is None:
            streamed = actual.increment
        for a, e, direct in zip(actual.increment, expected.increment, streamed, strict=True):
            np.testing.assert_allclose(a, e, rtol=3e-3, atol=3e-5)
            np.testing.assert_array_equal(a, direct)
        np.testing.assert_allclose(loss(p, x), objective.loss(p, x), rtol=2e-6)
        np.testing.assert_allclose(actual.relative_residual, expected.relative_residual, atol=2e-5)


@pytest.mark.parametrize("budget", [2, 100])
@pytest.mark.parametrize("placement", [_coupled_vectors.HOST, _coupled_vectors.DEVICE])
def test_cg_placements_match_dense_system_without_modifying_inputs(budget, placement):
    xp = placement.array
    rng = np.random.default_rng(50)
    design = rng.normal(size=(31, 21)).astype(np.float32)
    matrix = design.T @ design + np.eye(21, dtype=np.float32)
    b = rng.normal(size=21).astype(np.float32)
    rhs = (b[:12].reshape(2, 3, 2), b[12:].reshape(3, 3))
    diagonal = 1 / np.diag(matrix)
    inverse = (diagonal[:12].reshape(2, 3, 2), diagonal[12:].reshape(3, 3))
    if budget == 100:
        rhs = tuple(np.asfortranarray(a) for a in rhs)
    saved = b.copy()
    saved_rhs = tuple(a.copy() for a in rhs)

    def normal(pair):
        out = xp.asarray(matrix) @ xp.concatenate([a.ravel() for a in pair])
        blocks = out[:12].reshape(2, 3, 2), out[12:].reshape(3, 3)
        if placement.host:
            blocks = tuple(np.asfortranarray(a) for a in blocks)
            for block in blocks:
                block.setflags(write=False)
        return blocks

    result = solve_coupled_normal(
        normal,
        tuple(xp.asarray(a) for a in rhs),
        tuple(xp.asarray(a) for a in inverse),
        max_iters=budget,
        rtol=1e-5,
        placement=placement,
    )
    x = np.concatenate([a.ravel() for a in result.increment])
    np.testing.assert_array_equal(saved, b)
    for a, original in zip(rhs, saved_rhs, strict=True):
        np.testing.assert_array_equal(a, original)
    assert result.finite
    expected_residual = np.linalg.norm(matrix @ x - b) / np.linalg.norm(b)
    np.testing.assert_allclose(result.relative_residual, expected_residual, rtol=0.01, atol=2e-7)
    if budget == 100:
        np.testing.assert_allclose(x, np.linalg.solve(matrix, b), rtol=1e-3, atol=1e-5)
    else:
        assert result.iterations == budget
        assert result.relative_residual > 1e-5


@pytest.mark.parametrize("width", [7, 8])
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_finite_sampling_and_paired_derivatives_match_dense_oracle(width, backend):
    from scipy import signal

    from tomojax.forward import project_joseph
    from tomojax.geometry import Detector, Grid, ParallelGeometry, stack_view_poses

    rng = np.random.default_rng(6341)
    grid = Grid(5, 4, 2, 0.7, 0.8, 1.2)
    detector = Detector(width, 2, 0.7, 1.2, (0.13, -0.07))
    poses = stack_view_poses(ParallelGeometry(grid, detector, [0, 43, 135]), 3)
    poses = poses.at[:, 0, 3].set(jnp.array([0.21, -0.37, 0.13]))
    x = jnp.asarray(rng.normal(size=(5, 4, 2)), jnp.float32)
    y, z = (jnp.asarray(rng.normal(size=(3, 2, width)), jnp.float32) for _ in range(2))
    model = _matching_model.projectors(grid, detector, backend=backend, masked=lambda value: value)
    # Independently form the finite Shannon matrix on the physical fine detector.
    fine, length, pitch = 2 * width, 4 * width + 1, detector.du / 2
    begin = (length - fine) // 2
    centre = detector.center[0] + pitch * ((length - fine) / 2 - begin)
    extended = Detector(length, 2, pitch, detector.dv, (centre, detector.center[1]))
    rays = np.asarray(project_joseph(x, poses.at[:, 0, 3].set(0), grid, extended))
    delta = begin + np.arange(fine)[:, None] - np.arange(length)[None, :]
    shift = np.asarray(poses[:, 0, 3]) / pitch
    sinc = np.sinc(delta[None] - shift[:, None, None])
    translated = np.einsum("nij,nvj->nvi", sinc, rays)
    phase = np.exp(1j * np.pi * np.fft.fftfreq(fine))
    expected = signal.resample(np.fft.fft(translated) * phase, width, axis=-1, domain="freq").real
    np.testing.assert_allclose(model.forward(poses, x), expected, rtol=3e-6, atol=2e-6)
    fixed = model.fixed_pose(poses)
    value, du, ddu = fixed.value_and_derivatives(x)
    np.testing.assert_allclose(fixed.forward(x), (value, du), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(jnp.vdot(value, y), jnp.vdot(x, model.adjoint(poses, y)), atol=3e-6)
    np.testing.assert_allclose(
        jnp.vdot(value, y) + jnp.vdot(du, z),
        jnp.vdot(x, fixed.adjoint(y, z)),
        rtol=3e-6,
        atol=5e-6,
    )
    h = 0.003
    plus, minus = poses.at[:, 0, 3].add(h), poses.at[:, 0, 3].add(-h)
    np.testing.assert_allclose(
        du, (model.forward(plus, x) - model.forward(minus, x)) / (2 * h), atol=5e-4, rtol=3e-3
    )
    np.testing.assert_allclose(
        ddu,
        (model.derivative(plus, x) - model.derivative(minus, x)) / (2 * h),
        atol=1e-3,
        rtol=3e-3,
    )
    # The coupled positive metric must use this model's unit image, not bare Joseph.
    absolute = _matching_model.projectors(
        grid, detector, backend=backend, masked=lambda v: v, absolute=True
    )
    spec = SimpleNamespace(
        grid=grid,
        detector=detector,
        backend=backend,
        active_columns=(3, 5),
        refinement=_matching_model.JointRefinement(native_width=2 * width),
        jacobian=SimpleNamespace(
            cone_beam=False, translation_frame="detector", integrator="joseph"
        ),
    )
    arrays = SimpleNamespace(mask=jnp.float32(1), weights=jnp.float32(1), frames=None)
    bound = _coupled_program.positive_sensitivity(
        spec, arrays, poses, lambda v: v, x.shape, jnp.ones(3), jnp.zeros_like(x)
    )
    expected_bound = absolute.adjoint(poses, absolute.forward(poses, jnp.ones_like(x)))
    np.testing.assert_allclose(bound, expected_bound, rtol=2e-6, atol=2e-6)
    assert np.all(np.abs(value) <= np.asarray(absolute.forward(poses, jnp.abs(x))) + 2e-6)
    # Native affine nuisance follows the same map, including odd native widths.
    native = 2 * width + 1
    backgrounds = rng.normal(size=(3, 2, 1)) + rng.normal(size=(3, 1, 1)) * np.linspace(
        -1, 1, native
    )
    filtered = _matching_model.nuisance_filter(native, detector)
    np.testing.assert_allclose(
        filtered(_matching_model.resample_rows(backgrounds.astype(np.float32), width)), 0, atol=2e-6
    )
    np.testing.assert_allclose(jnp.vdot(filtered(y), z), jnp.vdot(y, filtered(z)), atol=2e-6)


@pytest.mark.parametrize("kind", ["l2", "l2_otsu", "huber"])
def test_fixed_volume_gn_uses_true_loss_and_two_sided_irls(monkeypatch, kind):
    # check-public-imports: allow-private
    from tomojax.alignment._objectives.loss_adapters import build_loss_adapter
    from tomojax.alignment.api import parse_loss_spec

    residual = np.array([-4.0, -0.2, 0.0, 0.4, 3.0], np.float32)
    columns = np.random.default_rng(1348).normal(size=(6, 5)).astype(np.float32)
    adapter = build_loss_adapter(parse_loss_spec(kind), jnp.zeros((1, 1, 5)))
    ctx = SimpleNamespace(
        loss_adapter=adapter, cfg=SimpleNamespace(gn_damping=0.2), active_mask=jnp.ones(6)
    )
    monkeypatch.setattr(
        _pose_objective,
        "build_pose_prediction_and_columns",
        lambda _: lambda *args: (jnp.asarray(residual), jnp.asarray(columns)),
    )
    batch = _pose_objective._build_gn_update_batch(ctx)
    arg = jnp.zeros((1, 6))
    system, rhs, value = batch(arg, arg, arg, arg, jnp.zeros(1), arg)
    weight = np.minimum(1, 1 / np.maximum(1, np.abs(residual))) if kind == "huber" else 1
    expected_value = (
        np.where(np.abs(residual) <= 1, 0.5 * residual**2, np.abs(residual) - 0.5).sum()
        if kind == "huber"
        else 0.5 * np.sum(residual**2)
    )
    np.testing.assert_allclose(
        system[0], (columns * weight) @ columns.T + 0.2 * np.eye(6), atol=2e-6
    )
    np.testing.assert_allclose(rhs[0], -columns @ (weight * residual), atol=2e-6)
    assert float(value[0]) == pytest.approx(expected_value)
    # The robust model gradient must equal the actual per-view objective gradient.
    if kind == "huber":
        gradient = jax.grad(
            lambda r: adapter.per_view_loss(r[None, None], jnp.zeros((1, 1, 5)), None).sum()
        )
        np.testing.assert_allclose(gradient(jnp.asarray(residual)), weight * residual, atol=1e-6)


def test_joint_gn_rejects_huber_with_fixed_volume_remedy():
    from tomojax.alignment.api import HuberLossSpec

    ctx = SimpleNamespace(cfg=None, loss_adapter=SimpleNamespace(spec=HuberLossSpec()))
    with pytest.raises(ValueError, match="fixed_volume"):
        _coupled_objective.build_coupled_objective(ctx)
