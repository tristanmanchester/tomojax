"""Independent dense Schur references for the coupled free-voxel update."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._pose._coupled_linear import solve_coupled_normal


@pytest.fixture(params=[None, 2], ids=["one_batch", "batches_of_two"])
def views_per_batch(request, monkeypatch):
    """The program's batch size, or two views a batch: three views then end in a shifted batch."""
    # check-public-imports: allow-private
    from tomojax.alignment._pose import _coupled_program

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


@pytest.mark.numerical
@pytest.mark.usefixtures("views_per_batch")
@pytest.mark.parametrize("scan_variant", [0, 1])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("solver", ["stacked", "pose_eliminated"])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("integrator", ["exact", "joseph"])
def test_physical_coupled_step_matches_independent_dense_model(  # noqa: PLR0915
    monkeypatch, stream, backend, solver, scan_variant, integrator
):
    from dataclasses import replace
    from pathlib import Path

    from tomojax.alignment import AlignConfig
    from tomojax.alignment._pose import _coupled_objective  # check-public-imports: allow-private

    # check-public-imports: allow-private
    from tomojax.alignment._pose._pose_context import _pose_objective_context

    # check-public-imports: allow-private
    from tomojax.alignment._pose._pose_loop import _build_alignment_runtime_context
    from tomojax.alignment.api import PWLSLossSpec
    from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses

    if backend == "pallas" and jax.default_backend() != "gpu":
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
        monkeypatch.setattr(_coupled_objective, "_POSE_CACHE_BYTES", 0)
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
        gn_joint_solver=solver,
        gn_joint_iterations=100,
        gn_joint_rtol=1e-6,
        gn_damping=0.2,
        # Accepted scalar-like options must remain usable in a hashable
        # program specification, including zero-dimensional NumPy arrays.
        gn_volume_damping=0.2 if scan_variant == 0 else np.asarray(0.2),
        gn_jacobian="central",
        gn_difference_step=0.003 if scan_variant == 0 else np.asarray(0.003),
        ray_integrator=integrator,
        projector_backend=backend,
        gather_dtype="fp32",
        pose_translation_frame="detector",
        tv_weight=0,
        loss=PWLSLossSpec(a=0.3, b=0.8),
        w_rot=0.03 + 0.005 * scan_variant,
        w_trans=0.05 + 0.003 * scan_variant,
    )
    common = dict(
        grid=grid,
        detector=det,
        projections=target,
        cfg=cfg,
        n_views=3,
        active_mask=jnp.asarray(active),
    )
    runtime = _build_alignment_runtime_context(geometry=geometry, det_grid_override=None, **common)
    ctx = _pose_objective_context(runtime=runtime, **common)
    ctx = replace(ctx, volume_mask=jnp.asarray(support))
    objective = _coupled_objective.build_coupled_objective(ctx)
    result = objective.update(jnp.asarray(p), jnp.asarray(x))
    assert result.finite
    assert objective.pose_columns_cached != stream
    weight = np.sqrt(1 / (0.3 * np.maximum(target, 0) + 0.8 + 1e-6)).ravel()
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


@pytest.mark.numerical
def test_coupled_solve_reports_actual_residual_at_budget_and_handles_zero_rhs():
    matrix = jnp.diag(jnp.array([1.0, 3.0, 8.0, 15.0]))

    def normal(blocks):
        result = matrix @ jnp.concatenate(blocks)
        return result[:2], result[2:]

    rhs = (jnp.ones(2), jnp.ones(2))
    result = solve_coupled_normal(normal, rhs, rhs, max_iters=1, rtol=1e-6)
    assert int(result.iterations) == 1 and result.relative_residual > 1e-2
    zero = (jnp.zeros(2), jnp.zeros(2))
    result = solve_coupled_normal(normal, zero, rhs, max_iters=10, rtol=1e-6)
    assert result.finite and result.iterations == 0 and result.relative_residual == 0


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
        gn_joint_solver=solver,
        gn_jacobian="central",
        gather_dtype="fp32",
        ray_integrator=integrator,
        projector_backend="jax",
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
    from tomojax.alignment import AlignConfig

    # check-public-imports: allow-private
    from tomojax.alignment._pose._pose_context import _pose_objective_context

    # check-public-imports: allow-private
    from tomojax.alignment._pose._pose_loop import _build_alignment_runtime_context
    from tomojax.geometry import Detector, Grid, ParallelGeometry

    grid, det = Grid(4, 4, 4, 1.0, 1.0, 1.0), Detector(5, 4, 1.0, 1.0)
    geometry = ParallelGeometry(grid, det, [0.0, 60.0, 120.0])
    projections = jnp.asarray(np.random.default_rng(3).uniform(0, 1, (3, 4, 5)), jnp.float32)
    cfg = AlignConfig(gn_coupling="joint", gather_dtype="fp32", projector_backend="jax")
    common = dict(
        grid=grid,
        detector=det,
        projections=projections,
        cfg=cfg,
        n_views=3,
        active_mask=jnp.ones(6, jnp.float32),
    )
    runtime = _build_alignment_runtime_context(geometry=geometry, det_grid_override=None, **common)
    return _pose_objective_context(runtime=runtime, **common)


def test_memory_check_compiles_the_update_unless_far_below_the_limit(monkeypatch):
    # check-public-imports: allow-private
    from tomojax.alignment._pose import _coupled_objective as coupled

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
        lowered.append(True)
        return lower(*args, **kwargs)

    monkeypatch.setattr(coupled.run_update, "lower", counting_lower)

    def free(nbytes):
        lowered.clear()
        monkeypatch.setattr(coupled, "_free_device_memory", lambda _array: nbytes)

    def assert_update_matches(objective):
        result = objective.update(p, x)
        for a, b in zip(result.increment, expected.increment, strict=True):
            # Within rounding: GPU atomics sum in no fixed order.
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6)

    # Far below the limit, the estimate is trusted.
    free(4 * estimate)
    assert_update_matches(coupled.build_coupled_objective(ctx))
    assert not lowered
    # Near it, XLA's own figure decides. Here the update needs more than the
    # estimate, which a shortcut without the columns or a margin would miss.
    free(int(1.5 * estimate))
    assert 13 * volume + 3 * sinogram < 1.5 * estimate
    with pytest.raises(coupled.AlignmentMemoryError, match="GiB is free") as error:
        coupled.build_coupled_objective(ctx)
    assert lowered
    assert error.value.available == int(1.5 * estimate) and error.value.needed > 1.5 * estimate
    # When it fits, the update compiled for the check is the one used.
    monkeypatch.setattr(coupled, "_MEMORY_MARGIN", 1000)
    free(error.value.needed)
    objective = coupled.build_coupled_objective(ctx)
    assert isinstance(objective.update.func, jax.stages.Compiled)
    assert_update_matches(objective)
    assert len(lowered) == 1
