"""Independent dense Schur references for the coupled free-voxel update."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.align._pose._coupled_linear import solve_coupled_normal


def test_joint_schedule_keeps_explicit_fixed_volume_stages():
    from tomojax.align import AlignConfig

    # check-public-imports: allow-private
    from tomojax.align._config import _resolved_schedule_for_cfg

    implicit = _resolved_schedule_for_cfg(AlignConfig(gn_coupling="joint"))
    explicit = _resolved_schedule_for_cfg(AlignConfig(gn_coupling="joint", schedule="pose_only"))
    assert implicit.stages[0].objective_kind == "joint_volume_pose"
    assert explicit.stages[0].objective_kind == "fixed_volume"


@pytest.mark.numerical
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("solver", ["stacked", "pose_eliminated"])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_physical_coupled_step_matches_independent_dense_model(  # noqa: PLR0915
    monkeypatch, stream, backend, solver
):
    from pathlib import Path

    from tomojax.align import AlignConfig
    from tomojax.align._pose import _coupled_objective  # check-public-imports: allow-private

    # check-public-imports: allow-private
    from tomojax.align._pose._pose_context import _pose_objective_context

    # check-public-imports: allow-private
    from tomojax.align._pose._pose_loop import _build_alignment_runtime_context
    from tomojax.align.api import PWLSLossSpec
    from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses

    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("CUDA required")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    from public_alignment_benchmark import physical_poses
    from voxel_truth import project_voxel_truth

    if stream:
        monkeypatch.setattr(_coupled_objective, "_POSE_CACHE_BYTES", 0)
    grid = Grid(2, 3, 2, 0.7, 1.1, 1.4, vol_origin=(-0.32, -0.71, -0.53))
    det = Detector(4, 3, 0.65, 0.91, (0.23, -0.17))
    geometry = LaminographyGeometry(grid, det, [7.2, 49.3, 113.7], tilt_deg=23)
    nominal = np.asarray(stack_view_poses(geometry, 3), np.float64)
    rng = np.random.default_rng(34)
    x = rng.uniform(0.2, 1.3, (2, 3, 2)).astype(np.float32)
    p = rng.normal(0, 0.01, (3, 5)).astype(np.float32)
    pose = physical_poses(nominal, p)
    predicted = project_voxel_truth(x, pose, grid, det)
    target = predicted + rng.normal(0, 0.03, predicted.shape)
    target = jnp.asarray(target, jnp.float32)
    active = np.array([1, 0, 1, 1, 0], np.float32)
    cfg = AlignConfig(
        gn_coupling="joint",
        gn_joint_solver=solver,
        gn_joint_iters=100,
        gn_joint_rtol=1e-6,
        gn_damping=0.2,
        gn_volume_damping=0.2,
        gn_jacobian="central",
        gn_difference_step=0.003,
        ray_integrator="exact",
        projector_backend=backend,
        gather_dtype="fp32",
        pose_translation_frame="detector",
        gauge_fix="none",
        lambda_tv=0,
        loss=PWLSLossSpec(a=0.3, b=0.8),
        w_rot=0.03,
        w_trans=0.05,
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
    objective = _coupled_objective.build_coupled_objective(ctx)
    result = objective.update(jnp.asarray(p), jnp.asarray(x))
    assert result.finite
    assert objective.pose_columns_cached != stream
    weight = np.sqrt(1 / (0.3 * np.maximum(target, 0) + 0.8 + 1e-6)).ravel()
    a = (
        np.column_stack(
            [
                project_voxel_truth(v.reshape(x.shape), pose, grid, det).ravel()
                for v in np.eye(x.size)
            ]
        )
        * weight[:, None]
    )
    displacement = cfg.gn_difference_step * min(grid.vx, grid.vy, grid.vz)
    radius = 0.5 * np.linalg.norm(np.array(x.shape) * [grid.vx, grid.vy, grid.vz])
    steps = np.array([displacement / radius] * 3 + [displacement] * 2)
    j = np.zeros((target.size, p.size))
    for i in range(p.size):
        view, dof = divmod(i, 5)
        plus, minus = p.astype(float), p.astype(float)
        plus[view, dof] += steps[dof]
        minus[view, dof] -= steps[dof]
        j[:, i] = (
            weight
            * active[dof]
            * (
                project_voxel_truth(x, physical_poses(nominal, plus), grid, det).ravel()
                - project_voxel_truth(x, physical_poses(nominal, minus), grid, det).ravel()
            )
            / (2 * steps[dof])
        )
    smooth = np.kron(np.array([[1, -2, 1]]), np.diag(np.array([0.03] * 3 + [0.05] * 2) * active))
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
@pytest.mark.parametrize("integrator", ["sampled", "exact"])
@pytest.mark.parametrize("solver", ["stacked", "pose_eliminated"])
def test_public_joint_huber_constraints_and_resume(integrator, solver):
    from tomojax.align import AlignConfig, align
    from tomojax.align.api import L2LossSpec
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
        lambda_tv=0.002,
        outer_iters=2,
        recon_iters=2,
        recon_L=100,
        loss=L2LossSpec(),
        early_stop=False,
        freeze_dofs=("beta",),
        bounds=(("dx", -0.01, 0.01),),
        gauge_fix="none",
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
