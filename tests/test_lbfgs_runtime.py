"""Exercise compiled L-BFGS updates with real line searches."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment.optimizers import (
    PoseLbfgsConfig,
    PoseOptimizationContext,
    _run_lbfgs_optax_loop,
    run_pose_lbfgs,
)

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T


@pytest.mark.numerical
def test_failed_lbfgs_line_search_does_not_report_unchanged_pose_as_converged():
    # A restricted valid domain models a failed pose/objective evaluation.
    # One zoom trial cannot backtrack to the known interior optimum.
    def objective(z):
        return jnp.where(jnp.all(jnp.abs(z) < 0.009), jnp.sum((z - 0.005) ** 2), jnp.inf)

    result = _run_lbfgs_optax_loop(
        z0=jnp.zeros(1),
        value_fn=objective,
        cfg=PoseLbfgsConfig(maxiter=4, maxls=1, ftol=1e-12, gtol=1e-8),
    )
    assert not result.success
    assert result.last_grad_norm > 0.001
    assert result.last_step_norm == 0.0
    np.testing.assert_array_equal(result.z, [0.0])
    assert "maxiter" in result.message

    solved = _run_lbfgs_optax_loop(
        z0=jnp.zeros(1),
        value_fn=objective,
        cfg=PoseLbfgsConfig(maxiter=4, maxls=5, ftol=1e-12, gtol=1e-8),
    )
    assert solved.success
    np.testing.assert_allclose(solved.z, [0.005], atol=1e-8)
    assert solved.last_grad_norm <= 1e-8


@pytest.mark.numerical
@pytest.mark.parametrize("explicit_gradient", [False, True])
def test_lbfgs_compiled_state_updates_converge_on_quadratic(explicit_gradient):
    target = jnp.asarray([0.3, -0.7, 1.4, -2.1], dtype=jnp.float32)
    scale = jnp.asarray([1.0, 3.0, 8.0, 17.0], dtype=jnp.float32)

    def objective(z):
        return 0.5 * jnp.sum(scale * (z - target) ** 2)

    def value_grad(z):
        return objective(z), scale * (z - target)

    result = _run_lbfgs_optax_loop(
        z0=jnp.ones(4),
        value_fn=objective,
        value_and_grad_fn=value_grad if explicit_gradient else None,
        cfg=PoseLbfgsConfig(maxiter=60, maxls=20, memory_size=5, ftol=1e-12, gtol=1e-5),
    )
    assert result.failure_message is None
    assert result.success
    assert result.nit > 1
    assert result.total_line_search_steps >= result.nit
    np.testing.assert_allclose(result.best_z, target, atol=2e-4)
    assert result.best_value < 1e-8


@pytest.mark.numerical
def test_reusable_lbfgs_kernels_consume_new_objective_inputs():
    # check-public-imports: allow-private
    from tomojax.alignment.optimizers import _LbfgsKernels

    def objective(z, target):
        return 0.5 * jnp.sum((z - target) ** 2)

    config = PoseLbfgsConfig(maxiter=10, maxls=10, ftol=1e-9, gtol=1e-5)
    kernels = _LbfgsKernels.build(objective, config)
    for target in [jnp.asarray([1.0, 2.0, -3.0]), jnp.asarray([-2.0, 0.5, 4.0])]:
        result = _run_lbfgs_optax_loop(
            z0=jnp.zeros(3),
            value_fn=objective,
            cfg=config,
            kernels=kernels,
            objective_args=(target,),
        )
        assert result.success
        np.testing.assert_allclose(result.best_z, target, atol=1e-5)


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("pose_model", "bounded"), [("per_view", False), ("polynomial", False), ("spline", True)]
)
def test_public_lbfgs_reuses_problem_across_volumes_and_motion_models(pose_model, bounded):
    import jax

    from tomojax.alignment import AlignConfig
    from tomojax.alignment.api import L2LossSpec, align
    from tomojax.geometry import Detector, Grid, ParallelGeometry

    grid = Grid(4, 4, 3, 1.0, 1.0, 1.0)
    detector = Detector(6, 5, 1.0, 1.0, (0.13, -0.21))
    geometry = ParallelGeometry(grid, detector, np.linspace(13.0, 169.0, 5))
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)])
    truth = jnp.asarray(np.random.default_rng(61).uniform(size=(4, 4, 3)), dtype=jnp.float32)
    data = jax.vmap(lambda t: forward_project_view_T(t, grid, detector, truth))(poses)
    initial = jnp.zeros((5, 5)).at[:, 2].set(0.01).at[:, 3].set(jnp.linspace(-0.1, 0.1, 5))
    cfg = AlignConfig(
        outer_iterations=2,
        iterations=1,
        lipschitz=100.0,
        opt_method="lbfgs",
        lbfgs_maxiter=2,
        projector_backend="jax",
        gather_dtype="fp32",
        views_per_batch=2,
        loss=L2LossSpec(),
        early_stop=False,
        optimise_dofs=("phi", "dx", "dz"),
        gauge_policy="anchor_mean",
        pose_model=pose_model,
        degree=2,
        knot_spacing=2,
        bounds={"dx": (-0.3, 0.3), "dz": (-0.3, 0.3)} if bounded else (),
    )
    volume, params, info = align(
        geometry, grid, detector, data, init_x=truth, init_pose_params=initial, config=cfg
    )
    assert len(info["outer_stats"]) == 2
    assert info["pose_model_basis_shape"] == [5, 5 if pose_model == "per_view" else 3]
    assert bool(jnp.all(jnp.isfinite(volume)))
    assert bool(jnp.all(jnp.isfinite(params)))
    np.testing.assert_array_equal(params[:, :2], np.zeros((5, 2)))
    if bounded:
        assert float(jnp.max(jnp.abs(params[:, 3:]))) <= 0.300001
    for stat in info["outer_stats"]:
        assert not stat.get("reconstruction_failed", False)
        assert not stat.get("lbfgs_fallback_to_gd", False)
        assert stat["lbfgs_backend"] == "optax"
        assert stat["loss_after"] <= stat["loss_before"] * (1 + 1e-5)


@pytest.mark.numerical
def test_pose_lbfgs_reduces_known_fixed_volume_pose_error():
    import jax

    from tomojax.alignment.api import apply_pose_updates
    from tomojax.geometry import Detector, Grid, LaminographyGeometry

    grid = Grid(9, 8, 7, 0.8, 1.1, 1.3)
    detector = Detector(13, 11, 0.7, 0.9, (0.13, -0.21))
    geometry = LaminographyGeometry(grid, detector, [13.0, 47.0, 96.0, 153.0], tilt_deg=23.0)
    nominal = jnp.asarray([geometry.pose_for_view(i) for i in range(4)])
    coords = np.meshgrid(
        *[(np.arange(n) - (n - 1) / 2) * spacing for n, spacing in [(9, 0.8), (8, 1.1), (7, 1.3)]],
        indexing="ij",
    )
    volume = sum(
        np.exp(-sum((v - c) ** 2 for v, c in zip(coords, centre, strict=True)) / width)
        for centre, width in [((1.1, -0.3, 0.7), 1.2), ((-1.3, 1.5, -1.0), 0.8)]
    )
    volume = jnp.asarray(volume, dtype=jnp.float32)
    shifts = [[0.12, -0.15], [-0.19, 0.09], [0.08, 0.14], [-0.11, -0.07]]
    truth = jnp.zeros((4, 5)).at[:, 3:].set(jnp.asarray(shifts))

    def project(params):
        poses = apply_pose_updates(nominal, params, translation_frame="detector")
        return jax.vmap(lambda pose: forward_project_view_T(pose, grid, detector, volume))(poses)

    data = project(truth)

    def objective(params):
        return 0.5 * jnp.mean((project(params) - data) ** 2)

    initial = jnp.zeros_like(truth)
    before = float(objective(initial))
    result = run_pose_lbfgs(
        pose_params_in=initial,
        motion_coeffs_in=None,
        loss_before_value=before,
        objective_fn=objective,
        eval_loss_fn=lambda params, _label: float(objective(params)),
        is_expected_failure=lambda _error: False,
        cfg=PoseLbfgsConfig(maxiter=60, maxls=20, ftol=1e-12, gtol=1e-7),
        context=PoseOptimizationContext(
            active_cols=np.asarray([3, 4]),
            frozen_pose_params=initial,
            bounds_lower=jnp.full(5, -jnp.inf),
            bounds_upper=jnp.full(5, jnp.inf),
            apply_param_constraints=lambda params: params,
        ),
    )
    assert result.accepted
    assert result.loss < before * 1e-5
    # The volume is fixed, so there is no free object-frame gauge to fit away.
    np.testing.assert_allclose(result.pose_params[:, 3:], truth[:, 3:], atol=2e-4)
    np.testing.assert_array_equal(result.pose_params[:, :3], np.zeros((4, 3)))


def test_lbfgs_initial_evaluation_failure_keeps_state_and_propagates_unknown_errors():
    def objective(_z):
        raise RuntimeError("bad start")

    cfg = PoseLbfgsConfig(maxiter=2, ftol=1e-9, gtol=1e-6, maxls=2)
    options = dict(z0=jnp.ones(2), value_fn=objective, cfg=cfg)
    failed = _run_lbfgs_optax_loop(
        **options, is_expected_failure=lambda exc: isinstance(exc, RuntimeError)
    )
    assert not failed.success and failed.nit == failed.eval_count == 0
    assert failed.initial_value is None and failed.initial_grad_norm is None
    assert failed.failure_message == "initial L-BFGS objective/gradient failed: bad start"
    np.testing.assert_array_equal(failed.z, options["z0"])
    with pytest.raises(RuntimeError, match="bad start"):
        _run_lbfgs_optax_loop(**options)
