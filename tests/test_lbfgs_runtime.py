"""Exercise compiled L-BFGS updates with real line searches."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment.optimizers import PoseLbfgsConfig, _run_lbfgs_optax_loop


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

    # check-public-imports: allow-private
    from tomojax.core.projector import forward_project_view_T
    from tomojax.geometry import Detector, Grid, ParallelGeometry

    grid = Grid(4, 4, 3, 1.0, 1.0, 1.0)
    detector = Detector(6, 5, 1.0, 1.0, (0.13, -0.21))
    geometry = ParallelGeometry(grid, detector, np.linspace(13.0, 169.0, 5))
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)])
    truth = jnp.asarray(np.random.default_rng(61).uniform(size=(4, 4, 3)), dtype=jnp.float32)
    data = jax.vmap(lambda t: forward_project_view_T(t, grid, detector, truth))(poses)
    initial = jnp.zeros((5, 5)).at[:, 2].set(0.01).at[:, 3].set(jnp.linspace(-0.1, 0.1, 5))
    cfg = AlignConfig(
        outer_iters=2,
        recon_iters=1,
        recon_L=100.0,
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
        geometry, grid, detector, data, init_x=truth, init_params5=initial, config=cfg
    )
    assert len(info["outer_stats"]) == 2
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
