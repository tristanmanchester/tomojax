"""Physical-scale and diagnostic regressions for differentiable reconstruction."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.forward import project_joseph
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FistaConfig, fista_tv
from tomojax.recon.fista_tv_core import (
    FistaCoreConfig,
    fista_objective_arrays,
    fista_tv_core_arrays,
)


@pytest.mark.numerical
@pytest.mark.parametrize("override", [False, True])
def test_core_one_voxel_reconstruction_preserves_physical_length_units(override):
    # An axis-aligned ray through one voxel has A=[scale]. Its least-squares
    # solution is exactly 3 after one step with L=scale**2, in any length units.
    scale = 1e-4
    grid, detector = Grid(1, 1, 1, scale, scale, scale), Detector(1, 1, scale, scale)
    poses = jnp.eye(4, dtype=jnp.float32)[None]
    lipschitz = scale**2
    cfg = FistaCoreConfig(
        iterations=1,
        tv_weight=0,
        lipschitz=1.0 if override else lipschitz,
        ray_integrator="joseph",
    )

    def reconstruct(L):
        return fista_tv_core_arrays(
            x0=jnp.zeros((1, 1, 1)),
            T_all=poses,
            det_grid=None,
            projections=jnp.full((1, 1, 1), 3 * scale),
            grid=grid,
            detector=detector,
            cfg=cfg,
            L_override=L if override else None,
        ).x.sum()

    actual = jax.jit(reconstruct)(jnp.float32(lipschitz))
    np.testing.assert_allclose(actual, 3.0, rtol=3e-6)
    if override:
        derivative = jax.jit(jax.grad(reconstruct))(jnp.float32(lipschitz))
        np.testing.assert_allclose(derivative * lipschitz, -3.0, rtol=3e-6)


def _diagnostic_problem(iterations, *, final_data, iteration_loss):
    grid, detector = Grid(2, 2, 1, 0.8, 1.1, 1.0), Detector(3, 1, 0.7, 1.0)
    poses = jnp.eye(4, dtype=jnp.float32)[None]
    initial = jnp.asarray([0.2, 0.8, 1.3, 0.6]).reshape(1, 2, 2)
    data = jnp.asarray([0.7, 1.0, 0.4]).reshape(1, 1, 3)
    cfg = FistaCoreConfig(
        iterations=iterations,
        tv_weight=0.3,
        huber_delta=0.25,
        lipschitz=10,
        ray_integrator="joseph",
        compute_iteration_loss=iteration_loss,
        compute_final_data_loss=final_data,
    )

    def solve(x, y):
        result = fista_tv_core_arrays(
            x0=x,
            T_all=poses,
            det_grid=None,
            projections=y,
            grid=grid,
            detector=detector,
            cfg=cfg,
        )
        return {
            "x": result.x,
            "loss": result.loss,
            "data_loss": result.data_loss,
            "regulariser_value": result.regulariser_value,
            "effective_iterations": result.effective_iterations,
        }

    actual = jax.jit(solve)(initial, data)
    residual = np.asarray(project_joseph(initial, poses, grid, detector, backend="jax")) - data
    expected_data = 0.5 * np.sum(np.asarray(residual, np.float64) ** 2)
    gradients = [
        np.diff(np.asarray(initial), axis=axis, append=np.take(initial, [-1], axis=axis))
        for axis in range(3)
    ]
    norm = np.sqrt(sum(gradient**2 for gradient in gradients))
    expected_tv = np.sum(np.where(norm <= 0.25, 0.5 * norm**2 / 0.25, norm - 0.125))
    return actual, initial, expected_data, expected_tv


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("iterations", "final_data", "iteration_loss"),
    [(1, False, False), (1, False, True), (0, False, False), (0, True, False)],
)
def test_core_data_diagnostic_never_contains_regularization(iterations, final_data, iteration_loss):
    actual, initial, expected_data, expected_tv = _diagnostic_problem(
        iterations, final_data=final_data, iteration_loss=iteration_loss
    )
    assert expected_tv > 0
    computed = final_data or iteration_loss
    np.testing.assert_allclose(actual["data_loss"], expected_data if computed else 0, rtol=3e-6)
    if iteration_loss:
        np.testing.assert_allclose(actual["loss"][0], expected_data + 0.3 * expected_tv, rtol=3e-6)
    if not iterations:
        np.testing.assert_array_equal(actual["x"], initial)
        assert actual["loss"].shape == (0,) and int(actual["effective_iterations"]) == 0
        np.testing.assert_allclose(actual["regulariser_value"], expected_tv, rtol=3e-6)


@pytest.mark.parametrize("lipschitz", [0.0, np.nan])
def test_core_rejects_invalid_lipschitz_config(lipschitz):
    with pytest.raises(ValueError, match="lipschitz"):
        replace(FistaCoreConfig(), lipschitz=lipschitz)


def test_core_rejects_nonsmooth_tv_instead_of_solving_an_unregularized_problem():
    # For A=I, y=(0,1), lam=.2, exact TV has the unique optimum (.2,.8).
    # Gradient-only FISTA previously returned (0,1) and claimed status='ok'.
    cfg = FistaCoreConfig(iterations=1, tv_weight=0.2, regulariser="tv", ray_integrator="joseph")
    # Objective-only evaluations still need exact TV, including coupled alignment.
    objective = fista_objective_arrays(
        T_all=jnp.eye(4)[None],
        grid=Grid(2, 1, 1, 1.0, 1.0, 1.0),
        detector=Detector(2, 1, 1.0, 1.0),
        volume=jnp.asarray([0.0, 1.0]).reshape(1, 1, 2),
        det_grid=None,
        projections=jnp.asarray([[[0.0, 1.0]]]),
        cfg=cfg,
    )
    np.testing.assert_allclose(objective, 0.2, rtol=1e-6)
    with pytest.raises(ValueError, match="huber_tv or tv_weight=0"):
        fista_tv_core_arrays(
            x0=jnp.zeros((1, 1, 2)),
            T_all=jnp.eye(4)[None],
            det_grid=None,
            projections=jnp.asarray([[[0.0, 1.0]]]),
            grid=Grid(2, 1, 1, 1.0, 1.0, 1.0),
            detector=Detector(2, 1, 1.0, 1.0),
            cfg=cfg,
        )


@pytest.mark.numerical
def test_zero_weight_tv_iteration_loss_has_only_the_data_gradient_at_a_flat_volume():
    grid, detector = Grid(2, 2, 2, 1.0, 1.0, 1.0), Detector(2, 2, 1.0, 1.0)
    poses = jnp.eye(4, dtype=jnp.float32)[None]
    cfg = FistaCoreConfig(iterations=1, tv_weight=0, regulariser="tv", ray_integrator="joseph")
    data = jnp.ones((1, 2, 2))

    def iteration_loss(initial):
        return fista_tv_core_arrays(
            x0=initial,
            T_all=poses,
            det_grid=None,
            projections=data,
            grid=grid,
            detector=detector,
            cfg=cfg,
        ).loss[0]

    # Every voxel lies on one unit-length plane contribution to a measured ray.
    actual = jax.jit(jax.grad(iteration_loss))(jnp.zeros((2, 2, 2)))
    np.testing.assert_allclose(actual, -np.ones((2, 2, 2)), rtol=1e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("grad_mode", ["batched", "stream"])
def test_public_fista_partial_view_batches_run_under_global_x64(grad_mode):
    grid, detector = Grid(1, 1, 1, 1.0, 1.0, 1.0), Detector(1, 1, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, np.zeros(5, np.float32))
    data = np.full((5, 1, 1), 3.0, np.float32)  # five unit rays through one voxel
    config = FistaConfig(
        iterations=1,
        tv_weight=0,
        projector_model="ray",
        backend="jax",
        ray_integrator="sampled",
        views_per_batch=2,
        power_iterations=3,
        grad_mode=grad_mode,
    )
    with jax.enable_x64():
        actual, info = fista_tv(geometry, grid, detector, data, config=config)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, 3.0, rtol=3e-6)
    np.testing.assert_allclose(info["lipschitz"], 5.0, rtol=3e-6)


@pytest.mark.numerical
def test_zero_view_weight_has_finite_smooth_parameter_derivatives():
    grid, detector = Grid(1, 1, 1, 1.0, 1.0, 1.0), Detector(1, 1, 1.0, 1.0)
    poses = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 4, 4))
    cfg = FistaCoreConfig(tv_weight=0, ray_integrator="joseph")

    def objective(parameter):
        return fista_objective_arrays(
            T_all=poses,
            grid=grid,
            detector=detector,
            volume=jnp.full((1, 1, 1), 0.4),
            det_grid=None,
            projections=jnp.asarray([1.0, 3.0]).reshape(2, 1, 1),
            cfg=cfg,
            view_weights=jnp.asarray([parameter**2, 1.0]),
        )

    # The exact loss is .5 * (.6**2 * parameter**2 + 2.6**2), smooth at parameter 0.
    value, gradient = jax.value_and_grad(objective)(jnp.float32(0))
    curvature = jax.grad(jax.grad(objective))(jnp.float32(0))
    np.testing.assert_allclose(value, 0.5 * 2.6**2, rtol=2e-6)
    np.testing.assert_allclose(gradient, 0.0, atol=0)
    np.testing.assert_allclose(curvature, 0.6**2, rtol=2e-6)
