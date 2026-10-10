"""Physical-scale and diagnostic regressions for differentiable reconstruction."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.forward import project_joseph
from tomojax.geometry import Detector, Grid
from tomojax.recon.fista_tv_core import FistaCoreConfig, fista_tv_core_arrays


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1.0, 1e-4])
@pytest.mark.parametrize("override", [False, True])
def test_core_one_voxel_reconstruction_preserves_physical_length_units(scale, override):
    # An axis-aligned ray through one voxel has A=[scale]. Its least-squares
    # solution is exactly 3 after one step with L=scale**2, in any length units.
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
    initial = jnp.asarray([0.2, 0.8, 1.3, 0.6]).reshape(2, 2, 1)
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
@pytest.mark.parametrize("iteration_loss", [False, True])
def test_core_uncomputed_final_data_term_never_contains_regularization(iteration_loss):
    actual, _, expected_data, expected_tv = _diagnostic_problem(
        1, final_data=False, iteration_loss=iteration_loss
    )
    assert expected_tv > 0
    np.testing.assert_allclose(
        actual["data_loss"], expected_data if iteration_loss else 0, rtol=3e-6
    )
    if iteration_loss:
        np.testing.assert_allclose(actual["loss"][0], expected_data + 0.3 * expected_tv, rtol=3e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("iteration_loss", [False, True])
@pytest.mark.parametrize("final_data", [False, True])
def test_core_zero_iterations_preserve_start_and_handle_optional_diagnostics(
    final_data, iteration_loss
):
    actual, initial, expected_data, expected_tv = _diagnostic_problem(
        0, final_data=final_data, iteration_loss=iteration_loss
    )
    np.testing.assert_array_equal(actual["x"], initial)
    assert actual["loss"].shape == (0,)
    assert int(actual["effective_iterations"]) == 0
    np.testing.assert_allclose(actual["data_loss"], expected_data if final_data else 0, rtol=3e-6)
    np.testing.assert_allclose(actual["regulariser_value"], expected_tv, rtol=3e-6)


@pytest.mark.parametrize("lipschitz", [0.0, -1.0, np.inf, np.nan])
def test_core_rejects_invalid_lipschitz_config(lipschitz):
    with pytest.raises(ValueError, match="lipschitz"):
        replace(FistaCoreConfig(), lipschitz=lipschitz)
