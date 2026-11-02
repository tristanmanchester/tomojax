"""The weighted data proximal agrees with exact-input independent references."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.forward import project_joseph
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import SPDHGConfig, spdhg_tv
from tomojax.recon.spdhg_tv import _prox_fstar_l2


def _prox_cell(cell):
    return _prox_fstar_l2(cell[0], cell[1], cell[2], cell[3])


@pytest.mark.numerical
def test_weighted_data_proximal_matches_float64_at_physical_scales():
    rng = np.random.default_rng(3862)
    cells = (10.0 ** rng.uniform(-3, 3, (512, 4))).astype(np.float32)
    cells[:, [0, 2]] *= rng.choice([-1, 1], (512, 2))
    cells[::7, 3] = 0
    u, sigma, y, w = cells.T.astype(np.float64)
    expected = np.where(w > 0, (u - sigma * y) * w / (sigma + w), 0)
    conditioning = np.abs(u * w / (sigma + w)) + np.abs(sigma * y * w / (sigma + w))
    actual = np.asarray(jax.jit(jax.vmap(_prox_cell))(jnp.asarray(cells)), dtype=np.float64)
    assert np.isfinite(actual).all()
    assert np.all(np.abs(actual - expected) <= 1e-6 * conditioning)


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1e-3, 1, 1e3])
def test_weighted_data_proximal_first_derivatives_at_zero_and_cancellation(scale):
    cells = np.asarray(
        [(0, 0.2, 0, 1), (0, 1, 1, 1), (1, 1, 0, 1), (2, 1, 2, 1), (1, 1, 2, 0)],
        dtype=np.float32,
    )
    cells[:, [1, 3]] *= np.float32(scale)
    u, sigma, y, w = cells.T.astype(np.float64)
    denominator = sigma + w
    expected = np.stack(
        [
            w / denominator,
            -w * (u + w * y) / denominator**2,
            -sigma * w / denominator,
            sigma * (u - sigma * y) / denominator**2,
        ],
        axis=-1,
    )
    expected[w == 0] = 0
    values = jnp.asarray(cells)
    forward = jax.jit(jax.vmap(jax.jacfwd(_prox_cell)))(values)
    reverse = jax.jit(jax.vmap(jax.grad(_prox_cell)))(values)
    for actual in [forward, reverse]:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-9)
    direction = jnp.asarray([0.7, -0.13, 0.23, 0.11])
    jvp = jax.jit(jax.vmap(lambda z: jax.jvp(_prox_cell, (z,), (direction,))[1]))(values)
    expected_jvp = expected @ np.asarray(direction, dtype=np.float64)
    condition = np.abs(expected) @ np.abs(np.asarray(direction, dtype=np.float64))
    assert np.all(np.abs(np.asarray(jvp) - expected_jvp) <= 2e-6 * np.maximum(condition, 1e-9))


@pytest.mark.numerical
@pytest.mark.parametrize("sigma", [0.2, 1e-3, 1e3])
def test_unweighted_data_proximal_is_the_same_as_unit_weights(sigma):
    u = jnp.asarray([0, 1, 2, 1e3], dtype=jnp.float32)
    y = jnp.asarray([0, 2, 2, 1e3], dtype=jnp.float32)
    implicit = jax.jit(lambda x: _prox_fstar_l2(x, sigma, y, None))(u)
    explicit = jax.jit(lambda x: _prox_fstar_l2(x, sigma, y, jnp.ones_like(x)))(u)
    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1e-3, 1, 1e3])
def test_weighted_data_proximal_zero_state_hessian_retains_analytic_scaling(scale):
    cell = jnp.asarray([0, scale, 0, scale], dtype=jnp.float32)
    step = float(np.float32(scale))
    expected = np.asarray(
        [
            [0, -0.25 / step, 0, 0.25 / step],
            [-0.25 / step, 0, -0.25, 0],
            [0, -0.25, 0, -0.25],
            [0.25 / step, 0, -0.25, 0],
        ],
        dtype=np.float64,
    )
    actual = np.asarray(jax.jit(jax.hessian(_prox_cell))(cell))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-9)


@pytest.mark.numerical
def test_weighted_data_proximal_is_local_to_each_observation():
    rng = np.random.default_rng(426)
    cells = rng.normal(size=(257, 4)).astype(np.float32)
    cells[:, [1, 3]] = rng.uniform(0.2, 2, (257, 2))
    cells[-1] = [1e3, 1e3, 1e3, 1]
    # Batched and independent scalar proximals must use the same arithmetic.
    expected = jax.jit(jax.vmap(_prox_cell))(jnp.asarray(cells))
    actual = jax.jit(lambda z: _prox_fstar_l2(z[:, 0], z[:, 1], z[:, 2], z[:, 3]))(
        jnp.asarray(cells)
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.numerical
@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_weighted_data_masked_outliers_keep_dense_reconstruction_and_truth_error(tilted, stream):
    shape = (2, 1, 1)
    grid = Grid(*shape, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.7, 0.9, (0.17, -0.23))
    angles = [13.0, 71.0, 127.0]
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=25)
        if tilted
        else ParallelGeometry(grid, detector, angles)
    )
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)], dtype=jnp.float32)

    def project(flat):
        return project_joseph(
            flat.reshape(shape[::-1]), poses, grid, detector, backend="jax"
        ).ravel()

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(2)), dtype=np.float64)
    truth = np.asarray([0.2, 1.0])
    data = matrix @ truth + np.random.default_rng(839).normal(0, 1e-3, matrix.shape[0])
    weights = np.linspace(0.3, 1.7, data.size).astype(np.float32)
    weights[::3] = 0
    # Masked outliers must not contribute to the reconstruction.
    data[weights == 0] = 1e3
    data = data.astype(np.float32)
    normal = matrix.T @ (weights[:, None] * matrix)
    expected = np.linalg.solve(normal, matrix.T @ (weights * data.astype(np.float64)))
    assert np.linalg.cond(normal) < 100
    assert np.linalg.norm(expected - truth) / np.linalg.norm(truth) < 0.01
    volume, info = spdhg_tv(
        geometry,
        grid,
        detector,
        data.reshape((3, 3, 4)),
        weights=weights.reshape((3, 3, 4)),
        config=SPDHGConfig(
            iterations=2000,
            views_per_batch=2,
            tv_weight=0,
            log_every=2000,
            projector_model="joseph",
            backend="jax",
            stream_projections=stream,
        ),
    )
    assert np.isfinite(info["loss"]).all()
    actual = np.asarray(volume).ravel()
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-5
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.01
