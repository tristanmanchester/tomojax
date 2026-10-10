"""The weighted data proximal agrees with exact-input independent references."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.recon.spdhg_tv import _prox_fstar_l2


def _prox_cell(cell):
    return _prox_fstar_l2(cell[0], cell[1], cell[2], cell[3])


@pytest.mark.numerical
def test_weighted_data_proximal_balances_normal_range_affine_terms():
    cells = np.asarray(
        [
            (1, 1e-13, 2, 1e-13),
            (1, 1e-30, 1, 1e-30),
            (1, 1e30, 1e20, 1),
            (1e30, 1e30, 0, 1e-30),
            (2, 3e38, 0, 2),
            (1e20, 1e20, 0, 1e20),
            (3e38, 3e38, 2, 3e38),
            (1e30, 1e30, 1e30, 0),
            (0, 1e-30, 0, 1e-30),
            (2, 1, 2, 1),
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(3862)
    random = np.ldexp(rng.uniform(0.5, 1, (512, 4)), rng.integers(-125, 128, (512, 4)))
    random[:, [0, 2]] *= rng.choice([-1, 1], (512, 2))
    cells = np.concatenate([cells, random.astype(np.float32)])
    u, sigma, y, w = cells.T.astype(np.float64)
    expected = np.where(w > 0, (u - sigma * y) * w / (sigma + w), 0)
    conditioning = np.abs(u * w / (sigma + w)) + np.abs(sigma * y * w / (sigma + w))
    limits = np.finfo(np.float32)
    selected = (np.abs(expected) <= limits.max) & (
        (np.abs(expected) >= limits.tiny) | (expected == 0)
    )
    actual = np.asarray(jax.jit(jax.vmap(_prox_cell))(jnp.asarray(cells)), dtype=np.float64)
    assert np.isfinite(actual[selected]).all()
    error = np.abs(actual[selected] - expected[selected])
    assert np.all(error <= 1e-6 * np.maximum(conditioning[selected], limits.tiny))


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1, 1e-13, 1e-30])
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
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-37)
    direction = jnp.asarray([0.7, -0.13, 0.23, 0.11])
    jvp = jax.jit(jax.vmap(lambda z: jax.jvp(_prox_cell, (z,), (direction,))[1]))(values)
    expected_jvp = expected @ np.asarray(direction, dtype=np.float64)
    condition = np.abs(expected) @ np.abs(np.asarray(direction, dtype=np.float64))
    assert np.all(np.abs(np.asarray(jvp) - expected_jvp) <= 2e-6 * np.maximum(condition, 1e-37))


@pytest.mark.numerical
def test_weighted_data_proximal_guard_boundaries_preserve_value_and_derivatives():
    limit = np.float32(2**61)
    bounds = [np.nextafter(limit, np.float32(0)), limit, np.nextafter(limit, np.float32(np.inf))]
    for bound in bounds:
        cell = jnp.asarray([0, bound, 0, bound], dtype=jnp.float32)
        expected = [0.5, 0, -float(bound) / 2, 0]
        np.testing.assert_allclose(jax.jit(jax.grad(_prox_cell))(cell), expected, rtol=2e-6)
        assert float(jax.jit(_prox_cell)(cell)) == 0


@pytest.mark.numerical
@pytest.mark.parametrize("sigma", [0.2, 1e-30, 1e30])
def test_unweighted_data_proximal_is_the_same_as_unit_weights(sigma):
    u = jnp.asarray([0, 1, 2, 1e30], dtype=jnp.float32)
    y = jnp.asarray([0, 2, 2, 1e20], dtype=jnp.float32)
    implicit = jax.jit(lambda x: _prox_fstar_l2(x, sigma, y, None))(u)
    explicit = jax.jit(lambda x: _prox_fstar_l2(x, sigma, y, jnp.ones_like(x)))(u)
    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1, 1e-13, 1e-20, 1e-30])
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
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-37)
