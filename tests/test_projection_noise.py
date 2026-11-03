"""Sampled Haar statistics retain units, masks and bounded array access."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.recon._noise import estimate_projection_noise


@pytest.mark.parametrize("shape", [(5, 17, 19), (5, 1, 19), (5, 17, 1)])
def test_haar_noise_matches_full_statistic_and_units(shape):
    values = np.random.default_rng(39).normal(size=shape).astype(np.float32)
    hh = values.astype(float)
    for axis in (1, 2):
        if shape[axis] > 1:
            size = shape[axis] // 2 * 2
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[axis], b[axis] = slice(0, size, 2), slice(1, size, 2)
            hh = (hh[tuple(a)] - hh[tuple(b)]) / np.sqrt(2)
    expected = np.median(abs(hh - np.median(hh))) / 0.6744897501960817
    assert estimate_projection_noise(values) == pytest.approx(expected)
    assert float(estimate_projection_noise(jnp.asarray(values))) == pytest.approx(
        expected, rel=2e-6
    )
    assert estimate_projection_noise(3 * values + 2) == pytest.approx(3 * expected, rel=2e-6)


def test_noise_masks_whole_cells_and_broadcast_weights():
    values = np.random.default_rng(40).normal(size=(4, 16, 18)).astype(np.float32)
    weights = np.ones((1, 16, 18), np.float32)
    weights[:, :8] = 0
    values[:, :8] = np.nan
    expected = estimate_projection_noise(values[:, 8:])
    assert estimate_projection_noise(values, weights=weights) == pytest.approx(expected)
    actual = estimate_projection_noise(jnp.asarray(values), weights=jnp.asarray(weights))
    assert float(actual) == pytest.approx(expected, rel=2e-6)
    compiled = jax.jit(lambda w: estimate_projection_noise(values, weights=w))
    assert float(compiled(jnp.asarray(weights))) == pytest.approx(expected, rel=2e-6)
    with pytest.raises(jax.errors.JaxRuntimeError, match="pass tv_weight explicitly"):
        compiled(jnp.zeros_like(weights)).block_until_ready()
    # A successful callback replaces the failed asynchronous runtime token.
    assert float(compiled(jnp.asarray(weights))) == pytest.approx(expected, rel=2e-6)
    jax.effects_barrier()
    with pytest.raises(ValueError, match="pass tv_weight explicitly"):
        estimate_projection_noise(values, weights=0)
    with pytest.raises(ValueError, match="pass tv_weight explicitly"):
        estimate_projection_noise(np.zeros((3, 1, 1)))


def test_sampled_noise_is_deterministic_and_frozen_under_jit():
    values = np.random.default_rng(41).normal(size=(13, 145, 147)).astype(np.float32)
    expected = estimate_projection_noise(values)
    assert expected == pytest.approx(1, rel=0.03)
    compiled = jax.jit(estimate_projection_noise)
    assert float(compiled(jnp.asarray(values))) == pytest.approx(expected, rel=2e-6)
    derivative = jax.grad(estimate_projection_noise)(jnp.asarray(values))
    np.testing.assert_array_equal(derivative, np.zeros_like(values))


def test_noise_never_materializes_a_large_stack():
    class SampleOnly:
        shape = (1000000, 2048, 2048)
        dtype = np.dtype(np.float32)

        def __getitem__(self, indices):
            assert all(index.shape == (65536,) for index in indices)
            return np.sin(indices[0] * 3 + indices[1] * 7 + indices[2] * 5)

        def __array__(self, *args, **kwargs):
            raise AssertionError("the whole input must not be copied")

    assert np.isfinite(estimate_projection_noise(SampleOnly()))
