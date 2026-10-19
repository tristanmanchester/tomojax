"""The shared transformation contract, independent of either CUDA projector."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.projector_autodiff import FirstOrderProjector
from tomojax.geometry import Detector, Grid


def _reference(c, x):
    return c[:, 0, None, None] * x[0] + c[:, 1, None, None] ** 2 * x[1]


class _DenseProjector(FirstOrderProjector):
    def expand(self, c, x, *cache):
        return _reference(c, cache[0] if cache else x)

    def coefficient_jvp(self, c, x, dc, dx):
        result = dc[:, 0, None, None] * x[0] + 2 * (c * dc)[:, 1, None, None] * x[1]
        return result if dx is None else result + _reference(c, dx)

    def coefficient_vjp(self, c, x, dy):
        return jnp.stack([jnp.sum(dy * x[0], (1, 2)), 2 * c[:, 1] * jnp.sum(dy * x[1], (1, 2))], 1)

    def adjoint(self, c, dy):
        return jnp.stack(
            [jnp.sum(dy * c[:, 0, None, None], 0), jnp.sum(dy * c[:, 1, None, None] ** 2, 0)]
        )


def _apply(c, x, *cache):
    args = (c, x, *cache)
    return _DenseProjector(
        tuple(jax.typeof(a) for a in args),
        grid=Grid(2, 3, 4, 1, 1, 1),
        detector=Detector(4, 3, 1, 1),
    )(*args)


@pytest.mark.parametrize("packed", [False, True])
def test_first_order_transformations_match_independent_dense_ad(packed):
    c = jnp.arange(10, dtype=jnp.float32).reshape(5, 2) / 20
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 20
    dc, dx = jnp.full_like(c, 0.2), jnp.full_like(x, 0.1)
    dy = jnp.arange(60, dtype=jnp.float32).reshape(5, 3, 4) / 30

    def project(c, x):
        return _apply(c, x, jax.lax.stop_gradient(x)) if packed else _apply(c, x)

    actual = jax.jit(lambda c, x, dc, dx: jax.jvp(project, (c, x), (dc, dx)))(c, x, dc, dx)
    expected = jax.jvp(_reference, (c, x), (dc, dx))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
    pullback = jax.vjp(project, c, x)[1](dy)
    expected_pullback = jax.vjp(_reference, c, x)[1](dy)
    _, linear = jax.linearize(project, c, x)
    for found in (pullback, jax.linear_transpose(linear, dc, dx)(dy)):
        for a, b in zip(found, expected_pullback, strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(linear(dc, dx), expected[1], rtol=2e-6, atol=1e-6)
    directions = jnp.stack([dc, 2 * dc])
    mapped = jax.jit(jax.vmap(lambda d: linear(d, dx)))(directions)

    def reference_jvp(d):
        return jax.jvp(_reference, (c, x), (d, dx))[1]

    np.testing.assert_allclose(mapped, jax.jit(jax.vmap(reference_jvp))(directions), rtol=2e-6)
    inputs = (jnp.stack([c, c + 1]), jnp.stack([x, x + 1]))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(project))(*inputs), jax.jit(jax.vmap(_reference))(*inputs)
    )


@pytest.mark.parametrize("packed", [False, True])
def test_volume_pullback_does_not_keep_primal_image_or_derived_cache(packed):
    c = jnp.arange(10, dtype=jnp.float32).reshape(5, 2) / 20
    x = jnp.ones((2, 3, 4), jnp.float32)
    cache = x.copy() if packed else None
    output, pullback = jax.vjp(lambda v: _apply(c, v, cache) if packed else _apply(c, v), x)
    output.block_until_ready()
    x.delete()
    if cache is not None:
        cache.delete()
    dy = jnp.ones((5, 3, 4), jnp.float32)
    expected = jax.vjp(lambda v: _reference(c, v), jnp.zeros((2, 3, 4), jnp.float32))[1](dy)
    np.testing.assert_allclose(jax.jit(pullback)(dy)[0], expected[0], rtol=2e-6)
