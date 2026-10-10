"""Pseudo-Huber projection objectives retain values and derivatives across scales."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.forward.api import pseudo_huber_loss, pseudo_huber_weights

pytestmark = pytest.mark.numerical


@pytest.mark.parametrize(
    ("residual", "delta"),
    [
        (0.0, 1.0),
        (1e-4, 1.0),
        (1e-10, 1.0),
        (1.0, 1e30),
        (1e-10, 1e30),
        (1e10, 1e30),
        (1e20, 1.0),
        (1e30, 1e-30),
        (1e-10, 1e-10),
        (1e10, 1e10),
        (1e20, 1e-10),
    ],
)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_pseudo_huber_matches_extended_precision_reference(residual, delta, sign) -> None:
    r = jnp.asarray(sign * residual, jnp.float32)
    d = float(np.float32(delta))
    rr, dd = np.longdouble(float(r)), np.longdouble(d)
    length = np.hypot(rr, dd)
    expected_value = float(dd * rr**2 / (length + dd))
    expected_weight = float(dd / length)
    expected_gradient = float(rr * dd / length)
    objective = functools.partial(pseudo_huber_loss, delta=d)
    value = jax.jit(objective)(r)
    gradient = jax.jit(jax.grad(objective))(r)
    weight = jax.jit(functools.partial(pseudo_huber_weights, delta=d))(r)
    np.testing.assert_allclose(value, expected_value, rtol=4e-6, atol=0.0)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=4e-6, atol=0.0)
    # A mathematically subnormal weight may be flushed to zero on the device.
    if expected_weight >= np.finfo(np.float32).tiny:
        np.testing.assert_allclose(weight, expected_weight, rtol=4e-6, atol=0.0)


@pytest.mark.parametrize("delta", [1e-30, 1e-20, 1.0, 1e20, 1e30])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 1.0, 2.0, -2.0])
def test_pseudo_huber_residual_hessian_is_scale_independent(delta, ratio) -> None:
    d = float(np.float32(delta))
    r = jnp.asarray(ratio * d, jnp.float32)
    ratio = float(r) / d
    expected = (1 + ratio**2) ** -1.5
    objective = functools.partial(pseudo_huber_loss, delta=d)
    actual = jax.jit(jax.grad(jax.grad(objective)))(r)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=0.0)


@pytest.mark.parametrize("scale", [1e-30, 1e-20, 1.0, 1e20, 1e30])
def test_pseudo_huber_delta_derivative_matches_reference(scale: float) -> None:
    r = jnp.asarray([0.0, 0.25, 1.0, 2.0, -2.0], jnp.float32) * scale
    d = jnp.full_like(r, scale)
    ratio = np.asarray(r, np.float64) / np.asarray(d, np.float64)
    expected = scale * (np.hypot(1.0, ratio) - 1.0) ** 2 / np.hypot(1.0, ratio)
    derivative = jax.jit(jax.vmap(jax.grad(lambda d, x: pseudo_huber_loss(x, delta=d))))
    np.testing.assert_allclose(derivative(d, r), expected, rtol=4e-6, atol=0.0)


@pytest.mark.parametrize("gain", [1e-4, 1.0, 1e20])
def test_pseudo_huber_projection_matching_recovers_known_subpixel_pose(gain: float) -> None:
    v, u = jnp.meshgrid(jnp.arange(-16, 16), jnp.arange(-24, 24), indexing="ij")

    def image(shift: jax.Array) -> jax.Array:
        return gain * jnp.exp(-0.5 * (((u - shift) / 3.0) ** 2 + (v / 4.0) ** 2))

    truth = jnp.asarray(0.35, jnp.float32)
    observed = image(truth)
    shifts = jnp.linspace(-1.2, 1.2, 49)
    costs = jax.jit(jax.vmap(lambda shift: jnp.sum(pseudo_huber_loss(image(shift) - observed))))(
        shifts
    )
    recovered = shifts[jnp.argmin(costs)]
    np.testing.assert_allclose(recovered, truth, rtol=0, atol=1e-6)
    assert np.isfinite(costs).all()
