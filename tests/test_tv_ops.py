from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tomojax.recon._tv_ops import (
    div3,
    grad3,
    huber_tv_grad,
    huber_tv_value,
    isotropic_tv_value,
)


def test_grad3_and_div3_are_negative_adjoints():
    u = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    px = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 1.0) / 10.0
    py = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 2.0) / 11.0
    pz = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 3.0) / 12.0

    dx, dy, dz = grad3(u)
    lhs = jnp.vdot(dx, px) + jnp.vdot(dy, py) + jnp.vdot(dz, pz)
    rhs = -jnp.vdot(u, div3(px, py, pz))

    assert lhs == pytest.approx(float(rhs), rel=1e-6, abs=1e-6)


def test_div3_handles_singleton_axes():
    px = jnp.ones((1, 2, 3), dtype=jnp.float32)
    py = jnp.ones((1, 1, 3), dtype=jnp.float32)
    pz = jnp.ones((1, 2, 1), dtype=jnp.float32)

    div = div3(px, py, pz)

    assert div.shape == (1, 2, 3)
    assert np.isfinite(np.asarray(div)).all()


def test_huber_tv_value_is_finite_on_simple_arrays():
    arrays = [
        jnp.zeros((3, 3, 3), dtype=jnp.float32),
        jnp.arange(27, dtype=jnp.float32).reshape(3, 3, 3) / 10.0,
        jnp.asarray(
            [
                [[0.0, 0.1], [0.2, 0.3]],
                [[0.4, 0.5], [0.6, 0.7]],
            ],
            dtype=jnp.float32,
        ),
    ]

    for arr in arrays:
        value = huber_tv_value(arr, delta=0.1)
        assert np.isfinite(float(value))
        assert float(value) >= 0.0


def test_huber_tv_grad_matches_autodiff_gradient():
    u = jnp.asarray(
        [
            [[0.0, 0.2, 0.4], [0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
            [[0.3, 0.5, 0.7], [0.4, 0.6, 0.8], [0.5, 0.7, 0.9]],
            [[0.6, 0.8, 1.0], [0.7, 0.9, 1.1], [0.8, 1.0, 1.2]],
        ],
        dtype=jnp.float32,
    )
    delta = 0.5

    manual = huber_tv_grad(u, delta)
    autodiff = jax.grad(lambda v: huber_tv_value(v, delta))(u)

    np.testing.assert_allclose(np.asarray(manual), np.asarray(autodiff), rtol=1e-5, atol=1e-5)


def test_huber_tv_approaches_isotropic_tv_for_small_delta():
    u = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    u = u.at[1:3, 1:3, 1:3].set(1.0)

    huber_value = huber_tv_value(u, delta=1e-5)
    tv_value = isotropic_tv_value(u)

    assert float(tv_value) > 0.0
    assert float(huber_value) == pytest.approx(float(tv_value), rel=1e-4, abs=1e-4)


def test_huber_tv_is_quadratic_near_zero_gradients():
    u = jnp.arange(27, dtype=jnp.float32).reshape(3, 3, 3) * 1e-3
    delta = 1.0
    gx, gy, gz = grad3(u)
    expected = 0.5 * jnp.sum(gx * gx + gy * gy + gz * gz) / delta

    assert float(huber_tv_value(u, delta)) == pytest.approx(float(expected), rel=1e-6, abs=1e-9)
