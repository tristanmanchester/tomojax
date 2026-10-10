"""Independent inverse-coordinate and derivative checks for alignment bounds."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.alignment.optimizers import BoundTransform


@pytest.mark.numerical
@pytest.mark.parametrize("upper_only", [False, True])
@pytest.mark.parametrize("distance", [2e-7, 1e-5, 0.01, 1.0, 20.0, 30.0, 90.0, 100.0, 1e20])
def test_one_sided_bound_inverse_coordinates_and_derivatives(distance, upper_only):
    sign = -1.0 if upper_only else 1.0
    transform = BoundTransform.from_bounds(
        jnp.asarray([-jnp.inf if upper_only else 0.0]),
        jnp.asarray([0.0 if upper_only else jnp.inf]),
        value_shape=(1,),
    )
    values = jnp.asarray([sign * distance], dtype=jnp.float32)
    distance64 = abs(float(np.asarray(values)[0]))
    clipped = max(distance64, float(np.float32(1e-6)))
    denominator = -np.expm1(-clipped)
    expected_z = clipped + np.log(denominator)
    expected_grad = sign / denominator if distance64 > float(np.float32(1e-6)) else 0.0
    expected_hess = -np.exp(-clipped) / denominator**2 if distance64 >= clipped else 0.0

    inverse = jax.jit(transform.to_unconstrained)
    np.testing.assert_allclose(inverse(values), [expected_z], rtol=2e-6, atol=2e-6)
    for derivative in (jax.jacfwd(inverse), jax.jacrev(inverse)):
        np.testing.assert_allclose(derivative(values), [[expected_grad]], rtol=3e-6, atol=1e-7)
    hessian = jax.jit(jax.hessian(lambda x: jnp.sum(inverse(x))))(values)
    np.testing.assert_allclose(hessian, [[expected_hess]], rtol=4e-6, atol=1e-7)
    np.testing.assert_allclose(
        transform.from_unconstrained(inverse(values)), [sign * clipped], rtol=3e-6, atol=1e-7
    )


@pytest.mark.numerical
@pytest.mark.parametrize("unbounded_value", [-1e20, -100.0, 0.0, 100.0, 1e20])
def test_mixed_bound_inverse_keeps_inactive_branches_out_of_derivatives(unbounded_value):
    transform = BoundTransform.from_bounds(
        jnp.asarray([-jnp.inf, 0.0, -jnp.inf, -2.0, 3.0]),
        jnp.asarray([jnp.inf, jnp.inf, 0.0, 6.0, 3.0]),
        value_shape=(5,),
    )
    values = jnp.asarray([unbounded_value, 100.0, -100.0, 1.0, 3.0])
    expected_z = [unbounded_value, 100.0, 100.0, np.log(3.0 / 5.0), 0.0]
    expected_jac = np.diag([1.0, 1.0, -1.0, 8.0 / 15.0, 0.0])
    expected_hess = np.diag([0.0, 0.0, 0.0, -1.0 / 9.0 + 1.0 / 25.0, 0.0])
    inverse = jax.jit(transform.to_unconstrained)
    np.testing.assert_allclose(inverse(values), expected_z, rtol=2e-6, atol=1e-6)
    for derivative in (jax.jacfwd(inverse), jax.jacrev(inverse)):
        np.testing.assert_allclose(derivative(values), expected_jac, rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(
        jax.hessian(lambda x: jnp.sum(inverse(x)))(values), expected_hess, rtol=4e-6, atol=1e-7
    )
    np.testing.assert_allclose(transform.from_unconstrained(inverse(values)), values, rtol=2e-6)
