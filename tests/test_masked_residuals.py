"""Missing detector pixels do not contaminate masked values or derivatives."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.forward.api import masked_whitened_residual, residual_loss

pytestmark = pytest.mark.numerical


def _masked_inputs() -> tuple[jax.Array, ...]:
    predicted = jnp.asarray([[1.0, np.nan, np.inf], [np.nan, 2.0, -np.inf]])
    observed = jnp.asarray([[0.0, np.nan, -np.inf], [np.inf, 0.0, np.nan]])
    mask = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    sigma = jnp.asarray([[2.0, 0.0, np.nan], [np.inf, 4.0, -np.inf]])
    return predicted, observed, mask, sigma


def test_masked_whitening_excludes_invalid_pixels_before_arithmetic() -> None:
    predicted, observed, mask, sigma = _masked_inputs()
    actual = jax.jit(masked_whitened_residual)(predicted, observed, mask=mask, sigma=sigma)
    np.testing.assert_array_equal(actual, [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0]])


@pytest.mark.parametrize("mode", ["l2", "pseudo_huber"])
def test_masked_loss_matches_only_the_available_pixels(mode: str) -> None:
    predicted, observed, mask, sigma = _masked_inputs()
    operation = functools.partial(residual_loss, mask=mask, sigma=sigma, mode=mode)
    actual = jax.jit(operation)(predicted, observed)
    safe_sigma = jnp.where(mask != 0, sigma, 1.0)
    reference = residual_loss(
        jnp.where(mask != 0, predicted, 0.0),
        jnp.where(mask != 0, observed, 0.0),
        mask=mask,
        sigma=safe_sigma,
        mode=mode,
    )
    np.testing.assert_array_equal(actual.residual, reference.residual)
    np.testing.assert_allclose(actual.loss, reference.loss, rtol=1e-6)
    np.testing.assert_allclose(actual.weights, reference.weights, rtol=1e-6)
    assert float(actual.valid_count) == 1.5


def test_masked_whitening_has_zero_derivatives_at_missing_pixels() -> None:
    predicted, observed, mask, sigma = _masked_inputs()

    def total(pred, obs, noise):
        return jnp.sum(masked_whitened_residual(pred, obs, mask=mask, sigma=noise))

    gradients = jax.jit(jax.grad(total, argnums=(0, 1, 2)))(predicted, observed, sigma)
    expected_pred = np.asarray([[0.5, 0.0, 0.0], [0.0, 0.125, 0.0]])
    expected_sigma = np.asarray([[-0.25, 0.0, 0.0], [0.0, -0.0625, 0.0]])
    for actual, expected in zip(
        gradients, (expected_pred, -expected_pred, expected_sigma), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    _, tangent = jax.jvp(
        lambda pred: masked_whitened_residual(pred, observed, mask=mask, sigma=sigma),
        (predicted,),
        (jnp.ones_like(predicted),),
    )
    np.testing.assert_array_equal(tangent, expected_pred)


@pytest.mark.parametrize("mode", ["l2", "pseudo_huber"])
def test_all_missing_pixels_give_zero_loss_and_gradient(mode: str) -> None:
    predicted = jnp.full((2, 3), np.nan)
    observed = jnp.full_like(predicted, np.inf)
    mask = jnp.zeros_like(predicted)

    def objective(pred):
        return residual_loss(pred, observed, mask=mask, sigma=0.0, mode=mode).loss

    value, gradient = jax.jit(jax.value_and_grad(objective))(predicted)
    assert float(value) == 0.0
    np.testing.assert_array_equal(gradient, np.zeros_like(predicted))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_masked_whitening_does_not_hide_unmasked_invalid_data(bad: float) -> None:
    predicted = jnp.asarray([[bad, 1.0]])
    actual = masked_whitened_residual(
        predicted, jnp.zeros_like(predicted), mask=jnp.ones_like(predicted)
    )
    assert not np.isfinite(actual[0, 0])
