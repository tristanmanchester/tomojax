"""Masked normal-consistent MAD stays usable inside compiled forward models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.forward.api import robust_residual_scale

pytestmark = pytest.mark.numerical


def _reference(values, mask=None) -> float:
    selected = np.asarray(values, np.float64).ravel()
    if mask is not None:
        selected = selected[np.asarray(mask, bool).ravel()]
    if selected.size == 0:
        return 1.0
    with np.errstate(invalid="ignore", over="ignore"):
        scale = 1.4826 * np.median(np.abs(selected - np.median(selected)))
    return float(scale) if np.isfinite(scale) and scale > 0 else 1.0


@pytest.mark.parametrize("scale", [1e-30, 1e-6, 1.0, 1e20, 1e30])
def test_compiled_masked_mad_matches_independent_medians_for_dynamic_masks(scale) -> None:
    values = np.array([[8, -2, 0, 17], [1, 3, -5, 11]], np.float32) * np.float32(scale)
    masks = np.array(
        [
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 0, 1, 1], [0, 1, 1, 0]],
            [[0, 1, 0, 1], [1, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        bool,
    )
    compiled = jax.jit(robust_residual_scale)
    for mask in masks:
        actual = compiled(values, mask=mask)
        np.testing.assert_allclose(actual, _reference(values, mask), rtol=1e-6)
    batched = jax.jit(jax.vmap(lambda mask: robust_residual_scale(values, mask=mask)))(masks)
    expected = [_reference(values, mask) for mask in masks]
    np.testing.assert_allclose(batched, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [3.0],
        [0.0, 0.0],
        [np.nan, 1.0],
        [np.inf, 1.0],
        [np.inf, 1.0, 2.0, 3.0, 4.0],
        [-np.inf, 1.0, 2.0, 3.0, 4.0],
        [np.inf, np.inf, np.inf, 3.0, 4.0],
    ],
)
@pytest.mark.parametrize("masked", [False, True])
def test_mad_preserves_empty_degenerate_and_active_nonfinite_fallback(values, masked) -> None:
    array = jnp.asarray(values, jnp.float32)
    mask = jnp.ones_like(array, bool) if masked else None
    actual = jax.jit(robust_residual_scale)(array, mask=mask)
    assert float(actual) == pytest.approx(_reference(values), rel=1e-6)


def test_missing_nonfinite_values_have_no_effect_on_mad_or_its_gradient() -> None:
    values = jnp.asarray([2.0, np.nan, 5.0, np.inf, 11.0, -np.inf])
    mask = jnp.asarray([1, 0, 1, 0, 1, 0], bool)
    value, gradient = jax.jit(jax.value_and_grad(lambda r: robust_residual_scale(r, mask=mask)))(
        values
    )
    np.testing.assert_allclose(value, _reference(values, mask), rtol=1e-6)
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(np.asarray(gradient)[~np.asarray(mask)], 0.0)
    direction = np.array([0.3, 0.0, -0.2, 0.0, 0.1, 0.0], np.float32)
    epsilon = 1e-3
    finite_difference = (
        _reference(np.asarray(values) + epsilon * direction, mask)
        - _reference(np.asarray(values) - epsilon * direction, mask)
    ) / (2 * epsilon)
    assert float(jnp.vdot(gradient, direction)) == pytest.approx(finite_difference, rel=1e-3)


def test_all_missing_mad_has_zero_gradient() -> None:
    values = jnp.full((2, 3), np.nan)
    value, gradient = jax.jit(
        jax.value_and_grad(lambda r: robust_residual_scale(r, mask=jnp.zeros_like(r, bool)))
    )(values)
    assert float(value) == 1.0
    np.testing.assert_array_equal(gradient, np.zeros((2, 3)))


def test_mad_rejects_a_wrong_shaped_mask_before_compilation() -> None:
    with pytest.raises(ValueError, match="mask must match residual shape"):
        jax.jit(robust_residual_scale)(jnp.zeros((2, 3)), mask=jnp.zeros((6,), bool))
