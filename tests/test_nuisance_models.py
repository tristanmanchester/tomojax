from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.nuisance import (
    BackgroundOffsetModel,
    GainOffsetModel,
    estimate_background_offset,
    estimate_gain_offset,
)


def test_gain_offset_model_identity_and_per_view_fit() -> None:
    predicted = jnp.asarray(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ],
        dtype=jnp.float32,
    )
    observed = predicted * jnp.asarray([2.0, 0.5], dtype=jnp.float32).reshape((2, 1, 1))
    observed = observed + jnp.asarray([1.0, -2.0], dtype=jnp.float32).reshape((2, 1, 1))

    identity = GainOffsetModel.identity(2)
    np.testing.assert_allclose(np.asarray(identity.apply(predicted)), np.asarray(predicted))

    fitted = estimate_gain_offset(predicted, observed, ridge=0.0)
    np.testing.assert_allclose(np.asarray(fitted.gain), [2.0, 0.5], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(fitted.offset), [1.0, -2.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(fitted.apply(predicted)), np.asarray(observed), rtol=1e-6)


@pytest.mark.parametrize("ridge", [0.0, 0.03])
@pytest.mark.parametrize("weight_scale", [1.0, 0.001])
def test_gain_offset_weighted_fit_matches_dense_reference(
    ridge: float, weight_scale: float
) -> None:
    predicted = np.asarray([[[0.0, 1.0, 3.0], [2.0, -1.0, 4.0]]], np.float32)
    observed = np.asarray([[[0.3, 2.0, 6.2], [4.1, -2.2, 7.5]]], np.float32)
    weights = np.asarray([[0.2, 0.0, 0.7], [0.4, 0.8, 0.1]], np.float32) * weight_scale
    design = np.column_stack([predicted.reshape(-1), np.ones(predicted.size)])
    weighted_design = design * np.sqrt(weights.reshape(-1).astype(np.float64))[:, None]
    weighted_data = observed.reshape(-1) * np.sqrt(weights.reshape(-1).astype(np.float64))
    # The gain alone is regularized; the offset is an unpenalized intercept.
    reference, *_ = np.linalg.lstsq(
        np.vstack([weighted_design, [np.sqrt(ridge), 0.0]]),
        np.append(weighted_data, 0.0),
        rcond=None,
    )

    fitted = estimate_gain_offset(predicted, observed, mask=weights, ridge=ridge)

    np.testing.assert_allclose(np.asarray(fitted.gain), reference[:1], rtol=2e-6)
    np.testing.assert_allclose(np.asarray(fitted.offset), reference[1:], rtol=2e-6, atol=1e-6)


def test_gain_offset_fit_preserves_variation_on_large_background() -> None:
    predicted = np.arange(16, dtype=np.float32).reshape(1, 4, 4) + 10000.0
    observed = 2.0 * predicted + 1.0

    fitted = estimate_gain_offset(predicted, observed, ridge=0.0)

    np.testing.assert_allclose(np.asarray(fitted.gain), [2.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(fitted.offset), [1.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(fitted.apply(predicted)), observed, rtol=1e-6)


@pytest.mark.parametrize("ridge", [0.0, 1e-6])
def test_gain_offset_constant_prediction_and_empty_view_are_finite(ridge: float) -> None:
    predicted = np.full((2, 2, 2), 3.0, np.float32)
    observed = np.asarray([[[4.0, 6.0], [8.0, 10.0]], [[1.0, 2.0], [3.0, 4.0]]], np.float32)
    weights = np.asarray([[[0.1, 0.2], [0.3, 0.4]], [[0.0, 0.0], [0.0, 0.0]]], np.float32)

    fitted = estimate_gain_offset(predicted, observed, mask=weights, ridge=ridge)

    np.testing.assert_allclose(np.asarray(fitted.gain), [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(fitted.offset), [8.0, 0.0], atol=1e-6)


def test_gain_offset_zero_weight_excludes_nonfinite_pixels() -> None:
    predicted = np.asarray([[[0.0, 1.0], [np.nan, np.inf]]], np.float32)
    observed = np.asarray([[[1.0, 3.0], [np.inf, np.nan]]], np.float32)
    weights = np.asarray([[0.1, 0.2], [0.0, 0.0]], np.float32)

    fitted = estimate_gain_offset(predicted, observed, mask=weights, ridge=0.0)

    np.testing.assert_allclose(np.asarray(fitted.gain), [2.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(fitted.offset), [1.0], rtol=1e-6)


def test_background_offset_model_fits_constant_and_vertical_gradient() -> None:
    predicted = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    model = BackgroundOffsetModel(
        constant=jnp.asarray([1.0, -0.5], dtype=jnp.float32),
        vertical_gradient=jnp.asarray([0.25, 0.75], dtype=jnp.float32),
    )
    observed = model.apply(predicted)

    fitted = estimate_background_offset(predicted, observed, ridge=0.0)

    np.testing.assert_allclose(np.asarray(fitted.constant), [1.0, -0.5], atol=1e-6)
    np.testing.assert_allclose(np.asarray(fitted.vertical_gradient), [0.25, 0.75], atol=1e-6)
    np.testing.assert_allclose(np.asarray(fitted.apply(predicted)), np.asarray(observed), atol=1e-6)
