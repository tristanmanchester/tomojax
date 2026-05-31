from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
