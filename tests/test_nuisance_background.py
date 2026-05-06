from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
import jax.numpy as jnp
import numpy as np

from tomojax.nuisance import BackgroundOffsetModel, estimate_background_offset


def test_background_offset_zero_model_applies_no_change() -> None:
    projections = jnp.arange(48, dtype=jnp.float32).reshape((2, 4, 6))

    model = BackgroundOffsetModel.zeros(2)

    np.testing.assert_allclose(model.apply(projections), projections)
    assert model.to_dict() == {
        "schema": "tomojax.background_offset_model.v1",
        "basis": "constant_plus_vertical_gradient",
        "constant": [0.0, 0.0],
        "vertical_gradient": [0.0, 0.0],
    }


def test_estimate_background_offset_recovers_constant_and_gradient() -> None:
    predicted = jnp.zeros((2, 5, 4), dtype=jnp.float32)
    true_model = BackgroundOffsetModel(
        constant=jnp.asarray([0.15, -0.08], dtype=jnp.float32),
        vertical_gradient=jnp.asarray([0.04, -0.02], dtype=jnp.float32),
    )
    observed = true_model.apply(predicted)

    fitted = estimate_background_offset(predicted, observed)

    np.testing.assert_allclose(fitted.constant, true_model.constant, atol=2.0e-6)
    np.testing.assert_allclose(
        fitted.vertical_gradient,
        true_model.vertical_gradient,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(fitted.apply(predicted), observed, atol=2.0e-6)


def test_estimate_background_offset_respects_detector_mask() -> None:
    predicted = jnp.zeros((1, 5, 4), dtype=jnp.float32)
    true_model = BackgroundOffsetModel(
        constant=jnp.asarray([0.3], dtype=jnp.float32),
        vertical_gradient=jnp.asarray([0.1], dtype=jnp.float32),
    )
    observed = true_model.apply(predicted)
    observed = observed.at[:, :, -1].set(100.0)
    mask = jnp.ones((5, 4), dtype=jnp.float32).at[:, -1].set(0.0)

    fitted = estimate_background_offset(predicted, observed, mask=mask)

    np.testing.assert_allclose(fitted.constant, true_model.constant, atol=2.0e-6)
    np.testing.assert_allclose(fitted.vertical_gradient, true_model.vertical_gradient, atol=2.0e-6)


def test_estimate_background_offset_uses_zero_for_empty_mask_view() -> None:
    predicted = jnp.zeros((2, 3, 3), dtype=jnp.float32)
    observed = jnp.ones_like(predicted)
    mask = jnp.ones_like(predicted).at[1].set(0.0)

    fitted = estimate_background_offset(predicted, observed, mask=mask)

    assert float(fitted.constant[1]) == 0.0
    assert float(fitted.vertical_gradient[1]) == 0.0
