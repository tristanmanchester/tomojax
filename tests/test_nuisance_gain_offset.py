from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
import jax.numpy as jnp
import numpy as np

from tomojax.nuisance import GainOffsetModel, estimate_gain_offset


def test_gain_offset_identity_applies_no_change() -> None:
    projections = jnp.arange(24, dtype=jnp.float32).reshape((2, 3, 4))

    model = GainOffsetModel.identity(2)

    np.testing.assert_allclose(model.apply(projections), projections)
    assert model.to_dict() == {
        "schema": "tomojax.gain_offset_model.v1",
        "gain": [1.0, 1.0],
        "offset": [0.0, 0.0],
    }


def test_estimate_gain_offset_recovers_per_view_affine_drift() -> None:
    predicted = jnp.asarray(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[1.5, 0.5, 3.0], [2.0, 4.5, 6.0]],
        ],
        dtype=jnp.float32,
    )
    true_model = GainOffsetModel(
        gain=jnp.asarray([1.08, 0.94], dtype=jnp.float32),
        offset=jnp.asarray([0.15, -0.08], dtype=jnp.float32),
    )
    observed = true_model.apply(predicted)

    fitted = estimate_gain_offset(predicted, observed)

    np.testing.assert_allclose(fitted.gain, true_model.gain, atol=2.0e-6)
    np.testing.assert_allclose(fitted.offset, true_model.offset, atol=2.0e-6)
    raw_residual = observed - predicted
    corrected_residual = observed - fitted.apply(predicted)
    assert float(jnp.mean(corrected_residual * corrected_residual)) < 1.0e-10
    assert float(jnp.mean(corrected_residual * corrected_residual)) < float(
        jnp.mean(raw_residual * raw_residual)
    )


def test_estimate_gain_offset_respects_detector_mask() -> None:
    predicted = jnp.asarray([[[0.0, 1.0, 2.0, 100.0], [3.0, 4.0, 5.0, -50.0]]])
    observed = 1.2 * predicted + 0.25
    observed = observed.at[:, :, -1].set(jnp.asarray([[500.0, -500.0]], dtype=jnp.float32))
    mask = jnp.asarray([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]], dtype=jnp.float32)

    fitted = estimate_gain_offset(predicted, observed, mask=mask)
    fitted_residual = (observed - fitted.apply(predicted)) * mask[None, :, :]

    np.testing.assert_allclose(fitted.gain, jnp.asarray([1.2], dtype=jnp.float32), atol=2.0e-6)
    np.testing.assert_allclose(fitted.offset, jnp.asarray([0.25], dtype=jnp.float32), atol=2.0e-6)
    assert float(jnp.mean(fitted_residual * fitted_residual)) < 1.0e-10


def test_estimate_gain_offset_uses_identity_for_empty_mask_view() -> None:
    predicted = jnp.ones((2, 2, 2), dtype=jnp.float32)
    observed = jnp.full((2, 2, 2), 3.0, dtype=jnp.float32)
    mask = jnp.ones_like(predicted).at[1].set(0.0)

    fitted = estimate_gain_offset(predicted, observed, mask=mask)

    assert float(fitted.gain[1]) == 1.0
    assert float(fitted.offset[1]) == 0.0
