from __future__ import annotations

import numpy as np
import pytest

from tomojax.nuisance import estimate_background_offset


@pytest.mark.parametrize("ridge", [0.0, 1e-6, 0.05])
@pytest.mark.parametrize("weight_scale", [1.0, 1e-4, 1e-8])
def test_background_weighted_fit_matches_dense_reference(ridge: float, weight_scale: float) -> None:
    rng = np.random.default_rng(42)
    predicted = rng.normal(size=(2, 4, 3)).astype(np.float32)
    observed = rng.normal(size=predicted.shape).astype(np.float32)
    weights = rng.uniform(0.0, 1.0, predicted.shape).astype(np.float32) * weight_scale
    ramp = np.broadcast_to(np.linspace(-1.0, 1.0, 4, dtype=np.float32)[:, None], (4, 3))
    design = np.column_stack([np.ones(12), ramp.reshape(-1)])
    reference = []
    for pred, obs, weight in zip(predicted, observed, weights, strict=True):
        sqrt_w = np.sqrt(weight.reshape(-1).astype(np.float64))
        fitted, *_ = np.linalg.lstsq(
            np.vstack([design * sqrt_w[:, None], np.sqrt(ridge) * np.eye(2)]),
            np.concatenate([(obs - pred).reshape(-1).astype(np.float64) * sqrt_w, np.zeros(2)]),
            rcond=None,
        )
        reference.append(fitted)

    fitted = estimate_background_offset(predicted, observed, mask=weights, ridge=ridge)

    np.testing.assert_allclose(np.asarray(fitted.constant), np.asarray(reference)[:, 0], rtol=2e-6)
    np.testing.assert_allclose(
        np.asarray(fitted.vertical_gradient), np.asarray(reference)[:, 1], rtol=2e-6
    )


@pytest.mark.parametrize("ridge", [0.0, 1e-6])
@pytest.mark.parametrize("rows", [1, 4])
def test_background_single_observed_row_has_minimum_norm_fit(rows: int, ridge: float) -> None:
    predicted = np.zeros((1, rows, 3), np.float32)
    observed = np.full_like(predicted, 5.0)
    weights = np.zeros_like(predicted)
    row = rows // 2
    weights[0, row] = [0.1, 0.3, 0.2]
    ramp = 0.0 if rows == 1 else float(np.linspace(-1.0, 1.0, rows, dtype=np.float32)[row])
    design = np.tile([1.0, ramp], (3, 1)) * np.sqrt(weights[0, row].astype(np.float64))[:, None]
    reference, *_ = np.linalg.lstsq(
        np.vstack([design, np.sqrt(ridge) * np.eye(2)]),
        np.append(5.0 * np.sqrt(weights[0, row].astype(np.float64)), [0.0, 0.0]),
        rcond=None,
    )

    fitted = estimate_background_offset(predicted, observed, mask=weights, ridge=ridge)

    np.testing.assert_allclose(np.asarray(fitted.constant), reference[:1], rtol=2e-6)
    np.testing.assert_allclose(np.asarray(fitted.vertical_gradient), reference[1:], atol=2e-6)
    np.testing.assert_allclose(np.asarray(fitted.apply(predicted))[0, row], 5.0, rtol=3e-6)


def test_background_empty_view_and_masked_nonfinite_pixels() -> None:
    predicted = np.zeros((2, 3, 2), np.float32)
    observed = np.asarray([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]] * 2, np.float32)
    weights = np.asarray([[[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]], [[0.0, 0.0]] * 3], np.float32)
    predicted[:, :, 0] = np.nan
    observed[:, :, 0] = np.inf

    fitted = estimate_background_offset(predicted, observed, mask=weights, ridge=0.0)

    np.testing.assert_allclose(np.asarray(fitted.constant), [2.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(fitted.vertical_gradient), [1.0, 0.0], rtol=1e-6)
