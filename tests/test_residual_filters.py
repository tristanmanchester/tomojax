from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
import jax.numpy as jnp
import numpy as np

from tomojax.forward import (
    ResidualFilterConfig,
    apply_residual_filter,
    apply_residual_filter_schedule,
)


def test_raw_residual_filter_is_identity_and_respects_mask() -> None:
    residual = jnp.arange(9, dtype=jnp.float32).reshape(1, 3, 3)
    mask = jnp.ones_like(residual).at[:, 1, 1].set(0.0)

    filtered = apply_residual_filter(
        residual,
        ResidualFilterConfig(kind="raw"),
        mask=mask,
    )

    expected = np.array(residual)
    expected[:, 1, 1] = 0.0
    np.testing.assert_allclose(np.asarray(filtered), expected)


def test_lowpass_gaussian_preserves_sum_and_spreads_impulse() -> None:
    residual = jnp.zeros((1, 7, 7), dtype=jnp.float32).at[:, 3, 3].set(1.0)

    filtered = apply_residual_filter(
        residual,
        ResidualFilterConfig(kind="lowpass_gaussian", sigma_px=1.0),
    )

    assert float(filtered[0, 3, 3]) < 1.0
    assert float(filtered[0, 3, 3]) > float(filtered[0, 0, 0])
    np.testing.assert_allclose(float(jnp.sum(filtered)), 1.0, rtol=1e-6)


def test_bandpass_difference_of_gaussians_has_zero_mean_for_impulse() -> None:
    residual = jnp.zeros((1, 9, 9), dtype=jnp.float32).at[:, 4, 4].set(1.0)

    filtered = apply_residual_filter(
        residual,
        ResidualFilterConfig(
            kind="bandpass_difference_of_gaussians",
            sigma_px=0.8,
            outer_sigma_px=1.6,
        ),
    )

    assert float(filtered[0, 4, 4]) > 0.0
    np.testing.assert_allclose(float(jnp.sum(filtered)), 0.0, atol=5e-7)


def test_filter_schedule_sums_weighted_components() -> None:
    residual = jnp.zeros((1, 5, 5), dtype=jnp.float32).at[:, 2, 2].set(1.0)
    configs = (
        ResidualFilterConfig(kind="lowpass_gaussian", weight=0.7, sigma_px=1.0),
        ResidualFilterConfig(
            kind="bandpass_difference_of_gaussians",
            weight=0.3,
            sigma_px=0.8,
            outer_sigma_px=1.6,
        ),
    )

    result = apply_residual_filter_schedule(residual, configs)

    assert result.configs == configs
    assert len(result.components) == 2
    np.testing.assert_allclose(
        np.asarray(result.residual),
        np.asarray(result.components[0] + result.components[1]),
    )
