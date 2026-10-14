"""Selective detector-response correction without erasing a centred feature."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.corrections import Stripes, correct_projections


@pytest.mark.parametrize("width", [15, 16])
def test_varying_response_preserves_centred_absorption(width: int) -> None:
    rng = np.random.default_rng(14)
    theta = np.linspace(0, 2 * np.pi, 721, endpoint=False)[:, None, None]
    u = np.arange(161)[None, None, :] - 80
    truth = 1.2 * np.exp(-0.5 * ((u - 24 * np.cos(theta)) / 32) ** 2)
    truth += 0.6 * np.exp(-0.5 * ((u + 15 * np.sin(theta)) / 9) ** 2)
    truth = truth * np.array([1, 0.8])[None, :, None]
    measured = -np.log(rng.poisson(50000 * np.exp(-truth)) / 50000)
    measured[..., 37:40] = 0.65 * measured[..., 37:40] + 0.08
    measured = measured.astype(np.float32)
    correct = jax.jit(Stripes(width, varying=True).apply)
    actual = np.asarray(correct(measured))
    done = correct_projections(measured, (Stripes(width, varying=True),))
    np.testing.assert_array_equal(done.projections, actual)
    assert done.records[0].settings == {"width": width, "varying": True}
    assert np.linalg.norm(actual - truth) < 0.6 * np.linalg.norm(measured - truth)
    np.testing.assert_array_equal(actual[..., 60:100], measured[..., 60:100])
    feature = (0.15 * np.exp(-0.5 * (u / 1.5) ** 2)).astype(np.float32)
    difference = np.asarray(correct(measured + feature)) - actual
    np.testing.assert_allclose(difference, np.broadcast_to(feature, measured.shape), atol=2e-6)
    np.testing.assert_array_equal(correct(measured[..., ::-1])[..., ::-1], actual)
    for level in (0.0, 0.4):
        saturated = measured.copy()
        saturated[..., 38] = level
        repaired = np.asarray(correct(saturated))
        assert np.linalg.norm(repaired - truth) < np.linalg.norm(saturated - truth)
        assert np.max(np.ptp(repaired[..., 38], axis=0)) == 0
        np.testing.assert_allclose(repaired[..., 38].mean(0), truth[..., 38].mean(0), rtol=0.1)
    for views in (1, 2, 10):
        constant = np.ones((views, 2, 17), dtype=np.float32)
        np.testing.assert_array_equal(correct(constant), constant)


def test_varying_response_directional_derivative() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(1, 0.1, (129, 1, 41))
    values[..., 9:11] = 0.4 * values[..., 9:11] + 0.3
    direction = rng.normal(0, 0.1, values.shape)
    with jax.enable_x64():
        correct = jax.jit(Stripes(15, varying=True).apply)
        _, actual = jax.jit(lambda x, d: jax.jvp(correct, (x,), (d,)))(
            jnp.asarray(values), jnp.asarray(direction)
        )
        epsilon = 1e-6
        reference = correct(values + epsilon * direction) - correct(values - epsilon * direction)
        reference /= 2 * epsilon
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-7)
