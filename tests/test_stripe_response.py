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


@pytest.mark.parametrize("columns", [41, 1031])
def test_varying_response_directional_derivative(columns: int) -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(1, 0.1, (129, 1, columns))
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


@pytest.mark.parametrize("width", [15, 16, 51])
@pytest.mark.parametrize("columns", [1031, 2053])
@pytest.mark.parametrize("defect", ["none", "left", "right", "interior", "dense"])
def test_varying_sparse_response_matches_full_distribution_fit(width, defect, columns):
    values = np.random.default_rng(71).normal(1, 0.07, (129, columns)).astype(np.float32)
    where = {
        "none": slice(0, 0),
        "left": slice(0, 4),
        "right": slice(-4, None),
        "interior": slice(130, 134),
        "dense": slice(None, None, 97),
    }[defect]
    values[:, where] = 0.5 * values[:, where] + 0.2
    if defect == "interior":
        values[:, 131] = 0.4

    def median(x):
        half = width // 2
        padded = np.pad(x, ((0, 0), (half, half)), mode="edge")
        result = np.median(np.lib.stride_tricks.sliding_window_view(padded, width, axis=1), -1)
        return (result[:, :-1] + result[:, 1:]) * 0.5 if width % 2 == 0 else result

    noise = np.mean(np.abs(np.diff(values, axis=0)), axis=0)
    local = median(noise[None])[0]
    ratio = np.log(np.maximum(noise / local, np.finfo(values.dtype).eps))
    deviation = np.abs(ratio - np.median(ratio))
    scale = 1.4826 * np.median(deviation)
    bad = (deviation > 6 * scale) & (scale > 0)
    bad = np.convolve(bad, np.ones(2 * max(1, width // 8) + 1), mode="same") > 0
    ranked = np.sort(values, axis=0)
    target = median(ranked)
    xmean, ymean = ranked.mean(0), target.mean(0)
    dx, dy = ranked - xmean, target - ymean
    variance = np.sum(dx * dx, axis=0)
    gain = np.sum(dx * dy, axis=0) / np.where(variance > 0, variance, 1)
    gain = np.where(bad & (noise > 0) & (variance > 0), gain, 1)
    bias = np.where(bad, ymean - gain * xmean, 0)
    actual = jax.jit(Stripes(width, varying=True).apply)(values[:, None])[:, 0]
    np.testing.assert_allclose(actual, values * gain + bias, rtol=2e-6, atol=2e-6)
