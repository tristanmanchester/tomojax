"""Adaptive CUDA reference against independent float64 QR fits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.corrections._rings import _reference, _reference_jax

WINDOWS = (30, 20, 14, 9, 6)


def _qr_reference(values, gamma=2.0, local=64):
    values = np.asarray(values, np.float64)
    rows, n = values.shape
    columns = np.arange(n)
    taps = {}
    fits = {}
    for win in WINDOWS:
        offset = np.arange(-win, win + 1)
        keep = abs(offset) > 1
        x = offset[keep] / win
        design = np.stack([np.ones_like(x), x, x * x], axis=1)
        q, r = np.linalg.qr(design)
        kernel = np.zeros(2 * win + 1)
        kernel[keep] = np.linalg.solve(r, q.T)[0]
        taps[win] = kernel
    offset = np.arange(-6, 7)
    neighbors = values[:, np.clip(columns[:, None] + offset, 0, n - 1)]
    residual = values - np.sum(neighbors * taps[6], axis=-1)
    sigma = 1.4826 * np.median(abs(residual), axis=1, keepdims=True)
    u = residual / (4.685 * np.maximum(sigma, 1e-3 * np.median(sigma) + 1e-12))
    weights = 1e-3 + np.where(abs(u) < 1, (1 - u * u) ** 2, 0.0)
    for win in WINDOWS:
        offset = np.r_[np.arange(-win, -1), np.arange(2, win + 1)]
        x = offset / win
        design = np.stack([np.ones_like(x), x, x * x], axis=1)
        indices = np.clip(columns[:, None] + offset, 0, n - 1)
        fit = np.empty_like(values)
        for row in range(rows):
            root = np.sqrt(weights[row, indices])
            q, r = np.linalg.qr(root[..., None] * design)
            rhs = np.einsum("nki,nk->ni", q, root * values[row, indices])
            fit[row] = np.linalg.solve(r, rhs[..., None])[:, 0, 0]
        fits[win] = fit
    residual = values - fits[6]
    indices = np.clip(columns[:, None] + np.arange(-local, local + 1, 8), 0, n - 1)
    scatter = (
        1.4826 * np.median(abs(residual[:, indices]), axis=-1) / np.sqrt(1 + np.sum(taps[6] ** 2))
    )
    centered = residual - residual.mean(1, keepdims=True)
    variance = np.maximum(np.mean(centered * centered, 1, keepdims=True), np.finfo(float).tiny)
    rho = [
        np.clip(np.mean(centered[:, k:] * centered[:, :-k], 1, keepdims=True) / variance, 0, 1)
        for k in (1, 2)
    ]
    result = fits[30].copy()
    for win in WINDOWS[1:]:
        d = np.pad(taps[win], 30 - win) - taps[30]
        power = np.sum(d * d) + 2 * (
            rho[0] * np.sum(d[1:] * d[:-1]) + rho[1] * np.sum(d[2:] * d[:-2])
        )
        noise = scatter * np.sqrt(power)
        difference = fits[win] - result
        result += np.sign(difference) * np.maximum(abs(difference) - gamma * noise, 0)
    return result


@pytest.mark.gpu
@pytest.mark.parametrize("width", [31, 129, 8193])
@pytest.mark.parametrize("local", [16, 64])
def test_cuda_reference_against_independent_qr(width, local):
    rng = np.random.default_rng(606)
    x = np.linspace(-1, 1, width)
    values = (0.6 + 0.2 * np.cos(8 * x)[None] + rng.normal(0, 0.01, (3, width))).astype("f")
    values[0, width // 3 : width // 2] += 12
    values[1, ::23] += 0.3
    values[2] *= 0.001
    actual = np.asarray(jax.jit(lambda x: _reference(x, 2.0, local))(jnp.asarray(values)))
    expected = _qr_reference(values, local=local)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=3e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("views", [1, 2, 3, 32, 513, 8193])
def test_global_median_has_no_view_sized_shared_array(views):
    # Quiet rows fall below the global floor; a repeated-row control alone
    # would not detect an incorrect median because its floor never binds.
    pattern = np.linspace(0.1, 0.8, 31, dtype="f")
    pattern[15] += 0.2
    noisy = (2 * views + 2) // 3
    values = np.broadcast_to(pattern, (views, 31)).copy()
    values[noisy:] *= 1e-4
    actual = np.asarray(jax.jit(lambda x: _reference(x, 2.0, 16))(jnp.asarray(values)))
    # For >2 rows, the global median is the repeated noisy-row sigma.
    sample = values if views <= 2 else values[[0, 0, -1]]
    expected = _qr_reference(sample, local=16)
    np.testing.assert_allclose(
        actual[:noisy], np.broadcast_to(expected[0], actual[:noisy].shape), rtol=1e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        actual[noisy:], np.broadcast_to(expected[-1], actual[noisy:].shape), rtol=1e-5, atol=3e-10
    )


def test_float64_reference_keeps_jax_arithmetic():
    with jax.enable_x64():
        values = jnp.asarray(np.random.default_rng(606).normal(size=(3, 31)), dtype=jnp.float64)
        actual = jax.jit(_reference)(values)
        expected = jax.jit(_reference_jax)(values)
        assert actual.dtype == jnp.float64
        np.testing.assert_array_equal(actual, expected)
