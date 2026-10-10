"""Spatial correction medians against independent host references."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import median_filter

from tomojax.corrections import Stripes, Zingers


def _zingers(values: np.ndarray, size: int) -> np.ndarray:
    median = median_filter(values, size=(1, size, size), mode="nearest")
    return np.where(values - median > 0.1, median, values)


def _stripes(values: np.ndarray, width: int) -> np.ndarray:
    ranked = np.sort(values, axis=0)
    half = width // 2
    padded = np.pad(ranked, ((0, 0), (0, 0), (half, width - 1 - half)), mode="edge")
    local = np.median(np.lib.stride_tricks.sliding_window_view(padded, width, axis=2), axis=-1)
    return values - np.median(ranked - local, axis=0)[None]


@pytest.mark.parametrize("shape", [(2, 1, 17), (2, 17, 1), (2, 9, 11)])
@pytest.mark.parametrize("size", [3, 5, 7])
def test_zinger_windows_match_scipy(shape: tuple[int, ...], size: int) -> None:
    values = np.random.default_rng(size).uniform(0.2, 0.8, shape).astype(np.float32)
    values.ravel()[::13] = 1.2
    actual = jax.jit(Zingers(size=size).apply)(jnp.asarray(values))
    np.testing.assert_array_equal(actual, _zingers(values, size))


@pytest.mark.parametrize("views", [1, 16, 17])
@pytest.mark.parametrize("width", [3, 4, 5, 8, 9, 15, 25])
def test_stripe_windows_match_numpy(views: int, width: int) -> None:
    values = np.random.default_rng(width).normal(size=(views, 2, 19)).astype(np.float32)
    actual = jax.jit(Stripes(width=width).apply)(jnp.asarray(values))
    np.testing.assert_allclose(actual, _stripes(values, width), rtol=1e-6, atol=2e-7)


@pytest.mark.parametrize("size", [3, 5, 7])
def test_zinger_directional_derivative_matches_reference(size: int) -> None:
    rng = np.random.default_rng(48)
    values = rng.uniform(0.2, 0.8, (2, 9, 11)).astype(np.float32)
    direction = rng.normal(size=values.shape).astype(np.float32)
    _, actual = jax.jit(lambda x, d: jax.jvp(Zingers(size=size).apply, (x,), (d,)))(
        jnp.asarray(values), jnp.asarray(direction)
    )
    epsilon = 1e-6
    reference = (
        _zingers(values.astype(np.float64) + epsilon * direction, size)
        - _zingers(values.astype(np.float64) - epsilon * direction, size)
    ) / (2 * epsilon)
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("width", [4, 9, 15])
def test_stripe_directional_derivative_matches_reference(width: int) -> None:
    rng = np.random.default_rng(49)
    values = rng.normal(size=(17, 2, 19)).astype(np.float32)
    direction = rng.normal(size=values.shape).astype(np.float32)
    _, actual = jax.jit(lambda x, d: jax.jvp(Stripes(width=width).apply, (x,), (d,)))(
        jnp.asarray(values), jnp.asarray(direction)
    )
    epsilon = 1e-6
    reference = (
        _stripes(values.astype(np.float64) + epsilon * direction, width)
        - _stripes(values.astype(np.float64) - epsilon * direction, width)
    ) / (2 * epsilon)
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("size", [3, 5, 7])
def test_zingers_leave_constant_images_unchanged(size: int) -> None:
    values = jnp.ones((2, 5, 6), dtype=jnp.float32)
    np.testing.assert_array_equal(jax.jit(Zingers(size=size).apply)(values), values)


@pytest.mark.parametrize("size", [5, 7, 9])
def test_zinger_medians_preserve_nonfinite_samples(size: int) -> None:
    values = np.random.default_rng(size).uniform(0.2, 0.8, (2, 9, 11)).astype(np.float32)
    values[0, 3, 4] = np.nan
    values[0, 7, 9] = np.inf
    values[1, 5, 6] = -np.inf
    half = size // 2
    padded = np.pad(values, ((0, 0), (half, half), (half, half)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size), axis=(1, 2))
    median = np.median(windows.reshape(*values.shape, size * size), axis=-1)
    with np.errstate(invalid="ignore"):
        reference = np.where(values - median > 0.1, median, values)
    actual = jax.jit(Zingers(size=size).apply)(jnp.asarray(values))
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("size", [5, 7, 9])
def test_zinger_binary_windows_match_majority(size: int) -> None:
    values = np.random.default_rng(size).integers(0, 2, (64, size, size)).astype(np.float32)
    values[:, size // 2, size // 2] = 1.0
    actual = jax.jit(Zingers(size=size).apply)(jnp.asarray(values))
    # With a bright centre, it is retained precisely when a strict majority
    # of the window is bright; otherwise the zero median replaces it.
    reference = (values.sum(axis=(1, 2)) > size * size // 2).astype(np.float32)
    np.testing.assert_array_equal(actual[:, size // 2, size // 2], reference)
