"""Spatial correction medians against independent host references."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import median_filter

from tomojax.corrections import Rings, Zingers


def _zingers(values: np.ndarray, size: int) -> np.ndarray:
    median = median_filter(values, size=(1, size, size), mode="nearest")
    return np.where(values - median > 0.1, median, values)


@pytest.mark.parametrize("shape", [(2, 1, 17), (2, 17, 1), (2, 9, 11)])
@pytest.mark.parametrize("size", [3, 5, 7])
def test_zinger_windows_match_scipy(shape: tuple[int, ...], size: int) -> None:
    values = np.random.default_rng(size).uniform(0.2, 0.8, shape).astype(np.float32)
    values.ravel()[::13] = 1.2
    actual = jax.jit(Zingers(size=size).apply)(jnp.asarray(values))
    np.testing.assert_array_equal(actual, _zingers(values, size))


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
    # With a bright center, it is retained precisely when a strict majority
    # of the window is bright; otherwise the zero median replaces it.
    reference = (values.sum(axis=(1, 2)) > size * size // 2).astype(np.float32)
    np.testing.assert_array_equal(actual[:, size // 2, size // 2], reference)


def test_rings_removes_a_broad_offset_under_moving_features() -> None:
    # A wide sample whose line integrals are level across the middle of the detector,
    # with pores and inclusions tracing sinusoids, and an 80-column stripe of
    # constant offset on that level.
    views, columns = 360, 1024
    angles = np.linspace(0, np.pi, views, endpoint=False)[:, None]
    u = np.arange(columns) - (columns - 1) / 2
    clean = np.clip((450 - np.abs(u)) / 50, 0, 1) * 2.0 * np.ones((views, 1))
    features = ((40, 300, 0.3, -0.004), (25, 200, 2.0, 0.006), (60, 100, 4.0, -0.003))
    for radius, distance, phase, density in features:
        center = distance * np.cos(angles + phase)
        clean += 2 * density * np.sqrt(np.clip(radius**2 - (u - center) ** 2, 0, None))
    offset = np.where((np.arange(columns) >= 700) & (np.arange(columns) < 780), 0.05, 0.0)
    noise = np.random.default_rng(93).normal(0, 0.002, clean.shape)
    measured = (clean + offset + noise).astype(np.float32)[:, None]

    corrected = np.asarray(Rings().apply(jnp.asarray(measured)))[:, 0]
    error = np.median(corrected - clean, axis=0)[640:840]
    assert np.sqrt(np.mean((error - np.median(error)) ** 2)) < 0.1 * 0.05
