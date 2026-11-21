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


def _window_median(values: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    padded = np.pad(values, ((0, 0), (0, 0), (half, half)), mode="edge")
    local = np.median(np.lib.stride_tricks.sliding_window_view(padded, width, axis=2), axis=-1)
    if width % 2 == 0:
        local = 0.5 * (local[..., :-1] + local[..., 1:])
    return local


def _box_means(values: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean over [p - scale/2, p + scale/2] of the piecewise-constant columns, edges repeated."""
    columns = values.shape[-1]
    middle = (columns - 1) / 2
    reach = -(-int(np.ceil(middle)) // scale)
    centres = middle + scale * np.arange(-reach, reach + 1)
    pad = scale * (reach + 2)
    extended = np.pad(values.astype(np.float64), ((0, 0), (0, 0), (pad, pad)), mode="edge")
    edges = np.arange(-pad, columns + pad + 1) - 0.5
    integral = np.concatenate([np.zeros(extended.shape[:-1] + (1,)), np.cumsum(extended, -1)], -1)

    def at(points: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda row: np.interp(points, edges, row), -1, integral)

    return centres, (at(centres + scale / 2) - at(centres - scale / 2)) / scale


def _stripes(values: np.ndarray, width: int) -> np.ndarray:
    ranked = np.sort(values, axis=0).astype(np.float64)
    columns = ranked.shape[-1]
    levels = 0
    while 9 * 2**levels < width:
        levels += 1
    offsets = np.zeros(ranked.shape[1:])
    for level in range(levels, -1, -1):
        scale = 2**level
        window = -(-width // scale) if level == levels else 9
        rest = ranked - offsets[None]
        if scale == 1:
            offsets += np.median(rest - _window_median(rest, window), axis=0)
            continue
        centres, blocks = _box_means(rest, scale)
        found = np.median(blocks - _window_median(blocks, window), axis=0)
        offsets += np.stack([np.interp(np.arange(columns), centres, f) for f in found])
    return values - offsets[None]


@pytest.mark.parametrize("shape", [(2, 1, 17), (2, 17, 1), (2, 9, 11)])
@pytest.mark.parametrize("size", [3, 5, 7])
def test_zinger_windows_match_scipy(shape: tuple[int, ...], size: int) -> None:
    values = np.random.default_rng(size).uniform(0.2, 0.8, shape).astype(np.float32)
    values.ravel()[::13] = 1.2
    actual = jax.jit(Zingers(size=size).apply)(jnp.asarray(values))
    np.testing.assert_array_equal(actual, _zingers(values, size))


@pytest.mark.parametrize("views", [1, 16, 17])
@pytest.mark.parametrize("rows", [1, 2, 3, 5, 9])
@pytest.mark.parametrize("width", [3, 4, 5, 8, 9, 15, 25, 55])
def test_stripe_windows_match_numpy(views: int, rows: int, width: int) -> None:
    values = np.random.default_rng(width).normal(size=(views, rows, 19)).astype(np.float32)
    actual = jax.jit(Stripes(width=width).apply)(jnp.asarray(values))
    np.testing.assert_allclose(actual, _stripes(values, width), rtol=1e-6, atol=2e-7)


@pytest.mark.parametrize("width", [3, 4, 8, 9])
def test_stripes_commute_with_detector_reversal(width: int) -> None:
    values = np.random.default_rng(615).normal(size=(17, 3, 19)).astype(np.float32)
    correct = jax.jit(Stripes(width).apply)
    reversed_result = np.asarray(correct(values[..., ::-1]))[..., ::-1]
    np.testing.assert_array_equal(correct(values), reversed_result)


@pytest.mark.parametrize("shape", [(257, 5, 1025), (257, 3, 4097)])
def test_stripe_large_rows_and_partial_groups_match_numpy(shape: tuple[int, ...]) -> None:
    values = np.random.default_rng(47).normal(size=shape).astype(np.float32)
    actual = jax.jit(Stripes(9).apply)(jnp.asarray(values))
    np.testing.assert_allclose(actual, _stripes(values, 9), rtol=1e-6, atol=2e-7)


def test_stripe_row_groups_preserve_float64_values() -> None:
    values = np.random.default_rng(46).normal(size=(129, 5, 1025))
    with jax.enable_x64():
        actual = jax.jit(Stripes(9).apply)(jnp.asarray(values))
        assert actual.dtype == values.dtype
        np.testing.assert_allclose(actual, _stripes(values, 9), rtol=1e-12, atol=2e-13)


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
@pytest.mark.parametrize("rows", [1, 2, 9])
def test_stripe_directional_derivative_matches_reference(width: int, rows: int) -> None:
    rng = np.random.default_rng(49)
    values = rng.normal(size=(17, rows, 19)).astype(np.float32)
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


@pytest.mark.parametrize("width", [25, 55, 201])
def test_wide_stripe_windows_commute_with_detector_reversal(width: int) -> None:
    values = np.random.default_rng(616).normal(size=(17, 3, 300)).astype(np.float32)
    correct = jax.jit(Stripes(width).apply)
    reversed_result = np.asarray(correct(values[..., ::-1]))[..., ::-1]
    np.testing.assert_allclose(correct(values), reversed_result, rtol=0, atol=1e-6)


@pytest.mark.parametrize("width", [101, 201])
def test_wide_stripe_windows_match_numpy(width: int) -> None:
    values = np.random.default_rng(width).normal(size=(33, 2, 300)).astype(np.float32)
    actual = jax.jit(Stripes(width=width).apply)(jnp.asarray(values))
    np.testing.assert_allclose(actual, _stripes(values, width), rtol=1e-6, atol=1e-6)


def _compiled_temporaries(step: Stripes, shape: tuple[int, ...]) -> int:
    values = jax.ShapeDtypeStruct(shape, jnp.float32)
    analysis = jax.jit(step.apply).lower(values).compile().memory_analysis()
    assert analysis is not None
    return int(analysis.temp_size_in_bytes)


def test_wide_stripe_windows_need_no_more_device_memory_than_narrow_ones() -> None:
    # A full-width row of a real scan: 1001 views of 2048 columns. Wide windows once
    # held a shifted copy of the row per column of the window.
    shape = (1001, 2, 2048)
    narrow = _compiled_temporaries(Stripes(9), shape)
    for width in (101, 401):
        assert _compiled_temporaries(Stripes(width), shape) <= 1.5 * narrow


def test_varying_stripe_memory_does_not_grow_with_the_window() -> None:
    shape = (1001, 2, 2048)
    narrow = _compiled_temporaries(Stripes(51, varying=True), shape)
    assert _compiled_temporaries(Stripes(151, varying=True), shape) <= 1.25 * narrow + (64 << 20)


def test_wide_windows_remove_a_broad_offset_under_moving_features() -> None:
    # A wide sample whose line integrals are level across the middle of the detector,
    # with pores and inclusions tracing sinusoids, and an 80-column stripe of
    # constant offset on that level. A nine-column window corrects only its edges.
    views, columns = 360, 1024
    angles = np.linspace(0, np.pi, views, endpoint=False)[:, None]
    u = np.arange(columns) - (columns - 1) / 2
    clean = np.clip((450 - np.abs(u)) / 50, 0, 1) * 2.0 * np.ones((views, 1))
    features = ((40, 300, 0.3, -0.004), (25, 200, 2.0, 0.006), (60, 100, 4.0, -0.003))
    for radius, distance, phase, density in features:
        centre = distance * np.cos(angles + phase)
        clean += 2 * density * np.sqrt(np.clip(radius**2 - (u - centre) ** 2, 0, None))
    offset = np.where((np.arange(columns) >= 700) & (np.arange(columns) < 780), 0.05, 0.0)
    noise = np.random.default_rng(93).normal(0, 0.002, clean.shape)
    measured = (clean + offset + noise).astype(np.float32)[:, None]

    def left(width: int) -> float:
        corrected = np.asarray(jax.jit(Stripes(width).apply)(jnp.asarray(measured)))[:, 0]
        error = np.median(corrected - clean, axis=0)[640:840]
        return float(np.sqrt(np.mean((error - np.median(error)) ** 2)) / 0.05)

    assert left(9) > 0.5
    assert left(201) < 0.1
