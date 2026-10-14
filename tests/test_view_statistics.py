"""View rejection retains exact medians on megapixel detector frames."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import RejectViews


@pytest.mark.parametrize("pixels", [1, 2**20, 2**20 + 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_view_statistics_keep_their_empty_shape_and_dtype(
    pixels: int, dtype: type[np.float32] | type[np.float64]
) -> None:
    values = np.empty((0, 1, pixels), dtype)
    with jax.enable_x64(dtype == np.float64):
        actual = jax.jit(RejectViews().statistic)(values)
    assert actual.shape == (0,)
    assert actual.dtype == values.dtype


@pytest.mark.parametrize("pixels", [2**20, 2**20 + 1])
def test_large_view_statistics_match_numpy(pixels: int) -> None:
    rng = np.random.default_rng(106)
    values = rng.normal(-0.1, 0.7, (3, 1, pixels)).astype(np.float32)
    values[1] = rng.integers(-2, 3, (1, pixels))  # repeated central values
    values[2] = 0.7  # a converged view alongside other, still-searching views
    actual = jax.jit(RejectViews().statistic)(values)
    np.testing.assert_array_equal(actual, np.median(values, axis=(1, 2)))


@pytest.mark.parametrize("pixels", [2**20, 2**20 + 1])
def test_large_view_statistics_preserve_nonfinite_medians(pixels: int) -> None:
    values = np.full((6, 1, pixels), 0.7, np.float32)
    values[0, 0, 0] = np.nan
    values[1] = np.inf
    values[2] = -np.inf
    values[3, 0, : pixels // 2] = -np.inf
    values[3, 0, pixels // 2 :] = np.inf
    values[4, 0, 0] = np.inf  # isolated infinities do not change the median
    values[4, 0, -1] = -np.inf
    values[5, 0, : pixels // 2] = -0.0
    values[5, 0, pixels // 2 :] = 0.0
    with np.errstate(invalid="ignore"):
        expected = np.median(values, axis=(1, 2))
    np.testing.assert_array_equal(jax.jit(RejectViews().statistic)(values), expected)


def test_large_statistics_preserve_median_jvp_and_reverse_gradient() -> None:
    rng = np.random.default_rng(107)
    values = rng.normal(size=(1, 1024, 1024)).astype(np.float32)
    direction = rng.normal(size=values.shape).astype(np.float32)
    statistic = RejectViews().statistic

    def reference(x):
        return jnp.median(x, axis=(1, 2))

    actual = jax.jit(lambda x, d: jax.jvp(statistic, (x,), (d,)))(values, direction)
    expected = jax.jit(lambda x, d: jax.jvp(reference, (x,), (d,)))(values, direction)
    for result, target in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, target)
    actual_gradient = jax.jit(jax.grad(lambda x: jnp.sum(statistic(x))))(values)
    expected_gradient = jax.jit(jax.grad(lambda x: jnp.sum(reference(x))))(values)
    np.testing.assert_array_equal(actual_gradient, expected_gradient)


def test_float64_view_statistics_do_not_round_to_float32() -> None:
    values = np.broadcast_to(1.0 + 2**-40, (1, 1, 2**20)).copy()
    with jax.enable_x64():
        actual = jax.jit(RejectViews().statistic)(values)
        assert actual.dtype == values.dtype
        np.testing.assert_array_equal(actual, np.median(values, axis=(1, 2)))


def test_large_view_rejection_keeps_geometry_and_nonfinite_record() -> None:
    values = np.full((13, 1024, 1024), 0.7, np.float32)
    values[3] = 3.0
    values[6, 0, 0] = np.nan
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 1, 1.0, 1.0, 1.0),
        tj.Detector(1024, 1024, 1.0, 1.0),
        np.arange(13) * 10.0,
    )
    actual = tj.Scan(values, geometry).corrected(RejectViews(), batch_views=4)
    kept = np.delete(np.arange(13), [3, 6])
    np.testing.assert_array_equal(actual.angles, geometry.angles[kept])
    np.testing.assert_array_equal(actual.projections, values[kept])
    assert actual.corrections[-1].found == {
        "rejected": [3, 6],
        "nonfinite": [6],
        "robust_scale": 0.0,
    }


def test_large_view_statistics_preserve_vmap_batching() -> None:
    values = np.random.default_rng(248).normal(-0.1, 0.7, (3, 2, 1024, 1024)).astype(np.float32)
    values[1] = 0.7
    values[2, 0] = 0.0
    values[2, 0, :512] = -0.0
    values[2, 1] = np.inf
    actual = jax.jit(jax.vmap(RejectViews().statistic))(values)
    np.testing.assert_array_equal(actual, np.median(values, axis=(2, 3)))


@pytest.mark.parametrize("pixels", [2**20, 2**20 + 1])
def test_large_view_statistics_remain_exact_with_spatially_spaced_spikes(pixels: int) -> None:
    values = np.full((2, 1, pixels), 0.7, np.float32)
    # Sparse regularly spaced high/low defects must not bias an exact view
    # statistic, even when a coarse spatial sample sees only those defects.
    indices = np.arange(512, dtype=np.int64) * (pixels - 1) // 511
    values[0, 0, indices] = 9.0
    values[1, 0, indices] = -9.0
    actual = jax.jit(RejectViews().statistic)(values)
    np.testing.assert_array_equal(actual, np.median(values, axis=(1, 2)))
