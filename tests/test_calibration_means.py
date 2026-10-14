"""Flat and dark averaging retain FP64 accumulation for every calibration layout."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tomojax as tj

pytestmark = pytest.mark.surface


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("several_sets", [False, True])
def test_calibration_means_match_independent_host_interpolation(
    dtype: type, layout: str, several_sets: bool
) -> None:
    rng = np.random.default_rng(8504)
    flats = rng.uniform(900.0, 1100.0, (9, 5, 7)).astype(dtype)
    darks = rng.uniform(5.0, 15.0, (7, 5, 7)).astype(dtype)
    counts = rng.uniform(300.0, 800.0, (5, 5, 7)).astype(np.float32)
    # A known unlit column exercises the calibration record and neutral fill.
    flats[:, :, 0], darks[:, :, 0], counts[:, :, 0] = 5, 15, 10
    if layout == "F":
        flats, darks = np.asfortranarray(flats), np.asfortranarray(darks)
    elif layout == "strided":
        flats = np.repeat(flats, 2, axis=2)[:, :, ::2]
        darks = np.repeat(darks, 2, axis=2)[:, :, ::2]
    originals = [values.copy() for values in (counts, flats, darks)]
    where = np.resize([10.0, 0.0], 9) if several_sets else np.zeros(9)
    positions = np.unique(where)
    reference_flats = np.stack(
        [np.asarray(flats[where == position], np.float64).mean(axis=0) for position in positions]
    )
    reference_dark = np.asarray(darks, np.float64).mean(axis=0)
    interpolated = np.empty(counts.shape, np.float64)
    for row in range(5):
        for col in range(7):
            interpolated[:, row, col] = np.interp(
                np.arange(5) + 0.5, positions, reference_flats[:, row, col]
            )
    transmission = (counts - reference_dark) / np.maximum(interpolated - reference_dark, 1e-6)
    transmission[:, :, 0] = 1.0  # the unlit column measures no absorption
    expected = -np.log(np.maximum(transmission, 1e-6))
    geometry = tj.ParallelGeometry(
        tj.Grid(4, 4, 2, 1.0, 1.0, 1.0), tj.Detector(7, 5, 1.0, 1.0), np.arange(5) * 36.0
    )
    actual = tj.Frames(
        counts, geometry, flats=flats, darks=darks, flat_positions=where if several_sets else None
    ).corrected(batch_views=3)
    np.testing.assert_allclose(actual.projections, expected, rtol=2e-6, atol=2e-7)
    assert actual.corrections[0].settings["flat_sets"] == len(positions)
    assert actual.corrections[0].found == {
        "pixels_without_light": 5 * len(positions),
        "pixels_without_reliable_light": 5 * len(positions),
    }
    assert actual.corrections[-1].found == {
        "nonpositive_transmission": 0,
        "nonfinite_set_to_zero": 0,
    }
    for values, original in zip((counts, flats, darks), originals, strict=True):
        np.testing.assert_array_equal(values, original)


def test_float32_calibration_accumulates_small_frames_in_float64() -> None:
    # A float32 running sum loses small frames after the large ones. Sixteen
    # large frames also keep the calibration mean resolved at three SEs; a
    # single outlier would correctly be treated as an unreliable flat field.
    flats = np.ones((4112, 2, 3), np.float32)
    flats[:16] = 1e8
    mean = (16e8 + 4096.0) / 4112.0
    counts = np.full((2, 2, 3), 0.9 * mean, np.float32)
    geometry = tj.ParallelGeometry(
        tj.Grid(4, 4, 2, 1.0, 1.0, 1.0), tj.Detector(3, 2, 1.0, 1.0), [0.0, 90.0]
    )
    actual = tj.Frames(counts, geometry, flats=flats).corrected()
    np.testing.assert_allclose(
        actual.projections, -np.log(float(counts[0, 0, 0]) / mean), rtol=2e-6
    )
    assert actual.corrections[-1].found["nonfinite_set_to_zero"] == 0


@pytest.mark.parametrize("several_sets", [False, True])
def test_device_calibration_batches_use_numpy_fp64_averaging(several_sets: bool) -> None:
    rng = np.random.default_rng(8673)
    flats = rng.uniform(0.9, 1.1, (99, 4, 7)).astype(np.float32)
    darks = rng.uniform(0.005, 0.015, (83, 4, 7)).astype(np.float32)
    counts = np.full((7, 4, 7), 0.5, np.float32)
    geometry = tj.ParallelGeometry(
        tj.Grid(7, 7, 4, 1, 1, 1), tj.Detector(7, 4, 1, 1), np.arange(7) * 180 / 7
    )
    positions = np.resize([0, 7], len(flats)) if several_sets else None
    with jax.enable_x64(False):
        expected = tj.Frames(
            counts, geometry, flats=flats, darks=darks, flat_positions=positions
        ).corrected(batch_views=3)
        actual = tj.Frames(
            counts,
            geometry,
            flats=jax.device_put(flats),
            darks=jax.device_put(darks),
            flat_positions=positions,
        ).corrected(batch_views=3)
        np.testing.assert_array_equal(actual.projections, expected.projections)
        assert actual.corrections == expected.corrections
        np.testing.assert_array_equal(
            tj.reconstruct(actual).volume, tj.reconstruct(expected).volume
        )
