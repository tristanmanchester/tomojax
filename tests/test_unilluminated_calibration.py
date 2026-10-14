"""Unreliable flat fields must not poison spatial transmission corrections."""

from __future__ import annotations

import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import Paganin


def _frames(counts: np.ndarray, flats: np.ndarray, darks: np.ndarray) -> tj.Frames:
    rows, cols = counts.shape[1:]
    geometry = tj.ParallelGeometry(
        tj.Grid(cols, cols, rows, 1.0, 1.0, 1.0),
        tj.Detector(cols, rows, 1.0, 1.0),
        np.arange(len(counts), dtype=float),
    )
    return tj.Frames(counts, geometry, flats=flats, darks=darks)


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_dark_flat_pixels_become_neutral_before_phase_retrieval(scale: float) -> None:
    counts = np.full((2, 9, 11), 500.0)
    flats = np.full((4, 9, 11), 1000.0)
    darks = np.zeros((4, 9, 11))
    # Mean signal is positive but not distinguishable from dark noise.
    flats[:, 2, 3] = [91.0, 111.0, 91.0, 111.0]
    darks[:, 2, 3] = [95.0, 105.0, 95.0, 105.0]
    counts[:, 2, 3] = 40000.0
    # Nonpositive flat-minus-dark is also unmeasured.
    flats[:, 6, 7] = 90.0
    darks[:, 6, 7] = 100.0
    phase = Paganin(300.0, 0.05, 20.0, 1e-6)
    actual = _frames(counts * scale, flats * scale, darks * scale).corrected(phase)
    expected = np.full_like(counts, 0.5, dtype=np.float32)
    expected[:, 2, 3] = expected[:, 6, 7] = 1.0
    expected = -np.log(np.asarray(phase.apply(expected)))
    np.testing.assert_allclose(actual.projections, expected, rtol=3e-6, atol=3e-7)
    assert actual.corrections[0].found["pixels_without_light"] == 1
    assert actual.corrections[0].found["pixels_without_reliable_light"] == 2


def test_resolved_low_count_signal_is_not_rejected_and_unmeasured_pixels_are_zero() -> None:
    counts = np.full((1, 3, 4), 0.0005)
    flats = np.full((3, 3, 4), 0.001)
    darks = np.zeros((3, 3, 4))
    flats[:, 1, 2] = 0.0
    actual = _frames(counts, flats, darks).corrected()
    expected = np.full_like(counts, np.log(2.0), dtype=np.float32)
    expected[:, 1, 2] = 0.0
    np.testing.assert_allclose(actual.projections, expected, rtol=3e-6)


def test_white_level_uses_dark_mean_uncertainty_and_records_unmeasured_pixels() -> None:
    counts = np.full((1, 3, 4), 0.5)
    darks = np.zeros((2, 3, 4))
    darks[:, 1, 2] = [0.0, 1.0]
    geometry = tj.ParallelGeometry(
        tj.Grid(4, 4, 3, 1.0, 1.0, 1.0), tj.Detector(4, 3, 1.0, 1.0), [0.0]
    )
    actual = tj.Frames(counts, geometry, darks=darks, white_level=1.0).corrected()
    assert actual.projections[0, 1, 2] == 0
    assert actual.corrections[0].found["pixels_without_reliable_light"] == 1


def test_flat_set_variances_follow_interpolation_weights() -> None:
    counts = np.full((1, 2, 3), 0.4)
    # Each flat mean 1 has SE 0.4 and is individually unresolved at 3 sigma.
    # Averaging independent before/after sets reduces the SE to sqrt(0.08),
    # so the midpoint flat is resolved. A linear variance interpolation would
    # wrongly reject it.
    flats = np.broadcast_to(np.array([0.6, 1.4, 0.6, 1.4])[:, None, None], (4, 2, 3))
    darks = np.zeros((1, 2, 3))
    frames = _frames(counts, flats, darks)
    from dataclasses import replace

    frames = replace(frames, flat_positions=np.array([0.0, 0.0, 1.0, 1.0]))
    actual = frames.corrected()
    np.testing.assert_allclose(actual.projections, -np.log(0.4), rtol=3e-6)


def test_calibration_moments_do_not_materialize_lazy_stacks() -> None:
    """Statistics retain lazy calibration sources and bound their FP64 workspace."""

    class LazyFlats:
        shape = (65, 128, 512)
        dtype = np.dtype(np.float32)

        def __init__(self) -> None:
            self.requests: list[int] = []

        def __len__(self) -> int:
            return self.shape[0]

        def __array__(self) -> np.ndarray:
            raise AssertionError("calibration stack was materialized")

        def __getitem__(self, indices: slice) -> np.ndarray:
            assert isinstance(indices, slice), "one flat set only needs basic slices"
            count = len(range(self.shape[0])[indices])
            self.requests.append(count)
            return np.full((count, *self.shape[1:]), 1000.0, np.float32)

    flats = LazyFlats()
    counts = np.full((1, 128, 512), 500.0, np.float32)
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 1, 1.0, 1.0, 1.0), tj.Detector(512, 128, 1.0, 1.0), [0.0]
    )
    actual = tj.Frames(counts, geometry, flats=flats).corrected()
    np.testing.assert_allclose(actual.projections, np.log(2.0), rtol=3e-6)
    assert sum(flats.requests) == 65
    assert max(flats.requests) < 65


def test_resolved_mean_survives_large_calibration_frame_variation() -> None:
    """The reliability threshold uses uncertainty of the mean, not frame noise."""
    counts = np.full((2, 3, 4), 500.0, np.float32)
    levels = np.tile([600.0, 1400.0], 25)
    flats = np.broadcast_to(levels[:, None, None], (50, 3, 4))
    darks = np.zeros((20, 3, 4), np.float32)
    # Individual frames' SD is about 404 counts, but fifty flats resolve the
    # 1000-count mean to about 57 counts. Beam variation is not dead light.
    actual = _frames(counts, flats, darks).corrected()
    np.testing.assert_allclose(actual.projections, np.log(2.0), rtol=3e-6)
    assert "pixels_without_reliable_light" not in actual.corrections[0].found
