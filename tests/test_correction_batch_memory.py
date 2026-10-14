"""Automatic raw-frame batches account for padded FFT arrays before reading data."""

from __future__ import annotations

import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import Paganin


class ConstantViews:
    """Lazy per-view constant fields, logging which data the correction reads."""

    dtype = np.dtype(np.float32)

    def __init__(self, rows: int, cols: int) -> None:
        self.shape = (65, rows, cols)
        self.levels = np.linspace(400.0, 800.0, self.shape[0], dtype=np.float32)
        self.requests: list[int] = []

    def __getitem__(self, views: slice) -> np.ndarray:
        levels = self.levels[views]
        self.requests.append(len(levels))
        return np.broadcast_to(levels[:, None, None], (len(levels), *self.shape[1:]))


@pytest.mark.parametrize("shape", [(256, 256), (192, 512)])
@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("batch", [None, 11])
def test_phase_retrieval_limits_automatic_reads_and_preserves_explicit_batches(
    shape: tuple[int, int], phase: bool, batch: int | None
) -> None:
    rows, cols = shape
    counts = ConstantViews(rows, cols)
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 2, 1.0, 1.0, 1.0), tj.Detector(cols, rows, 1.0, 1.0), np.arange(65)
    )
    steps = [Paganin(300.0, 0.001, 20.0, 1e-6)] if phase else []
    actual = tj.Frames(counts, geometry, white_level=1000.0).corrected(*steps, batch_views=batch)

    if batch is not None:
        assert max(counts.requests) == batch
    elif phase:
        # The small unpadded input fits one batch; its padded FFT arrays do not.
        assert max(counts.requests) < counts.shape[0]
    else:
        assert counts.requests == [counts.shape[0]]
    assert sum(counts.requests) == counts.shape[0]
    expected = -np.log(counts.levels / 1000.0)[:, None, None]
    np.testing.assert_allclose(
        actual.projections, np.broadcast_to(expected, counts.shape), rtol=2e-6, atol=2e-7
    )
    np.testing.assert_array_equal(actual.angles, np.arange(65))
    assert next(record for record in actual.corrections if record.name == "log").found == {
        "nonpositive_transmission": 0,
        "nonfinite_set_to_zero": 0,
    }
