"""Automatic raw-frame batches account for source dtype and padded FFT arrays."""

from __future__ import annotations

import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import Paganin


class ConstantViews:
    """Lazy per-view constant fields, logging which data the correction reads."""

    def __init__(self, rows: int, cols: int, dtype: str = "float32") -> None:
        self.shape = (65, rows, cols)
        self.dtype = np.dtype(dtype)
        self.levels = np.linspace(400.0, 800.0, self.shape[0], dtype=self.dtype)
        self.requests: list[int] = []

    def __getitem__(self, views: slice) -> np.ndarray:
        levels = self.levels[views]
        self.requests.append(len(levels))
        return np.broadcast_to(levels[:, None, None], (len(levels), *self.shape[1:]))


@pytest.mark.parametrize("shape", [(256, 256), (128, 512)])
@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("batch", [None, 11])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_phase_retrieval_limits_automatic_reads_and_preserves_explicit_batches(
    shape: tuple[int, int], phase: bool, batch: int | None, dtype: str
) -> None:
    rows, cols = shape
    counts = ConstantViews(rows, cols, dtype)
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 2, 1.0, 1.0, 1.0), tj.Detector(cols, rows, 1.0, 1.0), np.arange(65).tolist()
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


class BoundedDoubleViews(ConstantViews):
    """A valid lazy reader that cannot materialize more than one 256MiB read."""

    def __getitem__(self, views: slice) -> np.ndarray:
        size = len(self.levels[views]) * self.shape[1] * self.shape[2] * self.dtype.itemsize
        if size > 256 * 2**20:
            raise MemoryError("raw read exceeds available memory")
        return super().__getitem__(views)


def test_automatic_reads_budget_double_counts_before_conversion() -> None:
    counts = BoundedDoubleViews(512, 1024, "float64")
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 2, 1.0, 1.0, 1.0), tj.Detector(1024, 512, 1.0, 1.0), np.arange(65).tolist()
    )
    actual = tj.Frames(counts, geometry, white_level=1000.0).corrected()
    assert counts.requests == [64, 1]
    for start in range(0, 65, 8):
        levels = counts.levels[start : start + 8].astype(np.float32)
        expected = np.broadcast_to(
            -np.log(levels / 1000.0)[:, None, None], actual.projections[start : start + 8].shape
        )
        np.testing.assert_allclose(
            actual.projections[start : start + 8], expected, rtol=2e-6, atol=2e-7
        )
    np.testing.assert_array_equal(actual.angles, np.arange(65))
    assert actual.corrections[-1].found == {
        "nonpositive_transmission": 0,
        "nonfinite_set_to_zero": 0,
    }
