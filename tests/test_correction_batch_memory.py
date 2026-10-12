"""Automatic raw-frame batches account for padded FFT arrays before reading data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, Paganin, Step, Stripes


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


@dataclass(frozen=True)
class CountsFFT(Step):
    """A public custom step with a padded FFT workspace, in the counts domain."""

    domain: ClassVar = "counts"
    batches: ClassVar[list[int]] = []

    def workspace_pixels(self, rows: int, cols: int) -> int:
        return 4 * (rows + 2 * (rows // 2)) * (cols + 2 * (cols // 2))

    def apply(self, batch):
        # Observe actual calls, not just the shapes seen while tracing the JIT.
        jax.debug.callback(lambda n: CountsFFT.batches.append(int(n)), batch.shape[0], ordered=True)
        _, rows, cols = batch.shape
        pr, pc = rows // 2, cols // 2
        padded = jnp.pad(batch, ((0, 0), (pr, pr), (pc, pc)), mode="edge")
        result = jnp.fft.irfft2(jnp.fft.rfft2(padded), s=padded.shape[-2:])
        return result[:, pr : pr + rows, pc : pc + cols]


@dataclass(frozen=True)
class LineFFT(CountsFFT):
    """The same workspace requirement, after the logarithm."""

    domain: ClassVar = "line integrals"


@dataclass(frozen=True)
class FFTStatistic(LineFFT):
    """A view selector whose per-view statistic also needs device workspace."""

    selects_views: ClassVar = True

    def statistic(self, batch):
        return jnp.mean(self.apply(batch), axis=(1, 2))

    def select(self, statistics, views):
        return np.ones(len(views), bool), {}


@pytest.mark.parametrize(
    "where", ["counts", "head", "tail", "scan", "scan_tail", "scan_selector", "raw_selector"]
)
@pytest.mark.parametrize("batch", [None, 3])
def test_all_per_view_domains_respect_workspace_and_explicit_batches(
    monkeypatch, where: str, batch: int | None
) -> None:
    rows, cols = 17, 19
    counts = ConstantViews(rows, cols)
    # Twenty-four inputs fit the budget, but only one input with its padded FFTs.
    budget = 4 * rows * cols * 24
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", budget)
    CountsFFT.batches.clear()
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 2, 1.0, 1.0, 1.0), tj.Detector(cols, rows, 1.0, 1.0), np.arange(65)
    )
    expected = np.broadcast_to(-np.log(counts.levels / 1000.0)[:, None, None], counts.shape)
    # Include a zero-workspace step first: sizing must inspect the whole run.
    steps = [BeamHardening((1.0,)), LineFFT()]
    if where in {"tail", "scan_tail"}:
        steps.insert(0, Stripes())
    if where == "counts":
        steps = [CountsFFT()]
    if where in {"scan_selector", "raw_selector"}:
        steps = [FFTStatistic()]
    if where.startswith("scan"):
        actual = tj.Scan(expected.copy(), geometry).corrected(*steps, batch_views=batch)
    else:
        actual = tj.Frames(counts, geometry, white_level=1000.0).corrected(
            *steps, batch_views=batch
        )
        assert max(counts.requests) == (batch or (24 if where == "tail" else 1))
        assert sum(counts.requests) == counts.shape[0]
    assert max(CountsFFT.batches) == (batch or 1)
    assert sum(CountsFFT.batches) == counts.shape[0]
    if batch is None:
        pixels = rows * cols + CountsFFT().workspace_pixels(rows, cols)
        assert 4 * pixels * max(CountsFFT.batches) <= budget
    np.testing.assert_allclose(actual.projections, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(actual.angles, geometry.angles)


@pytest.mark.parametrize("shape", [(256, 256), (128, 512)])
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
