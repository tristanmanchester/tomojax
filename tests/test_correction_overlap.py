"""Whole-row corrections overlap bounded slabs without changing step order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.corrections import BeamHardening, Step, correct_projections


@dataclass(frozen=True)
class RowMean(Step):
    """A correction that needs every view, independently for each detector row."""

    domain: ClassVar = "line integrals"
    whole_rows: ClassVar = True

    def apply(self, batch):
        return batch + jnp.mean(batch, axis=0, keepdims=True)


@pytest.mark.parametrize("rows", [1, 4, 5])
def test_whole_rows_preserve_views_remainders_and_step_order(monkeypatch, rows: int) -> None:
    values = np.random.default_rng(51).uniform(0.1, 0.8, (7, rows, 11)).astype(np.float32)
    # Two rows per slab, including a final partial slab and a single-slab case.
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 7 * 11 * 8 * 2)
    steps = (BeamHardening((0.9,)), RowMean(), BeamHardening((1.0, 0.1)), RowMean())
    actual = correct_projections(values, steps)
    expected = values * 0.9
    expected += expected.mean(axis=0, keepdims=True)
    expected += 0.1 * expected**2
    expected += expected.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(actual.projections, expected, rtol=2e-6)
    np.testing.assert_array_equal(actual.kept, np.arange(7))
    assert tuple(record.name for record in actual.records) == (
        "beam_hardening",
        "row_mean",
        "beam_hardening",
        "row_mean",
    )
    np.testing.assert_array_equal(
        values, np.random.default_rng(51).uniform(0.1, 0.8, values.shape).astype(np.float32)
    )


def test_whole_rows_launch_next_slab_before_collecting_previous(monkeypatch) -> None:
    values = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 3 * 4 * 8 * 2)
    events = []
    pending = set()

    class Result:
        def __init__(self, batch, index):
            self.values = np.asarray(batch) * 2
            self.index = index

        def copy_to_host_async(self):
            pending.add(self.index)
            events.append(("start", self.index))
            assert len(pending) <= 2

        def __array__(self, dtype=None, copy=None):
            pending.remove(self.index)
            events.append(("collect", self.index))
            return np.asarray(self.values, dtype=dtype)

    calls = iter(range(3))
    monkeypatch.setattr(
        "tomojax.corrections._engine._apply",
        lambda steps, batch: Result(batch, next(calls)),
    )
    actual = correct_projections(values, (RowMean(),))
    np.testing.assert_array_equal(actual.projections, values * 2)
    assert not pending
    assert events == [
        ("start", 0),
        ("start", 1),
        ("collect", 0),
        ("start", 2),
        ("collect", 1),
        ("collect", 2),
    ]


def test_whole_rows_propagate_transfer_failure_without_changing_input(monkeypatch) -> None:
    values = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
    original = values.copy()
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 3 * 4 * 8 * 2)

    class FailedTransfer:
        def copy_to_host_async(self):
            pass

        def __array__(self, dtype=None, copy=None):
            raise RuntimeError("transfer failed")

    monkeypatch.setattr("tomojax.corrections._engine._apply", lambda *args: FailedTransfer())
    with pytest.raises(RuntimeError, match="transfer failed"):
        correct_projections(values, (RowMean(),))
    np.testing.assert_array_equal(values, original)


def test_whole_rows_larger_than_lookahead_budget_stay_serial(monkeypatch) -> None:
    values = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 3 * 4 * 4)
    events = []

    class Result:
        def __init__(self, batch, index):
            self.values, self.index = np.asarray(batch) * 2, index

        def __array__(self, dtype=None, copy=None):
            events.append(("collect", self.index))
            return np.asarray(self.values, dtype=dtype)

    def apply(steps, batch):
        index = len(events) // 2
        events.append(("start", index))
        assert batch.shape == (3, 1, 4)
        return Result(batch, index)

    monkeypatch.setattr("tomojax.corrections._engine._apply", apply)
    actual = correct_projections(values, (RowMean(),))
    np.testing.assert_array_equal(actual.projections, values * 2)
    assert events == [
        ("start", 0),
        ("collect", 0),
        ("start", 1),
        ("collect", 1),
        ("start", 2),
        ("collect", 2),
    ]
