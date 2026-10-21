"""Whole-view transmission corrections run before log in every streaming path."""

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.corrections import Step, correct_frames, corrected_row_source


@dataclass(frozen=True)
class NormalizeRows(Step):
    domain: ClassVar = "transmission"
    whole_rows: ClassVar = True

    def apply(self, batch):
        return batch / jnp.mean(batch, axis=0, keepdims=True)


@dataclass(frozen=True)
class Offset(Step):
    domain: ClassVar = "transmission"
    amount: float

    def apply(self, batch):
        return batch + self.amount


@dataclass(frozen=True)
class CountsOffset(Offset):
    domain: ClassVar = "counts"


@dataclass(frozen=True)
class PhaseRows(Step):
    domain: ClassVar = "line integrals"
    whole_rows: ClassVar = True

    def apply(self, batch):
        return 2 * batch


def inputs():
    t = np.random.default_rng(2026).uniform(0.2, 0.8, (5, 5, 7)).astype(np.float32)
    light = 100 + 20 * (np.arange(5) + 0.5)
    raw = (5 + light[:, None, None] * t).astype(np.float32)
    t = (raw.astype(np.float64) - 5) / light[:, None, None]
    calibration = dict(
        flats=np.stack([np.full((5, 7), 105), np.full((5, 7), 205)]),
        flat_positions=np.array([0, 5]),
        darks=np.full((1, 5, 7), 5),
        white_level=None,
        epsilon=1e-6,
    )
    return raw, t, calibration


@pytest.mark.parametrize("batch", [None, 2])
def test_transmission_row_stage_preserves_domain_and_within_domain_order(batch):
    raw, t, calibration = inputs()
    steps = (PhaseRows(), Offset(0.03), NormalizeRows(), CountsOffset(2), Offset(0.07))
    done = correct_frames(raw - 2, **calibration, steps=steps, batch_views=batch)
    expected = -2 * np.log((t + 0.03) / (t + 0.03).mean(0) + 0.07)
    np.testing.assert_allclose(done.projections, expected, atol=1e-6)
    assert [r.name for r in done.records] == [
        "counts_offset",
        "flat_dark",
        "offset",
        "normalize_rows",
        "offset",
        "log",
        "phase_rows",
    ]
    assert done.records[-2].found == {"nonpositive_transmission": 0, "nonfinite_set_to_zero": 0}


@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize("batch", [None, 2])
def test_row_transmission_and_phase_are_binned_after_log(factor, batch):
    raw, t, calibration = inputs()
    done = correct_frames(
        raw,
        **calibration,
        steps=(PhaseRows(), NormalizeRows()),
        batch_views=batch,
        binning=factor,
    )
    expected = -2 * np.log(t / t.mean(0))
    expected = expected[:, : 5 // factor * factor, : 7 // factor * factor]
    expected = expected.reshape(5, 5 // factor, factor, 7 // factor, factor).mean((2, 4))
    np.testing.assert_allclose(done.projections, expected, atol=1e-6)
    np.testing.assert_array_equal(done.kept, np.arange(5))


@pytest.mark.parametrize("host", [False, True])
def test_owned_row_source_preserves_transmission_phase_order_and_records(host):
    raw, t, calibration = inputs()
    with corrected_row_source(
        raw, **calibration, steps=(PhaseRows(), NormalizeRows()), host=host
    ) as read:
        done = read(1, 4)
        read(0, 2)
    np.testing.assert_allclose(
        done.projections, -2 * np.log(t[:, 1:4] / t[:, 1:4].mean(0)), atol=1e-6
    )
    assert [r.name for r in done.records] == ["flat_dark", "normalize_rows", "log", "phase_rows"]
