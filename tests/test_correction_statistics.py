"""Raw correction reuses device summaries without changing correction order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, RejectViews, Step, Stripes


def _frames() -> tj.Frames:
    counts = np.full((13, 3, 7), 500, np.uint16)
    counts[[3, 8]] = 5
    geometry = tj.ParallelGeometry(
        tj.Grid(7, 7, 3, 1.0, 1.0, 1.0),
        tj.Detector(7, 3, 1.0, 1.0),
        np.arange(13) * 10.0,
    )
    return tj.Frames(counts, geometry, white_level=1000.0)


@pytest.mark.parametrize(
    "steps",
    [
        (RejectViews(),),
        (BeamHardening((1.0, 0.1)), RejectViews()),
        (RejectViews(), BeamHardening((1.0, 0.1)), RejectViews()),
        (Stripes(), RejectViews()),
    ],
)
@pytest.mark.parametrize("batch", [4, 13])
def test_raw_selection_uses_the_statistic_at_its_place_in_the_pipeline(steps, batch) -> None:
    frames = _frames()
    expected = frames.corrected(batch_views=batch).corrected(*steps, batch_views=batch)
    actual = frames.corrected(*steps, batch_views=batch)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=2e-6, atol=2e-7)
    assert actual.corrections == expected.corrections


def test_first_view_selector_does_not_retransfer_corrected_batches(monkeypatch) -> None:
    frames = _frames()
    transfer_shapes = []
    asarray = jnp.asarray

    def record_transfer(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.dtype == np.float32:
            transfer_shapes.append(value.shape)
        return asarray(value, *args, **kwargs)

    monkeypatch.setattr(jnp, "asarray", record_transfer)
    actual = frames.corrected(RejectViews(), batch_views=4)
    assert actual.corrections[-1].found["rejected"] == [3, 8]
    assert transfer_shapes == []


@dataclass(frozen=True)
class _RangeSelector(Step):
    domain: ClassVar = "line integrals"
    selects_views: ClassVar = True

    def statistic(self, batch):
        return jnp.stack([jnp.min(batch, axis=(1, 2)), jnp.max(batch, axis=(1, 2))], axis=1)

    def select(self, statistics, views):
        keep = statistics[:, 1] - statistics[:, 0] < 0.5
        return keep, {"rejected": views[~keep].tolist()}


def test_device_summary_reuse_supports_custom_multicolumn_statistics() -> None:
    frames = _frames()
    frames.counts[5, 0, 0] = 5
    expected = frames.corrected().corrected(_RangeSelector(), batch_views=4)
    actual = frames.corrected(_RangeSelector(), batch_views=4)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.corrections == expected.corrections
    assert actual.corrections[-1].found["rejected"] == [5]
