"""A missing processed view must not poison every healthy view's rejection baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import RejectViews

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.surface


def _scan(projections: np.ndarray) -> tj.Scan:
    views, rows, cols = projections.shape
    return tj.Scan(
        projections,
        tj.ParallelGeometry(
            tj.Grid(cols, cols, rows, 1.0, 1.0, 1.0),
            tj.Detector(cols, rows, 1.0, 1.0),
            np.arange(views) * 180 / views,
        ),
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bad_view", [0, 6, 12])
def test_nonfinite_statistic_is_rejected_without_poisoning_good_views(value, bad_view) -> None:
    projections = np.full((13, 3, 7), 0.7, np.float32)
    projections[3] = 3.0  # a separate finite shutter-outlier control
    if np.isnan(value):
        projections[bad_view, 0, 0] = value  # even a single NaN makes the median nonfinite
    else:
        projections[bad_view] = value
    scan = _scan(projections)
    valid = np.arange(13) != bad_view
    expected = scan.selected(valid).corrected(RejectViews(), batch_views=4)
    with warnings.catch_warnings(record=True) as emitted:
        actual = scan.corrected(RejectViews(), batch_views=4)
    assert not emitted
    np.testing.assert_array_equal(actual.angles, expected.angles)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    found = actual.corrections[-1].found
    assert found["rejected"] == sorted([3, bad_view])
    assert found["nonfinite"] == [bad_view]
    assert found["robust_scale"] == expected.corrections[-1].found["robust_scale"] == 0.0


def test_consecutive_missing_views_keep_original_indices_and_saved_record(tmp_path: Path) -> None:
    projections = np.full((13, 3, 7), 0.7, np.float32)
    invalid = [0, 1, 4, 5, 12]
    projections[invalid] = np.nan
    projections[8] = 3.0
    scan = _scan(projections)
    valid = np.ones(13, bool)
    valid[invalid] = False
    expected = scan.selected(valid).corrected(RejectViews())
    actual = scan.corrected(RejectViews(), batch_views=3)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.corrections[-1].found == {
        "rejected": [0, 1, 4, 5, 8, 12],
        "robust_scale": 0.0,
        "nonfinite": invalid,
    }
    tj.save(tmp_path / "cleaned.nxs", actual)
    restored = tj.load(tmp_path / "cleaned.nxs")
    assert restored.corrections == actual.corrections
    np.testing.assert_array_equal(restored.projections, actual.projections)


def test_one_finite_view_is_retained_without_nonfinite_robust_scale() -> None:
    projections = np.full((9, 3, 7), np.nan, np.float32)
    projections[4] = 0.7
    with warnings.catch_warnings(record=True) as emitted:
        actual = _scan(projections).corrected(RejectViews(), batch_views=3)
    assert not emitted
    np.testing.assert_array_equal(actual.projections, projections[4:5])
    np.testing.assert_array_equal(actual.angles, [80.0])
    invalid = [0, 1, 2, 3, 5, 6, 7, 8]
    assert actual.corrections[-1].found == {
        "rejected": invalid,
        "robust_scale": 0.0,
        "nonfinite": invalid,
    }


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_all_invalid_views_raise_without_nan_mad_warnings(value) -> None:
    scan = _scan(np.full((7, 3, 7), value, np.float32))
    with (
        warnings.catch_warnings(record=True) as emitted,
        pytest.raises(ValueError, match="RejectViews would reject every view"),
    ):
        scan.corrected(RejectViews(), batch_views=3)
    assert not emitted
