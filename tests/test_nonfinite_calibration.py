"""Missing calibration pixels are recorded; zero-weight fields cannot poison valid views."""

from __future__ import annotations

import numpy as np
import pytest

import tomojax as tj


def _frames(flats, *, positions=None, at=None, darks=None):
    geometry = tj.ParallelGeometry(
        tj.Grid(9, 9, 3, 1, 1, 1), tj.Detector(9, 3, 1, 1), np.arange(5) * 36
    )
    return tj.Frames(
        np.full((5, 3, 9), 500, np.float32),
        geometry,
        flats=flats,
        flat_positions=positions,
        view_positions=at,
        darks=np.zeros((1, 3, 9), np.float32) if darks is None else darks,
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["flats", "darks"])
def test_nonfinite_calibration_is_neutral_and_reported(field, bad):
    flats = np.full((1, 3, 9), 1000, np.float32)
    darks = np.zeros((1, 3, 9), np.float32)
    (flats if field == "flats" else darks)[0, 1, 4] = bad
    scan = _frames(flats, darks=darks).corrected()
    np.testing.assert_array_equal(scan.projections[:, 1, 4], 0)
    np.testing.assert_allclose(scan.projections[:, 0], np.log(2), atol=1e-7)
    record = scan.corrections[0]
    assert record.found["pixels_without_reliable_light"] == 1
    assert record.settings["minimum_signal_snr"] == 3
    assert scan.corrections[-1].found == {
        "nonpositive_transmission": 0,
        "nonfinite_set_to_zero": 0,
    }


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bad_set", [0, 1])
def test_zero_weight_calibration_does_not_poison_healthy_endpoint(bad, bad_set):
    flats = np.full((4, 3, 9), 1000, np.float32)
    flats[2 * bad_set, 1, 4] = bad
    at = np.full(5, 1 - bad_set, np.float64)
    with np.errstate(invalid="ignore"):
        scan = _frames(flats, positions=np.array([0, 0, 1, 1]), at=at).corrected()
    np.testing.assert_allclose(scan.projections, np.log(2), atol=1e-7)
    assert scan.corrections[0].found["pixels_without_reliable_light"] == 1


def test_interior_view_still_needs_both_calibrations():
    flats = np.full((2, 3, 9), 1000, np.float32)
    flats[0, 1, 4] = np.nan
    scan = _frames(flats, positions=np.array([0, 1]), at=np.full(5, 0.5)).corrected()
    np.testing.assert_array_equal(scan.projections[:, 1, 4], 0)
    assert scan.corrections[0].found["pixels_without_reliable_light"] == 1
