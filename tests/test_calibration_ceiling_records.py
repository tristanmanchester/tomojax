"""Integer calibration ceilings are provenance, not silently repaired ray values."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import tomojax as tj


def _frames(flats, darks, *, positions=None, white_level=None, counts=None):
    geometry = tj.ParallelGeometry(
        tj.Grid(7, 7, 1, 1, 1, 1), tj.Detector(7, 1, 1, 1), np.arange(3) * 60
    )
    return tj.Frames(
        np.full((3, 1, 7), 500, np.uint16) if counts is None else counts,
        geometry,
        flats=flats,
        darks=darks,
        flat_positions=positions,
        white_level=white_level,
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_flat_dtype_ceiling_is_recorded_without_inventing_absorption(dtype, caplog):
    ceiling = np.iinfo(dtype).max
    flats = np.full((5, 1, 7), ceiling // 2, dtype)
    flats[:, 0, 3] = ceiling
    darks = np.zeros((1, 1, 7), dtype)
    counts = np.full((3, 1, 7), ceiling // 4, dtype)
    with caplog.at_level(logging.WARNING):
        scan = _frames(flats, darks, counts=counts).corrected()
    expected = _frames(
        flats.astype(np.float64), darks.astype(np.float64), counts=counts
    ).corrected()
    np.testing.assert_array_equal(scan.projections, expected.projections)
    record = scan.corrections[0]
    assert record.found["flat_pixels_at_dtype_ceiling"] == 1
    assert record.settings["flat_dtype_ceiling"] == ceiling
    assert "may be clipped" in caplog.text
    assert "dtype ceiling" in caplog.text


def test_dark_dtype_ceiling_is_separate_from_the_flat_limit():
    flats = np.full((3, 1, 7), 1000, np.uint16)
    darks = np.full((4, 1, 7), 17, np.uint8)
    darks[:, 0, 2] = 255
    scan = _frames(flats, darks).corrected()
    record = scan.corrections[0]
    assert record.found["dark_pixels_at_dtype_ceiling"] == 1
    assert record.settings["dark_dtype_ceiling"] == 255
    assert "flat_pixels_at_dtype_ceiling" not in record.found
    expected = _frames(flats, darks.astype(np.float64)).corrected()
    np.testing.assert_array_equal(scan.projections, expected.projections)


def test_explicit_white_level_still_reports_clipped_dark_calibration():
    darks = np.zeros((3, 1, 7), np.uint8)
    darks[:, 0, 2] = 255
    scan = _frames(None, darks, white_level=1000).corrected()
    assert scan.corrections[0].found["dark_pixels_at_dtype_ceiling"] == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_floating_calibration_does_not_infer_an_adc_limit(dtype, caplog):
    scan = _frames(np.full((3, 1, 7), 65535, dtype), np.zeros((1, 1, 7), dtype)).corrected()
    assert not any("ceiling" in key for key in scan.corrections[0].found)
    assert "dtype ceiling" not in caplog.text


def test_a_single_ceiling_sample_does_not_claim_a_whole_set_is_clipped(caplog):
    flats = np.full((5, 1, 7), 65534, np.uint16)
    flats[0, 0, 3] = 65535
    scan = _frames(flats, np.zeros((1, 1, 7), np.uint16)).corrected()
    assert not any("ceiling" in key for key in scan.corrections[0].found)
    assert "dtype ceiling" not in caplog.text


def test_flat_sets_count_ceiling_pixels_in_each_calibration_set():
    flats = np.full((6, 1, 7), 1000, np.uint16)
    flats[:3, 0, 2] = 65535
    flats[3:, 0, 4] = 65535
    scan = _frames(
        flats, np.zeros((1, 1, 7), np.uint16), positions=np.array([0, 0, 0, 10, 10, 10])
    ).corrected()
    assert scan.corrections[0].found["flat_pixels_at_dtype_ceiling"] == 2
    assert scan.corrections[0].settings["flat_positions"] == [0, 10]


class _LazyCalibration:
    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError("calibrations must remain lazily batched")

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        self.reads.append(index)
        return self.values[index]


def test_ceiling_diagnostics_reuse_the_calibration_mean_without_another_read():
    values = np.full((5, 1, 7), 1000, np.uint16)
    values[:, 0, 3] = 65535
    lazy = _LazyCalibration(values)
    scan = _frames(lazy, np.zeros((1, 1, 7), np.uint16)).corrected()
    assert scan.corrections[0].found["flat_pixels_at_dtype_ceiling"] == 1
    assert len(lazy.reads) == 1
    assert isinstance(lazy.reads[0], slice)


def test_array_like_calibrations_do_not_recover_lost_sensor_dtype():
    darks = np.full((3, 1, 7), 17, np.uint8)
    darks[:, 0, 2] = 255
    # Python integer lists infer the platform integer type, not the camera's
    # uint8 storage limit; do not invent ADC metadata after that conversion.
    scan = _frames(np.full((3, 1, 7), 1000, np.uint16), darks.tolist()).corrected()
    assert "dark_pixels_at_dtype_ceiling" not in scan.corrections[0].found


def test_ceiling_diagnostic_survives_scan_save_and_load(tmp_path):
    flats = np.full((5, 1, 7), 1000, np.uint16)
    flats[:, 0, 3] = 65535
    scan = _frames(flats, np.zeros((1, 1, 7), np.uint16)).corrected()
    assert scan.corrections[0].found["flat_pixels_at_dtype_ceiling"] == 1
    path = tmp_path / "corrected.nxs"
    tj.save(path, scan)
    restored = tj.load(path)
    assert restored.corrections == scan.corrections
    np.testing.assert_array_equal(restored.projections, scan.projections)
