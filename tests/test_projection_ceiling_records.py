"""Projection ceiling diagnostics refer to original counts, not repaired transmission."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import Paganin, RejectViews, Zingers


def _frames(counts, *, flats=None, darks=None):
    views, rows, cols = counts.shape
    geometry = tj.ParallelGeometry(
        tj.Grid(7, 7, 1, 1, 1, 1), tj.Detector(cols, rows, 1, 1), np.arange(views) * 10
    )
    return tj.Frames(
        counts,
        geometry,
        flats=np.full((3, rows, cols), 1000, np.uint16) if flats is None else flats,
        darks=np.zeros((2, rows, cols), np.uint16) if darks is None else darks,
    )


def _log(scan):
    return next(record for record in scan.corrections if record.name == "log")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_counts_ceiling_is_warned_without_changing_any_ray(dtype, caplog):
    ceiling = np.iinfo(dtype).max
    counts = np.full((5, 3, 7), ceiling // 4, dtype)
    counts[0, 1, 3] = counts[-1, 2, 4] = ceiling
    flats = np.full((3, 3, 7), ceiling // 2, dtype)
    darks = np.zeros((2, 3, 7), dtype)
    original = counts.copy()
    with caplog.at_level(logging.WARNING):
        actual = _frames(counts, flats=flats, darks=darks).corrected(batch_views=2)
    expected = _frames(counts.astype(np.float32), flats=flats, darks=darks).corrected(batch_views=3)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(counts, original)
    record = _log(actual)
    assert record.settings["counts_dtype_ceiling"] == ceiling
    assert record.found["counts_at_dtype_ceiling"] == 2
    assert "counts:" in caplog.text and "may be clipped" in caplog.text
    assert actual.projections[0, 1, 3] < 0  # valid >1 transmission must not be clipped


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_counts_do_not_infer_an_adc_ceiling(dtype, caplog):
    counts = np.full((3, 3, 7), 65535, dtype)
    scan = _frames(counts).corrected()
    assert "counts_dtype_ceiling" not in _log(scan).settings
    assert "counts_at_dtype_ceiling" not in _log(scan).found
    assert "counts:" not in caplog.text


def test_lower_adc_ceiling_cannot_be_inferred_after_widening_storage(caplog):
    counts = np.full((3, 3, 7), 65535, np.uint32)
    scan = _frames(counts).corrected()
    assert "counts_at_dtype_ceiling" not in _log(scan).found
    assert "counts:" not in caplog.text


def test_raw_ceiling_is_recorded_before_zingers_replace_it():
    counts = np.full((5, 7, 9), 500, np.uint16)
    counts[0, 3, 4] = 65535
    actual = _frames(counts).corrected(Zingers(), batch_views=2)
    expected = _frames(counts.astype(np.float32)).corrected(Zingers(), batch_views=2)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_allclose(actual.projections, np.log(2), rtol=1e-6)
    assert _log(actual).found["counts_at_dtype_ceiling"] == 1


def test_counts_and_calibration_ceiling_records_have_separate_scopes():
    counts = np.full((5, 3, 7), 500, np.uint16)
    counts[:, 1, 3] = 65535
    flats = np.full((3, 3, 7), 1000, np.uint16)
    flats[:, 1, 3] = 65535
    scan = _frames(counts, flats=flats).corrected()
    assert scan.corrections[0].found["flat_pixels_at_dtype_ceiling"] == 1
    assert _log(scan).found["counts_at_dtype_ceiling"] == 5


@pytest.mark.parametrize("layout", ["strided", "fortran", "big_endian"])
def test_count_diagnostics_preserve_strides_and_native_byte_order(layout):
    counts = np.full((5, 3, 7), 500, np.uint16)
    counts[0, 1, 2] = counts[-1, 2, 4] = 65535
    if layout == "strided":
        counts = np.repeat(counts, 2, axis=2)[:, :, ::2]
    elif layout == "fortran":
        counts = np.asfortranarray(counts)
    else:
        counts = counts.astype(">u2")
    actual = _frames(counts).corrected(batch_views=2)
    expected = _frames(counts.astype(np.float32)).corrected(batch_views=2)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert _log(actual).found["counts_at_dtype_ceiling"] == 2


class _LazyCounts:
    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.reads = []

    def __len__(self):
        return self.shape[0]

    def __array__(self, *args, **kwargs):
        raise AssertionError("counts diagnostics must not read the whole lazy stack")

    def __getitem__(self, index):
        self.reads.append(index)
        return self.values[index]


def test_counts_diagnostics_use_each_lazy_batch_exactly_once():
    values = np.full((7, 3, 7), 500, np.uint16)
    values[0, 1, 2] = values[-1, 2, 4] = 65535
    lazy = _LazyCounts(values)
    scan = _frames(lazy).corrected(batch_views=2)
    assert _log(scan).found["counts_at_dtype_ceiling"] == 2
    assert len(lazy.reads) == 4
    assert sum(len(range(7)[index]) for index in lazy.reads) == 7


def test_record_covers_original_readings_even_when_view_is_later_rejected():
    counts = np.full((31, 3, 7), 500, np.uint16)
    counts[15] = 65535
    scan = _frames(counts).corrected(RejectViews(neighbours=9, min_jump=0.1), batch_views=4)
    assert scan.projections.shape[0] == 30
    assert _log(scan).found["counts_at_dtype_ceiling"] == 21


def test_ceiling_diagnostic_and_corrected_values_survive_save_load(tmp_path):
    counts = np.full((5, 3, 7), 500, np.uint16)
    counts[0, 1, 2] = 65535
    scan = _frames(counts).corrected(batch_views=2)
    path = tmp_path / "projection_ceiling.nxs"
    tj.save(path, scan)
    restored = tj.load(path)
    assert restored.corrections == scan.corrections
    np.testing.assert_array_equal(restored.projections, scan.projections)


def test_positive_spatial_filter_keeps_raw_count_ceiling_provenance():
    counts = np.full((3, 7, 9), 60000, np.uint16)
    counts[:, 3, 4] = 65535
    flats = np.full((3, 7, 9), 65000, np.uint16)
    phase = Paganin(300, 0.6, 54.3, 13e-6)
    actual = _frames(counts, flats=flats).corrected(phase, batch_views=2)
    expected = _frames(counts.astype(np.float32), flats=flats).corrected(phase, batch_views=2)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert _log(actual).found["counts_at_dtype_ceiling"] == 3
    assert "flat_pixels_at_dtype_ceiling" not in actual.corrections[0].found


def test_above_flat_below_ceiling_is_valid_and_has_no_clipping_diagnostic(caplog):
    counts = np.full((3, 3, 7), 2000, np.uint16)
    actual = _frames(counts).corrected()
    np.testing.assert_allclose(actual.projections, -np.log(2), rtol=1e-6)
    assert "counts_at_dtype_ceiling" not in _log(actual).found
    assert "counts:" not in caplog.text


def test_ceiling_count_scratch_is_bounded_without_changing_user_batches(monkeypatch):
    counts = np.full((35, 1024, 1024), 500, np.uint16)
    counts[-1, 3, 4] = 65535
    original = np.count_nonzero
    scratch = []

    def tracked(values, *args, **kwargs):
        if isinstance(values, np.ndarray) and values.dtype.kind == "b":
            scratch.append(values.nbytes)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np, "count_nonzero", tracked)
    scan = _frames(counts).corrected(batch_views=35)
    assert _log(scan).found["counts_at_dtype_ceiling"] == 1
    assert max(scratch) <= 32 << 20
    assert scan.projections.shape == counts.shape
