"""Writable device selections use bounded downloads, including a partial batch."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tomojax as tj
import tomojax._scan as scan_module


@pytest.mark.parametrize("source", ["scan", "frames"])
@pytest.mark.parametrize("views_per_chunk", [1, 2, 3])
@pytest.mark.parametrize("keep_all", [False, True])
def test_device_selection_downloads_in_bounded_view_chunks(
    source, views_per_chunk, keep_all, monkeypatch
):
    dtype = np.uint16 if source == "frames" else np.float32
    values = np.arange(13 * 9 * 11, dtype=dtype).reshape(13, 9, 11)
    if source == "scan":
        values *= 0.001
    data = jax.device_put(values)
    geometry = tj.ParallelGeometry(
        tj.Grid(11, 11, 9, 1.0, 1.0, 1.0),
        tj.Detector(11, 9, 0.7, 1.1),
        [i * 180 / 13 for i in range(13)],
    )
    original = tj.Scan(data, geometry) if source == "scan" else tj.Frames(data, geometry)
    view_bytes = 9 * 11 * values.dtype.itemsize
    monkeypatch.setattr(
        scan_module, "_TRANSFER_BATCH_BYTES", views_per_chunk * view_bytes, raising=False
    )
    asarray, downloads = np.asarray, []

    def tracked(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            downloads.append(tuple(value.shape))
        return asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", tracked)
    kept = np.arange(13) if keep_all else np.array([1, 2, 5, 9, 12])
    result = original.selected(kept)
    expected_shapes = [
        (min(views_per_chunk, len(kept) - start), 9, 11)
        for start in range(0, len(kept), views_per_chunk)
    ]
    assert downloads == expected_shapes
    actual = result.projections if isinstance(result, tj.Scan) else result.counts
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == values.dtype and actual.flags.writeable
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, values[kept])
    np.testing.assert_array_equal(result.geometry.angles, np.asarray(geometry.angles)[kept])
    actual[0, 0, 0] = 9
    assert float(data[kept[0], 0, 0]) == values[kept[0], 0, 0]


def test_one_view_larger_than_the_target_still_makes_progress(monkeypatch):
    values = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    geometry = tj.ParallelGeometry(
        tj.Grid(5, 5, 3, 1.0, 1.0, 1.0), tj.Detector(5, 3, 1.0, 1.0), [0, 45, 90, 135]
    )
    monkeypatch.setattr(scan_module, "_TRANSFER_BATCH_BYTES", 1, raising=False)
    result = tj.Scan(jax.device_put(values), geometry).selected([0, 2])
    np.testing.assert_array_equal(result.projections, values[[0, 2]])
