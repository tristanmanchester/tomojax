"""Selected chunks overlap their download, retaining at most two device selections."""

from __future__ import annotations

import weakref

import jax
import numpy as np
import pytest

import tomojax as tj
import tomojax._scan as scan_module


@pytest.mark.parametrize("source", ["scan", "frames"])
@pytest.mark.parametrize("views_per_chunk", [1, 2, 3, 7])
def test_device_selection_downloads_overlap_but_stay_bounded(source, views_per_chunk, monkeypatch):
    values = np.arange(13 * 9 * 11, dtype=np.uint16 if source == "frames" else np.float32).reshape(
        13, 9, 11
    )
    data = jax.device_put(values)
    geometry = tj.ParallelGeometry(
        tj.Grid(11, 11, 9, 1, 1, 1), tj.Detector(11, 9, 1, 1), np.linspace(0, 180, 13)
    )
    original = tj.Scan(data, geometry) if source == "scan" else tj.Frames(data, geometry)
    monkeypatch.setattr(
        scan_module, "_TRANSFER_BATCH_BYTES", views_per_chunk * 9 * 11 * values.dtype.itemsize
    )
    events, live = [], []
    asarray, download = np.asarray, type(data).copy_to_host_async

    def scheduled(array):
        live.append(weakref.ref(array))
        assert sum(ref() is not None for ref in live) <= 2
        events.append(("schedule", tuple(array.shape)))
        return download(array)

    def collected(array, *args, **kwargs):
        if isinstance(array, jax.Array):
            events.append(("collect", tuple(array.shape)))
        return asarray(array, *args, **kwargs)

    monkeypatch.setattr(type(data), "copy_to_host_async", scheduled)
    monkeypatch.setattr(np, "asarray", collected)
    result = original.selected([0, 2, 4, 7, 12])
    shape = [(min(views_per_chunk, 5 - i), 9, 11) for i in range(0, 5, views_per_chunk)]
    expected = []
    for i, chunk in enumerate(shape):
        expected.append(("schedule", chunk))
        if i:
            expected.append(("collect", shape[i - 1]))
    expected.append(("collect", shape[-1]))
    assert events == expected
    actual = result.projections if source == "scan" else result.counts
    assert actual.flags.writeable and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, values[[0, 2, 4, 7, 12]])
    assert not any(ref() is not None for ref in live)
