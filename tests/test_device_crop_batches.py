"""Large device crops preserve read-only host windows without whole-stack staging."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tomojax as tj
import tomojax._scan as scan_module


def _geometry():
    return tj.ParallelGeometry(
        tj.Grid(11, 11, 9, 0.7, 0.7, 1.1),
        tj.Detector(11, 9, 0.7, 1.1, center=(0.2, -0.3)),
        np.linspace(0, 180, 13),
    )


@pytest.mark.parametrize("source", ["scan", "frames"])
@pytest.mark.parametrize("views_per_chunk", [1, 2, 3])
def test_large_device_crop_downloads_only_bounded_detector_windows(
    source, views_per_chunk, monkeypatch
):
    values = np.arange(13 * 9 * 11, dtype=np.uint16 if source == "frames" else np.float32).reshape(
        13, 9, 11
    )
    data = jax.device_put(values)
    flats, darks = values[:4] + 2000, values[:2]
    positions = np.array([0, 0, 13, 13])
    original = (
        tj.Scan(data, _geometry())
        if source == "scan"
        else tj.Frames(
            data,
            _geometry(),
            flats=jax.device_put(flats),
            darks=jax.device_put(darks),
            flat_positions=positions,
        )
    )
    monkeypatch.setattr(
        scan_module, "_TRANSFER_BATCH_BYTES", views_per_chunk * 7 * 5 * values.dtype.itemsize
    )
    asarray, downloads = np.asarray, []

    def tracked(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            downloads.append(tuple(value.shape))
            assert value.devices() == data.devices()
        return asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", tracked)
    result = original.cropped(slice(-8, -1), slice(-8, -3))
    expected = []
    for views in [13] if source == "scan" else [13, 4, 2]:
        batch = views if views <= views_per_chunk else views_per_chunk
        expected.extend((min(batch, views - start), 7, 5) for start in range(0, views, batch))
    assert downloads == expected
    actual = result.projections if source == "scan" else result.counts
    assert isinstance(actual, np.ndarray) and not actual.flags.writeable
    assert actual.dtype == values.dtype and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, values[:, 1:8, 3:8])
    reference = tj.Scan(values, _geometry()).cropped(slice(1, 8), slice(3, 8))
    assert result.geometry.detector == reference.geometry.detector
    assert result.geometry.grid == reference.geometry.grid
    if source == "frames":
        for field, wanted in ((result.flats, flats), (result.darks, darks)):
            assert not field.flags.writeable
            np.testing.assert_array_equal(field, wanted[:, 1:8, 3:8])
        assert result.flat_positions is positions
    np.testing.assert_array_equal(np.asarray(data), values)


@pytest.mark.parametrize("source", ["scan", "frames"])
def test_small_device_crop_keeps_the_single_read_only_host_view(source, monkeypatch):
    values = np.arange(13 * 9 * 11, dtype=np.float32).reshape(13, 9, 11)
    data = jax.device_put(values)
    original = tj.Scan(data, _geometry()) if source == "scan" else tj.Frames(data, _geometry())
    monkeypatch.setattr(scan_module, "_TRANSFER_BATCH_BYTES", 13 * 7 * 5 * values.dtype.itemsize)
    calls = []
    asynchronous = type(data).copy_to_host_async

    def scheduled(value):
        calls.append(tuple(value.shape))
        return asynchronous(value)

    monkeypatch.setattr(type(data), "copy_to_host_async", scheduled)
    result = original.cropped(slice(1, 8), slice(3, 8))
    assert calls == []
    actual = result.projections if source == "scan" else result.counts
    assert not actual.flags.writeable
    np.testing.assert_array_equal(actual, values[:, 1:8, 3:8])


def test_one_crop_view_larger_than_the_transfer_target_still_makes_progress(monkeypatch):
    values = np.arange(13 * 9 * 11, dtype=np.float32).reshape(13, 9, 11)
    monkeypatch.setattr(scan_module, "_TRANSFER_BATCH_BYTES", 1)
    result = tj.Scan(jax.device_put(values), _geometry()).cropped(slice(1, 8), slice(3, 8))
    np.testing.assert_array_equal(result.projections, values[:, 1:8, 3:8])
    assert not result.projections.flags.writeable
