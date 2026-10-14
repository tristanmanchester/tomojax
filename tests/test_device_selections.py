"""Select on the device before downloading; retain host selection ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest

import tomojax as tj

if TYPE_CHECKING:
    from pathlib import Path


def _geometry() -> tj.ParallelGeometry:
    return tj.ParallelGeometry(
        tj.Grid(11, 11, 9, 1.0, 1.0, 1.0),
        tj.Detector(11, 9, 0.7, 1.1, center=(0.2, -0.3)),
        [i * 180 / 13 for i in range(13)],
    )


def _values() -> np.ndarray:
    return np.arange(13 * 9 * 11, dtype=np.float32).reshape(13, 9, 11)


def _data(source: tj.Scan | tj.Frames) -> np.ndarray:
    data = source.projections if isinstance(source, tj.Scan) else source.counts
    assert isinstance(data, np.ndarray)
    return data


def _device_downloads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    asarray = np.asarray
    downloads = []

    def tracked(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            downloads.append(tuple(value.shape))
        return asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", tracked)
    return downloads


@pytest.mark.parametrize("source", ["scan", "frames"])
def test_device_crop_downloads_only_the_detector_window(source, monkeypatch):
    values, geometry = _values(), _geometry()
    device = jax.device_put(values)
    original = tj.Scan(device, geometry) if source == "scan" else tj.Frames(device, geometry)
    downloads = _device_downloads(monkeypatch)

    cropped = original.cropped(slice(2, 5), slice(3, 8))

    assert downloads == [(13, 3, 5)]
    actual = _data(cropped)
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, values[:, 2:5, 3:8])
    assert not actual.flags.writeable  # same read-only host view as before
    expected = tj.Scan(values, geometry).cropped(slice(2, 5), slice(3, 8))
    assert cropped.geometry.detector == expected.detector
    assert cropped.geometry.grid == geometry.grid


@pytest.mark.parametrize("source", ["scan", "frames"])
@pytest.mark.parametrize("selection", [slice(1, 11, 4), [1, 5, 9], np.arange(13) % 4 == 1])
def test_device_selection_downloads_only_kept_views_and_stays_writable(
    source, selection, monkeypatch
):
    values, geometry = _values(), _geometry()
    device = jax.device_put(values)
    positions = np.arange(13) * 1.3 + 0.2
    original = (
        tj.Scan(device, geometry)
        if source == "scan"
        else tj.Frames(device, geometry, view_positions=positions)
    )
    downloads = _device_downloads(monkeypatch)

    selected = original.selected(selection)

    assert downloads == [(3, 9, 11)]
    actual = _data(selected)
    assert isinstance(actual, np.ndarray) and actual.flags.writeable
    np.testing.assert_array_equal(actual, values[[1, 5, 9]])
    np.testing.assert_array_equal(selected.geometry.angles, np.asarray(geometry.angles)[[1, 5, 9]])
    if isinstance(selected, tj.Frames):
        np.testing.assert_array_equal(selected.view_positions, positions[[1, 5, 9]])
    actual[0, 0, 0] = -1
    assert float(device[1, 0, 0]) == values[1, 0, 0]


@pytest.mark.parametrize("source", ["scan", "frames"])
@pytest.mark.parametrize("storage", ["array", "read_only", "memmap"])
def test_host_crops_keep_aliases_but_selected_views_are_independent(
    source, storage, tmp_path: Path
):
    values = _values()
    if storage == "memmap":
        data = np.lib.format.open_memmap(tmp_path / "data.npy", mode="w+", shape=values.shape)
        data[:] = values
    else:
        data = values.copy()
        if storage == "read_only":
            data.flags.writeable = False
    original = tj.Scan(data, _geometry()) if source == "scan" else tj.Frames(data, _geometry())

    cropped = original.cropped(slice(2, 5), slice(3, 8))
    selected = original.selected([1, 5, 9])

    window = _data(cropped)
    kept = _data(selected)
    assert np.shares_memory(window, data)
    assert window.flags.writeable == data.flags.writeable
    assert not np.shares_memory(kept, data)
    assert kept.flags.writeable
    np.testing.assert_array_equal(window, values[:, 2:5, 3:8])
    np.testing.assert_array_equal(kept, values[[1, 5, 9]])
    kept[0, 0, 0] = -1
    assert data[1, 0, 0] == values[1, 0, 0]


def test_device_calibrations_crop_before_their_download(monkeypatch):
    values, geometry = _values(), _geometry()
    flats = values[:4] + 1500
    darks = values[:2] * 0.01
    original = tj.Frames(
        jax.device_put(values),
        geometry,
        flats=jax.device_put(flats),
        darks=jax.device_put(darks),
        flat_positions=np.array([0, 0, 13, 13]),
    )
    downloads = _device_downloads(monkeypatch)

    cropped = original.cropped(slice(2, 5), slice(3, 8))

    assert downloads == [(13, 3, 5), (4, 3, 5), (2, 3, 5)]
    np.testing.assert_array_equal(cropped.flats, flats[:, 2:5, 3:8])
    np.testing.assert_array_equal(cropped.darks, darks[:, 2:5, 3:8])
    np.testing.assert_array_equal(cropped.flat_positions, original.flat_positions)
