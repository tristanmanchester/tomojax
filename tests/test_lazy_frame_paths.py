"""Lazy raw stacks retain their acquisition identity across working-directory changes."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

import tomojax as tj
from tomojax.io.api import locate_frames, read_tiff_frames


def _write_scan(directory: Path, layout: str, *, other: bool) -> tuple[np.ndarray, ...]:
    directory.mkdir()
    counts = np.broadcast_to(
        np.array([6000, 8000] if other else [2000, 3000], np.uint16)[:, None, None], (2, 5, 7)
    ).copy()
    flats = np.full((1, 5, 7), 10000 if other else 6000, np.uint16)
    darks = np.full((1, 5, 7), 200 if other else 100, np.uint16)
    if layout == "hdf5":
        with h5py.File(directory / "scan.h5", "w") as file:
            file.create_dataset(
                "entry/instrument/detector/data", data=np.concatenate((flats, darks, counts))
            )
            file.create_dataset("entry/instrument/detector/image_key", data=[1, 2, 0, 0])
            file.create_dataset("entry/sample/transformations/rotation_angle", data=[0, 0, 0, 90])
    else:
        for name, values in (("counts", counts), ("flats", flats), ("darks", darks)):
            tifffile.imwrite(directory / f"{name}.tif", values, photometric="minisblack")
    return counts, flats, darks


@pytest.mark.parametrize("layout", ["hdf5", "tiff"])
def test_load_frames_preserves_relative_count_and_calibration_paths_after_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    original = tmp_path / "original"
    other = tmp_path / "other"
    counts, flats, darks = _write_scan(original, layout, other=False)
    _write_scan(other, layout, other=True)
    monkeypatch.chdir(original)
    if layout == "hdf5":
        frames = tj.load_frames("scan.h5")
    else:
        frames = tj.load_frames("counts.tif", angles=[0, 90], flats="flats.tif", darks="darks.tif")
    reference = tj.Frames(counts, frames.geometry, flats=flats, darks=darks)
    expected = reference.cropped(slice(1, 4), slice(2, 6)).corrected(batch_views=1)
    monkeypatch.chdir(other)  # valid same-named data exist, so a wrong read would be silent
    actual = frames.cropped(slice(1, 4), slice(2, 6)).corrected(batch_views=1)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.corrections == expected.corrections
    np.testing.assert_array_equal(np.asarray(frames.counts), counts)
    np.testing.assert_array_equal(np.asarray(frames.flats), flats)
    np.testing.assert_array_equal(np.asarray(frames.darks), darks)


@pytest.mark.parametrize("layout", ["hdf5", "tiff"])
def test_standalone_lazy_io_anchors_relative_source_without_loading_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    original = tmp_path / "original"
    other = tmp_path / "other"
    counts, flats, darks = _write_scan(original, layout, other=False)
    _write_scan(other, layout, other=True)
    monkeypatch.chdir(original)
    if layout == "hdf5":
        stack = locate_frames("scan.h5").stack
        expected = np.concatenate((flats, darks, counts))
    else:
        stack = read_tiff_frames("counts.tif", lazy=True)
        expected = counts
    monkeypatch.chdir(other)
    np.testing.assert_array_equal(
        np.asarray(stack.window(slice(1, 4), slice(2, 6))), expected[:, 1:4, 2:6]
    )
