from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

import tomojax as tj
from tomojax.io import load_tiff_stack
from tomojax.io.api import read_tiff_frames


@pytest.mark.parametrize("bigtiff", [False, True])
@pytest.mark.parametrize("columns", [3, 4, 7])
def test_grayscale_pages_are_views_not_colour_channels(
    tmp_path: Path, *, bigtiff: bool, columns: int
) -> None:
    expected = np.arange(3 * 5 * columns, dtype=np.uint16).reshape(3, 5, columns)
    path = tmp_path / "views.tif"
    tifffile.imwrite(path, expected, photometric="minisblack", bigtiff=bigtiff)

    actual = read_tiff_frames(path)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    dataset = load_tiff_stack(path, angles=[0.0, 60.0, 120.0])
    assert dataset.projections.shape == expected.shape
    assert dataset.projections.dtype == np.float32
    np.testing.assert_array_equal(dataset.projections, expected)


def test_every_series_is_read_in_file_and_page_order(tmp_path: Path) -> None:
    # Separate writes create separate series; imread's default selects only the first.
    with tifffile.TiffWriter(tmp_path / "view2.tif") as writer:
        for value in (2, 3):
            writer.write(np.full((5, 7), value, np.uint16), compression="deflate")
    tifffile.imwrite(tmp_path / "view10.tiff", np.full((5, 7), 10, np.uint16))
    tifffile.imwrite(tmp_path / "view1.TIF", np.full((5, 7), 1, np.uint16))

    actual = read_tiff_frames(tmp_path)
    np.testing.assert_array_equal(actual[:, 0, 0], [1, 2, 3, 10])
    dataset = load_tiff_stack(tmp_path, angles=[0.0, 30.0, 60.0, 90.0])
    np.testing.assert_array_equal(dataset.projections, actual)
    with pytest.raises(ValueError, match="does not match projection count 4"):
        load_tiff_stack(tmp_path, angles=[0.0, 30.0, 60.0])


@pytest.mark.parametrize("channels", [3, 4])
def test_colour_pages_are_not_detector_views(tmp_path: Path, channels: int) -> None:
    path = tmp_path / "colour.tif"
    tifffile.imwrite(path, np.ones((5, 7, channels), np.uint8), photometric="rgb")

    for reader in (read_tiff_frames, lambda p: load_tiff_stack(p, angles=[0.0])):
        with pytest.raises(ValueError, match="2-D images of one size"):
            reader(path)


@pytest.mark.parametrize("separate_files", [False, True])
def test_mismatched_page_sizes_fail_at_ingestion(tmp_path: Path, *, separate_files: bool) -> None:
    if separate_files:
        tifffile.imwrite(tmp_path / "1.tif", np.zeros((5, 7), np.uint16))
        tifffile.imwrite(tmp_path / "2.tif", np.zeros((6, 7), np.uint16))
        path = tmp_path
    else:
        path = tmp_path / "mismatch.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.zeros((5, 7), np.uint16))
            writer.write(np.zeros((6, 7), np.uint16))
    for reader in (read_tiff_frames, lambda p: load_tiff_stack(p, angles=[0.0, 90.0])):
        with pytest.raises(ValueError, match="2-D images of one size"):
            reader(path)


def test_raw_multipage_calibration_matches_in_memory_frames(tmp_path: Path) -> None:
    counts = np.full((3, 5, 7), 50, np.uint16)
    flats = np.full((2, 5, 7), 100, np.uint16)
    darks = np.full((2, 5, 7), 10, np.uint16)
    for name, values in (("views", counts), ("flats", flats), ("darks", darks)):
        tifffile.imwrite(tmp_path / f"{name}.tif", values, photometric="minisblack")
    frames = tj.load_frames(
        tmp_path / "views.tif",
        angles=[0.0, 60.0, 120.0],
        flats=tmp_path / "flats.tif",
        darks=tmp_path / "darks.tif",
    )
    reference = tj.Frames(counts, frames.geometry, flats=flats, darks=darks).corrected()

    actual = frames.corrected()
    np.testing.assert_array_equal(actual.projections, reference.projections)
    np.testing.assert_allclose(actual.projections, -np.log(40.0 / 90.0), rtol=1e-6)
    assert actual.corrections == reference.corrections
