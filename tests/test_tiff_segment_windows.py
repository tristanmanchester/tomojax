"""TIFF windows decode only intersecting scalar strips/tiles, without changing pixels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from tomojax.io.api import read_tiff_frames


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("compression", [None, "deflate", "lzma"])
@pytest.mark.parametrize("tiled", [False, True])
def test_segment_windows_preserve_lossless_pixels_and_tails(
    tmp_path: Path, dtype, byteorder: str, compression: str | None, *, tiled: bool
) -> None:
    expected = np.random.default_rng(16).integers(0, 200, (3, 37, 67)).astype(dtype)
    path = tmp_path / "counts.tif"
    tifffile.imwrite(
        path,
        expected,
        photometric="minisblack",
        byteorder=byteorder,
        compression=compression,
        tile=(16, 16) if tiled else None,
        rowsperstrip=8,
    )
    frames = read_tiff_frames(path, lazy=True)
    for rows, cols in [
        (slice(0, 1), slice(0, 1)),
        (slice(9, 28), slice(17, 50)),
        (slice(34, 37), slice(64, 67)),
        (slice(3, 19), slice(None)),
    ]:
        actual = np.asarray(frames.window(rows, cols))
        np.testing.assert_array_equal(actual, expected[:, rows, cols])
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_cropped_tiff_decompresses_only_intersecting_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tiled: bool, compressed: bool
) -> None:
    expected = np.arange(2 * 37 * 67, dtype=np.uint16).reshape(2, 37, 67)
    path = tmp_path / "counts.tif"
    tifffile.imwrite(
        path,
        expected,
        photometric="minisblack",
        rowsperstrip=8,
        tile=(16, 16) if tiled else None,
        compression="deflate" if compressed else None,
    )
    original = tifffile.TiffPage.decode
    calls = []

    def decoder(page):
        native = original.__get__(page, type(page))

        def decode(data, index, *args, **kwargs):
            calls.append((page.index, index))
            return native(data, index, *args, **kwargs)

        return decode

    monkeypatch.setattr(tifffile.TiffPage, "decode", property(decoder))
    frames = read_tiff_frames(path, lazy=True).window(slice(9, 28), slice(17, 50))
    np.testing.assert_array_equal(np.asarray(frames), expected[:, 9:28, 17:50])
    if tiled:
        # Two tile rows, three columns out of a five-column tile lattice.
        indices = [1, 2, 3, 6, 7, 8]
    elif compressed:
        indices = [1, 2, 3]
    else:
        indices = []  # ordinary contiguous strips use a direct measured-pixel map
    assert calls == [(page, index) for page in range(2) for index in indices]


@pytest.mark.parametrize("compressed", [False, True])
def test_cropped_tiff_preserves_empty_tiles_and_declared_nodata(
    tmp_path: Path, *, compressed: bool
) -> None:
    path = tmp_path / "sparse.tif"
    tiles = [
        np.arange(256, dtype=np.uint16).reshape(16, 16),
        None,
        None,
        np.full((16, 16), 7, np.uint16),
        None,
        np.full((16, 16), 99, np.uint16),
    ]
    tifffile.imwrite(
        path,
        iter(tiles),
        shape=(17, 33),
        dtype=np.uint16,
        tile=(16, 16),
        photometric="minisblack",
        compression="deflate" if compressed else None,
        extratags=[(42113, "s", 0, "17", False)],
    )
    with tifffile.TiffFile(path) as file:
        expected = file.pages[0].asarray()
    frames = read_tiff_frames(path, lazy=True)
    for rows, cols in [(slice(4, 16), slice(3, 19)), (slice(15, 17), slice(15, 33))]:
        np.testing.assert_array_equal(
            np.asarray(frames.window(rows, cols))[0], expected[rows, cols]
        )


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_cropped_tiff_preserves_horizontal_predictor(tmp_path: Path, byteorder: str) -> None:
    expected = np.arange(37 * 67, dtype=np.uint16).reshape(37, 67)
    path = tmp_path / "predicted.tif"
    tifffile.imwrite(
        path,
        expected,
        photometric="minisblack",
        byteorder=byteorder,
        compression="deflate",
        predictor=True,
        rowsperstrip=8,
    )
    frames = read_tiff_frames(path, lazy=True).window(slice(11, 25), slice(13, 39))
    np.testing.assert_array_equal(np.asarray(frames)[0], expected[11:25, 13:39])


@pytest.mark.parametrize("codec", ["jpeg", "lzw", "zstd", "floatpred"])
@pytest.mark.parametrize("tiled", [False, True])
def test_optional_tiff_codecs_preserve_native_decoded_values(
    tmp_path: Path, codec: str, *, tiled: bool
) -> None:
    pytest.importorskip("imagecodecs")
    dtype = np.float32 if codec == "floatpred" else np.uint8 if codec == "jpeg" else np.uint16
    pixels = np.random.default_rng(12).integers(0, 200, (2, 37, 67)).astype(dtype)
    path = tmp_path / "codec.tif"
    tifffile.imwrite(
        path,
        pixels,
        photometric="minisblack",
        rowsperstrip=8,
        compression="deflate" if codec == "floatpred" else codec,
        predictor=True if codec == "floatpred" else None,
        tile=(16, 16) if tiled else None,
    )
    with tifffile.TiffFile(path) as file:
        native = np.stack([page.asarray() for page in file.pages])
    frames = read_tiff_frames(path, lazy=True).window(slice(9, 28), slice(17, 50))
    np.testing.assert_array_equal(np.asarray(frames), native[:, 9:28, 17:50])
