"""Raw TIFF loading and detector windows retain only header metadata until read."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

import tomojax as tj
from tomojax.io.api import read_tiff_frames


@pytest.mark.parametrize("compression", [None, "deflate"])
@pytest.mark.parametrize("tiled", [False, True])
def test_lazy_tiff_raw_windows_and_selection_match_arrays_without_eager_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compression: str | None, *, tiled: bool
) -> None:
    shape = (17, 33)
    pattern = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    counts = np.stack([500 + pattern + i for i in range(4)])
    flats = np.stack([1500 + pattern, 1600 + pattern])
    darks = np.stack([10 + pattern, 20 + pattern])
    for name, pixels in (("counts", counts), ("flats", flats), ("darks", darks)):
        tifffile.imwrite(
            tmp_path / f"{name}.tif",
            pixels,
            photometric="minisblack",
            compression=compression,
            tile=(16, 16) if tiled else None,
        )
    calls = []
    original = tifffile.TiffPage.asarray

    def asarray(page, *args, **kwargs):
        calls.append(kwargs.get("out"))
        return original(page, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffPage, "asarray", asarray)
    frames = tj.load_frames(
        tmp_path / "counts.tif",
        angles=[0, 45, 90, 135],
        flats=tmp_path / "flats.tif",
        darks=tmp_path / "darks.tif",
    )
    narrow = frames.selected([0, 2, 3]).cropped(slice(1, 17), slice(2, 33))
    narrow = narrow.cropped(slice(2, 8), slice(3, 10))
    assert calls == []
    assert narrow.flats.shape == (2, 6, 7)
    assert narrow.darks.shape == (2, 6, 7)
    np.testing.assert_array_equal(np.asarray(narrow.counts), counts[[0, 2, 3], 3:9, 5:12])
    np.testing.assert_array_equal(np.asarray(narrow.flats), flats[:, 3:9, 5:12])
    np.testing.assert_array_equal(np.asarray(narrow.darks), darks[:, 3:9, 5:12])
    # Compressed/tiled pages must not create temporary mapped output files.
    assert all(
        out == "memmap" if compression is None and not tiled else out is None for out in calls
    )
    with pytest.raises(ValueError, match="without copying"):
        np.array(narrow.flats, copy=False)
    reference = replace(frames, counts=counts, flats=flats, darks=darks).corrected()
    actual = narrow.corrected(batch_views=1)
    np.testing.assert_array_equal(actual.projections, reference.projections[[0, 2, 3], 3:9, 5:12])
    assert actual.corrections == reference.corrections


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_lazy_tiff_views_follow_natural_file_order_all_series_and_numpy_indices(
    tmp_path: Path, byteorder: str
) -> None:
    pages = np.arange(4 * 5 * 7, dtype=np.uint16).reshape(4, 5, 7)
    tifffile.imwrite(tmp_path / "view10.tiff", pages[3], byteorder=byteorder)
    tifffile.imwrite(tmp_path / "view1.TIF", pages[0], byteorder=byteorder)
    with tifffile.TiffWriter(tmp_path / "view2.tif", byteorder=byteorder, bigtiff=True) as writer:
        writer.write(pages[1], compression="deflate")
        writer.write(pages[2], compression="deflate")
    stack = read_tiff_frames(tmp_path, lazy=True)
    assert stack.shape == pages.shape
    assert len(stack) == 4 and stack.ndim == 3 and stack.dtype == pages.dtype
    for selected in [
        slice(None),
        slice(None, None, 2),
        slice(None, None, -1),
        -1,
        0,
        np.array([0, 2, 3]),
        np.array([False, True, False, True]),
    ]:
        np.testing.assert_array_equal(stack[selected], pages[selected])
    np.testing.assert_array_equal(stack.frames_at(np.array([0, 2, 3]))[:], pages[[0, 2, 3]])
    assert isinstance(read_tiff_frames(tmp_path), np.ndarray)
    np.testing.assert_array_equal(np.asarray(stack, dtype=np.float32), pages.astype(np.float32))
    np.testing.assert_array_equal(stack[:0], pages[:0])
    with pytest.raises(IndexError):
        _ = stack[4]
    with pytest.raises(IndexError):
        _ = stack[-5]


def test_lazy_tiff_is_safe_for_concurrent_independent_windows(tmp_path: Path) -> None:
    values = np.arange(8 * 17 * 33, dtype=np.uint16).reshape(8, 17, 33)
    path = tmp_path / "pages.tif"
    tifffile.imwrite(path, values, photometric="minisblack", compression="deflate")
    stack = read_tiff_frames(path, lazy=True)
    windows = [stack.window(slice(i, i + 3), slice(i, i + 4)) for i in range(6)]
    with ThreadPoolExecutor(max_workers=6) as reader:
        results = list(reader.map(np.asarray, windows))
    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, values[:, i : i + 3, i : i + 4])


@pytest.mark.parametrize("invalid", ["colour", "shape", "dtype"])
def test_lazy_tiff_rejects_inconsistent_headers_without_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    tifffile.imwrite(tmp_path / "1.tif", np.zeros((5, 7), np.uint16))
    if invalid == "colour":
        tifffile.imwrite(tmp_path / "2.tif", np.zeros((5, 7, 3), np.uint8), photometric="rgb")
    else:
        tifffile.imwrite(
            tmp_path / "2.tif",
            np.zeros(
                (6 if invalid == "shape" else 5, 7), np.float32 if invalid == "dtype" else np.uint16
            ),
        )

    def no_pixels(*args, **kwargs):
        raise AssertionError("invalid stack metadata must not decode pixels")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", no_pixels)
    with pytest.raises(ValueError, match="2-D images of one size and dtype"):
        tj.load_frames(tmp_path, angles=[0, 90], white_level=100)
