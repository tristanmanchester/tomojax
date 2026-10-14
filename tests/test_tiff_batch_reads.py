"""Full-page stacking retains native TIFF decoding and conservative mixed-layout reads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import TiffFile, TiffWriter

from tomojax.io.api import read_tiff_frames

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("dtype", ["bool", "uint16", ">u2", "float32", "float64"])
@pytest.mark.parametrize("layout", ["strips", "deflate", "tiles", "mixed", "strip_lengths"])
def test_full_tiff_batch_reads_preserve_pixels_order_and_crops(
    tmp_path: Path, dtype: str, layout: str
) -> None:
    pattern = np.arange(9 * 33 * 35).reshape(9, 33, 35) % 997 + 100
    expected = np.asarray(
        pattern % 2 if dtype == "bool" else pattern * 0.001 + 0.1 if "float" in dtype else pattern,
        dtype=dtype,
    )
    path = tmp_path / "frames.tif"
    with TiffWriter(path, bigtiff=True, byteorder=">" if dtype.startswith(">") else "<") as file:
        for index, frame in enumerate(expected):
            file.write(
                frame,
                compression="deflate"
                if layout == "deflate" or (layout == "mixed" and index % 2)
                else None,
                tile=(16, 16) if layout == "tiles" else None,
                rowsperstrip=5 if layout == "strip_lengths" and index % 2 else 11,
                metadata=None,
                photometric="minisblack",
            )
    stack = read_tiff_frames(path, lazy=True)
    selections = [slice(None), slice(None, None, -1), [8, 1, 1, 4], np.arange(9) % 2 == 0, -1]
    for selection in selections:
        actual = stack[selection]
        np.testing.assert_array_equal(actual, expected[selection])
        assert actual.flags.c_contiguous and actual.flags.writeable
    cropped = stack.window(slice(0, 25), slice(0, 31)).window(slice(3, 13), slice(2, 19))
    np.testing.assert_array_equal(cropped[:], expected[:, 3:13, 2:19])


def test_tiff_batch_reads_preserve_each_sparse_pages_nodata(tmp_path: Path) -> None:
    path = tmp_path / "sparse.tif"
    fills = [123, 456, 789]
    with TiffWriter(path, bigtiff=True) as file:
        for fill in fills:
            file.write(
                iter([b""] * 3),
                shape=(33, 35),
                dtype=np.uint16,
                compression="deflate",
                rowsperstrip=11,
                metadata=None,
                extratags=[(42113, "s", 0, str(fill), False)],
            )
    stack = read_tiff_frames(path, lazy=True)
    # Native decoder hashes are equal despite the distinct page fill values.
    with TiffFile(path) as file:
        assert len({page.hash for page in file.pages}) == 1
    expected = np.broadcast_to(np.asarray(fills, np.uint16)[:, None, None], (3, 33, 35))
    np.testing.assert_array_equal(stack[:], expected)
    np.testing.assert_array_equal(
        stack.window(slice(2, 19), slice(3, 22))[:], expected[:, 2:19, 3:22]
    )


def test_native_tiff_groups_bound_headers_and_leave_crops_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "frames.tif"
    expected = np.broadcast_to(np.arange(301, dtype=np.uint16)[:, None, None], (301, 33, 35))
    with TiffWriter(path, bigtiff=True) as file:
        for frame in expected:
            file.write(frame, compression="deflate", rowsperstrip=11, metadata=None)
    stack = read_tiff_frames(path, lazy=True)
    batches = []
    original = TiffFile.asarray

    def asarray(file: TiffFile, *args: object, **kwargs: object) -> np.ndarray:
        batches.append(len(kwargs["key"]))
        return original(file, *args, **kwargs)

    monkeypatch.setattr(TiffFile, "asarray", asarray)
    np.testing.assert_array_equal(stack[:], expected)
    assert sum(batches) == len(expected) and max(batches) <= 256
    batches.clear()
    np.testing.assert_array_equal(stack.window(slice(0, 5), slice(None))[:], expected[:, :5])
    assert not batches


@pytest.mark.parametrize("codec", ["jpeg", "lzw", "zstd", "floatpred"])
@pytest.mark.parametrize("tiled", [False, True])
def test_full_tiff_batches_preserve_optional_native_decoders(
    tmp_path: Path, codec: str, *, tiled: bool
) -> None:
    pytest.importorskip("imagecodecs")
    dtype = np.float32 if codec == "floatpred" else np.uint8 if codec == "jpeg" else np.uint16
    pixels = np.random.default_rng(94).integers(0, 200, (5, 37, 67)).astype(dtype)
    path = tmp_path / "frames.tif"
    with TiffWriter(path, bigtiff=True) as file:
        for frame in pixels:
            file.write(
                frame,
                compression="deflate" if codec == "floatpred" else codec,
                predictor=True if codec == "floatpred" else None,
                tile=(16, 16) if tiled else None,
                rowsperstrip=8,
                photometric="minisblack",
                metadata=None,
            )
    with TiffFile(path) as file:
        expected = np.stack([page.asarray() for page in file.pages])
    stack = read_tiff_frames(path, lazy=True)
    np.testing.assert_array_equal(stack[:], expected)
    np.testing.assert_array_equal(stack[[4, 1, 1, 3]], expected[[4, 1, 1, 3]])
