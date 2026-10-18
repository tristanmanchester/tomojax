"""Compressed detector crops read only their intersecting storage segments."""

from dataclasses import replace

import numpy as np
import pytest
import tifffile

from tomojax.io.api import read_tiff_frames


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
def test_compressed_crop_decodes_only_intersecting_segments(tmp_path, monkeypatch, tiled, flipped):
    values = np.arange(3 * 35 * 49, dtype=np.uint16).reshape(3, 35, 49)
    path = tmp_path / "windows.tif"
    layout = {"tile": (16, 16)} if tiled else {"rowsperstrip": 4}
    tifffile.imwrite(path, values, compression="deflate", photometric="minisblack", **layout)
    calls = []
    read = tifffile.FileHandle.read_segments

    def selected_segments(self, offsets, bytecounts, indices=None, **kwargs):
        calls.append(list(indices))
        return read(self, offsets, bytecounts, indices, **kwargs)

    monkeypatch.setattr(tifffile.FileHandle, "read_segments", selected_segments)
    stack = read_tiff_frames(path, lazy=True).frames_at(np.array([2, 0]))
    stack = replace(stack, flip_rows=flipped)
    oriented = values[:, ::-1] if flipped else values
    crop = stack.window(slice(31, 35), slice(30, 49))
    np.testing.assert_array_equal(np.asarray(crop), oriented[[2, 0], 31:35, 30:49])
    expected = [5, 6, 7, 9, 10, 11] if tiled else [7, 8]
    if flipped:
        expected = [1, 2, 3] if tiled else [0]
    assert calls == [expected, expected]
