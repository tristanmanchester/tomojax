"""Compressed detector crops read only their intersecting storage segments."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

import tomojax as tj
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


def test_tiff_reads_are_binary_and_bounded_not_sidecar_probes(tmp_path, monkeypatch):
    values = np.arange(2 * 5 * 7, dtype=np.uint16).reshape(2, 5, 7)
    for i, view in enumerate(values):
        tifffile.imwrite(tmp_path / f"view{i}.tif", view)
    original = Path.open
    buffers = []

    def sidecar_only(path, *args, **kwargs):
        if path.suffix == ".tif":
            assert args[0] == "rb", "TIFF pixels must not be probed as vendor text"
            buffers.append(kwargs.get("buffering", -1))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", sidecar_only)
    frames = tj.load_frames(
        tmp_path,
        angles=[0, 180],
        detector=tj.Detector(7, 5, 1, 1),
        white_level=100,
        darks=0,
    )
    np.testing.assert_array_equal(np.asarray(frames.counts), values)
    assert len(buffers) == 2 * len(values)  # headers, then indexed pixels
    assert all(0 < size <= 64 * 1024 for size in buffers)
