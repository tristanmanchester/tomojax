"""Formatted VDS dataset names retain physical backing-file ownership."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.io.api import locate_frames

if TYPE_CHECKING:
    from pathlib import Path

HDF5 = cast(Any, h5py)  # Native VDS/external-storage APIs are missing from supplied stubs.


def _acquisition(root: Path, filename_printf: bool, nested: bool, *, spelling: str = "plain"):
    shape = (8, 2, 3)
    expected = np.full(shape, 1000.0, np.float32)
    raw = root / "counts.bin"
    raw.write_bytes(expected.tobytes())
    unused = root / "unused.bin"
    unused.write_bytes(np.zeros(shape, np.float32).tobytes())
    files = (
        [root / f"source{block}.h5" for block in range(2)]
        if filename_printf
        else [root / "source.h5"]
    )
    for index, path in enumerate(files):
        with HDF5.File(path, "w") as file:
            for block in [index] if filename_printf else range(2):
                name = f"group{block}/data{block}" if nested else f"data{block}"
                file.create_dataset(
                    name, shape=(4, 2, 3), dtype="f4", external=[(str(raw), block * 96, 96)]
                )
            if nested and not filename_printf:
                file.require_group("group2")  # native end probe must reach a missing leaf
            # This dataset has no native matching block (a mismatched repeated
            # substitution, or a noncanonical leading-zero integer).
            decoy = (
                "group0/data1"
                if nested
                else "data1"
                if filename_printf and index == 0
                else "data01"
            )
            if decoy not in file:
                file.create_dataset(
                    decoy, shape=(4, 2, 3), dtype="f4", external=[(str(unused), 0, 96)]
                )
    path = root / "views.h5"
    filename = "source%b.h5" if filename_printf else "source.h5"
    dataset = "/group%b/data%b" if nested else "/data%b"
    if spelling == "slashes":
        dataset = dataset.replace("/", "//")
    elif spelling == "dots":
        dataset = dataset.replace("/", "/./")

    with HDF5.File(path, "w") as file:
        layout = HDF5.VirtualLayout(shape=shape, maxshape=(None, 2, 3), dtype="f4")
        source = HDF5.VirtualSource(filename, dataset, shape=(4, 2, 3))
        space = HDF5.h5s.create_simple(shape, (HDF5.h5s.UNLIMITED, 2, 3))
        space.select_hyperslab(
            (0, 0, 0), count=(HDF5.h5s.UNLIMITED, 1, 1), stride=(4, 2, 3), block=(4, 2, 3)
        )
        layout.dcpl.set_virtual(space, filename.encode(), dataset.encode(), source.sel.id)
        file.create_virtual_dataset("data", layout)
    stack = locate_frames(str(path), data_path="/data").stack
    np.testing.assert_array_equal(np.asarray(stack), expected)  # independently native HDF5 read
    geometry = tj.ParallelGeometry(
        tj.Grid(3, 3, 2, 1, 1, 1), tj.Detector(3, 2, 1, 1), np.arange(8).tolist()
    )
    return tj.Frames(stack, geometry, white_level=2000), raw, unused, expected


@pytest.mark.parametrize("filename_printf", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("role", ["counts", "flats", "darks"])
def test_formatted_dataset_backing_file_is_rejected_without_reading_or_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename_printf: bool, nested: bool, role: str
) -> None:
    frames, raw, _, expected = _acquisition(tmp_path, filename_printf, nested)
    source = frames.counts
    if role != "counts":
        counts = np.full(expected.shape, 500 if role == "flats" else 1500, np.float32)
        frames = replace(frames, counts=counts, **{role: source})
    frames = frames.selected([0, 2, 5])
    out = np.memmap(raw, mode="r+", dtype="f4", shape=expected.shape)[:3]
    before = raw.read_bytes()

    def unread(*args: object, **kwargs: object) -> None:
        raise AssertionError("alias must fail before any acquisition read")

    monkeypatch.setattr(type(source), "__getitem__", unread)
    with pytest.raises(ValueError, match="separate .*files"):
        frames.corrected(out=out)
    assert raw.read_bytes() == before


@pytest.mark.parametrize("filename_printf", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_native_unused_formatted_dataset_keeps_independent_output_usable(
    tmp_path: Path, filename_printf: bool, nested: bool
) -> None:
    frames, raw, unused, pixels = _acquisition(tmp_path, filename_printf, nested)
    expected = frames.corrected()
    out = np.memmap(unused, mode="r+", dtype="f4", shape=pixels.shape)
    actual = frames.corrected(out=out)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(np.asarray(frames.counts), pixels)
    assert raw.read_bytes() == pixels.tobytes()
    assert actual.corrections == expected.corrections


@pytest.mark.parametrize("spelling", ["slashes", "dots"])
def test_native_dataset_path_normalization_preserves_backing_file_identity(
    tmp_path: Path, spelling: str
) -> None:
    frames, raw, _, pixels = _acquisition(tmp_path, True, True, spelling=spelling)
    out = np.memmap(raw, mode="r+", dtype="f4", shape=pixels.shape)
    with pytest.raises(ValueError, match="separate .*files"):
        frames.corrected(out=out)
    assert raw.read_bytes() == pixels.tobytes()
