"""Regular view selections retain native HDF5 decoding and bounded extra storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from tomojax.io.api import locate_frames

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("dtype", ["u2", ">u2", "f4", ">f4"])
@pytest.mark.parametrize("chunks", [None, (1, 17, 33), (3, 17, 33), (1, 5, 11)])
@pytest.mark.parametrize("filtered", [False, True])
def test_regular_selections_preserve_native_decoding_and_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
    chunks: tuple[int, int, int] | None,
    filtered: bool,
) -> None:
    values = np.arange(67 * 17 * 33).reshape(67, 17, 33) % 997 + 100
    expected = np.asarray(values * 0.125 + 0.1 if "f" in dtype else values, dtype=dtype)
    path = tmp_path / "frames.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "data",
            data=expected,
            chunks=chunks,
            compression="gzip" if filtered else None,
            shuffle=filtered,
            fletcher32=filtered,
        )
    stack = locate_frames(str(path), data_path="/data").stack
    allocated = []
    original_empty = np.empty

    def empty(*args: object, **kwargs: object) -> np.ndarray:
        result = original_empty(*args, **kwargs)
        allocated.append(result.nbytes)
        return result

    monkeypatch.setattr(np, "empty", empty)
    for windowed in [False, True]:
        if windowed:
            reader = stack.window(slice(1, 16), slice(2, 30)).window(slice(3, 9), slice(1, 17))
            selected = expected[:, 4:10, 3:19]
        else:
            reader, selected = stack, expected
        for stride in [2, 3, 5, 11]:
            allocated.clear()
            actual = reader.frames_at(slice(2, 60, stride))[:]
            np.testing.assert_array_equal(actual, selected[2:60:stride])
            assert actual.dtype == expected.dtype
            assert actual.flags.c_contiguous
            assert allocated[0] == actual.nbytes
            assert len(allocated) <= 2
            if len(allocated) == 2:
                assert allocated[1] <= min(16 * 2**20, actual.nbytes)


@pytest.mark.parametrize("kind", ["external", "virtual"])
def test_regular_reads_preserve_linked_sources_and_virtual_fill(tmp_path: Path, kind: str) -> None:
    expected = (np.arange(67 * 17 * 33).reshape(67, 17, 33) % 997 + 100).astype(np.uint16)
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("images", data=expected, chunks=(1, 17, 33), compression="gzip")
    linked = tmp_path / "linked.h5"
    with h5py.File(linked, "w") as file:
        if kind == "external":
            file["data"] = h5py.ExternalLink(path.name, "/images")
        else:
            layout = h5py.VirtualLayout(shape=expected.shape, dtype=expected.dtype)
            source = h5py.VirtualSource(str(path), "images", shape=expected.shape)
            layout[:30] = source[:30]
            layout[40:] = source[40:]
            file.create_virtual_dataset("data", layout, fillvalue=317)
            expected[30:40] = 317
    stack = locate_frames(str(linked), data_path="/data").stack
    actual = stack.window(slice(3, 12), slice(5, 19)).frames_at(slice(None, None, 3))[:]
    np.testing.assert_array_equal(actual, expected[::3, 3:12, 5:19])


def test_sparse_and_irregular_reads_need_no_padded_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = np.arange(67 * 7 * 9, dtype=np.uint16).reshape(67, 7, 9)
    path = tmp_path / "frames.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=expected, chunks=(1, 7, 9), fillvalue=123)
    stack = locate_frames(str(path), data_path="/data").stack
    allocated = []
    original_empty = np.empty

    def empty(*args: object, **kwargs: object) -> np.ndarray:
        result = original_empty(*args, **kwargs)
        allocated.append(result.nbytes)
        return result

    monkeypatch.setattr(np, "empty", empty)
    for indices in [[], [0, 66], [1, 2, 5, 7, 8]]:
        allocated.clear()
        actual = stack[np.asarray(indices, np.int64)]
        np.testing.assert_array_equal(actual, expected[indices])
        assert allocated == [actual.nbytes]
    for indices in [[0, 0], [3, 2]]:
        with pytest.raises(ValueError, match="increasing order"):
            stack[np.asarray(indices, np.int64)]


def test_regular_read_workspace_remains_bounded_for_large_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The selected result exceeds the scratch cap and needs several read groups.
    expected = np.broadcast_to(np.arange(66, dtype=np.uint16)[:, None, None], (66, 512, 512))
    path = tmp_path / "frames.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=expected, chunks=(1, 512, 512))
    stack = locate_frames(str(path), data_path="/data").stack
    allocated = []
    original_empty = np.empty

    def empty(*args: object, **kwargs: object) -> np.ndarray:
        result = original_empty(*args, **kwargs)
        allocated.append(result.nbytes)
        return result

    monkeypatch.setattr(np, "empty", empty)
    actual = stack[::2]
    np.testing.assert_array_equal(actual, expected[::2])
    assert allocated[0] == actual.nbytes > 16 * 2**20
    assert len(allocated) <= 2
    if len(allocated) == 2:
        assert allocated[1] <= 16 * 2**20
