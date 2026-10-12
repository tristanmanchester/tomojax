"""Single-view chunks are never reused; multi-view chunks keep their sparse-read cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

from tomojax.io.api import locate_frames

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("chunk_views", "compressed"), [(None, False), (1, False), (1, True), (3, False), (3, True)]
)
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("windowed", [False, True])
def test_frame_cache_policy_preserves_reads_crops_filters_and_external_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunk_views: int | None,
    compressed: bool,
    external: bool,
    windowed: bool,
) -> None:
    expected = np.arange(8 * 7 * 9, dtype=">u2").reshape(8, 7, 9)
    path = tmp_path / "frames.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "data",
            data=expected,
            chunks=None if chunk_views is None else (chunk_views, 7, 9),
            compression="gzip" if compressed else None,
        )
    if external:
        linked = tmp_path / "linked.h5"
        with h5py.File(linked, "w") as file:
            file["data"] = h5py.ExternalLink(path.name, "/data")
        path = linked
    stack = locate_frames(str(path), data_path="/data").stack
    assert stack.chunk_views == chunk_views
    stack = stack.frames_at(np.array([0, 2, 3, 6, 7]))
    if windowed:
        stack = stack.window(slice(1, 7), slice(2, 9)).window(slice(1, 5), slice(1, 6))
        selected = expected[[0, 2, 3, 6, 7], 2:6, 3:8]
    else:
        stack = stack.window(slice(None), slice(None))  # a full window is still the bulk path
        selected = expected[[0, 2, 3, 6, 7]]
    assert stack.is_window == windowed
    cache_sizes = []
    original_get = h5py.Dataset.__getitem__
    original_direct = h5py.Dataset.read_direct

    def get(data: h5py.Dataset, *args: object, **kwargs: object) -> np.ndarray:
        cache_sizes.append(data.id.get_access_plist().get_chunk_cache()[1])
        return original_get(data, *args, **kwargs)

    def direct(data: h5py.Dataset, *args: object, **kwargs: object) -> None:
        cache_sizes.append(data.id.get_access_plist().get_chunk_cache()[1])
        return original_direct(data, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", get)
    monkeypatch.setattr(h5py.Dataset, "read_direct", direct)
    for index in [slice(None), slice(1, 4), slice(None, None, 2), np.array([0, 2, 4]), -1, 0]:
        actual = stack[index]
        np.testing.assert_array_equal(actual, selected[index])
        assert actual.dtype == expected.dtype
    assert cache_sizes
    assert all(size == 0 if chunk_views == 1 and windowed else size > 0 for size in cache_sizes)


def test_cache_free_frame_reads_preserve_unwritten_fill_values(tmp_path: Path) -> None:
    path = tmp_path / "unwritten.h5"
    expected = np.full((8, 7, 9), 123, np.uint16)
    expected[0] = np.arange(7 * 9).reshape(7, 9)
    with h5py.File(path, "w") as file:
        data = file.create_dataset(
            "data", shape=expected.shape, dtype=expected.dtype, chunks=(1, 7, 9), fillvalue=123
        )
        data[0] = expected[0]
    stack = locate_frames(str(path), data_path="/data").stack.window(slice(1, 5), slice(2, 8))
    np.testing.assert_array_equal(stack[:], expected[:, 1:5, 2:8])
    np.testing.assert_array_equal(stack[[0, 2, 6]], expected[[0, 2, 6], 1:5, 2:8])
