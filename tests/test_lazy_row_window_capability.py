"""Lazy readers advertise cheap row windows from native storage headers, never pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import pytest
import tifffile

from tomojax.io.api import locate_frames, read_tiff_frames

if TYPE_CHECKING:
    from pathlib import Path

# Native HDF5 metadata APIs are missing from h5py's supplied typing.
HDF5 = cast(Any, h5py)


@pytest.mark.parametrize(
    ("layout", "efficient"),
    [
        ("contiguous", True),
        ("single_view", True),
        ("multi_view", False),
        ("gzip", False),
        ("shuffle", False),
        ("checksum", False),
        ("scaleoffset", False),
        ("virtual", False),
    ],
)
@pytest.mark.parametrize("linked", [False, True])
def test_hdf_row_window_capability_uses_native_layout_and_all_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, efficient: bool, linked: bool
) -> None:
    expected = np.arange(8 * 7 * 9, dtype=np.uint16).reshape(8, 7, 9)
    path = tmp_path / "frames.h5"
    with HDF5.File(path, "w") as file:
        if layout == "virtual":
            file.create_dataset("source", data=expected)
            virtual = HDF5.VirtualLayout(shape=expected.shape, dtype=expected.dtype)
            virtual[:] = HDF5.VirtualSource(".", "/source", shape=expected.shape)
            file.create_virtual_dataset("data", virtual)
        else:
            options = {
                "gzip": {"compression": "gzip"},
                "shuffle": {"shuffle": True},
                "checksum": {"fletcher32": True},
                "scaleoffset": {"scaleoffset": 0},
            }.get(layout, {})
            file.create_dataset(
                "data",
                data=expected,
                chunks=None
                if layout == "contiguous"
                else (3 if layout == "multi_view" else 1, 7, 9),
                **options,
            )
    if linked:
        link = tmp_path / "linked.h5"
        with HDF5.File(link, "w") as file:
            file["data"] = HDF5.ExternalLink(path.name, "/data")
        path = link

    def unread(*args: object, **kwargs: object) -> None:
        raise AssertionError("capability discovery must not read acquisition pixels")

    with monkeypatch.context() as headers:
        headers.setattr(h5py.Dataset, "__getitem__", unread)
        headers.setattr(h5py.Dataset, "read_direct", unread)
        stack = locate_frames(str(path), data_path="/data").stack
        assert getattr(stack, "_efficient_row_windows", False) is efficient
        selected = stack.frames_at(np.array([0, 2, 5]))
        window = selected.window(slice(1, 6), slice(2, 8)).window(slice(1, 4), slice(1, 5))
        assert getattr(window, "_efficient_row_windows", False) is efficient
    np.testing.assert_array_equal(np.asarray(window), expected[[0, 2, 5], 2:5, 3:7])


@pytest.mark.parametrize("compression", [None, "deflate"])
@pytest.mark.parametrize("tiled", [False, True])
def test_tiff_row_window_capability_requires_native_memmappable_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compression: str | None, tiled: bool
) -> None:
    expected = np.arange(4 * 33 * 35, dtype=np.uint16).reshape(4, 33, 35)
    path = tmp_path / "frames.tif"
    tifffile.imwrite(
        path,
        expected,
        photometric="minisblack",
        compression=compression,
        tile=(16, 16) if tiled else None,
    )

    def unread(*args: object, **kwargs: object) -> None:
        raise AssertionError("capability discovery must not decode acquisition pixels")

    with monkeypatch.context() as headers:
        headers.setattr(tifffile.TiffPage, "asarray", unread)
        stack = read_tiff_frames(path, lazy=True)
        assert getattr(stack, "_efficient_row_windows", False) is (
            compression is None and not tiled
        )
        window = stack.frames_at(np.array([0, 2, 3])).window(slice(1, 17), slice(2, 20))
        window = window.window(slice(2, 8), slice(3, 10))
        assert getattr(window, "_efficient_row_windows", False) is getattr(
            stack, "_efficient_row_windows", False
        )
    np.testing.assert_array_equal(np.asarray(window), expected[[0, 2, 3], 3:9, 5:12])


def test_tiff_capability_tracks_only_selected_pages(tmp_path: Path) -> None:
    expected = np.arange(3 * 7 * 9, dtype=np.uint16).reshape(3, 7, 9)
    with tifffile.TiffWriter(tmp_path / "mixed.tif") as file:
        for index, frame in enumerate(expected):
            file.write(frame, compression="deflate" if index == 1 else None)
    stack = read_tiff_frames(tmp_path, lazy=True)
    assert not getattr(stack, "_efficient_row_windows", False)
    selected = stack.frames_at(np.array([0, 2])).window(slice(1, 5), slice(2, 8))
    assert getattr(selected, "_efficient_row_windows", False)
    np.testing.assert_array_equal(np.asarray(selected), expected[[0, 2], 1:5, 2:8])
    assert not getattr(stack.frames_at(np.array([0, 1])), "_efficient_row_windows", False)
