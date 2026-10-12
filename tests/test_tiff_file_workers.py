"""Native full-file decode budget stays bounded; windows use caller-owned reads/maps."""

from threading import Lock, get_ident

import numpy as np
import pytest
import tifffile

from tomojax.io.api import read_tiff_frames


@pytest.mark.parametrize("compression", [None, "deflate"])
@pytest.mark.parametrize("pages_per_file", [1, 3])
def test_independent_files_reuse_one_native_worker_budget(
    tmp_path, monkeypatch, compression, pages_per_file
):
    data = np.arange(6 * pages_per_file * 17 * 23, dtype="u2").reshape(6 * pages_per_file, 17, 23)
    for file in range(6):
        tifffile.imwrite(
            tmp_path / f"{file}.tif",
            data[file * pages_per_file : (file + 1) * pages_per_file],
            photometric="minisblack",
            compression=compression,
            rowsperstrip=4,
        )
    source = read_tiff_frames(tmp_path, lazy=True)
    calls = []
    state = [0, 0]
    lock = Lock()
    original_array = tifffile.TiffPage.asarray
    original_enter, original_exit = tifffile.TiffFile.__enter__, tifffile.TiffFile.__exit__

    def asarray(page, *args, **kwargs):
        with lock:
            calls.append(kwargs.get("maxworkers"))
        return original_array(page, *args, **kwargs)

    def enter(file):
        result = original_enter(file)
        with lock:
            state[0] += 1
            state[1] = max(state)
        return result

    def leave(file, *args):
        try:
            return original_exit(file, *args)
        finally:
            with lock:
                state[0] -= 1

    monkeypatch.setattr(tifffile.TIFF, "MAXWORKERS", 3)
    monkeypatch.setattr(tifffile.TiffPage, "asarray", asarray)
    monkeypatch.setattr(tifffile.TiffFile, "__enter__", enter)
    monkeypatch.setattr(tifffile.TiffFile, "__exit__", leave)
    for selection in [
        slice(None),
        slice(None, None, -1),
        [1, 4, 1, 3],
        np.arange(len(data)) % 2 == 0,
    ]:
        calls.clear()
        state[:] = [0, 0]
        np.testing.assert_array_equal(source[selection], data[selection])
        assert calls and state[0] == 0
        if compression:
            assert 1 <= state[1] <= 3
            assert all(value == 1 for value in calls)
        else:
            assert state[1] == 1


@pytest.mark.parametrize("compression", [None, "deflate"])
def test_cropped_file_sequences_keep_native_window_reads(tmp_path, monkeypatch, compression):
    data = np.arange(6 * 17 * 23, dtype="u2").reshape(6, 17, 23)
    for file in range(6):
        tifffile.imwrite(
            tmp_path / f"{file}.tif", data[file], compression=compression, rowsperstrip=4
        )
    source = read_tiff_frames(tmp_path, lazy=True).window(slice(1, 15), slice(2, 21))
    source = source.window(slice(None), slice(None))
    caller = get_ident()
    reads = []
    maps = []
    original = tifffile.TiffFile.__enter__
    original_map = np.memmap.__new__

    def enter(file):
        assert get_ident() == caller, "windows must not create a separate file-reader pool"
        reads.append(file.filename)
        return original(file)

    def mapped(cls, *args, **kwargs):
        assert get_ident() == caller, "windows must not create a separate map-reader pool"
        maps.append(kwargs.get("mode"))
        return original_map(cls, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffFile, "__enter__", enter)
    monkeypatch.setattr(np.memmap, "__new__", mapped)
    np.testing.assert_array_equal(source[:], data[:, 1:15, 2:21])
    assert len(reads) == (6 if compression else 0)
    assert maps == ([] if compression else ["r"] * 6)
