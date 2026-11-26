"""Unfiltered HDF5 frames read at their stored byte offsets equal the values written."""

import os

import h5py
import numpy as np
import pytest

from tomojax.io.api import locate_frames


@pytest.fixture
def bytes_read(monkeypatch):
    """Bytes read by positional reads, the stored-offset path; HDF5's own reads are not counted."""
    total = [0]
    preadv = os.preadv

    def counted(fd, buffers, offset):
        count = preadv(fd, buffers, offset)
        total[0] += count
        return count

    monkeypatch.setattr(os, "preadv", counted)
    return total


def _stack(path, dataset="/data"):
    return locate_frames(str(path), data_path=dataset).stack


def test_chunk_edges_frame_selections_and_windows_match_the_written_values(tmp_path, bytes_read):
    # Chunks span several frames and stop short of every edge; values are big-endian.
    values = np.arange(11 * 13 * 17, dtype=">i4").reshape(11, 13, 17) * 7 - 500
    with h5py.File(tmp_path / "scan.h5", "w") as file:
        file.create_dataset("data", data=values, chunks=(3, 5, 6))
    stack = _stack(tmp_path / "scan.h5")
    frames = np.array([0, 2, 3, 7, 10])
    for rows, cols in [
        (slice(None), slice(None)),
        (slice(4, 11), slice(None)),
        (slice(1, 2), slice(5, 13)),
        (slice(12, 13), slice(16, 17)),
        (slice(3, 9), slice(2, 15)),
    ]:
        actual = stack.frames_at(frames).window(rows, cols)[:]
        assert actual.dtype == values.dtype
        np.testing.assert_array_equal(actual, values[frames][:, rows, cols])
    np.testing.assert_array_equal(stack[4:9], values[4:9])
    assert bytes_read[0] > 0


def test_a_band_of_rows_reads_only_those_rows_of_whole_frame_chunks(tmp_path, bytes_read):
    values = np.random.default_rng(1).integers(0, 65535, (20, 64, 48), np.uint16)
    with h5py.File(tmp_path / "scan.h5", "w") as file:
        file.create_dataset("data", data=values, chunks=(1, 64, 48))
    band = _stack(tmp_path / "scan.h5").window(slice(30, 34), slice(None))[:]
    np.testing.assert_array_equal(band, values[:, 30:34])
    assert bytes_read[0] == band.nbytes


def test_contiguous_data_behind_a_user_block_and_2d_sinograms(tmp_path, bytes_read):
    frames = np.random.default_rng(2).random((9, 10, 12)).astype(np.float32)
    sinogram = np.random.default_rng(3).integers(-9, 9, (15, 31), np.int16)
    with h5py.File(tmp_path / "scan.h5", "w", userblock_size=512) as file:
        file.create_dataset("data", data=frames)
        file.create_dataset("sino", data=sinogram)
    stack = _stack(tmp_path / "scan.h5")
    np.testing.assert_array_equal(stack.frames_at(np.array([1, 4, 8]))[:], frames[[1, 4, 8]])
    np.testing.assert_array_equal(stack.window(slice(2, 7), slice(3, 9))[:], frames[:, 2:7, 3:9])
    rows = _stack(tmp_path / "scan.h5", "/sino")
    np.testing.assert_array_equal(
        rows.window(slice(None), slice(4, 20))[:], sinogram[:, None, 4:20]
    )
    assert bytes_read[0] > 0


def test_frames_behind_an_external_link(tmp_path, bytes_read):
    values = np.random.default_rng(4).integers(0, 4000, (30, 16, 20), np.uint16)
    with h5py.File(tmp_path / "raw.h5", "w") as raw:
        raw.create_dataset("entry/detector/detector", data=values, chunks=(1, 16, 20))
    with h5py.File(tmp_path / "scan.nxs", "w") as master:
        master["entry/imaging/data"] = h5py.ExternalLink("raw.h5", "/entry/detector/detector")
    stack = _stack(tmp_path / "scan.nxs", "/entry/imaging/data")
    selected = stack.frames_at(np.arange(0, 30, 4)).window(slice(5, 9), slice(None))
    band = selected[:]
    np.testing.assert_array_equal(band, values[::4, 5:9])
    assert bytes_read[0] == band.nbytes


def test_unwritten_chunks_and_unwritten_contiguous_data_read_as_the_fill_value(tmp_path):
    values = np.full((6, 8, 9), 17, np.uint16)
    values[2:4, :4] = np.arange(2 * 4 * 9, dtype=np.uint16).reshape(2, 4, 9)
    with h5py.File(tmp_path / "scan.h5", "w") as file:
        chunked = file.create_dataset("data", (6, 8, 9), np.uint16, chunks=(2, 4, 9), fillvalue=17)
        chunked[2:4, :4] = values[2:4, :4]
        file.create_dataset("empty", (3, 4, 5), np.float32, fillvalue=-1.5)
    np.testing.assert_array_equal(_stack(tmp_path / "scan.h5")[:], values)
    np.testing.assert_array_equal(
        _stack(tmp_path / "scan.h5", "/empty")[:], np.full((3, 4, 5), -1.5)
    )


def test_filtered_frames_are_decoded_by_hdf5(tmp_path, bytes_read):
    values = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)
    with h5py.File(tmp_path / "scan.h5", "w") as file:
        file.create_dataset("data", data=values, chunks=(1, 6, 7), compression="gzip")
    stack = _stack(tmp_path / "scan.h5")
    np.testing.assert_array_equal(stack.window(slice(1, 4), slice(2, 7))[:], values[:, 1:4, 2:7])
    assert bytes_read[0] == 0


def test_a_file_rewritten_in_place_is_read_at_its_new_chunk_offsets(tmp_path):
    path = tmp_path / "scan.h5"
    first = np.arange(8 * 4 * 5, dtype=np.uint16).reshape(8, 4, 5)
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=first, chunks=(1, 4, 5))
    np.testing.assert_array_equal(_stack(path)[:], first)
    # The same path and dataset, the chunks written in another order at other offsets.
    second = first[::-1] + 1000
    with h5py.File(path, "w") as file:
        file.create_dataset("pad", data=np.zeros(977, np.uint8))
        data = file.create_dataset("data", (8, 4, 5), np.uint16, chunks=(1, 4, 5))
        for frame in (5, 0, 7, 2, 6, 1, 4, 3):
            data[frame] = second[frame]
    np.testing.assert_array_equal(_stack(path)[:], second)
