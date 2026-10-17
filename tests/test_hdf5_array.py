"""The bounded decoder preserves HDF5 slice, fill and filter-mask semantics."""

import h5py
import numpy as np

from tomojax.io.api import read_hdf5_array


def test_gzip_slices_preserve_compound_endian_fill_and_uncompressed_chunks(tmp_path):
    dtype = np.dtype([("real", ">f4"), ("imag", ">f4")])
    fill = np.array((3, -2), dtype=dtype)[()]
    with h5py.File(tmp_path / "chunks.h5", "w") as stream:
        data = stream.create_dataset(
            "array",
            (20, 41, 256),
            dtype=dtype,
            chunks=(5, 7, 31),
            compression="gzip",
            fillvalue=fill,
        )
        block = np.zeros((20, 13, 256), dtype=dtype)
        block["real"] = np.arange(block.size).reshape(block.shape)
        block["imag"] = -block["real"]
        data[:, :13, :] = block
        data.id.write_direct_chunk((0, 0, 0), np.full(data.chunks, fill).tobytes(), filter_mask=1)
    with h5py.File(tmp_path / "chunks.h5", "r") as stream:
        data = stream["array"]
        selection = (slice(1, 19), slice(1, 40), slice(3, 255))
        actual = read_hdf5_array(data, selection)
        assert actual.dtype == dtype and actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, data[selection])


def test_other_filters_and_strided_reads_keep_hdf5_semantics(tmp_path):
    expected = np.arange(40 * 41 * 256, dtype=np.float32).reshape(40, 41, 256)
    with h5py.File(tmp_path / "fallback.h5", "w") as stream:
        for checksum in (False, True):
            data = stream.create_dataset(
                str(checksum),
                data=expected,
                chunks=(5, 7, 31),
                compression="gzip",
                fletcher32=checksum,
            )
            np.testing.assert_array_equal(read_hdf5_array(data), data[:])
    with h5py.File(tmp_path / "fallback.h5", "r") as stream:
        for checksum in (False, True):
            data = stream[str(checksum)]
            for selection in (None, (slice(None, None, 2), slice(1, 31), slice(3, 253))):
                key = (...,) if selection is None else selection
                np.testing.assert_array_equal(read_hdf5_array(data, selection), data[key])
