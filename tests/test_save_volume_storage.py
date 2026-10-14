"""Synchronous persistence borrows large volumes and bounds strided HDF5 staging."""

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.io import ProjectionDataset, load_dataset, save_dataset
from tomojax.io.api import NXTomoMetadata


@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_saving_memmap_does_not_copy_volume_into_metadata(tmp_path, monkeypatch, suffix, order):
    shape = (7, 6, 5)
    volume = np.memmap(
        tmp_path / "volume.dat", mode="w+", dtype=np.float32, shape=shape, order=order
    )
    volume[:] = np.arange(volume.size, dtype=np.float32).reshape(shape) / volume.size
    expected = volume.copy()
    dataset = ProjectionDataset(
        np.ones((3, 5, 7), np.float32),
        np.array([0, 60, 120]),
        volume=volume,
        grid=tj.Grid(*shape, 1, 1, 1),
        detector=tj.Detector(7, 5, 1, 1),
    )
    original_array = np.array

    def no_full_metadata_copy(value, *args, **kwargs):
        if (
            isinstance(value, np.ndarray)
            and value.shape == shape
            and np.shares_memory(value, volume)
        ):
            raise AssertionError("saving must not copy the full volume as metadata")
        return original_array(value, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(np, "array", no_full_metadata_copy)
        save_dataset(tmp_path / ("saved" + suffix), dataset)
    loaded = load_dataset(tmp_path / ("saved" + suffix))
    np.testing.assert_array_equal(loaded.volume, expected)
    np.testing.assert_array_equal(volume, expected)
    assert dataset.volume is volume


@pytest.mark.parametrize("axes", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
@pytest.mark.parametrize("storage", ["C", "F", "reverse"])
def test_volume_storage_and_disk_axes_are_independent(tmp_path, monkeypatch, axes, storage):
    volume = np.arange(7 * 6 * 5, dtype=np.float32).reshape(7, 6, 5) / 100
    projections = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7) / 100
    if storage == "F":
        volume = np.asfortranarray(volume)
        projections = np.asfortranarray(projections)
    elif storage == "reverse":
        volume = volume[::-1, :, ::-1]
        projections = projections[::-1, :, ::-1]
    dataset = ProjectionDataset(
        projections,
        np.array([0, 60, 120]),
        volume=volume,
        grid=tj.Grid(7, 6, 5, 1, 1, 1),
        detector=tj.Detector(7, 5, 1, 1),
        _metadata=NXTomoMetadata(volume_axes_order=axes),
    )
    create = h5py.Group.create_dataset

    def no_implicit_hdf5_copy(group, name, *args, **kwargs):
        data = kwargs.get("data")
        if name in {"volume", "data"} and data is not None:
            assert data.flags.c_contiguous, "h5py would copy this whole strided volume"
        return create(group, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "create_dataset", no_implicit_hdf5_copy)
    save_dataset(tmp_path / "saved.nxs", dataset)
    loaded = load_dataset(tmp_path / "saved.nxs")
    np.testing.assert_array_equal(loaded.volume, volume)
    np.testing.assert_array_equal(loaded.projections, projections)
    assert loaded.to_nxtomo_metadata().volume_axes_order == axes


def test_hdf5_volume_staging_is_bounded_and_chunk_aligned(tmp_path, monkeypatch):
    # Just over the staging bound; zeros compress cheaply and still exercise
    # multiple writes and ragged chunk boundaries on all disk axes.
    shape = (257, 256, 257)
    volume = np.zeros(shape, np.float32)
    volume[::3, ::5, ::7] = 0.375
    dataset = ProjectionDataset(np.zeros((1, 1, 1), np.float32), np.zeros(1), volume=volume)
    write = h5py.Dataset.__setitem__
    staged = []

    def bounded_write(target, region, value):
        if target.name.endswith("/volume"):
            assert value.flags.c_contiguous
            staged.append(value.nbytes)
            assert value.nbytes <= 64 * 1024**2
            for axis, item in enumerate(region):
                assert item.start % target.chunks[axis] == 0
                assert item.stop == target.shape[axis] or item.stop % target.chunks[axis] == 0
        return write(target, region, value)

    monkeypatch.setattr(h5py.Dataset, "__setitem__", bounded_write)
    path = tmp_path / "saved.nxs"
    save_dataset(path, dataset)
    assert len(staged) > 1 and sum(staged) == volume.nbytes
    with h5py.File(path) as handle:
        saved = handle["entry/processing/tomojax/volume"]
        for start in range(0, shape[2], 4):
            np.testing.assert_array_equal(
                saved[start : start + 4], volume[:, :, start : start + 4].T
            )
