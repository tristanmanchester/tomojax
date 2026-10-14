"""GPU reconstruction persistence does not materialize a full host volume."""

import importlib

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.io import ProjectionDataset, load_dataset, save_dataset
from tomojax.io.api import NXTomoMetadata


def result(volume, kind):
    nx, ny, nz = volume.shape
    grid, detector = tj.Grid(nx, ny, nz, 1, 1, 1), tj.Detector(nx, nz, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, [0, 60, 120])
    scan = tj.Scan(np.ones((3, nz, nx), np.float32), geometry)
    if kind == "alignment":
        return tj.Alignment(volume, scan, np.zeros((3, 6), np.float32), {})
    return tj.Reconstruction(volume, scan, "fbp", {})


@pytest.mark.parametrize("kind", ["reconstruction", "alignment"])
@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_device_volume_public_save_round_trip(tmp_path, kind, suffix):
    expected = np.arange(7 * 6 * 5, dtype=np.float32).reshape(7, 6, 5)
    volume = jnp.asarray(expected)
    item = result(volume, kind)
    path = tmp_path / ("saved" + suffix)
    tj.save(path, item)
    np.testing.assert_array_equal(tj.load_reconstruction(path).volume, expected)
    assert item.volume is volume
    np.testing.assert_array_equal(volume, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("axes", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("execution", ["cpu", "gpu"])
def test_device_volume_keeps_placement_and_disk_axes(
    tmp_path, monkeypatch, axes, committed, execution
):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    expected = np.arange(41 * 39 * 37, dtype=np.float32).reshape(41, 39, 37)
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        volume = jax.device_put(expected, gpu) if committed else jnp.asarray(expected)
    dataset = ProjectionDataset(
        np.ones((3, 37, 41), np.float32),
        np.array([0, 60, 120]),
        volume=volume,
        grid=tj.Grid(41, 39, 37, 1, 1, 1),
        detector=tj.Detector(41, 37, 1, 1),
        _metadata=NXTomoMetadata(volume_axes_order=axes),
    )
    contiguous = np.ascontiguousarray
    downloads = []

    def record(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            downloads.append(value.devices())
        return contiguous(value, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", record)
    path = tmp_path / "saved.nxs"
    with jax.default_device(jax.devices(execution)[0]):
        save_dataset(path, dataset)
    assert downloads and all(devices == {gpu} for devices in downloads)
    loaded = load_dataset(path)
    np.testing.assert_array_equal(loaded.volume, expected)
    assert loaded.to_nxtomo_metadata().volume_axes_order == axes
    assert dataset.volume is volume


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["reconstruction", "alignment"])
def test_public_device_volume_download_and_transpose_are_bounded(tmp_path, monkeypatch, kind):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    shape, bound = (257, 256, 257), 64 * 1024**2
    expected = np.zeros(shape, np.float32)
    expected[::3, ::5, ::7] = 0.375
    volume = jax.device_put(expected)
    item = result(volume, kind)
    asarray, contiguous, transpose = np.asarray, np.ascontiguousarray, type(volume).transpose
    downloads, transposes = [], []
    copy_async = type(volume).copy_to_host_async
    pending = {}
    pending_counts = []

    def checked_async(value):
        if value.ndim == 3:
            assert value.nbytes <= bound // 2
            pending[id(value)] = value.nbytes
            pending_counts.append(len(pending))
            assert len(pending) <= 2
            assert sum(pending.values()) <= bound
        return copy_async(value)

    def check(value):
        if isinstance(value, jax.Array) and value.ndim == 3:
            assert value.size * value.dtype.itemsize <= bound

    def checked_array(value, *args, **kwargs):
        check(value)
        return asarray(value, *args, **kwargs)

    def checked_contiguous(value, *args, **kwargs):
        check(value)
        if isinstance(value, jax.Array) and value.ndim == 3:
            downloads.append(value.size * value.dtype.itemsize)
        host = contiguous(value, *args, **kwargs)
        pending.pop(id(value), None)
        return host

    def checked_transpose(value, *args, **kwargs):
        check(value)
        transposes.append(value.shape)
        return transpose(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "asarray", checked_array)
        patch.setattr(np, "ascontiguousarray", checked_contiguous)
        patch.setattr(type(volume), "transpose", checked_transpose)
        patch.setattr(type(volume), "copy_to_host_async", checked_async)
        path = tmp_path / "saved.nxs"
        tj.save(path, item)
    assert len(downloads) > 1 and sum(downloads) == expected.nbytes
    assert transposes
    assert not pending and max(pending_counts) == 2
    with h5py.File(path) as handle:
        saved = handle["entry/processing/tomojax/volume"]
        for start in range(0, shape[2], 4):
            np.testing.assert_array_equal(
                saved[start : start + 4], expected[:, :, start : start + 4].T
            )
    assert item.volume is volume


def test_cpu_device_volume_keeps_zero_copy_contiguous_save(tmp_path, monkeypatch):
    volume = jax.device_put(
        np.arange(210, dtype=np.float32).reshape(7, 6, 5), jax.devices("cpu")[0]
    )
    dataset = ProjectionDataset(
        np.zeros((1, 1, 1), np.float32),
        np.zeros(1),
        volume=volume,
        _metadata=NXTomoMetadata(volume_axes_order="xyz"),
    )
    module = importlib.import_module("tomojax._data._io_nxtomo")
    monkeypatch.setattr(
        module, "_write_strided_array", lambda *a, **k: pytest.fail("CPU tile copy")
    )
    create = h5py.Group.create_dataset

    def direct(group, name, *args, **kwargs):
        if name == "volume":
            assert isinstance(kwargs["data"], np.ndarray)
            assert np.shares_memory(kwargs["data"], np.asarray(volume))
        return create(group, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "create_dataset", direct)
    path = tmp_path / "cpu.nxs"
    save_dataset(path, dataset)
    np.testing.assert_array_equal(load_dataset(path).volume, volume)
