"""Device scans share the bounded HDF writer with device volumes."""

from dataclasses import replace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.io import ProjectionDataset


def item_of(projections, kind):
    views, nv, nu = projections.shape
    grid, detector = tj.Grid(7, 6, nv, 1, 1, 1), tj.Detector(nu, nv, 1, 1)
    angles = np.linspace(0, 180, views, dtype=np.float32)
    source = ProjectionDataset(
        np.zeros((1, 1, 1), np.float32),
        np.zeros(1),
        sample_name="original",
        geometry_metadata={"acquisition_id": "scan-17"},
    )
    scan = tj.Scan(
        projections, tj.ParallelGeometry(grid, detector, angles), name="saved", source=source
    )
    if kind == "scan":
        return scan
    volume = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    if kind == "alignment":
        return tj.Alignment(volume, scan, np.zeros((views, 6), np.float32), {})
    return tj.Reconstruction(volume, scan, "fbp", {})


@pytest.mark.parametrize("kind", ["scan", "reconstruction", "alignment"])
@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_device_projections_round_trip_and_provenance(tmp_path, kind, suffix):
    expected = np.arange(11 * 5 * 7, dtype=np.float32).reshape(11, 5, 7)
    projections = jnp.asarray(expected)
    item = item_of(projections, kind)
    path = tmp_path / ("saved" + suffix)
    tj.save(path, item)
    loaded = tj.load(path)
    np.testing.assert_array_equal(loaded.projections, expected)
    assert loaded.name == "saved"
    assert loaded.source.geometry_metadata["acquisition_id"] == "scan-17"
    scan = item if kind == "scan" else item.scan
    assert scan.projections is projections
    assert scan.source.sample_name == "original"
    assert scan.source.projections.shape == (1, 1, 1)


@pytest.mark.gpu
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("execution", ["cpu", "gpu"])
def test_device_projections_keep_source_placement(tmp_path, monkeypatch, committed, execution):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    expected = np.arange(11 * 5 * 7, dtype=np.float32).reshape(11, 5, 7)
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        projections = jax.device_put(expected, gpu) if committed else jnp.asarray(expected)
        scan = item_of(projections, "scan")
    contiguous = np.ascontiguousarray
    downloads = []

    def record(value, *args, **kwargs):
        if isinstance(value, jax.Array) and value.ndim == 3:
            downloads.append(value.devices())
        return contiguous(value, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", record)
    path = tmp_path / "saved.nxs"
    with jax.default_device(jax.devices(execution)[0]):
        tj.save(path, scan)
    assert downloads and all(devices == {gpu} for devices in downloads)
    np.testing.assert_array_equal(tj.load(path).projections, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["scan", "reconstruction", "alignment"])
def test_public_device_projection_downloads_are_bounded(tmp_path, monkeypatch, kind):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    shape, bound = (259, 256, 257), 64 * 1024**2
    expected = np.zeros(shape, np.float32)
    expected[::3, ::5, ::7] = 0.375
    projections = jax.device_put(expected)
    item = item_of(projections, kind)
    asarray, contiguous = np.asarray, np.ascontiguousarray
    downloads = []

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
        return contiguous(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "asarray", checked_array)
        patch.setattr(np, "ascontiguousarray", checked_contiguous)
        path = tmp_path / "saved.nxs"
        tj.save(path, item)
    volume_bytes = 0 if kind == "scan" else item.volume.size * item.volume.dtype.itemsize
    assert len(downloads) > 1 and sum(downloads) == expected.nbytes + volume_bytes
    with h5py.File(path) as handle:
        saved = handle["entry/instrument/detector/data"]
        for start in range(0, shape[0], 4):
            np.testing.assert_array_equal(saved[start : start + 4], expected[start : start + 4])


def test_cpu_device_projection_save_keeps_zero_copy(tmp_path, monkeypatch):
    expected = np.arange(11 * 5 * 7, dtype=np.float32).reshape(11, 5, 7)
    projections = jax.device_put(expected, jax.devices("cpu")[0])
    scan = replace(item_of(expected, "scan"), projections=projections)
    create = h5py.Group.create_dataset
    writes = []

    def direct(group, name, *args, **kwargs):
        if name == "data":
            assert isinstance(kwargs["data"], np.ndarray)
            assert np.shares_memory(kwargs["data"], np.asarray(projections))
            writes.append(name)
        return create(group, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "create_dataset", direct)
    path = tmp_path / "cpu.nxs"
    tj.save(path, scan)
    assert writes == ["data"]
    np.testing.assert_array_equal(tj.load(path).projections, expected)
