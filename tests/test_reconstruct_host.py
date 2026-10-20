"""Public reconstruction keeps oversized volumes on the host and fills caller storage."""

from dataclasses import replace
import importlib

import jax
import numpy as np
import pytest

import tomojax as tj
import tomojax.backends
from tomojax.backends.host_stream import should_stream


def test_measurement_placement_reserves_solver_workspace(monkeypatch):
    data = np.ones((4, 4, 4), np.float32)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", lambda: 1024)
    assert not should_stream(data)
    assert should_stream(data, reserve_bytes=512)
    assert should_stream(data, reserve_bytes=2048)
    assert not should_stream(jax.device_put(data), reserve_bytes=2048)


@pytest.mark.parametrize("order", ["C", "F"])
def test_host_validation_preserves_mask_coordinates_across_layouts(order):
    check = importlib.import_module("tomojax.recon._host_arrays").validate_host_views
    data = np.array([[0, np.nan], [0, 0]], np.float32, order=order)
    check(data, weights=np.array([[1, 0], [1, 1]], np.float32))
    data[0, 1] = 1
    check(data)
    data[1, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        check(data)


def _scan(kind, nz=19):
    grid = tj.Grid(9, 7, nz, 0.8, 1.2, 0.7, vol_center=(1.2, -0.4, 0.31))
    detector = tj.Detector(11, 20, 0.9, 1.0, (0.23, -0.19))
    angles = np.linspace(0, 180, 13, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    if kind == "lamino":
        geometry = tj.LaminographyGeometry(grid, detector, angles * 2, tilt_deg=25)
    elif kind == "cone":
        geometry = tj.ConeGeometry(grid, detector, angles * 2, tj.ConeBeam(30, 45))
    data = np.random.default_rng(132).normal(size=(13, 20, 11)).astype(np.float32)
    return tj.Scan(data, geometry)


@pytest.mark.parametrize("kind", ["parallel", "lamino"])
@pytest.mark.parametrize("device_input", [False, True])
def test_volume_larger_than_the_device_is_reconstructed_in_host_slabs(
    monkeypatch, kind, device_input
):
    scan = _scan(kind)
    if device_input:
        scan = replace(scan, projections=jax.device_put(scan.projections))
    expected = np.asarray(tj.reconstruct(scan, "fbp", filter="hann").volume)
    # Import before patching the facade budget: the host module binds its own query.
    importlib.import_module("tomojax.recon.fbp_host")
    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", lambda: 4096)
    actual = tj.reconstruct(scan, "fbp", filter="hann")
    assert isinstance(actual.volume, np.ndarray) and actual.info["host_slabs"]
    np.testing.assert_allclose(actual.volume, expected, atol=3e-6, rtol=5e-5)


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone", "fourier"])
@pytest.mark.parametrize("storage", ["C", "F", "strided"])
def test_out_is_filled_and_returned(tmp_path, kind, storage):
    scan = _scan("parallel" if kind == "fourier" else kind, nz=4)
    method = "fourier" if kind == "fourier" else "fbp"
    expected = tj.reconstruct(scan, method)
    shape = (scan.grid.nx, scan.grid.ny, scan.grid.nz)
    if storage != "strided":
        path = tmp_path / "volume.dat"
        out = np.memmap(path, mode="w+", dtype=np.float32, shape=shape, order=storage)
    else:
        backing = np.full((*shape[:2], 2 * shape[2]), -71, np.float32)
        out = backing[:, :, ::2]
    result = tj.reconstruct(scan, method, out=out)
    assert result.volume is out and result.info["host_slabs"]
    np.testing.assert_allclose(out, np.asarray(expected.volume), rtol=5e-5, atol=3e-6)
    if storage == "strided":
        np.testing.assert_array_equal(backing[:, :, 1::2], -71)


def test_out_is_refused_where_it_cannot_be_honoured():
    scan = _scan("parallel")
    out = np.zeros((9, 7, 19), np.float32)
    with pytest.raises(ValueError, match="method .cgls. takes out for cone-beam geometry only"):
        tj.reconstruct(scan, "cgls", out=out)
    grid = tj.Grid(13, 20, 11, 1, 1, 1)
    with pytest.raises(ValueError, match="must not overlap"):
        tj.reconstruct(scan, grid=grid, out=scan.projections)


def test_fortran_output_with_ragged_axial_slabs(tmp_path):
    scan = _scan("parallel", nz=19)  # Balanced slabs contain 10 and 9 slices.
    shape = (scan.grid.nx, scan.grid.ny, scan.grid.nz + 2)
    path = tmp_path / "volume.dat"
    backing = np.memmap(path, mode="w+", dtype=np.float32, shape=shape, order="F")
    backing[:] = -71
    expected = tj.reconstruct(scan, "fbp", filter="hann")
    result = tj.reconstruct(scan, "fbp", filter="hann", out=backing[:, :, 1:-1])
    np.testing.assert_allclose(result.volume, np.asarray(expected.volume), rtol=5e-5, atol=3e-6)
    np.testing.assert_array_equal(backing[:, :, [0, -1]], -71)


def test_fortran_output_propagates_write_failures():
    scan = _scan("parallel", nz=19)
    writes = []

    class FailingStorage(np.ndarray):
        def __setitem__(self, key, value):
            writes.append(value.shape)
            raise OSError("output write failed")

    out = np.empty((9, 7, 19), np.float32, order="F").view(FailingStorage)
    with pytest.raises(OSError, match="output write failed"):
        tj.reconstruct(scan, "fbp", out=out)
    assert len(writes) == 1
