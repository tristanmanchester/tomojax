"""Caller-owned output reaches analytic host backends through the workflow API."""

from dataclasses import replace

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import FourierConfig


def _scan(kind):
    grid = tj.Grid(7, 6, 5, 0.8, 1.1, 0.9, vol_center=(0.2, -0.4, 0.1))
    detector = tj.Detector(12, 7, 1, 1, (0.3, -0.1))
    angles = np.linspace(0, 180, 12, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    if kind == "lamino":
        geometry = tj.LaminographyGeometry(grid, detector, angles * 2, tilt_deg=20)
    elif kind == "cone":
        geometry = tj.ConeGeometry(grid, detector, angles * 2, tj.ConeBeam(30, 45))
    data = np.random.default_rng(152).normal(size=(12, 7, 12)).astype(np.float32)
    return tj.Scan(data, geometry)


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone", "fourier"])
@pytest.mark.parametrize("storage", ["C", "F", "strided"])
def test_analytic_workflow_returns_exact_caller_storage(tmp_path, kind, storage):
    scan = _scan(kind)
    method = "fourier" if kind == "fourier" else "fbp"
    grid = replace(scan.grid, nx=5, nz=4)
    expected = tj.reconstruct(scan, method, grid=grid)
    shape = (grid.nx, grid.ny, grid.nz)
    if storage != "strided":
        out = np.memmap(
            tmp_path / "volume.dat", mode="w+", dtype=np.float32, shape=shape, order=storage
        )
    else:
        backing = np.full((shape[0], shape[1], shape[2] * 2), -71, np.float32)
        out = backing[:, :, ::2]
    result = tj.reconstruct(scan, method, grid=grid, out=out)
    assert result.volume is out
    assert result.grid == grid and result.info["host_slabs"]
    assert "out" not in result.info
    np.testing.assert_allclose(result.volume, np.asarray(expected.volume), rtol=5e-5, atol=3e-6)
    if storage == "strided":
        np.testing.assert_array_equal(backing[:, :, 1::2], -71)
    else:
        out.flush()
        reopened = np.memmap(
            tmp_path / "volume.dat", mode="r", dtype=np.float32, shape=shape, order=storage
        )
        np.testing.assert_array_equal(reopened, result.volume)


@pytest.mark.gpu
def test_cupy_fourier_workflow_keeps_caller_memmap(tmp_path):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    scan = _scan("fourier")
    shape = (scan.grid.nx, scan.grid.ny, scan.grid.nz)
    out = np.memmap(tmp_path / "volume.dat", mode="w+", dtype=np.float32, shape=shape)
    config = FourierConfig(backend="cupy", slices_per_batch=2)
    expected = tj.reconstruct(scan, "fourier", config=config)
    actual = tj.reconstruct(scan, "fourier", config=config, out=out)
    assert actual.volume is out
    np.testing.assert_array_equal(out, expected.volume)


@pytest.mark.parametrize("method", ["cgls", "fista", "spdhg"])
def test_iterative_workflow_rejects_unsupported_output(method):
    scan = _scan("parallel")
    out = np.zeros((7, 6, 5), np.float32)
    with pytest.raises(ValueError, match=f"method '{method}' does not take out"):
        tj.reconstruct(scan, method, out=out)
    np.testing.assert_array_equal(out, 0)


@pytest.mark.parametrize("method", ["fbp", "fourier"])
def test_public_output_preserves_host_aliasing_rejection(method):
    scan = _scan("parallel")
    shared = scan.projections
    grid = replace(scan.grid, nx=12, ny=7, nz=12)
    with pytest.raises(ValueError, match="must not overlap"):
        tj.reconstruct(scan, method, grid=grid, out=shared)


def test_host_output_does_not_silently_drop_replayed_detector_roll():
    class ReplayGeometry(tj.ParallelGeometry):
        detector_roll_deg = 15.0

    scan = _scan("parallel")
    scan = replace(scan, geometry=ReplayGeometry(scan.grid, scan.detector, scan.geometry.angles))
    out = np.full((7, 6, 5), -71, np.float32)
    with pytest.raises(ValueError, match="explicit detector grids"):
        tj.reconstruct(scan, out=out)
    np.testing.assert_array_equal(out, -71)


@pytest.mark.parametrize("number", [1, 2])
def test_explicit_devices_preserve_output_ownership_and_validate_whole_shape(number):
    if len(jax.devices()) < number:
        pytest.skip("needs multiple devices")
    scan = _scan("parallel")
    devices = jax.devices()[:number]
    expected = tj.reconstruct(scan).volume
    out = np.empty((7, 6, 5), np.float32)
    actual = tj.reconstruct(scan, out=out, devices=devices)
    assert actual.volume is out
    np.testing.assert_allclose(out, np.asarray(expected), rtol=5e-5, atol=3e-6)
    with pytest.raises(ValueError, match="volume shape"):
        tj.reconstruct(scan, out=np.empty((7, 6, 6), np.float32), devices=devices)


@pytest.mark.parametrize("kind", ["parallel", "lamino"])
def test_host_output_accepts_cpu_device_projections(kind):
    scan = _scan(kind)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = tj.reconstruct(scan, out=np.empty((7, 6, 5), np.float32)).volume
        resident = replace(scan, projections=jax.device_put(scan.projections))
        out = np.empty((7, 6, 5), np.float32)
        result = tj.reconstruct(resident, out=out)
    assert result.volume is out
    np.testing.assert_array_equal(out, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["parallel", "lamino"])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("execution", ["cpu", "gpu"])
def test_resident_host_output_bounds_projection_downloads(monkeypatch, kind, committed, execution):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = tj.Grid(9, 7, 11, 1, 1, 1)
    detector = tj.Detector(21, 13, 1, 1)
    angles = np.linspace(0, 360 if kind == "lamino" else 180, 70, endpoint=False)
    geometry = (
        tj.LaminographyGeometry(grid, detector, angles, tilt_deg=20)
        if kind == "lamino"
        else tj.ParallelGeometry(grid, detector, angles)
    )
    data = np.random.default_rng(87).normal(size=(70, 13, 21)).astype(np.float32)
    scan = tj.Scan(data, geometry)
    gpu = jax.devices("gpu")[0]
    with jax.default_device(gpu):
        resident = jax.device_put(data, gpu) if committed else jax.numpy.asarray(data)
    original_asarray, original_contiguous = np.asarray, np.ascontiguousarray
    downloads = []

    def record(value):
        if isinstance(value, jax.Array) and value.ndim == 3 and value.shape[-1] == 21:
            downloads.append((value.shape, value.devices()))

    def asarray(value, *args, **kwargs):
        record(value)
        return original_asarray(value, *args, **kwargs)

    def contiguous(value, *args, **kwargs):
        record(value)
        return original_contiguous(value, *args, **kwargs)

    target = jax.devices(execution)[0]
    with jax.default_device(target):
        expected = tj.reconstruct(scan, out=np.empty((9, 7, 11), np.float32)).volume
        monkeypatch.setattr(np, "asarray", asarray)
        monkeypatch.setattr(np, "ascontiguousarray", contiguous)
        out = np.empty((9, 7, 11), np.float32)
        result = tj.reconstruct(replace(scan, projections=resident), out=out)
    assert result.volume is out
    assert downloads and max(shape[0] for shape, _ in downloads) <= 32
    assert all(devices == {gpu} for _, devices in downloads)
    np.testing.assert_allclose(out, expected, rtol=5e-5, atol=3e-6)
