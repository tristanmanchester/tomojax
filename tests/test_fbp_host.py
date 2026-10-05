"""Physical geometry, bounded execution and host-storage checks for slab FBP."""

from __future__ import annotations

from dataclasses import replace
import importlib

import jax
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FBPConfig, FBPHostConfig, fbp, fbp_host


def scan(kind="fractional"):
    grid = Grid(9, 7, 11, 0.8, 1.2, 0.7, vol_center=(1.2, -0.4, 0.31))
    detector = Detector(9, 10, 0.9, 1.0, (0.23, -0.19))
    if kind == "integer":
        grid = replace(grid, vz=1.0, vol_center=(1.2, -0.4, 0.0))
        detector = replace(detector, nv=11, det_center=(0.23, 0.0))
    elif kind == "outside":
        grid = replace(grid, vz=3.0, vol_origin=(-1.7, -2.1, -25.0))
    elif kind == "one_row":
        detector = replace(detector, nv=1)
    angles = np.array([0.0, 29.0, 73.0, 118.0, 163.0], dtype=np.float32)
    geometry = ParallelGeometry(grid, detector, angles)
    data = np.random.default_rng(132).normal(size=(5, detector.nv, detector.nu)).astype(np.float32)
    return grid, detector, geometry, data


@pytest.mark.parametrize("kind", ["integer", "fractional", "outside", "one_row"])
@pytest.mark.parametrize("depth", [1, 4, 20])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_host_slabs_match_full_physical_reconstruction(kind, depth, backend):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, geometry, data = scan(kind)
    reference = np.asarray(
        fbp(geometry, grid, detector, data, config=FBPConfig(backprojector="jax"))
    )
    actual = fbp_host(
        geometry,
        grid,
        detector,
        data,
        config=FBPHostConfig(slices_per_batch=depth, views_per_batch=3, backprojector=backend),
    )
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, reference, atol=4e-6, rtol=5e-5)


def test_host_slabs_reuse_compilation_and_bound_device_shapes(monkeypatch):
    module = importlib.import_module("tomojax.recon.fbp_host")
    grid, detector, geometry, data = scan()
    kernel = module._run_fbp_streamed
    calls = []
    before = kernel._cache_size()

    def record(poses, projections, *args, **kwargs):
        calls.append((projections.shape, kwargs["grid"], kwargs["detector"]))
        return kernel(poses, projections, *args, **kwargs)

    monkeypatch.setattr(module, "_run_fbp_streamed", record)
    cfg = FBPHostConfig(slices_per_batch=3, views_per_batch=2, backprojector="jax")
    actual = fbp_host(geometry, grid, detector, data, config=cfg)
    assert len(calls) == 4
    assert all(shape[1] <= 4 and slab.nz == 3 for shape, slab, _ in calls)
    assert len({(slab, det) for _, slab, det in calls}) == 1
    after = kernel._cache_size()
    assert after - before <= 1
    repeated = fbp_host(geometry, grid, detector, data * 1.2, config=cfg)
    assert kernel._cache_size() == after
    np.testing.assert_allclose(repeated, actual * 1.2, atol=2e-6, rtol=2e-5)


def test_host_slabs_write_memmap_and_preserve_custom_scale(tmp_path):
    grid, detector, geometry, data = scan()
    source = np.memmap(tmp_path / "data.bin", mode="w+", dtype=np.float32, shape=data.shape)
    source[:] = data
    target = np.memmap(
        tmp_path / "volume.bin", mode="w+", dtype=np.float32, shape=(grid.nx, grid.ny, grid.nz)
    )
    cfg = FBPHostConfig(slices_per_batch=4, views_per_batch=2, scale=0.27, filter_name="hann")
    result = fbp_host(geometry, grid, detector, source, config=cfg, out=target)
    assert result is target
    target.flush()
    reloaded = np.memmap(tmp_path / "volume.bin", mode="r", dtype=np.float32, shape=target.shape)
    expected = np.asarray(
        fbp(geometry, grid, detector, data, config=FBPConfig(scale=0.27, filter_name="hann"))
    )
    np.testing.assert_allclose(reloaded, expected, atol=3e-6, rtol=3e-5)
    np.testing.assert_array_equal(source, data)


@pytest.mark.parametrize("mapping", [False, True])
def test_host_slabs_reject_overlapping_output_before_writing(tmp_path, mapping):
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(3, 4, 1, 1)
    geometry = ParallelGeometry(grid, detector, [0, 31, 73, 114, 155])
    if mapping:
        path = tmp_path / "same.bin"
        data = np.memmap(path, mode="w+", dtype=np.float32, shape=(5, 4, 3))
        out = np.memmap(path, mode="r+", dtype=np.float32, shape=data.shape)
    else:
        data = np.ones((5, 4, 3), dtype=np.float32)
        out = data.view()
    data[:] = 1
    with pytest.raises(ValueError, match="overlap|separate files"):
        fbp_host(geometry, grid, detector, data, out=out)
    np.testing.assert_array_equal(data, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slices_per_batch", 0),
        ("views_per_batch", -1),
        ("scale", np.nan),
        ("backprojector", "invalid"),
    ],
)
def test_host_slabs_reject_invalid_configuration(field, value):
    grid, detector, geometry, data = scan()
    with pytest.raises(ValueError, match="fbp_host"):
        fbp_host(geometry, grid, detector, data, config=replace(FBPHostConfig(), **{field: value}))


def test_host_slabs_reject_tilted_geometry_and_nonfinite_samples():
    grid, detector, geometry, data = scan()
    tilted = LaminographyGeometry(grid, detector, geometry.thetas_deg, tilt_deg=30)
    with pytest.raises(ValueError, match="ParallelGeometry"):
        fbp_host(tilted, grid, detector, data)
    data[0, 4, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fbp_host(geometry, grid, detector, data)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "readonly", "device_input"])
def test_host_slabs_reject_incompatible_storage(invalid):
    grid, detector, geometry, data = scan()
    out = np.empty((grid.nx, grid.ny, grid.nz), dtype=np.float32)
    if invalid == "dtype":
        out = out.astype(np.float64)
    elif invalid == "shape":
        out = out[:, :, :-1]
    elif invalid == "readonly":
        out.flags.writeable = False
    else:
        data = jax.device_put(data)
    with pytest.raises((ValueError, TypeError), match="fbp_host"):
        fbp_host(geometry, grid, detector, data, out=out)


@pytest.mark.parametrize("length_scale", [0.5, 2.0])
def test_host_slabs_recover_independent_gaussian_attenuation(length_scale):
    n, nz, views = 40, 7, 80
    spacing = 0.8 * length_scale
    grid = Grid(n, n, nz, spacing, spacing, spacing)
    detector = Detector(n, nz, spacing, spacing)
    angles = np.linspace(0, 180, views, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    x = (np.arange(n) - (n - 1) / 2) * spacing
    z = (np.arange(nz) - (nz - 1) / 2) * spacing
    sigma = 4 * length_scale
    axial = np.exp(-0.5 * (z / (1.1 * length_scale)) ** 2)
    profile = np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * (x / sigma) ** 2)
    data = np.broadcast_to(axial[:, None] * profile, (views, nz, n)).astype(np.float32)
    truth = np.exp(-0.5 * (x[:, None, None] ** 2 + x[None, :, None] ** 2) / sigma**2) * axial
    actual = fbp_host(
        geometry,
        grid,
        detector,
        data,
        config=FBPHostConfig(slices_per_batch=3, views_per_batch=17),
    )
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.01
