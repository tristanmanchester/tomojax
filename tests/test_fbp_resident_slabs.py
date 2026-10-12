"""Budgeted native resident slabs retain host-output and numerical contracts."""

from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FBPConfig, FBPHostConfig, fbp_host

host = importlib.import_module("tomojax.recon.fbp_host")
fbp_module = importlib.import_module("tomojax.recon.fbp")


def gaussian():
    n, nz, views = 40, 7, 83
    grid, detector = Grid(n, n, nz, 0.8, 0.8, 0.8), Detector(n, nz, 0.8, 0.8)
    geometry = ParallelGeometry(grid, detector, np.linspace(0, 180, views, endpoint=False))
    x = (np.arange(n) - (n - 1) / 2) * 0.8
    z = (np.arange(nz) - (nz - 1) / 2) * 0.8
    axial = np.exp(-0.5 * ((z - 0.7) / 3.1) ** 2)
    profile = np.sqrt(2 * np.pi) * 4 * np.exp(-0.5 * (x / 4) ** 2)
    data = np.broadcast_to(axial[:, None] * profile, (views, nz, n)).astype(np.float32)
    truth = np.exp(-0.5 * (x[:, None, None] ** 2 + x[None, :, None] ** 2) / 16) * axial
    return grid, detector, geometry, data, truth


def truncated_gaussian():
    grid, detector = Grid(48, 48, 7, 1, 1, 1), Detector(24, 7, 1, 1, (-2.3, 0))
    angles = np.linspace(0, 180, 97, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    x = np.arange(grid.nx) - (grid.nx - 1) / 2
    z = np.arange(grid.nz) - (grid.nz - 1) / 2
    u = np.arange(detector.nu) - (detector.nu - 1) / 2 + detector.center[0]
    theta = np.deg2rad(angles)
    axial = np.exp(-0.5 * ((z - 0.4) / 3.1) ** 2)
    centre = 2 * np.cos(theta) + np.sin(theta)
    profile = np.sqrt(2 * np.pi) * 16 * np.exp(-0.5 * ((u[None] - centre[:, None]) / 16) ** 2)
    data = (profile[:, None] * axial[None, :, None]).astype(np.float32)
    truth = np.exp(-0.5 * ((x[:, None, None] - 2) ** 2 + (x[None, :, None] + 1) ** 2) / 16**2)
    truth = truth * axial
    return grid, detector, geometry, data, truth


@pytest.mark.parametrize("filter_name", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("padding", ["zero", "edge"])
def test_resident_loop_preserves_endpoint_prior(monkeypatch, filter_name, padding):
    grid, detector, geometry, data, truth = truncated_gaussian()
    config = FBPHostConfig(
        slices_per_batch=3,
        views_per_batch=17,
        backprojector="jax",
        filter=filter_name,
        padding=padding,
    )
    expected = fbp_host(geometry, grid, detector, data, config=config)
    original = host._fbp_from_host

    def resident(poses, rows, *args, **kwargs):
        kwargs["resident_slab"] = True
        return original(poses, jax.device_put(rows[:]), *args, **kwargs)

    monkeypatch.setattr(host, "_fbp_from_host", resident)
    actual = fbp_host(geometry, grid, detector, data, config=config)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert abs(np.linalg.norm(actual - truth) - np.linalg.norm(expected - truth)) < 2e-5


@pytest.mark.gpu
@pytest.mark.parametrize("filter_name", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("padding", ["zero", "edge"])
def test_public_resident_output_preserves_endpoint_prior(monkeypatch, filter_name, padding):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    import tomojax as tj

    grid, detector, geometry, data, truth = truncated_gaussian()
    config = FBPConfig(filter=filter_name, padding=padding)
    expected = tj.reconstruct(tj.Scan(data, geometry), config=config).volume
    calls = []
    original = fbp_module._fbp_from_resident

    def resident(*args, **kwargs):
        calls.append(kwargs.get("padding", "zero"))
        return original(*args, **kwargs)

    monkeypatch.setattr(fbp_module, "_fbp_from_resident", resident)
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: 2**30)
    out = np.empty((grid.nx, grid.ny, grid.nz), np.float32, order="F")
    result = tj.reconstruct(tj.Scan(jax.device_put(data), geometry), config=config, out=out)
    assert result.volume is out and result.info["config"].padding == padding
    assert calls == [padding]
    np.testing.assert_allclose(out, np.asarray(expected), rtol=2e-6, atol=2e-6)
    assert abs(np.linalg.norm(out - truth) - np.linalg.norm(np.asarray(expected) - truth)) < 2e-5


@pytest.mark.parametrize("batch", [17, 32, 64])
def test_resident_loop_retains_independent_gaussian_truth(monkeypatch, batch):
    grid, detector, geometry, data, truth = gaussian()
    config = FBPHostConfig(slices_per_batch=3, views_per_batch=batch, backprojector="jax")
    expected = fbp_host(geometry, grid, detector, data, config=config)
    # Exercise the shared resident loop on CPU too, without changing production's
    # CUDA-only budget policy or the public CPU-array zero-copy exposure.
    original = host._fbp_from_host

    def resident(poses, rows, *args, **kwargs):
        kwargs["resident_slab"] = True
        return original(poses, jax.device_put(rows[:]), *args, **kwargs)

    monkeypatch.setattr(host, "_fbp_from_host", resident)
    actual = fbp_host(geometry, grid, detector, data, config=config)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.01


@pytest.mark.gpu
@pytest.mark.parametrize("free", [None, 0, "below", "enough"])
@pytest.mark.parametrize("native", [False, True])
def test_resident_rows_budget_before_allocation(monkeypatch, free, native):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, _, data, _ = gaussian()
    projections = jax.device_put(data)
    detector = replace(detector, nv=3)
    grid = replace(grid, nz=3)
    spectrum = jnp.ones(65, jnp.float32)
    required = 2 * (4 * 83 * 3 * 40) + 3 * (4 * 40 * 40 * 3) + 16 * 17 * 3 * (65 + 40)
    available = 2 * required - (1 if free == "below" else 0) if isinstance(free, str) else free
    probes = []

    def memory():
        probes.append(jax.config.jax_default_device)
        return available

    monkeypatch.setattr(host, "device_free_memory_bytes", memory)
    actual = host._resident_axial_rows(projections, 2, detector, spectrum, grid, 17, native=native)
    assert (actual is not None) == (native and free == "enough")
    assert probes == ([next(iter(projections.devices()))] if native else [])
    if actual is not None:
        np.testing.assert_array_equal(actual, data[:, 2:5])


@pytest.mark.gpu
@pytest.mark.parametrize("reason", ["outside", "dtype", "cpu_target", "cpu_source"])
def test_ineligible_resident_rows_keep_bounded_fallback(monkeypatch, reason):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, _, data, _ = gaussian()
    projections = jax.device_put(data.astype(np.int16) if reason == "dtype" else data)
    spectrum = jnp.ones(65, jnp.float32)
    if reason == "cpu_target":
        spectrum = jax.device_put(spectrum, jax.devices("cpu")[0])
    if reason == "cpu_source":
        projections = jax.device_put(projections, jax.devices("cpu")[0])
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: pytest.fail("queried memory"))
    assert (
        host._resident_axial_rows(
            projections, 1 if reason == "outside" else 0, detector, spectrum, grid, 17, native=True
        )
        is None
    )


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["native", "fractional", "strided", "tight"])
def test_resident_host_output_matches_fallback(monkeypatch, case):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, geometry, data, truth = gaussian()
    if case == "fractional":
        grid = replace(grid, vol_center=(0, 0, 0.2))
    elif case == "strided":
        grid = replace(grid, nz=3, vz=1.6)
    geometry = replace(geometry, grid=grid)
    config = FBPHostConfig(slices_per_batch=3, views_per_batch=17)
    expected = fbp_host(geometry, grid, detector, data, config=config)
    calls = []
    original = fbp_module._fbp_from_resident

    def resident(*args, **kwargs):
        calls.append(args[1].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(fbp_module, "_fbp_from_resident", resident)
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: 0 if case == "tight" else 2**30)
    out = np.empty_like(expected)
    actual = fbp_host(geometry, grid, detector, jax.device_put(data), config=config, out=out)
    assert actual is out
    assert bool(calls) == (case == "native")
    assert all(shape == (83, 3, 40) for shape in calls)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    if case == "native":
        assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.01


@pytest.mark.gpu
def test_resident_nonfinite_tail_rejected_before_slab_write(monkeypatch):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, geometry, data, _ = gaussian()
    data[-1, 1, 3] = np.nan
    out = np.full((40, 40, 7), 123, np.float32)
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: 2**30)
    with pytest.raises(ValueError, match="finite in FP32"):
        fbp_host(
            geometry,
            grid,
            detector,
            jax.device_put(data),
            out=out,
            config=FBPHostConfig(slices_per_batch=3, views_per_batch=17),
        )
    np.testing.assert_array_equal(out, 123)
