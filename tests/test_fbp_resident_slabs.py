"""Budgeted native resident slabs retain host-output and numerical contracts."""

from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FBPHostConfig, fbp_host

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


@pytest.mark.parametrize("contrast", [0.01, 0.001])
@pytest.mark.parametrize("batch", [17, 32])
@pytest.mark.parametrize("storage", ["host", "device"])
def test_host_fbp_preserves_weak_feature_signal(contrast, batch, storage):
    """A small whole-image error must not hide loss of low-contrast structure."""
    grid, detector, geometry, background, background_truth = gaussian()
    x = (np.arange(grid.nx) - (grid.nx - 1) / 2) * grid.vx
    z = (np.arange(grid.nz) - (grid.nz - 1) / 2) * grid.vz
    axial = np.exp(-0.5 * ((z - 0.7) / 1.1) ** 2)
    sigma = 1.3
    # Continuous line integrals, not the numerical forward projector. The
    # feature is narrow in all three dimensions against a broader background.
    profile = np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * (x / sigma) ** 2)
    feature = np.broadcast_to(axial[:, None] * profile, background.shape).astype(np.float32)
    truth = np.exp(-0.5 * (x[:, None, None] ** 2 + x[None, :, None] ** 2) / sigma**2) * axial
    config = FBPHostConfig(slices_per_batch=3, views_per_batch=batch)

    def reconstruct(data):
        source = jax.device_put(data) if storage == "device" else data
        return fbp_host(geometry, grid, detector, source, config=config)

    base = reconstruct(background)
    actual = reconstruct(background + contrast * feature)
    expected = background_truth + contrast * truth
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.01
    # Subtract reconstructions, not input sinograms: quantizing their sum can
    # erase the feature even when reconstruction of the feature alone is good.
    signal = (actual.astype(np.float64) - base) / contrast
    assert np.linalg.norm(signal - truth) / np.linalg.norm(truth) < 0.08


@pytest.mark.gpu
@pytest.mark.parametrize("free", [None, 0, "below", "enough"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("source", ["host", "device"])
def test_resident_rows_budget_before_allocation(monkeypatch, free, native, source):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, _, data, _ = gaussian()
    projections = data if source == "host" else jax.device_put(data)
    detector = replace(detector, nv=3)
    grid = replace(grid, nz=3)
    spectrum = jnp.ones(65, jnp.float32)
    row_copies = 3 if source == "host" else 2
    required = row_copies * (4 * 83 * 3 * 40) + 3 * (4 * 40 * 40 * 3) + 16 * 17 * 3 * (65 + 40)
    available = 2 * required - (1 if free == "below" else 0) if isinstance(free, str) else free
    probes = []

    def memory():
        probes.append(jax.config.jax_default_device)
        return available

    monkeypatch.setattr(host, "device_free_memory_bytes", memory)
    actual = host._resident_axial_rows(projections, 2, detector, spectrum, grid, 17, native=native)
    assert (actual is not None) == (native and free == "enough")
    assert probes == ([next(iter(spectrum.devices()))] if native else [])
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
@pytest.mark.parametrize("source", ["host", "device"])
def test_resident_host_output_matches_fallback(monkeypatch, case, source):
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
    projections = data if source == "host" else jax.device_put(data)
    actual = fbp_host(geometry, grid, detector, projections, config=config, out=out)
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


@pytest.mark.parametrize("layout", ["c", "f", "negative"])
def test_host_row_upload_preserves_strides_and_bounds_staging(monkeypatch, layout):
    data = np.arange(101 * 7 * 13, dtype=np.float32).reshape(101, 7, 13)
    if layout == "f":
        data = np.asfortranarray(data)
    elif layout == "negative":
        data = data[::-1, ::-1, ::-1]
    limit = 5 * 3 * 13 * 4
    monkeypatch.setattr(host, "_HOST_UPLOAD_BYTES", limit)
    contiguous = np.ascontiguousarray
    copies = []

    def bounded(value, *args, **kwargs):
        assert value.nbytes <= limit
        copies.append(value.shape[0])
        return contiguous(value, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", bounded)
    actual = host._upload_axial_rows(data, 2, 5, jax.devices("cpu")[0])
    np.testing.assert_array_equal(actual, data[:, 2:5])
    assert copies == [5] * 20 + [1]


def test_host_row_upload_oversized_view_keeps_fallback(monkeypatch):
    data = np.ones((5, 7, 13), np.float32)
    monkeypatch.setattr(host, "_HOST_UPLOAD_BYTES", 3 * 13 * 4 - 1)
    monkeypatch.setattr(np, "ascontiguousarray", lambda *a, **k: pytest.fail("host copy"))
    assert host._upload_axial_rows(data, 2, 5, jax.devices("cpu")[0]) is None


def test_host_row_upload_host_allocation_failure_keeps_fallback(monkeypatch):
    data = np.ones((11, 7, 13), np.float32)
    monkeypatch.setattr(host, "_HOST_UPLOAD_BYTES", 5 * 3 * 13 * 4)
    contiguous = np.ascontiguousarray
    calls = []

    def unavailable(value, *args, **kwargs):
        calls.append(value.shape)
        if len(calls) == 2:
            raise MemoryError("host staging quota")
        return contiguous(value, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", unavailable)
    assert host._upload_axial_rows(data, 2, 5, jax.devices("cpu")[0]) is None
    assert len(calls) == 2


@pytest.mark.gpu
def test_host_row_upload_honors_target_inside_cpu_context(monkeypatch):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, _, data, _ = gaussian()
    detector, grid = replace(detector, nv=3), replace(grid, nz=3)
    spectrum = jnp.ones(65, jnp.float32)
    gpu = next(iter(spectrum.devices()))
    monkeypatch.setattr(host, "device_free_memory_bytes", lambda: 2**30)
    with jax.default_device(jax.devices("cpu")[0]):
        result = host._resident_axial_rows(data, 2, detector, spectrum, grid, 17, native=True)
    assert result.devices() == {gpu}
    np.testing.assert_array_equal(result, data[:, 2:5])
