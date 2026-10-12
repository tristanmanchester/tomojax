"""Public reconstruction dispatch keeps oversized parallel volumes on the host."""

from dataclasses import replace

import jax
import numpy as np
import pytest

import tomojax as tj
import tomojax.backends
from tomojax.recon import FBPConfig


def _scan(kind):
    grid = tj.Grid(9, 7, 19, 0.8, 1.2, 0.7, vol_center=(1.2, -0.4, 0.31))
    detector = tj.Detector(11, 20, 0.9, 1.0, (0.23, -0.19))
    angles = np.linspace(0, 180, 13, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    if kind == "lamino":
        geometry = tj.LaminographyGeometry(grid, detector, angles * 2, tilt_deg=25)
    data = np.random.default_rng(132).normal(size=(13, 20, 11)).astype(np.float32)
    return tj.Scan(data, geometry)


@pytest.mark.parametrize("kind", ["parallel", "lamino"])
@pytest.mark.parametrize("device_input", [False, True])
def test_public_reconstruction_uses_host_slabs_without_changing_values(
    monkeypatch, kind, device_input
):
    scan = _scan(kind)
    if device_input:
        scan = replace(scan, projections=jax.device_put(scan.projections))
    cfg = FBPConfig(filter="hann", scale=0.17)
    expected = np.asarray(tj.reconstruct(scan, config=cfg).volume)
    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", lambda: 4096)
    actual = tj.reconstruct(scan, config=cfg)
    assert isinstance(actual.volume, np.ndarray)
    assert actual.info["host_slabs"]
    assert actual.info["config"] == cfg
    assert actual.scan is scan
    np.testing.assert_allclose(actual.volume, expected, atol=3e-6, rtol=5e-5)


@pytest.mark.parametrize("kind", ["parallel", "lamino"])
@pytest.mark.parametrize("number", [1, 2])
def test_host_dispatch_honours_explicit_devices(monkeypatch, kind, number):
    if len(jax.devices()) < number:
        pytest.skip("needs multiple devices")
    scan = _scan(kind)
    expected = np.asarray(tj.reconstruct(scan).volume)
    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", lambda: 4096)
    actual = tj.reconstruct(scan, devices=jax.devices()[:number])
    assert isinstance(actual.volume, np.ndarray) and actual.info["host_slabs"]
    np.testing.assert_allclose(actual.volume, expected, atol=3e-6, rtol=5e-5)


def test_dispatch_probes_memory_on_explicit_devices(monkeypatch):
    if len(jax.devices()) < 2:
        pytest.skip("needs multiple devices")
    first, second = jax.devices()[:2]
    checked = []

    def free_memory():
        device = jax.config.jax_default_device
        checked.append(device)
        return 4096 if device == second else 2**30

    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", free_memory)
    scan = _scan("parallel")
    with jax.default_device(first):
        actual = tj.reconstruct(scan, devices=[first, second])
        assert jax.config.jax_default_device == first
    assert isinstance(actual.volume, np.ndarray) and actual.info["host_slabs"]
    assert checked[:2] == [first, second]


@pytest.mark.parametrize("kind", ["parallel", "lamino"])
@pytest.mark.parametrize("caller_owned", [False, True])
def test_single_explicit_device_keeps_host_allocation_at_the_backend(
    monkeypatch, kind, caller_owned
):
    from tomojax.recon import api

    scan = _scan(kind)
    shape = (scan.grid.nx, scan.grid.ny, scan.grid.nz)
    expected = np.empty(shape, np.float32, order="F")
    out = expected if caller_owned else None
    target = jax.devices("cpu")[-1]
    original_device = jax.config.jax_default_device
    calls = []

    def host(geometry, grid, detector, data, *, config, out):
        calls.append(jax.config.jax_default_device)
        # Let the backend allocate its preferred storage, or honor caller storage.
        if caller_owned:
            assert np.shares_memory(out, expected)
        else:
            assert out is None
        assert data is scan.projections
        expected.fill(0.25)
        return expected

    monkeypatch.setattr(api, "fbp_host", host)
    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", lambda: 4096)
    result = tj.reconstruct(scan, devices=[target], out=out)
    assert result.volume is expected
    assert result.volume.flags.f_contiguous
    np.testing.assert_array_equal(result.volume, 0.25)
    assert calls == [target]
    assert jax.config.jax_default_device is original_device
