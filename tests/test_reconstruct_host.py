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
