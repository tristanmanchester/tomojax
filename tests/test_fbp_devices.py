"""FBP executes on requested devices, independently of input residency."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FBPConfig, fbp


def _gaussian():
    grid = Grid(17, 15, 3, 0.7, 0.9, 0.8, vol_center=(-0.3, 0.7, 2.3))
    detector = Detector(33, 5, 0.7, 0.8, center=(1.1, 2.3))
    angles = np.linspace(7, 187, 37, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du + detector.center[0]
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv + detector.center[1]
    phi = np.deg2rad(angles)
    centre = 1.2 * np.cos(phi) + 0.7 * np.sin(phi)
    radial = 0.02 * np.sqrt(2 * np.pi) * 2.3 * np.exp(-((u - centre[:, None]) ** 2) / (2 * 2.3**2))
    axial = np.exp(-((v - 2.7) ** 2) / (2 * 1.3**2))
    projections = (radial[:, None, :] * axial[None, :, None]).astype(np.float32)
    x = (np.arange(grid.nx) - (grid.nx - 1) / 2) * grid.vx + grid.vol_center[0]
    y = (np.arange(grid.ny) - (grid.ny - 1) / 2) * grid.vy + grid.vol_center[1]
    z = (np.arange(grid.nz) - (grid.nz - 1) / 2) * grid.vz + grid.vol_center[2]
    truth = 0.02 * np.exp(-((x[:, None] - 1.2) ** 2 + (y[None, :] + 0.7) ** 2) / (2 * 2.3**2))
    truth = truth[:, :, None] * np.exp(-((z - 2.7) ** 2) / (2 * 1.3**2))
    return geometry, grid, detector, projections, truth


@dataclass(frozen=True)
class _Device:
    platform: str

    @property
    def client(self):
        return SimpleNamespace(platform_version="CUDA 12.0")


@pytest.mark.parametrize("input_cuda", [False, True])
@pytest.mark.parametrize("target_cuda", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_fbp_execution_devices_determine_backend(monkeypatch, input_cuda, target_cuda, explicit):
    """Dispatch-only probe runs without requiring CUDA hardware."""
    module = importlib.import_module("tomojax.recon.fbp")
    geometry, grid, detector, data, _ = _gaussian()
    device = _Device("gpu" if input_cuda else "cpu")
    poses = SimpleNamespace(shape=(len(data), 4, 4), devices=lambda: (device,))
    monkeypatch.setattr(module, "stack_view_poses", lambda *args: poses)
    cpu = jax.devices("cpu")[0]
    # The execution device's metadata suffices for this dispatch-only probe;
    # no fake device reaches a JAX mesh or a numerical kernel.
    target = _Device("gpu" if target_cuda else "cpu")
    config = FBPConfig(devices=target if explicit else None)
    selected = []

    def planned(*args, cuda):
        selected.append(cuda)
        return SimpleNamespace(
            backprojected=lambda *args: jax.device_put(
                np.zeros((grid.nx, grid.ny, grid.nz), np.float32), cpu
            )
        )

    def split(devices, n):
        return (
            SimpleNamespace(summed=lambda run: run(cpu, range(n))) if devices is not None else None
        )

    monkeypatch.setattr(module, "_plan", planned)
    monkeypatch.setattr(module, "view_split", split)
    fbp(geometry, grid, detector, data, config=config)
    assert selected == [target_cuda if explicit else input_cuda]


@pytest.fixture
def cuda_device():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA and CPU devices")
    devices = jax.devices("gpu")
    if not all(device.client.platform_version.lower().startswith("cuda") for device in devices):
        pytest.skip("requires CUDA")
    return devices[0]


@pytest.mark.gpu
@pytest.mark.parametrize("on_host", [False, True])
@pytest.mark.parametrize("target_cuda", [False, True])
def test_cross_platform_fbp_uses_explicit_target(monkeypatch, cuda_device, on_host, target_cuda):
    module = importlib.import_module("tomojax.recon.fbp")
    cpu = jax.devices("cpu")[0]
    target, source = (cuda_device, cpu) if target_cuda else (cpu, cuda_device)
    geometry, grid, detector, data, truth = _gaussian()
    with jax.default_device(cpu):
        reference = np.asarray(
            fbp(geometry, grid, detector, data, config=FBPConfig(backprojector="jax"))
        )
    original, selected = module._plan, []

    def planned(*args, **kwargs):
        plan = original(*args, **kwargs)
        selected.append(plan.backend)
        return plan

    monkeypatch.setattr(module, "_plan", planned)
    with jax.default_device(source):
        projections = data if on_host else jax.device_put(data, source)
        result = fbp(geometry, grid, detector, projections, config=FBPConfig(devices=target))
    assert selected == ["pallas" if target_cuda else "jax"]
    assert result.devices() == {target}
    volume = np.asarray(result)
    np.testing.assert_allclose(volume, reference, rtol=2e-5, atol=2e-8)
    assert np.linalg.norm(volume - truth) / np.linalg.norm(truth) < 0.02


@pytest.mark.gpu
@pytest.mark.parametrize("on_host", [False, True])
def test_pallas_accepts_cpu_input_on_explicit_cuda(cuda_device, on_host):
    cpu = jax.devices("cpu")[0]
    geometry, grid, detector, data, truth = _gaussian()
    with jax.default_device(cpu):
        projections = data if on_host else jax.device_put(data, cpu)
        result = fbp(
            geometry,
            grid,
            detector,
            projections,
            config=FBPConfig(devices=cuda_device, backprojector="pallas"),
        )
    assert result.devices() == {cuda_device}
    assert np.linalg.norm(np.asarray(result) - truth) / np.linalg.norm(truth) < 0.02


@pytest.mark.gpu
@pytest.mark.parametrize("on_host", [False, True])
def test_pallas_refuses_explicit_cpu_even_with_cuda_input(cuda_device, on_host):
    cpu = jax.devices("cpu")[0]
    geometry, grid, detector, data, _ = _gaussian()
    with jax.default_device(cuda_device):
        projections = data if on_host else jax.device_put(data, cuda_device)
        with pytest.raises(ValueError, match="CUDA execution devices"):
            fbp(
                geometry,
                grid,
                detector,
                projections,
                config=FBPConfig(devices=cpu, backprojector="pallas"),
            )
