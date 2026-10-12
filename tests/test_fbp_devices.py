"""FBP executes on requested devices, independently of input residency."""

from __future__ import annotations

from contextlib import nullcontext
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
    monkeypatch.setattr(jax, "default_device", lambda _: nullcontext())
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


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("source_index", [0, 1])
def test_oblique_cache_query_follows_actual_execution(monkeypatch, committed, source_index):
    """The pose temporary can reside on a different CPU than committed input."""
    import jax.numpy as jnp

    from tomojax.geometry import LaminographyGeometry

    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two CPU devices")
    source, context = devices[source_index], devices[1 - source_index]
    grid = Grid(9, 7, 5, 0.7, 0.9, 0.8)
    detector = Detector(17, 11, 0.7, 0.8)
    geometry = LaminographyGeometry(
        grid, detector, np.linspace(0, 180, 13, endpoint=False), tilt_deg=30, tilt_about="z"
    )
    data = np.random.default_rng(32).random((13, 11, 17)).astype(np.float32)
    with jax.default_device(source):
        projections = jnp.asarray(data)
    if committed:
        projections = jax.device_put(projections, source)
    assert projections.committed == committed
    expected = source if committed else context
    cache = importlib.import_module("tomojax.recon._fbp_pallas")
    queries = []

    def queried(device):
        queries.append(device)

    monkeypatch.setattr(cache, "_cuda_l2_bytes", queried)
    with jax.default_device(context):
        actual = fbp(geometry, grid, detector, projections)
    assert actual.devices() == {expected}
    assert queries == [expected]
    reference = fbp(geometry, grid, detector, data, config=FBPConfig(devices=expected))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


@pytest.mark.gpu
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("source_cuda", [False, True])
def test_default_fbp_execution_matches_jax_input_placement(
    monkeypatch, cuda_device, committed, source_cuda
):
    """Uncommitted input follows the caller; committed input fixes execution."""
    import jax.numpy as jnp

    module = importlib.import_module("tomojax.recon.fbp")
    cache = importlib.import_module("tomojax.recon._fbp_pallas")
    cpu = jax.devices("cpu")[0]
    source, context = (cuda_device, cpu) if source_cuda else (cpu, cuda_device)
    expected = source if committed else context
    geometry, grid, detector, data, truth = _gaussian()
    with jax.default_device(cpu):
        reference = np.asarray(
            fbp(geometry, grid, detector, data, config=FBPConfig(backprojector="jax"))
        )
    with jax.default_device(source):
        projections = jnp.asarray(data)
    if committed:
        projections = jax.device_put(projections, source)
    assert projections.committed == committed
    original_plan, original_query = module._plan, cache._cuda_l2_bytes
    selected, queries = [], []

    def planned(*args, **kwargs):
        plan = original_plan(*args, **kwargs)
        selected.append(plan.backend)
        return plan

    def queried(device):
        queries.append(device)
        return original_query(device)

    monkeypatch.setattr(module, "_plan", planned)
    monkeypatch.setattr(cache, "_cuda_l2_bytes", queried)
    with jax.default_device(context):
        result = fbp(geometry, grid, detector, projections)
    assert result.devices() == {expected}
    assert selected == ["pallas" if expected == cuda_device else "jax"]
    assert queries == ([expected] if expected == cuda_device else [])
    volume = np.asarray(result)
    np.testing.assert_allclose(volume, reference, rtol=2e-5, atol=2e-8)
    assert np.linalg.norm(volume - truth) / np.linalg.norm(truth) < 0.02


@pytest.mark.gpu
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("source_cuda", [False, True])
def test_default_pallas_requirement_matches_execution(cuda_device, committed, source_cuda):
    import jax.numpy as jnp

    cpu = jax.devices("cpu")[0]
    source, context = (cuda_device, cpu) if source_cuda else (cpu, cuda_device)
    expected = source if committed else context
    geometry, grid, detector, data, truth = _gaussian()
    with jax.default_device(source):
        projections = jnp.asarray(data)
    if committed:
        projections = jax.device_put(projections, source)
    with jax.default_device(context):
        if expected == cpu:
            with pytest.raises(ValueError, match="CUDA execution devices"):
                fbp(geometry, grid, detector, projections, config=FBPConfig(backprojector="pallas"))
        else:
            result = fbp(
                geometry, grid, detector, projections, config=FBPConfig(backprojector="pallas")
            )
            assert result.devices() == {expected}
            assert np.linalg.norm(np.asarray(result) - truth) / np.linalg.norm(truth) < 0.02


@pytest.mark.parametrize("oblique", [False, True])
def test_default_fbp_preserves_distributed_input_sharding(oblique):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from tomojax.geometry import LaminographyGeometry

    devices = jax.devices("cpu")[:2]
    if len(devices) < 2:
        pytest.skip("requires two CPU devices")
    grid = Grid(9, 7, 5, 0.7, 0.9, 0.8)
    detector = Detector(17, 11, 0.7, 0.8)
    angles = np.linspace(0, 180, 12, endpoint=False)
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30, tilt_about="z")
        if oblique
        else ParallelGeometry(grid, detector, angles)
    )
    data = np.random.default_rng(32).random((12, 11, 17)).astype(np.float32)
    mesh = Mesh(np.array(devices), ("views",))
    projections = jax.device_put(data, NamedSharding(mesh, PartitionSpec("views")))
    with jax.default_device(devices[0]):
        result = fbp(geometry, grid, detector, projections)
        reference = fbp(geometry, grid, detector, data)
    assert result.devices() == set(devices)
    assert result.sharding == NamedSharding(mesh, PartitionSpec())
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference), rtol=2e-5, atol=2e-7)
