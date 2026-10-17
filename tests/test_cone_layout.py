"""Prepared cone layouts preserve the exact pair and avoid device-driven planning."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.core import cone
from tomojax.recon import SIRTConfig, sirt


def test_cone_layout_uses_only_host_poses():
    grid = tj.Grid(17, 19, 21, 1, 1.1, 0.9)
    detector = tj.Detector(31, 29, 2, 2)
    geometry = tj.ConeGeometry(grid, detector, np.arange(0, 360, 30), tj.ConeBeam(100, 200))
    ordinary = cone.cone_model(geometry)
    assert ordinary.forward_workspace_bytes(grid, 12, detector) == 0
    geometry = replace(geometry, tilt_deg=60)

    class Poses:
        def __init__(self, table):
            self.table = table

        def __getattr__(self, name):
            return getattr(geometry, name)

        def pose_for_view(self, i):
            return self.table[i]

    host = cone.cone_model(Poses(geometry.poses()))
    assert host.forward_workspace_bytes(grid, 12, detector) == 4 * 17 * 19 * 21
    assert host.forward_workspace_bytes(replace(grid, vz=2), 12, detector) == 0
    assert host.forward_workspace_bytes(grid, 1, tj.Detector(1, 1, 2, 2)) == 0
    device = cone.cone_model(Poses(jnp.asarray(geometry.poses())))
    assert device.layout_hint is None
    device_angles = replace(geometry, angles=jnp.asarray(geometry.angles))
    assert cone.cone_model(device_angles).layout_hint is None


@pytest.mark.gpu
@pytest.mark.parametrize(("tilt", "low_memory"), [(0, False), (60, False), (60, True)])
def test_prepared_cone_projection_and_subset_updates_match_original(monkeypatch, tilt, low_memory):
    if not cone.use_cuda_cone():
        pytest.skip("requires CUDA cone kernels")
    if low_memory:
        from tomojax import backends

        monkeypatch.setattr(backends, "device_free_memory_bytes", lambda: 0)
    grid = tj.Grid(23, 19, 17, 0.8, 1.1, 0.9, vol_center=(0.2, -0.1, 0.3))
    detector = tj.Detector(35, 31, 1.7, 1.9, center=(0.3, -0.2))
    geometry = tj.ConeGeometry(
        grid,
        detector,
        np.linspace(0, 360, 65, endpoint=False),
        tj.ConeBeam(100, 200),
        tilt_deg=tilt,
    )
    rng = np.random.default_rng(58)
    volume = jnp.asarray(rng.normal(size=(23, 19, 17)), jnp.float32)
    images = jnp.asarray(rng.normal(size=(65, 31, 35)), jnp.float32)
    projected = jax.jit(lambda x: tj.project(geometry, x))
    actual = projected(volume)
    np.testing.assert_allclose(projected(2 * volume), 2 * actual, rtol=2e-6, atol=1e-5)
    gradient = jax.grad(lambda x: jnp.vdot(projected(x), images))(volume)
    expected_gradient = jax.jit(lambda y: tj.backproject(geometry, y))(images)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=1e-4)
    config = SIRTConfig(iterations=2, subsets=3, views_per_batch=8)
    updated = sirt(geometry, grid, detector, images, config=config)[0]
    monkeypatch.setattr(cone, "_layout_hint", lambda *_: None)
    expected = jax.jit(lambda x: tj.project(geometry, x))(volume)
    reference = sirt(geometry, grid, detector, images, config=config)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(updated, reference, rtol=2e-5, atol=1e-5)
