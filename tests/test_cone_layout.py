"""Native cone layouts preserve physical axes, the exact pair and solver updates."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.core import cone
from tomojax.recon import SIRTConfig, sirt


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_cone_volume_indices_follow_physical_zyx_axes(backend):
    grid, detector = tj.Grid(7, 5, 3, 0.7, 1.1, 0.9), tj.Detector(1, 1, 1, 1)
    volume = np.arange(105, dtype=np.float32).reshape(3, 5, 7) / 105
    rows = np.zeros((3, 25), np.float32)
    rows[:, :3] = rows[:, 3:6] = [3, 2, 1]
    for axis in range(3):
        rows[axis, axis], rows[axis, 3 + axis] = -100, 100
    actual = cone.cone_project(volume, jnp.asarray(rows), grid, detector, backend=backend)
    expected = [volume[1, 2].sum() * 0.7, volume[1, :, 3].sum() * 1.1, volume[:, 2, 3].sum() * 0.9]
    np.testing.assert_allclose(np.asarray(actual).ravel(), expected, rtol=2e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("tilt", [0, 60])
def test_native_cone_projection_and_subset_batching(tilt):
    if not cone.use_cuda_cone():
        pytest.skip("requires CUDA cone kernels")
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
    volume = jnp.asarray(rng.normal(size=(17, 19, 23)), jnp.float32)
    images = jnp.asarray(rng.normal(size=(65, 31, 35)), jnp.float32)
    projected = jax.jit(lambda x: tj.project(geometry, x))
    actual = projected(volume)
    np.testing.assert_allclose(projected(2 * volume), 2 * actual, rtol=2e-6, atol=1e-5)
    gradient = jax.grad(lambda x: jnp.vdot(projected(x), images))(volume)
    expected_gradient = jax.jit(lambda y: tj.backproject(geometry, y))(images)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=1e-4)
    config = SIRTConfig(iterations=2, subsets=3, views_per_batch=8)
    updated = sirt(geometry, grid, detector, images, config=config)[0]

    def direct(x):
        coeff = cone.cone_coefficients(geometry.poses(), grid, detector, geometry.beam)
        return cone.cone_project(x, coeff, grid, detector, backend="cuda")

    expected = jax.jit(direct)(volume)
    reference = sirt(geometry, grid, detector, images, config=replace(config, views_per_batch=16))[
        0
    ]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(updated, reference, rtol=2e-5, atol=1e-5)
