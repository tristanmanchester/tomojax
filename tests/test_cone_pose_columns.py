"""Batched CUDA central differences retain the per-view finite-difference model."""

from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._geometry.parametrizations import apply_pose_update

# check-public-imports: allow-private
from tomojax.alignment._pose._pose_jacobian import PoseJacobianOptions, pose_prediction_and_columns
from tomojax.core.cone import use_cuda_cone
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid

pytestmark = [pytest.mark.numerical, pytest.mark.gpu]


@pytest.mark.parametrize("translation_frame", ["object", "detector"])
@pytest.mark.parametrize("width", [5, 6])
@pytest.mark.parametrize("jacobian", ["central", "autodiff"])
@pytest.mark.parametrize("grid_case", ["small", "large"])
@pytest.mark.parametrize(
    "direction", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (0.2, 0.7, -0.8)]
)
def test_cuda_columns_retain_individual_central_differences(
    monkeypatch, translation_frame, width, jacobian, direction, grid_case
):
    monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    if not use_cuda_cone():
        pytest.skip("CUDA cone columns require an NVIDIA GPU and CuPy")
    if grid_case == "small":
        grid = Grid(7, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.23, -0.17, 0.31))
        detector = Detector(9, 7, 1.3, 0.8)
    else:
        grid = Grid(101, 101, 101, 50 / 101, 53 / 101, 47 / 101, vol_center=(0.23, -0.17, 0.31))
        detector = Detector(192, 243, 0.6, 0.6)
    beam = np.asarray(direction, np.float64)
    beam /= np.linalg.norm(beam)
    u = np.cross(beam, np.eye(3)[np.argmin(np.abs(beam))])
    u /= np.linalg.norm(u)
    v = np.cross(beam, u)
    u, v = np.cos(0.13) * u + np.sin(0.13) * v, -np.sin(0.13) * u + np.cos(0.13) * v
    frame = jnp.asarray(np.stack([40 * beam, u, v, -80 * beam]), jnp.float32)
    nominal = jnp.eye(4)
    params = jnp.zeros(width)
    if direction == (0.2, 0.7, -0.8):
        params = jnp.asarray([0.012, -0.018, 0.031, 0.11, -0.07, 0.04][:width])
    volume = jnp.asarray(
        np.random.default_rng(533).uniform(-0.1, 1.1, (grid.nx, grid.ny, grid.nz)), jnp.float32
    )
    target = jnp.asarray(
        np.random.default_rng(85).normal(size=(detector.nv, detector.nu)), jnp.float32
    )
    weight = jnp.asarray(
        np.random.default_rng(86).uniform(0.05, 0.99, (detector.nv, detector.nu)), jnp.float32
    )
    weight = weight.at[0].set(0)
    options = PoseJacobianOptions(
        0.001, translation_frame, True, 1, "fp32", "joseph", jacobian, True
    )
    candidate = jax.jit(
        partial(
            pose_prediction_and_columns,
            grid=grid,
            detector=detector,
            det_grid=None,
            options=options,
        )
    )

    @jax.jit
    def individual(p, nominal, frame, volume, target, weight):
        def prediction(q):
            pose = apply_pose_update(nominal, q, translation_frame=translation_frame)
            image = forward_project_view_T(
                pose, grid, detector, volume, frames=frame, projector_backend="pallas"
            )
            return (weight * (image - target)).ravel()

        displacement = 0.001 * min(grid.vx, grid.vy, grid.vz)
        radius = 0.5 * math.sqrt(
            (grid.nx * grid.vx) ** 2 + (grid.ny * grid.vy) ** 2 + (grid.nz * grid.vz) ** 2
        )
        steps = jnp.asarray([displacement / radius] * 3 + [displacement] * (width - 3), jnp.float32)
        columns = jax.vmap(
            lambda direction, step: (
                prediction(p + step * direction) - prediction(p - step * direction)
            )
            / (2 * step)
        )(jnp.eye(width, dtype=jnp.float32), steps)
        return prediction(p), columns

    actual = candidate(params, nominal, frame, volume, target, weight)
    expected = individual(params, nominal, frame, volume, target, weight)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
