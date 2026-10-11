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
from tomojax.core.cone import frame_coefficients, use_cuda_cone
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    Detector,
    Grid,
    LaminographyGeometry,
    stack_view_poses,
)

pytestmark = pytest.mark.numerical


@pytest.mark.gpu
@pytest.mark.parametrize("width", [5, 6])
@pytest.mark.parametrize("jacobian", ["central", "autodiff"])
@pytest.mark.parametrize(
    ("grid_case", "direction"),
    [
        ("small", (1, 0, 0)),
        ("small", (0.2, 0.7, -0.8)),
        ("large", (0.2, 0.7, -0.8)),
        ("tie", (1, 1, 1)),
    ],
)
def test_cuda_columns_retain_individual_central_differences(
    monkeypatch, width, jacobian, direction, grid_case
):
    monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    if not use_cuda_cone():
        pytest.skip("CUDA cone columns require an NVIDIA GPU and CuPy")
    if grid_case == "small":
        grid = Grid(7, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.23, -0.17, 0.31))
        detector = Detector(9, 7, 1.3, 0.8)
    elif grid_case == "tie":
        # One ray and isotropic spacing make (1,1,1) an EXACT index-axis tie.
        # Odd/even voxel centres give the axes distinct quadrature phases.
        grid = Grid(7, 6, 5, 1.0, 1.0, 1.0)
        detector = Detector(1, 1, 1.3, 0.8)
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
    rng = np.random.default_rng(533)
    volume = jnp.asarray(rng.uniform(-0.1, 1.1, (grid.nx, grid.ny, grid.nz)), jnp.float32)
    target = jnp.asarray(rng.normal(size=(detector.nv, detector.nu)), jnp.float32)
    weight = jnp.asarray(rng.uniform(0.05, 0.99, target.shape), jnp.float32)
    if grid_case == "tie":
        row = np.asarray(frame_coefficients(nominal[None], frame, grid, detector))[0]
        ray = row[3:6] - row[:3]
        np.testing.assert_array_equal(np.abs(ray), np.repeat(abs(ray[0]), 3))
    else:
        weight = weight.at[0].set(0)
    options = PoseJacobianOptions(0.001, "object", True, 1, "fp32", "joseph", jacobian, True)
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
            pose = apply_pose_update(nominal, q, translation_frame="object")
            image = forward_project_view_T(
                pose, grid, detector, volume, frames=frame, projector_backend="pallas"
            )
            return image.ravel()

        displacement = 0.001 * min(grid.vx, grid.vy, grid.vz)
        radius = 0.5 * math.sqrt(
            (grid.nx * grid.vx) ** 2 + (grid.ny * grid.vy) ** 2 + (grid.nz * grid.vz) ** 2
        )
        steps = jnp.asarray([displacement / radius] * 3 + [displacement] * (width - 3), jnp.float32)
        columns = jax.vmap(
            lambda direction, step: weight.ravel()
            * (prediction(p + step * direction) - prediction(p - step * direction))
            / (2 * step)
        )(jnp.eye(width, dtype=jnp.float32), steps)
        return weight.ravel() * (prediction(p) - target.ravel()), columns

    actual = candidate(params, nominal, frame, volume, target, weight)
    expected = individual(params, nominal, frame, volume, target, weight)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cone", [False, True])
def test_central_pose_columns_do_not_depend_on_measurement_offsets(cone):
    # For fixed pixel weights, d[W(A(p)x-y)]/dp is independent of y.
    # Subtracting a large measurement before differencing can round away
    # the perturbations, even though the represented predictions distinguish p.
    grid = Grid(7, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.23, -0.17, 0.31))
    detector = Detector(9, 7, 1.3, 0.8, (0.13, -0.19))
    geometry = (
        ConeGeometry(grid, detector, [31.7], ConeBeam(40, 60), tilt_deg=23)
        if cone
        else LaminographyGeometry(grid, detector, [31.7], tilt_deg=23)
    )
    nominal = stack_view_poses(geometry, 1)[0]
    frame = (
        jnp.asarray([[0, 40, 0], [1, 0, 0], [0, 0, 1], [0, -80, 0]], jnp.float32) if cone else None
    )
    rng = np.random.default_rng(687)
    volume = jnp.asarray(rng.uniform(0.2, 1.1, (grid.nx, grid.ny, grid.nz)), jnp.float32)
    params = jnp.asarray([0.012, -0.018, 0.031, 0.11, -0.07, 0.04], jnp.float32)
    weight = jnp.asarray(rng.uniform(0.1, 1, (detector.nv, detector.nu)), jnp.float32).at[0].set(0)
    target = jnp.asarray(rng.normal(size=weight.shape), jnp.float32)
    options = PoseJacobianOptions(0.001, "object", True, 1, "fp32", "joseph", "central", cone)
    evaluate = jax.jit(
        partial(
            pose_prediction_and_columns,
            grid=grid,
            detector=detector,
            det_grid=None,
            options=options,
        )
    )
    residual, columns = evaluate(params, nominal, frame, volume, target, weight)
    assert np.linalg.norm(columns) > 0
    np.testing.assert_array_equal(np.asarray(residual).reshape(weight.shape)[0], 0)
    np.testing.assert_array_equal(np.asarray(columns).reshape(6, *weight.shape)[:, 0], 0)
    for offset in (1.0, 1e5):
        shifted_residual, shifted_columns = evaluate(
            params, nominal, frame, volume, target + offset, weight
        )
        np.testing.assert_array_equal(shifted_columns, columns)
        assert not np.array_equal(shifted_residual, residual)
