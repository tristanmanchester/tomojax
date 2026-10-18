"""Batching active parallel pose tangents preserves physical derivatives."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.alignment.api import apply_pose_update
from tomojax.core.joseph import forward_project_planes, plane_coefficients, plane_pose_columns
from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses


@pytest.mark.gpu
@pytest.mark.parametrize("integrator", ["joseph", "joseph_cubic"])
@pytest.mark.parametrize("indices", [(3, 4), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5)])
def test_parallel_pose_columns_match_reference(integrator, indices):
    if jax.default_backend() != "gpu":
        pytest.skip("CUDA required")
    grid = Grid(77, 6, 5, 0.7, 1.1, 0.9, vol_center=(0.23, -0.17, 0.31))
    detector = Detector(9, 7, 0.8, 1.3, (0.15, -0.21))
    nominal = stack_view_poses(LaminographyGeometry(grid, detector, [47], tilt_deg=31), 1)[0]
    rng = np.random.default_rng(543)
    volume = jnp.asarray(rng.uniform(-0.1, 1.1, (77, 6, 5)), jnp.float32)
    target = jnp.asarray(rng.normal(size=(7, 9)), jnp.float32)
    weight = jnp.asarray(rng.uniform(0.05, 0.99, target.shape), jnp.float32).at[0].set(0)
    params = jnp.asarray([0.012, -0.018, 0.031, 0.11, -0.07, 0.04], jnp.float32)
    interpolation = "cubic" if integrator == "joseph_cubic" else "linear"

    def coefficients(p):
        pose = apply_pose_update(nominal, p, translation_frame="detector")
        return plane_coefficients(pose[None], grid, detector)[0]

    coeff = coefficients(params)
    tangent = jax.jacfwd(coefficients)(params).T[jnp.asarray(indices)]
    images = jax.jit(
        partial(
            plane_pose_columns,
            grid=grid,
            detector=detector,
            backend="cuda",
            interpolation=interpolation,
        )
    )(coeff[None], volume, tangent[None])[0]
    actual = (
        weight.ravel() * (images[0].ravel() - target.ravel()),
        weight.ravel() * images[1:].reshape(len(indices), -1),
    )

    def prediction(p):
        return weight.ravel() * (
            forward_project_planes(
                coefficients(p)[None],
                volume,
                grid,
                detector,
                backend="jax",
                interpolation=interpolation,
            ).ravel()
            - target.ravel()
        )

    expected = (
        prediction(params),
        jax.jit(jax.vmap(lambda d: jax.jvp(prediction, (params,), (d,))[1]))(
            jnp.eye(6, dtype=jnp.float32)[jnp.asarray(indices)]
        ),
    )
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=5e-4, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_fused_columns_keep_cell_and_support_boundary_derivatives(interpolation):
    if jax.default_backend() != "gpu":
        pytest.skip("CUDA required")
    grid, detector = Grid(73, 4, 3, 1, 1, 1), Detector(1, 1, 1, 1)
    coeff = jnp.asarray(
        [[0, 0, 0, 0, b, 0, 0, 0, 1, 1, 0, 0, 0, 0] for b in [-2, -1, 0, 1, 3, 4]], jnp.float32
    )
    tangent = jnp.zeros((6, 2, 14), jnp.float32).at[:, 0, 4].set(1).at[:, 1, 8].set(1)
    volume = jnp.ones((73, 4, 3), jnp.float32)
    function = partial(
        plane_pose_columns, grid=grid, detector=detector, interpolation=interpolation
    )
    actual = jax.jit(partial(function, backend="cuda"))(coeff, volume, tangent)
    expected = jax.jit(partial(function, backend="jax"))(coeff, volume, tangent)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
