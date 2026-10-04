"""Discrete projector contracts checked against JAX's independent transpose."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.core.projector import backproject_view_T, forward_project_view_T
from tomojax.geometry import Detector, Grid


def _oblique_case(shape: tuple[int, int, int]) -> tuple[Grid, Detector, jax.Array]:
    grid = Grid(*shape, 0.8, 1.1, 1.3, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(shape[0] + 3, shape[2] + 2, 0.9, 1.1, (0.23, -0.41))
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_euler("xyz", [17.0, 11.0, 43.0], degrees=True).as_matrix()
    pose[:3, 3] = [0.13, -0.31, 0.53]
    return grid, detector, jnp.asarray(pose)


@pytest.mark.parametrize("n_steps", [None, 19])
def test_explicit_adjoint_matches_autodiff_for_long_oblique_rays(n_steps: int | None) -> None:
    shape = (48, 39, 5)
    grid, detector, pose = _oblique_case(shape)
    rng = np.random.default_rng(19)
    volume = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    image = jnp.asarray(rng.normal(size=(detector.nv, detector.nu)), dtype=jnp.float32)

    def project(x: jax.Array) -> jax.Array:
        return forward_project_view_T(pose, grid, detector, x, step_size=0.17, n_steps=n_steps)

    expected = jax.jit(lambda y: jax.vjp(project, volume)[1](y)[0])(image)
    actual = jax.jit(
        lambda y: backproject_view_T(pose, grid, detector, y, step_size=0.17, n_steps=n_steps)
    )(image)

    rel_error = jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)
    assert float(rel_error) < 3e-6
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=5e-5)


@pytest.mark.parametrize("gather_dtype", ["fp32", "fp16", "bf16"])
def test_adjoint_matches_autodiff_for_gather_precision(gather_dtype: str) -> None:
    grid, detector, pose = _oblique_case((5, 4, 3))
    rng = np.random.default_rng(32)
    volume = jnp.asarray(rng.normal(size=(5, 4, 3)), dtype=jnp.float32)
    image = jnp.asarray(rng.normal(size=(detector.nv, detector.nu)), dtype=jnp.float32)

    # Include several nonzero traversal samples and allow the expected accumulation
    # roundoff of half-precision adjoints.
    def project(x: jax.Array) -> jax.Array:
        return forward_project_view_T(pose, grid, detector, x, gather_dtype=gather_dtype)

    expected = jax.jit(lambda y: jax.vjp(project, volume)[1](y)[0])(image)
    actual = jax.jit(
        lambda y: backproject_view_T(pose, grid, detector, y, gather_dtype=gather_dtype)
    )(image)
    tolerance = {"fp32": 3e-6, "fp16": 3e-3, "bf16": 3e-2}[gather_dtype]
    assert float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)) < tolerance


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("unroll", [None, 2])
def test_pallas_adjoint_and_weighted_loss_match_autodiff(
    interpret: bool, unroll: int | None
) -> None:
    # check-public-imports: allow-private
    from tomojax.core.pallas import api as pallas

    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    shape = (5, 4, 3)
    grid, detector, pose = _oblique_case(shape)
    rng = np.random.default_rng(45)
    volume = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    images = jnp.asarray(rng.normal(size=(2, detector.nv, detector.nu)), dtype=jnp.float32)
    poses = jnp.stack([pose, pose.at[0, 3].add(0.4)])
    weights = jnp.asarray([0.3, 1.7], dtype=jnp.float32)[:, None, None]

    def project(x: jax.Array) -> jax.Array:
        return jax.vmap(lambda t: forward_project_view_T(t, grid, detector, x, step_size=0.4))(
            poses
        )

    expected_bp = jax.jit(lambda y: jax.vjp(project, volume)[1](y)[0])(images)
    actual_bp = sum(
        pallas.backproject_view_T_pallas(
            poses[i],
            grid,
            detector,
            images[i],
            step_size=0.4,
            interpret=interpret,
            tile_shape=(4, 4),
            num_warps=1,
            unroll=unroll,
        )
        for i in range(2)
    )
    np.testing.assert_allclose(actual_bp, expected_bp, atol=1e-5, rtol=5e-5)
    expected_loss, expected_grad = jax.jit(
        jax.value_and_grad(lambda x: 0.5 * jnp.sum(((project(x) - images) * weights) ** 2))
    )(volume)
    actual_loss, actual_grad = pallas.forward_project_loss_and_grad_T_pallas(
        poses,
        grid,
        detector,
        volume,
        images,
        weights=weights,
        options=pallas.PallasProjectorOptions(
            step_size=0.4,
            interpret=interpret,
            tile_shape=(4, 4),
            num_warps=1,
            kernel_variant="generic",
            unroll=unroll,
        ),
    )
    np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-5, rtol=5e-5)
    np.testing.assert_allclose(actual_grad, expected_grad, atol=3e-5, rtol=1e-4)


@pytest.mark.numerical
@pytest.mark.parametrize("gather_dtype", ["fp32", "fp16", "bf16"])
def test_stack_adjoint_matches_sum_with_general_geometry(gather_dtype: str) -> None:
    # check-public-imports: allow-private
    from tomojax.core.projector import get_detector_grid_device, sum_backproject_views_T

    grid = Grid(7, 6, 5, 0.7, 1.1, 1.4)
    detector = Detector(9, 7, 0.8, 1.2, (0.13, -0.29))
    rng = np.random.default_rng(83)
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    poses[:, :3, :3] = Rotation.from_euler(
        "xyz", [[17, 29, 43], [-11, 27, 71], [21, -9, 131]], degrees=True
    ).as_matrix()
    poses[:, :3, 3] = rng.normal(size=(3, 3)) * 0.2
    poses = jnp.asarray(poses, dtype=jnp.float32)
    images = jnp.asarray(rng.normal(size=(3, 7, 9)), dtype=jnp.float32)
    u, v = get_detector_grid_device(detector)
    det_grid = (u + 0.05 * jnp.sin(v), v + 0.03 * jnp.cos(u))
    options = dict(gather_dtype=gather_dtype, det_grid=det_grid, n_steps=13, step_size=0.4)
    reference = jax.jit(
        lambda poses, images: jnp.sum(
            jax.vmap(lambda t, y: backproject_view_T(t, grid, detector, y, **options))(
                poses, images
            ),
            axis=0,
        )
    )
    expected = reference(poses, images)
    actual = jax.jit(lambda t, y: sum_backproject_views_T(t, grid, detector, y, **options))(
        poses, images
    )
    # CUDA half-precision scatter additions are order dependent: even repeated
    # calls to the same single-view kernel can differ by half-precision ulps.
    tolerance = {"fp32": 3e-6, "fp16": 3e-3, "bf16": 3e-2}[gather_dtype]
    assert float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)) < tolerance
    if gather_dtype == "fp32":
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("views", [1, 3, 17])
@pytest.mark.parametrize("layout", ["detector_vu", "detector_uv"])
def test_pallas_shared_adjoint_matches_independent_transpose(views, layout):
    # check-public-imports: allow-private
    from tomojax.core.pallas import api as pallas

    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, pose = _oblique_case((7, 6, 5))
    rng = np.random.default_rng(219)
    poses = jnp.stack([pose.at[0, 3].add(i * 0.17) for i in range(views)])
    images = jnp.asarray(rng.normal(size=(views, detector.nv, detector.nu)), dtype=jnp.float32)
    volume = jnp.asarray(rng.normal(size=(7, 6, 5)), dtype=jnp.float32)

    def project(x):
        return jax.vmap(lambda t: forward_project_view_T(t, grid, detector, x))(poses)

    expected = jax.jit(lambda y: jax.vjp(project, volume)[1](y)[0])(images)
    actual = jax.jit(
        lambda y: pallas.sum_backproject_views_T_pallas(
            poses, grid, detector, y, layout_variant=layout
        )
    )(images)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    assert float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)) < 3e-6
