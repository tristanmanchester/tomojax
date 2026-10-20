"""Regional products retain the full-grid operator and its pose derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.joseph import forward_project_planes, plane_coefficients, sum_backproject_planes
from tomojax.core.joseph_regions import PlaneRegion
from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses


@pytest.mark.numerical
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    ("interpolation", "absolute"), [("linear", False), ("cubic", False), ("cubic", True)]
)
@pytest.mark.parametrize(
    ("origin", "shape", "tilt"), [((0, 0, 0), (9, 11, 3), -35), ((2, 1, 3), (5, 8, 3), 75)]
)
def test_region_submatrix_and_columns(backend, interpolation, absolute, origin, shape, tilt):
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(8712)
    grid = Grid(9, 11, 7, 0.7, 0.9, 1.2, vol_origin=(-2.3, -4.6, -3.7))
    detector = Detector(12, 9, 0.6, 0.8, center=(0.13, -0.29))
    poses = stack_view_poses(
        LaminographyGeometry(grid, detector, [0, 45, 98, 183], tilt_deg=tilt), 4
    )
    coefficients = plane_coefficients(poses, grid, detector)
    region = PlaneRegion(
        grid=grid,
        detector=detector,
        backend=backend,
        volume_origin=origin,
        detector_origin=(2, 3),
        detector_shape=(5, 7),
        interpolation=interpolation,
        absolute_weights=absolute,
    )
    tile = jnp.asarray(rng.normal(size=shape), jnp.float32)
    index = tuple(slice(o, o + n) for o, n in zip(origin, shape, strict=True))
    full = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32).at[index].set(tile)
    kwargs = dict(backend=backend, interpolation=interpolation, absolute_weights=absolute)
    expected = forward_project_planes(coefficients, full, grid, detector, **kwargs)[:, 2:7, 3:10]
    np.testing.assert_allclose(region.project(coefficients, tile), expected, rtol=3e-5, atol=2e-5)
    images = jnp.asarray(rng.normal(size=expected.shape), jnp.float32)
    padded = jnp.zeros((4, detector.nv, detector.nu), jnp.float32).at[:, 2:7, 3:10].set(images)
    expected_at = sum_backproject_planes(coefficients, padded, grid, detector, **kwargs)[index]
    initial = jnp.asarray(rng.normal(size=shape), jnp.float32)
    transpose = region.transpose(coefficients, images, accumulate=initial)
    np.testing.assert_allclose(transpose, initial + expected_at, rtol=3e-5, atol=2e-5)
    tangent = jnp.asarray(rng.normal(0, 0.02, (4, 2, 14)), jnp.float32).at[:, :, 0].set(0)

    def reference(c):
        return forward_project_planes(c, full, grid, detector, **{**kwargs, "backend": "jax"})[
            :, 2:7, 3:10
        ]

    derivative = jax.jit(
        jax.vmap(lambda d: jax.jvp(reference, (coefficients,), (d,))[1], in_axes=1, out_axes=1)
    )(tangent)
    columns = region.columns(coefficients, tile, tangent=tangent)
    np.testing.assert_allclose(columns[:, 0], expected, rtol=3e-5, atol=2e-5)
    np.testing.assert_allclose(columns[:, 1:], derivative, rtol=3e-5, atol=2e-5)
    seed = jnp.ones_like(columns) * 0.25
    accumulated = jax.jit(
        lambda value: region.columns(coefficients, tile, tangent=tangent, accumulate=value),
        donate_argnums=(0,),
    )(seed)
    np.testing.assert_allclose(accumulated, columns + 0.25, rtol=2e-6, atol=2e-6)
    lhs, rhs = jnp.vdot(expected, images), jnp.vdot(tile, transpose - initial)
    np.testing.assert_allclose(lhs, rhs, rtol=3e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_partial_blocks_outside_detector_field(interpolation):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector = Grid(81, 77, 65, 1, 1, 1), Detector(13, 11, 1, 1)
    poses = stack_view_poses(
        LaminographyGeometry(grid, detector, np.linspace(0, 360, 33), tilt_deg=-35), 33
    )
    coefficients = plane_coefficients(poses, grid, detector)
    rng = np.random.default_rng(49712)
    images = jnp.asarray(rng.normal(size=(33, 11, 13)), jnp.float32)
    full = sum_backproject_planes(
        coefficients, images, grid, detector, backend="cuda", interpolation=interpolation
    )
    for origin in [(0, 0, 0), (35, 31, 27), (70, 68, 62)]:
        shape = (11, 9, 3)
        region = PlaneRegion(
            grid=grid, detector=detector, backend="cuda",
            volume_origin=origin, interpolation=interpolation,
        )  # fmt: skip
        actual = region.transpose(coefficients, images, accumulate=jnp.zeros(shape, jnp.float32))
        index = tuple(slice(o, o + n) for o, n in zip(origin, shape, strict=True))
        np.testing.assert_allclose(actual, full[index], rtol=3e-6, atol=2e-6)
