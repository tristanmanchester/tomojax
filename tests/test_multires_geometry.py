"""Physical invariants of coarsened volumes and sampled detector rays."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.multires import (
    bin_projections,
    bin_volume,
    scale_detector,
    scale_grid,
    upsample_volume,
)
from tomojax.geometry import Detector, Grid, grid_volume_origin


@pytest.mark.parametrize("factor", [1, 2, 3, 4, 16])
@pytest.mark.parametrize(
    "placement",
    [
        {},
        {"vol_center": (4.0, -2.0, 1.0)},
        {"vol_origin": (-7.0, 3.0, 2.0), "vol_center": (99.0, 99.0, 99.0)},
    ],
)
def test_coarsening_preserves_all_volume_faces(factor, placement):
    fine = Grid(9, 6, 1, 0.7, 1.3, 2.1, **placement)
    coarse = scale_grid(fine, factor)
    faces = []
    for grid in [fine, coarse]:
        counts = np.asarray([grid.nx, grid.ny, grid.nz])
        spacing = np.asarray([grid.vx, grid.vy, grid.vz])
        lower = np.asarray(grid_volume_origin(grid)) - spacing / 2
        faces.append(np.stack([lower, lower + counts * spacing]))
    np.testing.assert_allclose(faces[0], faces[1], atol=1e-14)
    assert (coarse.nx, coarse.ny, coarse.nz) == tuple(math.ceil(n / factor) for n in (9, 6, 1))


@pytest.mark.parametrize("factor", [1, 2, 3, 4, 16])
@pytest.mark.parametrize("shape", [(7, 9), (6, 8), (1, 2), (2, 1)])
def test_coarse_detector_matches_every_selected_measured_ray(factor, shape):
    nv, nu = shape
    fine = Detector(nu, nv, 0.75, 1.25, (3.0, -4.0))
    coarse = scale_detector(fine, factor)

    def coordinates(det):
        return np.meshgrid(
            (np.arange(det.nu) - (det.nu - 1) / 2) * det.du + det.center[0],
            (np.arange(det.nv) - (det.nv - 1) / 2) * det.dv + det.center[1],
        )

    # Coordinate-valued images independently identify the original physical ray
    # for every sampled pixel, including both ends of each odd-sized axis.
    original = np.stack(coordinates(fine))
    sampled = np.asarray(bin_projections(jnp.asarray(original), factor))
    np.testing.assert_allclose(sampled, np.stack(coordinates(coarse)), atol=1e-6)
    assert sampled.shape == (2, math.ceil(nv / factor), math.ceil(nu / factor))
    assert len(np.unique(sampled[0])) == coarse.nu
    assert len(np.unique(sampled[1])) == coarse.nv


def test_volume_resize_preserves_constant_density_and_world_linear_interior():
    grid = Grid(17, 15, 13, 0.7, 1.3, 2.1, vol_origin=(-2.0, 3.0, -4.0))
    coarse = scale_grid(grid, 3)

    def affine_volume(g):
        x, y, z = np.meshgrid(
            *[
                o + np.arange(n) * d
                for o, n, d in zip(
                    grid_volume_origin(g), (g.nx, g.ny, g.nz), (g.vx, g.vy, g.vz), strict=True
                )
            ],
            indexing="ij",
        )
        return 0.2 * x - 0.4 * y + 0.6 * z

    upsampled = upsample_volume(jnp.asarray(affine_volume(coarse)), 3, (17, 15, 13))
    # Resize uses half-pixel centres and extends the nearest value outside the
    # coarse centres; only the interior is expected to reproduce affine data.
    np.testing.assert_allclose(
        upsampled[2:-2, 2:-2, 2:-2], affine_volume(grid)[2:-2, 2:-2, 2:-2], rtol=2e-6, atol=2e-6
    )
    downsampled = bin_volume(jnp.full((17, 15, 13), 2.75), 3)
    assert downsampled.shape == (coarse.nx, coarse.ny, coarse.nz)
    np.testing.assert_allclose(downsampled, 2.75, rtol=1e-6)
    np.testing.assert_allclose(upsample_volume(downsampled, 3, (17, 15, 13)), 2.75, rtol=1e-6)


@pytest.mark.parametrize(
    ("shape", "target"), [((3, 4, 2), (7, 9, 5)), ((7, 9, 5), (3, 4, 2)), ((1, 5, 3), (7, 2, 1))]
)
def test_sparse_resize_preserves_linear_values_and_transpose(shape, target):
    x = jnp.asarray(np.random.default_rng(903).uniform(-1, 1, shape), jnp.float32)
    weight = jnp.asarray(np.random.default_rng(904).normal(size=target), jnp.float32)

    def reference(v):
        return jax.image.resize(v, target, method="linear", antialias=False)

    np.testing.assert_allclose(upsample_volume(x, 2, target), reference(x), rtol=1e-5, atol=4e-7)
    actual = jax.grad(lambda v: jnp.sum(upsample_volume(v, 2, target) * weight))(x)
    expected = jax.grad(lambda v: jnp.sum(reference(v) * weight))(x)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.int32])
def test_other_resize_dtypes_retain_the_existing_cast(dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 4, 2)
    expected = jax.image.resize(x, (7, 9, 5), method="linear", antialias=False).astype(dtype)
    np.testing.assert_array_equal(upsample_volume(x, 2, (7, 9, 5)), expected)
