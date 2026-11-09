"""Freezing ray geometry must preserve every image/pose operator and mask."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._joint._operators import Operators
from tomojax.core.cone import beam_frame
from tomojax.geometry import ConeBeam, Detector, Grid, LaminographyGeometry, stack_view_poses


@pytest.mark.parametrize("cone", [False, True])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_frozen_geometry_preserves_batched_normal_and_cross_actions(cone, batch_size):
    grid = Grid(5, 4, 3, 0.7, 0.9, 1.1, vol_center=(0.2, -0.1, 0.3))
    detector = Detector(6, 5, 0.8, 0.9, center=(0.12, -0.17))
    geometry = LaminographyGeometry(grid, detector, [13, 79, 211], tilt_deg=-31)
    nominal = np.asarray(stack_view_poses(geometry, 3))
    frames = np.broadcast_to(beam_frame(ConeBeam(31, 47), detector), (3, 4, 3))
    rng = np.random.default_rng(850)
    with jax.enable_x64(True):
        ops = Operators(
            grid,
            detector,
            nominal,
            lever=4,
            native=(0.4, 0.45),
            batch_size=batch_size,
            frames=frames if cone else None,
            interpolation="linear" if cone else "cubic",
        )
        q = ops.pad_poses(rng.normal(0, 0.05, (3, ops.dofs)))
        frozen = ops.prepare(q)
        direct = ops.coefficients(ops.nominal, q, ops.valid, ops.scales, ops.frames)
        for left, right in zip(jax.tree.leaves(frozen), jax.tree.leaves(direct), strict=True):
            # Eager and fused coefficient arithmetic can differ by a few FP32
            # ulps; the image actions below compare the two compiled paths.
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=3e-7)
        x = jnp.asarray(rng.normal(size=ops.shape), jnp.float32)
        sw = ops.pad_images(rng.uniform(0.1, 1, (3, detector.nv, detector.nu)))
        columns = jnp.asarray(
            rng.normal(size=(ops.padded_n, ops.dofs, detector.nv, detector.nu)), jnp.float32
        )
        p = jnp.asarray(rng.normal(size=3 * ops.dofs), jnp.float64)
        np.testing.assert_array_equal(
            ops.initial_gradient(q, x, sw, None, 1.0),
            ops.initial_gradient(frozen, x, sw, None, 1.0),
        )
        fresh = ops.weighted_forward(q, x, sw, columns)
        cached = ops.weighted_forward(frozen, x, sw, columns)
        for left, right in zip(fresh, cached, strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(
            ops.weighted_back(q, fresh[0], sw, columns, p),
            ops.weighted_back(frozen, fresh[0], sw, columns, p),
        )
        np.testing.assert_array_equal(
            ops.pose_back(q, sw, columns, p), ops.pose_back(frozen, sw, columns, p)
        )
        left = jnp.vdot(p, cached[1])
        right = jnp.vdot(
            x.astype(jnp.float64), ops.pose_back(frozen, sw, columns, p).astype(jnp.float64)
        )
        np.testing.assert_allclose(left, right, rtol=5e-6, atol=2e-5)
