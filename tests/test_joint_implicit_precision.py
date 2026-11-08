"""Uniform measurement precision needs no scan-sized resident array."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.alignment._joint._operators import Operators  # check-public-imports: allow-private
from tomojax.alignment._joint._refine import _EdgeObjective  # check-public-imports: allow-private
from tomojax.core.cone import beam_frame
from tomojax.geometry import ConeBeam, Detector, Grid, LaminographyGeometry, stack_view_poses


@pytest.mark.parametrize("cone", [False, True])
def test_implicit_precision_matches_explicit_ones_in_joint_and_edge_objectives(cone):
    grid = Grid(5, 4, 3, 0.7, 0.9, 1.1)
    detector = Detector(7, 6, 0.8, 0.9, center=(0.12, -0.17))
    nominal = np.asarray(
        stack_view_poses(LaminographyGeometry(grid, detector, [17, 91, 231], tilt_deg=-29), 3)
    )
    frames = np.broadcast_to(beam_frame(ConeBeam(31, 47), detector), (3, 4, 3))
    rng = np.random.default_rng(851)
    with jax.enable_x64(True):
        ops = Operators(
            grid,
            detector,
            nominal,
            lever=4,
            native=(0.4, 0.45),
            batch_size=2,
            frames=frames if cone else None,
            interpolation="linear" if cone else "cubic",
            ray_binning=(2, 2),
        )
        assert ops.pad_precision(None) is None
        explicit = ops.pad_precision(np.ones((3, detector.nv, detector.nu), np.float32))
        q = ops.pad_poses(rng.normal(0, 0.02, (3, ops.dofs)))
        x = jnp.asarray(rng.uniform(0.1, 1, ops.shape), jnp.float32)
        y = ops.pad_images(rng.normal(size=(3, detector.nv, detector.nu)))
        for function in (ops.linearize, ops.initial_gradient, ops.objective):
            left = function(q, x, y, None, 0.8)
            right = function(q, x, y, explicit, 0.8)
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
                np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
        edge = _EdgeObjective(ops)
        for function in (edge.stats, edge.value):
            for a, b in zip(
                jax.tree.leaves(function(q, x, y, None, 0.8)),
                jax.tree.leaves(function(q, x, y, explicit, 0.8)),
                strict=True,
            ):
                np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
        for function in (edge.consistency, edge.scale):
            np.testing.assert_allclose(
                function(q, x, y, None), function(q, x, y, explicit), rtol=2e-6, atol=2e-6
            )
