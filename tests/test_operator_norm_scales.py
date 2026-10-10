"""Norm estimates must retain physical units without squaring away the signal."""

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.operator_norm import estimate_normal_norm
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FistaConfig, fista_tv


@pytest.mark.numerical
@pytest.mark.parametrize("support_scale", [0.0, 1e-12, 1e-4, 1.0, 1e12])
@pytest.mark.parametrize("upper_bound", [False, True])
def test_ray_normal_norm_preserves_support_scale_without_absolute_floor(support_scale, upper_bound):
    # A single axis-aligned voxel gives A=[1] and M=[support_scale].
    # Both the spectral norm and the positive row-sum bound are M**2 exactly.
    grid, detector = Grid(1, 1, 1, 1.0, 1.0, 1.0), Detector(1, 1, 1.0, 1.0)
    actual = estimate_normal_norm(
        jnp.eye(4, dtype=jnp.float32)[None],
        jnp.ones((1, 1, 1)),
        None,
        jnp.full((1, 1, 1), support_scale),
        grid=grid,
        detector=detector,
        batch_size=1,
        iterations=20,
        unroll=1,
        checkpoint=False,
        gather_dtype="fp32",
        ray_integrator="joseph",
        upper_bound=upper_bound,
    )
    np.testing.assert_allclose(actual, support_scale**2, rtol=3e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
@pytest.mark.parametrize("model", ["ray", "joseph"])
def test_auto_step_fista_recovers_same_one_voxel_image_in_any_length_units(scale, model):
    grid, detector = Grid(1, 1, 1, scale, scale, scale), Detector(1, 1, scale, scale)
    geometry = ParallelGeometry(grid, detector, [0.0])
    # For this axis-aligned one-voxel case both projector models integrate [scale].
    actual, info = fista_tv(
        geometry,
        grid,
        detector,
        jnp.full((1, 1, 1), 3 * scale),
        config=FistaConfig(
            iterations=1,
            tv_weight=0,
            projector_model=model,
            projector_backend="jax",
            power_iterations=3,
        ),
    )
    np.testing.assert_allclose(actual, 3.0, rtol=3e-6)
    np.testing.assert_allclose(info["lipschitz"], scale**2, rtol=3e-6)
