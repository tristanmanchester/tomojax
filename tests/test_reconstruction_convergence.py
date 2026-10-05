"""Solver convergence on anisotropic parallel and tilted scans."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FistaConfig, SPDHGConfig, fista_tv, spdhg_tv

# check-public-imports: allow-private
from tomojax.recon._projection import projection_operators


@pytest.mark.numerical
@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("solver", ["fista", "spdhg"])
@pytest.mark.parametrize("model", ["joseph", "ray"])
def test_tv_solver_reduces_residual_and_recovers_phantom(
    tilted: bool, solver: str, model: str
) -> None:
    grid = Grid(12, 12, 5, 0.8, 1.1, 1.3)
    detector = Detector(16, 7, 0.8, 1.3)
    angles = np.linspace(0, 180, 16, endpoint=False)
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=25)
        if tilted
        else ParallelGeometry(grid, detector, angles)
    )
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(16)], dtype=jnp.float32)
    x, y, z = jnp.meshgrid(
        jnp.arange(12) - 5.5, jnp.arange(12) - 5.5, jnp.arange(5) - 2, indexing="ij"
    )
    truth = jnp.exp(-((x / 2.3) ** 2 + (y / 1.7) ** 2 + (z / 1.5) ** 2) / 2)
    project = jax.jit(projection_operators(poses, grid, detector, None, "jax", 16, model)[0])
    # These data isolate optimizer convergence. Independent analytic projection
    # and reconstruction checks live in the benchmark and FBP accuracy tests.
    data = project(truth)
    if solver == "fista":
        volume, _ = fista_tv(
            geometry,
            grid,
            detector,
            data,
            config=FistaConfig(
                iters=30,
                lambda_tv=0.001,
                positivity=True,
                power_iters=5,
                views_per_batch=4,
                projector_model=model,
                projector_backend="jax",
            ),
        )
        residual_limit, volume_limit = 0.005, 0.04
    else:
        volume, _ = spdhg_tv(
            geometry,
            grid,
            detector,
            data,
            config=SPDHGConfig(
                iters=100,
                lambda_tv=0.001,
                positivity=True,
                views_per_batch=4,
                seed=18,
                projector_model=model,
                projector_backend="jax",
            ),
        )
        residual_limit, volume_limit = 0.03, 0.08
    assert float(jnp.min(volume)) >= 0.0
    assert float(jnp.linalg.norm(project(volume) - data) / jnp.linalg.norm(data)) < residual_limit
    assert float(jnp.linalg.norm(volume - truth) / jnp.linalg.norm(truth)) < volume_limit
