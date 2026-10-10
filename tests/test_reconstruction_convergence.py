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
                iterations=30,
                tv_weight=0.001,
                nonnegative=True,
                power_iterations=5,
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
                iterations=100,
                tv_weight=0.001,
                nonnegative=True,
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


@pytest.mark.numerical
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("regulariser", ["tv", "huber_tv"])
@pytest.mark.parametrize("solver", ["fista", "spdhg"])
def test_tv_reconstruction_small_signal_scales_match_dense_solution(stream, regulariser, solver):
    shape = (2, 1, 1)
    grid = Grid(*shape, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.7, 0.9, (0.17, -0.23))
    geometry = ParallelGeometry(grid, detector, [13.0, 71.0, 127.0])
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)], dtype=jnp.float32)
    forward, _ = projection_operators(poses, grid, detector, None, "jax", 3, "joseph")
    matrix = np.asarray(jax.jacfwd(lambda x: forward(x.reshape(shape)).ravel())(jnp.zeros(2)))
    matrix = matrix.astype(np.float64)
    data = matrix @ np.asarray([0.2, 1.0])
    lam, delta = 0.05, 0.2
    # The minimizer remains on the increasing, linear branch of TV/Huber-TV.
    # This dense normal-equation solution is independent of proximal updates.
    expected = np.linalg.solve(matrix.T @ matrix, matrix.T @ data - lam * np.asarray([-1.0, 1.0]))
    assert expected[1] - expected[0] > delta
    if solver == "fista":
        config = FistaConfig(
            iterations=150,
            tv_prox_iterations=30,
            lipschitz=np.linalg.norm(matrix, 2) ** 2,
            projector_model="joseph",
            projector_backend="jax",
            stream_projections=stream,
            regulariser=regulariser,
        )
    else:
        config = SPDHGConfig(
            iterations=1000,
            views_per_batch=3,
            log_every=0,
            projector_model="joseph",
            projector_backend="jax",
            stream_projections=stream,
            regulariser=regulariser,
        )
    for scale in [1.0, 1e-8, 1e-14]:
        config.tv_weight, config.huber_delta = lam * scale, delta * scale
        actual, _ = (fista_tv if solver == "fista" else spdhg_tv)(
            geometry,
            grid,
            detector,
            (data * scale).reshape((3, 3, 4)).astype(np.float32),
            config=config,
        )
        # Check the reconstruction, not just a smaller objective or residual.
        np.testing.assert_allclose(
            np.asarray(actual).ravel() / scale, expected, rtol=3e-5, atol=3e-6
        )
