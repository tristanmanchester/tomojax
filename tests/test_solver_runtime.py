"""Cached solvers must consume fresh geometry, measurements, and support."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.operator_norm import estimate_normal_norm

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FistaConfig, SPDHGConfig, fista_tv, spdhg_tv

# check-public-imports: allow-private
from tomojax.recon._projection import projection_operators


@pytest.mark.numerical
@pytest.mark.parametrize("solver", ["fista", "spdhg"])
@pytest.mark.parametrize("model", ["ray", "joseph"])
def test_repeated_solver_calls_use_new_arrays(solver, model):
    grid = Grid(2, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.9, 1.2, (0.17, 0.23))
    rng = np.random.default_rng(92)
    previous = None
    for run in range(2):
        geometry = ParallelGeometry(grid, detector, np.asarray([17.0, 69.0, 137.0]) + run * 8)
        poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)])
        data = jnp.asarray(rng.normal(size=(3, 3, 4)), dtype=jnp.float32)
        support = jnp.asarray(rng.uniform(0.3, 1.0, size=(2, 3, 2)), dtype=jnp.float32)

        forward, _ = projection_operators(poses, grid, detector, None, "jax", 3, model)

        def project(flat, forward=forward):
            return forward(flat.reshape((2, 3, 2))).ravel()

        matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(12)))
        if solver == "fista":
            cfg = FistaConfig(
                iters=1,
                L=20.0,
                lambda_tv=0.0,
                views_per_batch=2,
                positivity=False,
                support=support,
                projector_model=model,
                projector_backend="jax",
            )
            actual, _ = fista_tv(geometry, grid, detector, data, config=cfg)
            expected = support.ravel() * (matrix.T @ data.ravel()) / 20.0
        else:
            weights = jnp.asarray(rng.uniform(0.5, 1.5, size=data.shape), dtype=jnp.float32)
            cfg = SPDHGConfig(
                iters=1,
                tau=0.1,
                sigma_data=0.2,
                sigma_tv=0.2,
                lambda_tv=0.0,
                views_per_batch=3,
                positivity=False,
                support=support,
                projector_model=model,
                projector_backend="jax",
            )
            actual, _ = spdhg_tv(geometry, grid, detector, data, weights=weights, config=cfg)
            dual = 0.2 * data * weights / (0.2 + weights)
            expected = 0.1 * support.ravel() * (matrix.T @ dual.ravel())
        np.testing.assert_allclose(actual.ravel(), expected, rtol=2e-5, atol=2e-6)
        if previous is not None:
            assert not np.allclose(previous, actual)
        previous = np.asarray(actual)


@pytest.mark.numerical
def test_shared_power_method_matches_dense_operator_with_tail_and_support():
    grid = Grid(2, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.9, 1.2, (0.17, 0.23))
    for offset in [0.0, 11.0]:
        geometry = ParallelGeometry(grid, detector, np.asarray([17.0, 69.0, 137.0]) + offset)
        poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)])
        support = jnp.linspace(0.2, 1.0, 12).reshape((2, 3, 2))

        def project(flat, poses=poses, support=support):
            return jax.vmap(
                lambda t: forward_project_view_T(
                    t, grid, detector, flat.reshape((2, 3, 2)) * support
                )
            )(poses).ravel()

        matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(12)))
        expected = np.linalg.svd(matrix, compute_uv=False)[0] ** 2
        actual = estimate_normal_norm(
            poses,
            jnp.ones((2, 3, 2)),
            None,
            support,
            grid=grid,
            detector=detector,
            batch_size=2,
            iters=50,
            unroll=1,
            checkpoint=True,
            gather_dtype="fp32",
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-5)


@pytest.mark.numerical
@pytest.mark.parametrize("tilt", [False, True])
def test_positive_normal_bound_dominates_dense_spectrum_for_irregular_geometry(tilt):
    grid = Grid(2, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(5, 3, 0.6, 1.2, (0.17, 0.23))
    angles = np.array([13.0, 68.0, 142.0])
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if tilt
        else ParallelGeometry(grid, detector, angles)
    )
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)])
    support = jnp.linspace(-1.0, 0.7, 12).reshape((2, 3, 2))

    def project(flat):
        return jax.vmap(
            lambda t: forward_project_view_T(t, grid, detector, flat.reshape((2, 3, 2)) * support)
        )(poses).ravel()

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(12))).astype(np.float64)
    expected = np.max(np.abs(matrix).T @ np.abs(matrix) @ np.ones(12))
    actual = float(
        estimate_normal_norm(
            poses,
            jnp.zeros((2, 3, 2)),
            None,
            support,
            grid=grid,
            detector=detector,
            batch_size=2,
            iters=1,
            unroll=1,
            checkpoint=True,
            gather_dtype="fp32",
            upper_bound=True,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-5)
    assert actual >= np.linalg.svd(matrix, compute_uv=False)[0] ** 2 * (1 - 3e-5)


@pytest.mark.numerical
@pytest.mark.parametrize("batch", [2, 5])
def test_fista_streams_host_projections_like_device_projections(tmp_path, batch):
    grid = Grid(5, 4, 3, 0.8, 1.1, 1.3)
    detector = Detector(6, 4, 0.9, 1.2, (0.17, -0.2))
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 360, 7, endpoint=False), 30)
    data = np.random.default_rng(5).random((7, 4, 6), dtype=np.float32)
    stored = np.memmap(tmp_path / "views.f32", mode="w+", dtype=np.float32, shape=data.shape)
    stored[:] = data
    results = {}
    for stream, projections in [(False, jnp.asarray(data)), (True, stored)]:
        config = FistaConfig(
            iters=4,
            lambda_tv=0.01,
            positivity=True,
            projector_model="joseph",
            projector_backend="jax",
            views_per_batch=batch,
            stream_projections=stream,
        )
        results[stream] = fista_tv(geometry, grid, detector, projections, config=config)
    np.testing.assert_allclose(results[True][0], results[False][0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(results[True][1]["loss"], results[False][1]["loss"], rtol=1e-5)


@pytest.mark.numerical
@pytest.mark.parametrize("weighted", [False, True])
def test_spdhg_streams_data_weights_and_duals_from_host(tmp_path, weighted):
    grid = Grid(5, 4, 3, 0.8, 1.1, 1.3)
    detector = Detector(6, 4, 0.9, 1.2, (0.17, -0.2))
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 360, 9, endpoint=False), 30)
    rng = np.random.default_rng(6)
    data = rng.random((9, 4, 6), dtype=np.float32)
    weights = rng.uniform(0.5, 1.5, data.shape).astype(np.float32) if weighted else None
    stored = np.memmap(tmp_path / "views.f32", mode="w+", dtype=np.float32, shape=data.shape)
    stored[:] = data
    results = {}
    for stream, projections in [(False, jnp.asarray(data)), (True, stored)]:
        config = SPDHGConfig(
            iters=20,
            lambda_tv=0.01,
            views_per_batch=4,
            projector_model="joseph",
            projector_backend="jax",
            stream_projections=stream,
        )
        w = None if weights is None else (weights if stream else jnp.asarray(weights))
        results[stream] = spdhg_tv(geometry, grid, detector, projections, weights=w, config=config)
    np.testing.assert_allclose(results[True][0], results[False][0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(results[True][1]["loss"], results[False][1]["loss"], rtol=1e-6)
