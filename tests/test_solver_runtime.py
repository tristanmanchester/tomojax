"""Cached solvers must consume fresh geometry, measurements, and support."""

from __future__ import annotations

from dataclasses import replace
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.core.operator_norm import estimate_normal_norm

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    ConeSegments,
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
)
from tomojax.recon import FistaConfig, SPDHGConfig, fista_tv, spdhg_tv

# check-public-imports: allow-private
from tomojax.recon._projection import normal_operator_norm, projection_operators


@pytest.mark.numerical
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("weight_scale", [1, 1e-13, 1e-30])
def test_spdhg_small_weights_preserve_the_dense_weighted_minimizer(stream, weight_scale):
    shape = (1, 1, 2)
    grid = Grid(*shape[::-1], 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.7, 0.9, (0.17, -0.23))
    geometry = ParallelGeometry(grid, detector, [13.0, 71.0, 127.0])
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)], dtype=jnp.float32)
    forward, _ = projection_operators(poses, grid, detector, None, "jax", 3, "joseph")
    matrix = np.asarray(jax.jacfwd(lambda x: forward(x.reshape(shape)).ravel())(jnp.zeros(2)))
    matrix = matrix.astype(np.float64)
    truth = np.asarray([0.2, 1.0])
    data = (matrix @ truth + np.random.default_rng(839).normal(0, 0.05, 36)).astype(np.float32)
    weights = (np.linspace(0.3, 1.7, 36) * weight_scale).astype(np.float32)
    # FP64 reference uses the actual rounded inputs, independent of the prox.
    reference_weights = weights.astype(np.float64) / weight_scale
    normal = matrix.T @ (reference_weights[:, None] * matrix)
    expected = np.linalg.solve(normal, matrix.T @ (reference_weights * data))
    assert np.linalg.cond(normal) < 100
    volume, info = spdhg_tv(
        geometry,
        grid,
        detector,
        data.reshape((3, 3, 4)),
        weights=weights.reshape((3, 3, 4)),
        config=SPDHGConfig(
            iterations=2000,
            views_per_batch=3,
            tv_weight=0,
            tau=0.2 / (weight_scale * np.linalg.norm(matrix, 2) ** 2),
            sigma_data=0.2 * weight_scale,
            sigma_tv=0.1,
            log_every=2000,
            projector_model="joseph",
            backend="jax",
            stream_projections=stream,
        ),
    )
    assert np.isfinite(info["loss"]).all()
    actual = np.asarray(volume).ravel()
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-5
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.07


@pytest.mark.numerical
@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("device_weights", [False, True])
def test_spdhg_zero_weight_mask_recovers_dense_weighted_solution(tilted, stream, device_weights):
    shape = (1, 1, 2)
    grid = Grid(*shape[::-1], 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.7, 0.9, (0.17, -0.23))
    angles = [13.0, 71.0, 127.0]
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=25)
        if tilted
        else ParallelGeometry(grid, detector, angles)
    )
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)], dtype=jnp.float32)
    forward, _ = projection_operators(poses, grid, detector, None, "jax", 3, "joseph")
    matrix = np.asarray(jax.jacfwd(lambda x: forward(x.reshape(shape)).ravel())(jnp.zeros(2)))
    matrix = matrix.astype(np.float64)
    truth = np.asarray([0.2, 1.0])
    data = matrix @ truth + np.random.default_rng(839).normal(0, 1e-3, matrix.shape[0])
    weights = np.linspace(0.3, 1.7, data.size).astype(np.float32)
    weights[::3] = 0
    data[weights == 0] = 1e4  # Masked observations must not affect the minimizer.
    normal = matrix.T @ (weights[:, None] * matrix)
    expected = np.linalg.solve(normal, matrix.T @ (weights * data))
    assert np.linalg.cond(normal) < 100
    assert np.linalg.norm(expected - truth) / np.linalg.norm(truth) < 0.01
    weights = weights.reshape((3, 3, 4))
    volume, info = spdhg_tv(
        geometry,
        grid,
        detector,
        data.reshape((3, 3, 4)).astype(np.float32),
        weights=jnp.asarray(weights) if device_weights else weights,
        config=SPDHGConfig(
            iterations=2000,
            views_per_batch=2,
            tv_weight=0,
            log_every=2000,
            projector_model="joseph",
            backend="jax",
            stream_projections=stream,
        ),
    )
    assert np.isfinite(info["loss"]).all()
    assert np.linalg.norm(np.asarray(volume).ravel() - expected) / np.linalg.norm(expected) < 5e-5
    assert np.linalg.norm(np.asarray(volume).ravel() - truth) / np.linalg.norm(truth) < 0.01


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
                iterations=1,
                lipschitz=20.0,
                tv_weight=0.0,
                views_per_batch=2,
                nonnegative=False,
                support=support,
                projector_model=model,
                backend="jax",
            )
            actual, _ = fista_tv(geometry, grid, detector, data, config=cfg)
            expected = support.ravel() * (matrix.T @ data.ravel()) / 20.0
        else:
            weights = jnp.asarray(rng.uniform(0.5, 1.5, size=data.shape), dtype=jnp.float32)
            cfg = SPDHGConfig(
                iterations=2,
                tau=0.1,
                sigma_data=0.2,
                sigma_tv=0.2,
                tv_weight=0.0,
                views_per_batch=3,
                nonnegative=False,
                support=support,
                projector_model=model,
                backend="jax",
            )
            actual, _ = spdhg_tv(geometry, grid, detector, data, weights=weights, config=cfg)
            dual = 0.2 * data * weights / (0.2 + weights)
            expected = 2 * 0.1 * support.ravel() * (matrix.T @ dual.ravel())
        np.testing.assert_allclose(actual.ravel(), expected, rtol=2e-5, atol=2e-6)
        if previous is not None:
            assert not np.allclose(previous, actual)
        previous = np.asarray(actual)


@pytest.mark.numerical
@pytest.mark.parametrize("ray_integrator", ["sampled", "joseph", "joseph_cubic"])
def test_shared_power_method_matches_dense_operator_with_tail_and_support(ray_integrator):
    grid = Grid(2, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.9, 1.2, (0.17, 0.23))
    for offset in [0.0, 11.0]:
        geometry = ParallelGeometry(grid, detector, np.asarray([17.0, 69.0, 137.0]) + offset)
        poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)])
        support = jnp.linspace(0.2, 1.0, 12).reshape((2, 3, 2))

        def project(flat, poses=poses, support=support):
            return jax.vmap(
                lambda t: forward_project_view_T(
                    t,
                    grid,
                    detector,
                    flat.reshape((2, 3, 2)) * support,
                    ray_integrator=ray_integrator,
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
            iterations=50,
            unroll=1,
            checkpoint=True,
            gather_dtype="fp32",
            ray_integrator=ray_integrator,
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-5)


@pytest.mark.numerical
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tilt", [False, True])
@pytest.mark.parametrize("ray_integrator", ["sampled", "joseph", "joseph_cubic"])
def test_positive_normal_bound_dominates_dense_spectrum_for_irregular_geometry(
    tilt, ray_integrator, monkeypatch, batch_size, masked
):
    grid = Grid(2, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(5, 3, 0.6, 1.2, (0.17, 0.23))
    angles = np.array([13.0, 68.0, 142.0])
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if tilt
        else ParallelGeometry(grid, detector, angles)
    )
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)])
    support = jnp.linspace(-1.0, 0.7, 12).reshape((2, 3, 2)) if masked else None

    def project(flat):
        return jax.vmap(
            lambda t: forward_project_view_T(
                t,
                grid,
                detector,
                flat.reshape((2, 3, 2)) * (1 if support is None else support),
                ray_integrator=ray_integrator,
            )
        )(poses).ravel()

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(12))).astype(np.float64)
    expected = np.max(np.abs(matrix).T @ np.abs(matrix) @ np.ones(12))
    if jax.default_backend() == "gpu":
        monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
        for operation in ("forward_jax", "adjoint_jax"):
            monkeypatch.setattr(
                f"tomojax.core.joseph.{operation}",
                lambda *a, **kw: pytest.fail("CUDA norm must use the bounded-workspace pair"),
            )
    actual = float(
        estimate_normal_norm(
            poses,
            jnp.zeros((2, 3, 2)),
            None,
            support,
            grid=grid,
            detector=detector,
            batch_size=batch_size,
            iterations=1,
            unroll=1,
            checkpoint=True,
            gather_dtype="fp32",
            upper_bound=True,
            ray_integrator=ray_integrator,
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
            iterations=4,
            tv_weight=0.01,
            nonnegative=True,
            projector_model="joseph",
            backend="jax",
            views_per_batch=batch,
            stream_projections=stream,
        )
        results[stream] = fista_tv(geometry, grid, detector, projections, config=config)
    # Within rounding: GPU atomics sum in no fixed order.
    np.testing.assert_allclose(results[True][0], results[False][0], rtol=1e-5, atol=1e-6)
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
            iterations=20,
            tv_weight=0.01,
            views_per_batch=4,
            projector_model="joseph",
            backend="jax",
            stream_projections=stream,
        )
        w = None if weights is None else (weights if stream else jnp.asarray(weights))
        results[stream] = spdhg_tv(geometry, grid, detector, projections, weights=w, config=config)
    # The two compiled programs round differently in float32 (by ~2e-7 on a GPU).
    np.testing.assert_allclose(results[True][0], results[False][0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(results[True][1]["loss"], results[False][1]["loss"], rtol=1e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("model", ["ray", "joseph"])
@pytest.mark.parametrize("tilted", [False, True])
def test_spdhg_partial_steps_match_dense_first_update_and_preserve_each_override(model, tilted):
    grid = Grid(2, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.9, 1.2, (0.17, -0.23))
    angles = np.array([13.0, 68.0, 142.0])
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if tilted
        else ParallelGeometry(grid, detector, angles)
    )
    data = np.random.default_rng(78).normal(size=(3, 3, 4)).astype(np.float32)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)])
    forward, _ = projection_operators(poses, grid, detector, None, "jax", 3, model)
    matrix = np.asarray(jax.jacfwd(lambda x: forward(x.reshape((2, 3, 2))).ravel())(jnp.zeros(12)))
    config = SPDHGConfig(
        iterations=2,
        tv_weight=0,
        views_per_batch=3,
        log_every=1,
        projector_model=model,
        backend="jax",
    )
    weights = np.random.default_rng(91).uniform(0.5, 1.5, data.shape).astype(np.float32)
    _, automatic = spdhg_tv(geometry, grid, detector, data, weights=weights, config=config)
    overrides = {"tau": 0.07, "sigma_data": 0.13, "sigma_tv": 0.19}
    for selected in itertools.product([False, True], repeat=3):
        given = {
            name: value
            for (name, value), use in zip(overrides.items(), selected, strict=True)
            if use
        }
        actual, info = spdhg_tv(
            geometry, grid, detector, data, weights=weights, config=replace(config, **given)
        )
        for name, value in overrides.items():
            assert info[name] == pytest.approx(value if name in given else automatic[name])
        assert info["sigma_data_base"] == info["sigma_data"]
        tau, sigma = info["tau"], info["sigma_data"]
        dual = sigma * data * weights / (sigma + weights)
        expected = 2 * tau * (matrix.T @ dual.ravel())
        np.testing.assert_allclose(actual.ravel(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("model", ["ray", "joseph"])
@pytest.mark.parametrize("regulariser", ["tv", "huber_tv"])
def test_spdhg_partial_tv_step_controls_dense_primal_update(model, regulariser):
    shape = (2, 3, 2)
    grid = Grid(*shape[::-1], 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.9, 1.2, (0.17, -0.23))
    geometry = LaminographyGeometry(grid, detector, [13.0, 68.0, 142.0], tilt_deg=30)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)])
    forward, _ = projection_operators(poses, grid, detector, None, "jax", 3, model)
    matrix = np.asarray(jax.jacfwd(lambda x: forward(x.reshape(shape)).ravel())(jnp.zeros(12)))
    # Independent dense forward differences: each row's dual has three components.
    edges = np.zeros((3, 12, 12))
    for index in np.ndindex(shape):
        row = np.ravel_multi_index(index, shape)
        for axis in range(3):
            neighbor = list(index)
            neighbor[axis] += 1
            if neighbor[axis] < shape[axis]:
                edges[axis, row, row] = -1
                edges[axis, row, np.ravel_multi_index(tuple(neighbor), shape)] = 1
    rng = np.random.default_rng(734)
    initial = rng.normal(size=shape).astype(np.float32)
    data = rng.normal(size=(3, 3, 4)).astype(np.float32)
    config = SPDHGConfig(
        iterations=2,
        tv_weight=0.04,
        regulariser=regulariser,
        huber_delta=0.08,
        views_per_batch=3,
        log_every=1,
        projector_model=model,
        backend="jax",
    )
    for given in [{"sigma_tv": 0.19}, {"tau": 0.07, "sigma_data": 0.13, "sigma_tv": 0.19}]:
        volume, info = spdhg_tv(
            geometry, grid, detector, data, init_x=initial, config=replace(config, **given)
        )
        assert info["sigma_tv"] == pytest.approx(given["sigma_tv"])
        tv_dual = info["sigma_tv"] * (edges @ initial.ravel())
        if regulariser == "huber_tv":
            tv_dual *= config.tv_weight / (config.tv_weight + info["sigma_tv"] * config.huber_delta)
        tv_dual /= np.maximum(1.0, np.linalg.norm(tv_dual, axis=0) / config.tv_weight)
        sigma = info["sigma_data"]
        data_dual = sigma * (matrix @ initial.ravel() - data.ravel()) / (sigma + 1)
        tv_gradient = sum(edge.T @ dual for edge, dual in zip(edges, tv_dual, strict=True))
        expected = initial.ravel() - 2 * info["tau"] * (matrix.T @ data_dual + tv_gradient)
        np.testing.assert_allclose(volume.ravel(), expected, rtol=2e-5, atol=3e-6)


@pytest.mark.numerical
def test_batched_normal_norm_matches_dense_scaled_spectrum():
    # Physical length units and an unnormalised start must not change the estimate.
    scale, start_scale = 1e-4, 1e3
    matrix = jnp.asarray([[1.0, 0.3], [0.3, 2.0]], dtype=jnp.float32) * scale
    support = jnp.asarray([0.4, 1.0]).reshape(1, 1, 2)

    def forward(x):
        return matrix @ x.ravel()

    def adjoint(y):
        return (matrix.T @ y).reshape(1, 1, 2)

    actual = jax.jit(
        lambda start: normal_operator_norm(
            forward, adjoint, (1, 1, 2), iterations=30, mask=support, start=start
        )
    )(jnp.full((1, 1, 2), start_scale))
    expected = np.linalg.norm(np.asarray(matrix, np.float64) * np.asarray(support).ravel(), 2) ** 2
    np.testing.assert_allclose(actual, expected, rtol=3e-6)


@pytest.mark.numerical
@pytest.mark.parametrize(("regulariser", "stream"), [("tv", False), ("huber_tv", True)])
def test_spdhg_disabled_tv_cannot_corrupt_finite_data_diagnostics(regulariser, stream):
    scale = 1e-4
    grid, detector = Grid(2, 1, 1, scale, scale, scale), Detector(2, 1, scale, scale)
    geometry = ParallelGeometry(grid, detector, [0.0])
    initial = np.array([0.0, 1e20], dtype=np.float32).reshape(1, 1, 2)
    data = (initial.ravel() * np.float32(scale)).reshape(1, 1, 2)
    actual, info = spdhg_tv(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=SPDHGConfig(
            iterations=1,
            tv_weight=0,
            regulariser=regulariser,
            tau=0.1,
            sigma_data=0.1,
            sigma_tv=0.1,
            log_every=1,
            stream_projections=stream,
            backend="jax",
        ),
    )
    # Each ray sees one plane of length scale: this warm start is already an
    # exact fit. Its absent regularizer must never turn data_loss=0 into NaN.
    np.testing.assert_array_equal(actual, initial)
    np.testing.assert_array_equal(info["loss"], [0.0])


def _three_view_problem():
    grid, detector = Grid(2, 1, 1, 0.8, 1.1, 1.3), Detector(4, 3, 0.7, 0.9, (0.17, -0.23))
    return grid, detector, ParallelGeometry(grid, detector, [13.0, 71.0, 127.0])


@pytest.mark.numerical
@pytest.mark.parametrize(("stream", "missing"), [(False, np.nan), (True, -np.inf)])
def test_spdhg_masked_invalid_data_preserve_solution_and_diagnostics(stream, missing):
    grid, detector, geometry = _three_view_problem()
    truth = np.asarray([0.2, 1.0], np.float32).reshape(1, 1, 2)
    data = np.array(tj.project(geometry, truth))
    weights = np.linspace(0.3, 1.7, data.size).astype(np.float32).reshape(data.shape)
    weights.reshape(-1)[::4] = 0.0
    cfg = SPDHGConfig(
        iterations=100,
        views_per_batch=2,
        tv_weight=0,
        log_every=10,
        projector_model="joseph",
        backend="jax",
        stream_projections=stream,
    )
    reference, ref_info = spdhg_tv(geometry, grid, detector, data, weights=weights, config=cfg)
    data[weights == 0] = missing
    actual, info = spdhg_tv(geometry, grid, detector, data, weights=weights, config=cfg)
    # Independent CUDA norm estimates can differ by an ulp between solves.
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(info["loss"], ref_info["loss"], rtol=1e-5, atol=1e-10)
    assert np.isfinite(info["loss"]).all()


@pytest.mark.numerical
@pytest.mark.parametrize("stream", [False, True])
def test_spdhg_all_missing_data_preserve_finite_warm_start(stream):
    grid, detector, geometry = _three_view_problem()
    initial = np.asarray([0.2, 1.0], dtype=np.float32).reshape(1, 1, 2)
    actual, info = spdhg_tv(
        geometry,
        grid,
        detector,
        np.full((3, 3, 4), np.nan, dtype=np.float32),
        weights=np.zeros((3, 3, 4), dtype=np.float32),
        init_x=initial,
        config=SPDHGConfig(
            iterations=2,
            views_per_batch=2,
            tv_weight=0,
            log_every=1,
            stream_projections=stream,
            backend="jax",
        ),
    )
    np.testing.assert_array_equal(actual, initial)
    np.testing.assert_array_equal(info["loss"], [0.0, 0.0])


@pytest.mark.numerical
@pytest.mark.parametrize(("weighted", "invalid"), [(False, np.nan), (True, 1e40)])
def test_spdhg_streaming_still_rejects_unmasked_nonfinite_data(weighted, invalid):
    grid, detector, geometry = _three_view_problem()
    data = np.zeros((3, 3, 4), dtype=np.float64)
    data[0, 0, 0] = invalid
    weights = np.ones_like(data) if weighted else None
    cfg = SPDHGConfig(
        iterations=1,
        views_per_batch=3,
        tv_weight=0,
        stream_projections=True,
        backend="jax",
    )
    with pytest.raises(ValueError, match="finite"):
        spdhg_tv(geometry, grid, detector, data, weights=weights, config=cfg)
    # Reject before ordered dual callbacks: their failed token would otherwise
    # make this valid solve (or a later caller's solve) fail with the old error.
    data[0, 0, 0] = 0.0
    actual, info = spdhg_tv(geometry, grid, detector, data, weights=weights, config=cfg)
    np.testing.assert_array_equal(actual, 0.0)
    np.testing.assert_array_equal(info["loss"], [0.0])


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("regulariser", "model", "stream"), [("tv", "ray", False), ("huber_tv", "joseph", True)]
)
def test_fista_disabled_tv_does_not_evaluate_regulariser(monkeypatch, regulariser, stream, model):
    import importlib

    module = importlib.import_module("tomojax.recon.fista_tv")
    grid, detector = Grid(2, 1, 1, 1.0, 1.0, 1.0), Detector(2, 1, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, [0.0])
    initial = np.asarray([0.25, 0.75], dtype=np.float32).reshape(1, 1, 2)
    data = initial.reshape(1, 1, 2)

    def disabled(*_args):
        raise AssertionError("the disabled regulariser must not be evaluated")

    monkeypatch.setattr(module, "isotropic_tv_value", disabled)
    monkeypatch.setattr(module, "huber_tv_value", disabled)
    module._run_fista_scan.clear_cache()
    try:
        actual, info = fista_tv(
            geometry,
            grid,
            detector,
            data,
            init_x=initial,
            config=FistaConfig(
                iterations=2,
                lipschitz=1.0,
                tv_weight=0.0,
                regulariser=regulariser,
                projector_model=model,
                backend="jax",
                stream_projections=stream,
            ),
        )
        # Each ray sees one unit-length voxel; this ordinary-range warm start
        # is exactly the independent least-squares solution on both paths.
        np.testing.assert_array_equal(actual, initial)
        np.testing.assert_array_equal(info["loss"], [0.0, 0.0])
    finally:
        module._run_fista_scan.clear_cache()


@pytest.mark.numerical
@pytest.mark.parametrize(("batch", "stream"), [(2, False), (4, True)])
def test_spdhg_segmented_cone_batches_follow_dense_individual_arrangements(batch, stream):
    grid = Grid(3, 2, 2, 0.8, 1.1, 1.3)
    detectors = [Detector(4, 3, 0.7, 0.9, (0.17, -0.23)), Detector(4, 3, 0.7, 0.9, (-0.4, 0.31))]
    segments = [
        ConeGeometry(grid, detectors[0], [13.0, 71.0, 127.0], ConeBeam(9.0, 15.0), tilt_deg=17.0),
        ConeGeometry(
            grid, detectors[1], [7.0, 49.0, 97.0, 155.0], ConeBeam(11.0, 24.0), tilt_deg=-19.0
        ),
    ]
    geometry = ConeSegments(tuple(segments))
    # Build each physical arrangement independently through the public projector:
    # never give a sliced pose stack the combined scan's source/detector model.
    columns = [
        np.concatenate(
            [
                np.asarray(tj.project(segment, basis.reshape(2, 2, 3))).reshape(-1)
                for segment in segments
            ]
        )
        for basis in np.eye(12, dtype=np.float32)
    ]
    matrix = np.stack(columns, axis=1).astype(np.float64)
    truth = np.linspace(0.2, 1.0, 12)
    data = (matrix @ truth + np.random.default_rng(18).normal(0, 0.005, 84)).astype(np.float32)
    weights = np.linspace(0.3, 1.4, 84).astype(np.float32)
    weights[::7] = 0.0
    iterations, tau, sigma = 9, 0.03, 0.2
    cfg = SPDHGConfig(
        iterations=iterations,
        views_per_batch=batch,
        tv_weight=0.0,
        tau=tau,
        sigma_data=sigma,
        nonnegative=True,
        seed=17,
        stream_projections=stream,
    )
    actual, _ = spdhg_tv(
        geometry,
        grid,
        detectors[0],
        data.reshape(7, 3, 4),
        weights=weights.reshape(7, 3, 4),
        config=cfg,
    )
    # Independent FP64 block-dual/primal recurrence, including the shifted final
    # batch mask. A boundary-crossing block contains views from both arrangements.
    x, gradient_bar, dual = np.zeros(12), np.zeros(12), np.zeros(84)
    blocks = (7 + batch - 1) // batch
    schedule = np.random.default_rng(17).integers(0, blocks, iterations, dtype=np.int32)
    for block in schedule:
        x = np.maximum(x - tau * gradient_bar, 0.0)
        start, stop = int(block) * batch, min((int(block) + 1) * batch, 7)
        rows = slice(start * 12, stop * 12)
        old_dual = dual[rows].copy()
        dual[rows] = (
            (dual[rows] + sigma * (matrix[rows] @ x - data[rows]))
            * weights[rows]
            / (weights[rows] + sigma)
        )
        gradient_bar = matrix.T @ dual + cfg.theta * blocks * (
            matrix[rows].T @ (dual[rows] - old_dual)
        )
    np.testing.assert_allclose(np.asarray(actual).ravel(), x, rtol=2e-6, atol=2e-7)


class _RareMeasuredView(ParallelGeometry):
    def pose_for_view(self, i):
        pose = np.eye(4)
        pose[0, 3] = 0.0 if i == 0 else 20.0
        return tuple(map(tuple, pose))


@pytest.mark.numerical
def test_spdhg_automatic_steps_stay_stable_when_one_view_measures_the_volume():
    # One ray measures the voxel and the other 18 views miss it, so the least-squares
    # minimizer is 3. Block norms without the selection probability give unsafe steps.
    grid, detector = Grid(1, 1, 1, 1, 1, 1), Detector(1, 1, 1, 1)
    geometry = _RareMeasuredView(grid, detector, [0.0] * 19)
    data = np.zeros((19, 1, 1), np.float32)
    data[0] = 3.0
    volume, info = spdhg_tv(
        geometry,
        grid,
        detector,
        data,
        config=SPDHGConfig(
            iterations=1500,
            tv_weight=0,
            views_per_batch=1,
            projector_model="joseph",
            backend="jax",
            seed=8,
            log_every=0,
        ),
    )
    assert info["tau"] * info["sigma_data"] / info["selection_prob"] < 1
    np.testing.assert_allclose(volume, 3.0, rtol=2e-6, atol=2e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("regulariser", ["tv", "huber_tv"])
def test_spdhg_tv_trajectory_matches_dense_primal_dual_reference(regulariser):
    shape = (1, 2, 2)
    grid = Grid(*shape[::-1], 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.7, 0.9, (0.17, -0.23))
    geometry = LaminographyGeometry(grid, detector, [13, 71, 127, 193, 241], tilt_deg=25)
    columns = [np.asarray(tj.project(geometry, b.reshape(shape))).ravel() for b in np.eye(4)]
    matrix = np.stack(columns, axis=1).astype(np.float64)
    edges = np.zeros((3, 4, 4))
    for index in np.ndindex(shape):
        for axis in range(3):
            neighbor = list(index)
            neighbor[axis] += 1
            if neighbor[axis] < shape[axis]:
                row = np.ravel_multi_index(index, shape)
                edges[axis, row, row] = -1
                edges[axis, row, np.ravel_multi_index(tuple(neighbor), shape)] = 1
    data = np.random.default_rng(938).normal(0.3, 0.2, matrix.shape[0]).astype(np.float32)
    weights = np.linspace(0, 1.7, data.size).astype(np.float32)
    initial = np.asarray([-0.3, 0.7, 1.3, 0.4], np.float32)
    cfg = SPDHGConfig(
        iterations=14,
        tv_weight=0.2,
        regulariser=regulariser,
        huber_delta=0.4,
        tau=0.015,
        sigma_data=0.07,
        sigma_tv=0.08,
        views_per_batch=2,
        seed=12,
        nonnegative=True,
        log_every=1,
    )
    # Independent dense recurrence: primal first, then the sampled block dual and
    # the deterministic TV dual, each extrapolated by its inverse selection probability.
    x = np.maximum(initial, 0).astype(np.float64)
    dual, tv_dual, gradient_bar = np.zeros(data.size), np.zeros((3, 4)), np.zeros(4)
    losses = []
    for block in np.random.default_rng(12).integers(0, 3, 14, dtype=np.int32):
        x = np.maximum(x - cfg.tau * gradient_bar, 0)
        rows = slice(int(block) * 24, (int(block) + 1) * 24)
        a, w, y = matrix[rows], weights[rows], data[rows]
        old_dual, old_tv = dual[rows].copy(), tv_dual.copy()
        dual[rows] = (old_dual + cfg.sigma_data * (a @ x - y)) * w / (w + cfg.sigma_data)
        differences = edges @ x
        tv_dual += cfg.sigma_tv * differences
        if regulariser == "huber_tv":
            tv_dual *= cfg.tv_weight / (cfg.tv_weight + cfg.sigma_tv * cfg.huber_delta)
        tv_dual /= np.maximum(1, np.linalg.norm(tv_dual, axis=0) / cfg.tv_weight)
        gradient_bar = matrix.T @ dual + sum(e.T @ p for e, p in zip(edges, tv_dual, strict=True))
        gradient_bar += 3 * (a.T @ (dual[rows] - old_dual))
        gradient_bar += sum(e.T @ dp for e, dp in zip(edges, tv_dual - old_tv, strict=True))
        lengths = np.linalg.norm(differences, axis=0)
        if regulariser == "huber_tv":
            lengths = np.where(
                lengths <= cfg.huber_delta,
                lengths**2 / (2 * cfg.huber_delta),
                lengths - 0.5 * cfg.huber_delta,
            )
        losses.append(1.5 * np.sum(w * (a @ x - y) ** 2) + cfg.tv_weight * lengths.sum())
    actual, info = spdhg_tv(
        geometry,
        grid,
        detector,
        data.reshape(5, 3, 4),
        weights=weights.reshape(5, 3, 4),
        init_x=initial.reshape(shape),
        config=cfg,
    )
    np.testing.assert_allclose(np.asarray(actual).ravel(), x, rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(info["loss"], losses, rtol=2e-5, atol=3e-6)


@pytest.mark.numerical
def test_spdhg_automatic_norm_bounds_every_block_scaled_by_its_probability():
    shape = (1, 1, 2)
    grid = Grid(*shape[::-1], 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.7, 0.9, (0.17, -0.23))
    geometry = ConeGeometry(
        grid, detector, [13, 71, 127, 193, 241], ConeBeam(7, 12, detector_roll_deg=13), tilt_deg=23
    )
    columns = [np.asarray(tj.project(geometry, b.reshape(shape))).ravel() for b in np.eye(2)]
    matrix = np.stack(columns, axis=1).astype(np.float64)
    _, info = spdhg_tv(
        geometry,
        grid,
        detector,
        np.zeros((5, 3, 4), np.float32),
        config=SPDHGConfig(iterations=1, tv_weight=0, views_per_batch=2, log_every=0),
    )
    block_norms = [np.linalg.norm(matrix[start * 12 : (start + 2) * 12], 2) for start in (0, 2, 4)]
    assert info["A_norm"] >= np.linalg.norm(matrix, 2)
    assert info["A_norm"] >= np.sqrt(3) * max(block_norms)


@pytest.mark.numerical
def test_spdhg_runs_with_global_x64_and_returns_fp32():
    grid = Grid(2, 1, 1, 0.8, 1.1, 1.3)
    detector = Detector(2, 1, 0.8, 1.3)
    geometry = ParallelGeometry(grid, detector, [0.0] * 5)
    density = np.asarray([0.2, 1.0])
    data = np.broadcast_to(grid.vy * density, (5, 1, 2)).astype(np.float32).copy()
    weights = np.linspace(0.3, 1.7, 10).astype(np.float32).reshape(data.shape)
    lam = 0.1
    # Both free voxels are beyond the TV corner: stationarity gives the exact minimizer.
    expected = density + np.asarray([lam, -lam]) / (grid.vy**2 * weights.sum(axis=(0, 1)))
    with jax.enable_x64():
        actual, _ = spdhg_tv(
            geometry,
            grid,
            detector,
            data,
            weights=weights,
            init_x=np.asarray([-1.0, 3.0], np.float32).reshape(1, 1, 2),
            config=SPDHGConfig(
                iterations=1500,
                tv_weight=lam,
                nonnegative=True,
                views_per_batch=2,
                projector_model="joseph",
                backend="jax",
                log_every=0,
            ),
        )
    assert actual.dtype == np.dtype("float32")
    np.testing.assert_allclose(np.asarray(actual).ravel(), expected, rtol=3e-5, atol=3e-6)


@pytest.mark.numerical
def test_fista_joseph_integrator_uses_the_matched_batched_operators():
    grid = Grid(3, 4, 2, 0.8, 1.1, 1.3)
    detector = Detector(6, 3, 0.9, 1.2, (0.17, -0.23))
    geometry = ParallelGeometry(grid, detector, [13.0, 68.0, 142.0])
    data = np.random.default_rng(78).normal(size=(3, 3, 6)).astype(np.float32)
    config = FistaConfig(
        iterations=4, lipschitz=100.0, tv_weight=0.001, ray_integrator="joseph", views_per_batch=2
    )
    actual, _ = fista_tv(geometry, grid, detector, data, config=config)
    # Explicit ray model keeps the reference implementation; a fixed bound
    # isolates the operators from their norm estimates.
    reference = replace(config, projector_model="ray", backend="jax")
    expected, _ = fista_tv(geometry, grid, detector, data, config=reference)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
