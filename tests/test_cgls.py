"""Check CGLS against independently assembled dense least-squares systems."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax import progress
from tomojax.forward import project_joseph
from tomojax.geometry import (
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    stack_view_poses,
)
from tomojax.recon import CGLSConfig, cgls

# check-public-imports: allow-private
from tomojax.recon._cgls_jax import _operators, _run_cgls, _solve, _solve_initial


@pytest.mark.numerical
@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_underdetermined_cgls_preserves_initial_null_component(warm_start, interpolation, backend):
    """A small data residual must not hide a different unmeasured image component."""
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    shape = (2, 3, 3)
    grid = Grid(*shape[::-1], 0.8, 1.2, 1.4)
    detector = Detector(3, 2, 0.9, 1.3, (0.17, -0.23))
    geometry = LaminographyGeometry(grid, detector, [13, 83], tilt=23)
    poses = jnp.asarray([geometry.pose_matrix(i) for i in range(2)])

    def project(flat):
        volume = flat.reshape(shape)
        return project_joseph(
            volume, poses, grid, detector, backend="jax", interpolation=interpolation
        ).ravel()

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(np.prod(shape))), dtype=np.float64)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=True)
    rank = np.count_nonzero(singular > singular[0] * 1e-10)
    null_basis = vt[rank:].T
    assert null_basis.shape[1] >= 6
    rng = np.random.default_rng(861)
    data = rng.normal(size=(2, 2, 3)).astype(np.float32)
    initial = rng.normal(size=shape).astype(np.float32) if warm_start else np.zeros(shape)
    expected = (
        initial.ravel()
        + np.linalg.lstsq(matrix, data.ravel() - matrix @ initial.ravel(), rcond=1e-10)[0]
    )
    volume, info = cgls(
        geometry,
        grid,
        detector,
        data,
        init_x=initial if warm_start else None,
        config=CGLSConfig(
            iterations=100,
            tolerance=1e-7,
            views_per_batch=1,
            backend=backend,
            interpolation=interpolation,
        ),
    )
    actual = np.asarray(volume).ravel()
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
    increment = actual - initial.ravel()
    # Use a relative projection norm: an absolute threshold changes with the
    # measurement units and the amplification of the observable solution.
    assert np.linalg.norm(null_basis.T @ increment) < 2e-5 * np.linalg.norm(increment)
    assert info["termination"] in {"converged", "roundoff_limit"}


def small_problem(offset=0.0):
    grid = Grid(3, 2, 2, 0.8, 1.1, 1.3)
    detector = Detector(5, 4, 0.7, 1.2, (0.17, -0.23))
    geometry = LaminographyGeometry(
        grid, detector, np.asarray([13, 49, 83, 122, 161]) + offset, tilt=23
    )
    poses = jnp.asarray([geometry.pose_matrix(i) for i in range(5)])

    def project(flat):
        return project_joseph(flat.reshape((2, 2, 3)), poses, grid, detector).ravel()

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(12))).astype(np.float64)
    return grid, detector, geometry, matrix


@pytest.mark.numerical
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("damping", [0.0, 0.7])
@pytest.mark.parametrize("warm_start", [False, True])
def test_cgls_matches_dense_lstsq_with_tail_batches(stream, damping, warm_start):
    grid, detector, geometry, matrix = small_problem()
    rng = np.random.default_rng(842)
    data = rng.normal(size=(5, 4, 5)).astype(np.float32)
    initial = rng.normal(size=(2, 2, 3)).astype(np.float32) if warm_start else None
    expected = np.linalg.lstsq(
        np.concatenate([matrix, damping * np.eye(12)]),
        np.concatenate([data.ravel(), np.zeros(12)]),
        rcond=None,
    )[0]
    volume, info = cgls(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=CGLSConfig(
            iterations=60,
            tolerance=1e-7,
            damping=damping,
            views_per_batch=3,
            backend="jax",
            stream=stream,
        ),
    )
    np.testing.assert_allclose(volume.ravel(), expected, rtol=2e-4, atol=3e-5)
    normal_residual = (
        matrix.T @ (data.ravel() - matrix @ np.asarray(volume).ravel())
        - damping**2 * np.asarray(volume).ravel()
    )
    assert np.linalg.norm(normal_residual) < 2e-6 * np.linalg.norm(matrix.T @ data.ravel())
    assert info["termination"] in {"converged", "roundoff_limit"}
    assert info["effective_iterations"] < 60


def test_unattainable_tolerance_stops_at_roundoff_without_diverging():
    grid, detector, geometry, matrix = small_problem()
    data = np.random.default_rng(842).normal(size=(5, 4, 5)).astype(np.float32)
    expected = np.linalg.lstsq(
        np.concatenate([matrix, 0.7 * np.eye(12)]),
        np.concatenate([data.ravel(), np.zeros(12)]),
        rcond=None,
    )[0]
    for _ in range(8):
        volume, info = cgls(
            geometry,
            grid,
            detector,
            data,
            config=CGLSConfig(
                iterations=160,
                tolerance=0,
                damping=0.7,
                views_per_batch=3,
                backend="jax",
            ),
        )
        np.testing.assert_allclose(volume.ravel(), expected, rtol=2e-4, atol=3e-5)
        assert info["termination"] in {"converged", "roundoff_limit"}
        assert info["effective_iterations"] < 160


def test_zero_data_and_zero_iteration_limit_preserve_valid_results():
    grid, detector, geometry, _ = small_problem()
    volume, info = cgls(geometry, grid, detector, jnp.zeros((5, 4, 5)))
    np.testing.assert_array_equal(volume, np.zeros((2, 2, 3)))
    assert info["converged"] and info["effective_iterations"] == 0
    initial = jnp.ones((2, 2, 3))
    volume, info = cgls(
        geometry,
        grid,
        detector,
        jnp.zeros((5, 4, 5)),
        init_x=initial,
        config=CGLSConfig(iterations=0),
    )
    np.testing.assert_array_equal(volume, initial)
    assert info["termination"] == "iteration_limit"


def test_changed_data_geometry_and_budget_reuse_compiled_solver():
    from contextlib import nullcontext
    from threading import get_ident

    programs = (_run_cgls.unwatched, _solve_initial, _solve)
    for program in programs:
        program.clear_cache()
    owner = get_ident()
    labels = []

    def observe(event):
        assert get_ident() == owner
        if isinstance(event, progress.Started) and event.name == "prepare":
            assert event.total is None and event.unit == ""
            labels.append(event.label)

    outputs = []
    for watched, offset, iterations, expected_sizes in [
        (False, 0.0, 35, [1, 0, 0]),
        (True, 0.0, 35, [1, 1, 1]),
        (True, 7.0, 50, [1, 1, 1]),
        (False, 7.0, 50, [1, 1, 1]),
    ]:
        grid, detector, geometry, matrix = small_problem(offset)
        truth = np.linspace(-0.3, 1.7 + offset / 10, 12)
        data = (matrix @ truth).reshape((5, 4, 5)).astype(np.float32)
        labels.clear()
        with progress.watch(observe) if watched else nullcontext():
            result, info = cgls(
                geometry,
                grid,
                detector,
                data,
                config=CGLSConfig(
                    iterations=iterations,
                    views_per_batch=3,
                    backend="jax",
                ),
            )
        np.testing.assert_allclose(result.ravel(), truth, rtol=2e-4, atol=2e-5)
        assert info["converged"]
        assert labels == (["initialize solver", "first update"] if watched else [])
        assert [program._cache_size() for program in programs] == expected_sizes
        outputs.append(np.asarray(result))
    assert _run_cgls.unwatched._cache_size() == 1
    assert not np.allclose(outputs[0], outputs[-1])


@pytest.mark.parametrize(
    ("bad_config", "message"),
    [
        ({"iterations": -1}, "iterations must be nonnegative"),
        ({"views_per_batch": 0}, "views_per_batch positive"),
        ({"tolerance": -0.1}, "tolerance must be finite and nonnegative"),
        ({"damping": np.inf}, "damping must be finite and nonnegative"),
        (
            {"absolute_tolerance": np.nan},
            "absolute_tolerance must be finite and nonnegative",
        ),
        ({"backend": "invalid"}, "backend must be"),
        ({"iterations": -1, "backend": "invalid"}, "iterations must be nonnegative"),
    ],
)
def test_cgls_rejects_invalid_configuration(bad_config, message):
    grid, detector, geometry, _ = small_problem()
    with pytest.raises(ValueError, match=message):
        cgls(geometry, grid, detector, jnp.zeros((5, 4, 5)), config=CGLSConfig(**bad_config))


def test_cgls_rejects_nonfinite_input():
    grid, detector, geometry, _ = small_problem()
    with pytest.raises(ValueError, match="finite"):
        cgls(geometry, grid, detector, jnp.full((5, 4, 5), jnp.nan))


@pytest.mark.gpu
def test_cuda_cgls_matches_dense_damped_solution():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, geometry, matrix = small_problem()
    data = np.random.default_rng(184).normal(size=(5, 4, 5)).astype(np.float32)
    damping = 0.3
    expected = np.linalg.lstsq(
        np.concatenate([matrix, damping * np.eye(12)]),
        np.concatenate([data.ravel(), np.zeros(12)]),
        rcond=None,
    )[0]
    actual, info = cgls(
        geometry,
        grid,
        detector,
        data,
        config=CGLSConfig(
            iterations=60,
            tolerance=1e-7,
            damping=damping,
            views_per_batch=3,
            backend="cuda",
        ),
    )
    np.testing.assert_allclose(actual.ravel(), expected, rtol=3e-4, atol=3e-5)
    assert info["projector_backend"] == "cuda"
    normal_residual = (
        matrix.T @ (data.ravel() - matrix @ np.asarray(actual).ravel())
        - damping**2 * np.asarray(actual).ravel()
    )
    assert np.linalg.norm(normal_residual) < 2e-6 * np.linalg.norm(matrix.T @ data.ravel())
    assert info["termination"] in {"converged", "roundoff_limit"}


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_bright_region_does_not_hide_weak_region_updates(backend):
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    from tomojax.geometry import ParallelGeometry

    grid = Grid(4, 3, 2, 1.0, 1.0, 1.0)
    detector = Detector(9, 2, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, [0.0, 31.0, 67.0, 95.0, 147.0])
    poses = stack_view_poses(geometry, 5)
    initial = np.zeros((2, 3, 4), np.float32)
    initial[0] = 1e7
    truth = initial.copy()
    truth[1] = 0.01 * np.arange(1, 13).reshape(4, 3).T
    # Each detector row sees only one axial slice. The bright slice is already
    # correct and cannot set the precision limit for the independent weak slice.
    project = jax.jit(lambda t, x: _operators(t, grid, detector, None, backend, 5, "joseph")[0](x))
    data = project(poses, jnp.asarray(truth))
    actual, info = cgls(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=CGLSConfig(
            iterations=60,
            tolerance=1e-6,
            views_per_batch=5,
            backend=backend,
        ),
    )
    np.testing.assert_array_equal(np.asarray(actual)[0], initial[0])
    np.testing.assert_allclose(np.asarray(actual)[1], truth[1], rtol=2e-4, atol=1e-7)
    assert info["converged"]
    assert info["normal_residual_is_recomputed"]
    assert info["residual_recomputations"] >= 1


def test_cancellation_estimate_does_not_stop_decreasing_normal_residual():
    from tomojax.geometry import ParallelGeometry

    # Fixed CPU arithmetic reproduces the premature roundoff stop independently
    # of the GPU reduction ordering in the existing bright/weak-region test.
    with jax.default_device(jax.devices("cpu")[0]):
        grid, detector = Grid(4, 3, 2, 1, 1, 1), Detector(9, 2, 1, 1)
        geometry = ParallelGeometry(grid, detector, [0, 31, 67, 95, 147])
        poses = stack_view_poses(geometry, 5)
        initial = np.zeros((2, 3, 4), np.float32)
        initial[0] = 1e7
        truth = initial.copy()
        truth[1] = 0.01 * np.arange(1, 13).reshape(4, 3).T
        project = jax.jit(
            lambda t, x: _operators(t, grid, detector, None, "jax", 5, "joseph")[0](x)
        )
        data = project(poses, jnp.asarray(truth))
        config = CGLSConfig(tolerance=1e-7, backend="jax", views_per_batch=5)
        actual, info = cgls(geometry, grid, detector, data, init_x=initial, config=config)
        from dataclasses import replace

        _, previous = cgls(
            geometry, grid, detector, data, init_x=initial,
            config=replace(config, iterations=info["effective_iterations"] - 1),
        )  # fmt: skip
    np.testing.assert_array_equal(np.asarray(actual)[0], initial[0])
    np.testing.assert_allclose(np.asarray(actual)[1], truth[1], rtol=2e-4, atol=1e-7)
    assert info["normal_residual_is_recomputed"] and previous["normal_residual_is_recomputed"]
    if info["converged"]:
        assert info["normal_residual_norm"] <= info["normal_residual_tolerance"]
    else:
        # FP32 reduction order can put the attainable floor on either side of
        # the tolerance. A decreasing verified norm must never count as stalled.
        assert info["termination"] == "roundoff_limit"
        assert info["normal_residual_norm"] >= previous["normal_residual_norm"]


@pytest.mark.parametrize("stream", [False, True])
def test_healthy_small_normal_residual_does_not_trigger_per_iteration_replacement(stream):
    import tomojax as tj

    scan = tj.simulate("shepp", size=16, n_views=17).scan
    clean = np.asarray(scan.projections)
    noise = np.random.default_rng(12).standard_normal(clean.shape, dtype=np.float32)
    data = clean + 0.01 * clean.std() * noise
    config = CGLSConfig(iterations=128, tolerance=0, stream=stream)
    _, info = cgls(scan.geometry, scan.grid, scan.detector, data, config=config)
    assert info["effective_iterations"] == 128
    assert 0 < info["residual_recomputations"] < 128 // 4


@pytest.mark.parametrize("damping", [0.0, 0.3])
def test_streamed_cgls_solves_the_same_least_squares_problem(tmp_path, damping):
    grid, detector = Grid(12, 11, 8, 0.8, 1.1, 1.3), Detector(14, 10, 0.9, 1.2, (0.1, -0.2))
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 360, 40, endpoint=False), 30)
    data = np.random.default_rng(0).random((40, 10, 14), dtype=np.float32)
    stored = np.memmap(tmp_path / "views.f32", mode="w+", dtype=np.float32, shape=data.shape)
    stored[:] = data
    results = {}
    for stream, projections in [(False, jnp.asarray(data)), (True, stored)]:
        config = CGLSConfig(iterations=300, damping=damping, views_per_batch=7, stream=stream)
        results[stream] = cgls(geometry, grid, detector, projections, config=config)
    assert results[True][1]["formulation"] == "streamed_normal_equations"
    assert results[True][1]["converged"] and results[False][1]["converged"]
    np.testing.assert_allclose(results[True][0], results[False][0], rtol=1e-4, atol=1e-5)


def test_auto_streaming_reserves_volume_state(monkeypatch):
    from dataclasses import replace

    import tomojax.backends
    from tomojax.geometry import ConeBeam, ConeGeometry

    # The same small sinogram fits comfortably; only the volume state changes.
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(tomojax.backends, "device_free_memory_bytes", lambda: 6000)
    detector = Detector(5, 4, 1, 1)
    data = np.ones((3, 4, 5), np.float32)
    for size, expected in [(3, "cgls"), (8, "streamed_normal_equations")]:
        grid = Grid(size, size, size, 1, 1, 1)
        geometry = ConeGeometry(grid, detector, [0, 60, 120], ConeBeam(30, 50))
        config = CGLSConfig(iterations=2, backend="jax")
        automatic, info = cgls(geometry, grid, detector, data, config=config)
        assert info["formulation"] == expected
        resident, info = cgls(
            geometry,
            grid,
            detector,
            data,
            config=replace(config, stream=False),
        )
        assert info["formulation"] == "cgls"
        np.testing.assert_allclose(automatic, resident, rtol=2e-5, atol=2e-6)


def test_streamed_cgls_rejects_nonfinite_views():
    grid, detector = Grid(6, 6, 4, 1, 1, 1), Detector(8, 6, 1, 1)
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 360, 8, endpoint=False), 30)
    data = np.ones((8, 6, 8), np.float32)
    data[3, 2, 2] = np.nan
    with pytest.raises(Exception, match="finite"):
        cgls(geometry, grid, detector, data, config=CGLSConfig(stream=True))
    # Replace the failed callback token before JAX's process-exit barrier.
    data[3, 2, 2] = 1
    result, _ = cgls(geometry, grid, detector, data, config=CGLSConfig(stream=True))
    np.testing.assert_array_equal(np.isfinite(result), True)
    jax.effects_barrier()


# Its child process needs CUDA, which the suite's own JAX would otherwise hold.
@pytest.mark.gpu
@pytest.mark.fresh_gpu
@pytest.mark.parametrize("data_term", ["absolute", "per_view_contrast"])
def test_host_cuda_cgls_does_not_import_jax(data_term):
    import subprocess
    import sys

    code = f"""
import sys, json, numpy as np
import tomojax as tj
from tomojax.geometry import Grid, Detector, LaminographyGeometry
from tomojax.recon import CGLSConfig, cgls
g, d = Grid(4, 4, 3, 1, 1, 1), Detector(9, 7, 1, 1)
geometry = LaminographyGeometry(g, d, np.linspace(0, 180, 7, endpoint=False), tilt=23)
x, info = cgls(geometry, g, d, np.zeros((7, 7, 9), np.float32),
    init_x=np.ones((3, 4, 4), np.float32),
    config=CGLSConfig(iterations=2, backend='cuda', views_per_batch=3, data_term='{data_term}'))
json.dumps(info, allow_nan=False)
assert 'jax' not in sys.modules and info['execution_backend'] == 'cupy'
assert isinstance(x, np.ndarray) and np.isfinite(x).all() and np.linalg.norm(x) < 7
# A repeated solve finds the first one's memory back with the driver and stays on CuPy.
result = tj.reconstruct(tj.Scan(np.ones((7, 7, 9), np.float32), geometry), 'cgls', iterations=2)
assert 'jax' not in sys.modules and result.info['execution_backend'] == 'cupy'
assert isinstance(result.volume, np.ndarray) and np.isfinite(result.volume).all()
assert 'cupyx.scipy.sparse' not in sys.modules
"""
    # The test process may already hold the GPU: the child must not preallocate.
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.gpu
@pytest.mark.parametrize("tilt", [0, 27])
def test_native_cuda_cgls_matches_independent_damped_system(tilt):
    from tomojax.forward import project_joseph

    g, d = Grid(5, 4, 3, 0.8, 1.1, 1.3), Detector(7, 5, 0.9, 1.0, (0.2, -0.3))
    geometry = LaminographyGeometry(g, d, [13, 49, 83, 122, 161], tilt=tilt)
    poses = np.asarray([geometry.pose_matrix(i) for i in range(5)], np.float32)
    rng = np.random.default_rng(761)
    data = rng.normal(size=(5, 5, 7)).astype(np.float32)
    matrix = np.asarray(
        jax.jacfwd(
            lambda x: project_joseph(x.reshape(3, 4, 5), poses, g, d, backend="jax").ravel()
        )(jnp.zeros(60))
    )
    expected = np.linalg.lstsq(
        np.concatenate((matrix, 0.3 * np.eye(60))),
        np.concatenate((data.ravel(), np.zeros(60))),
        rcond=None,
    )[0]
    x, info = cgls(
        geometry,
        g,
        d,
        data,
        config=CGLSConfig(
            iterations=100, tolerance=1e-6, damping=0.3, backend="cuda", views_per_batch=3
        ),
    )
    assert info["execution_backend"] == "cupy"
    np.testing.assert_allclose(x.ravel(), expected, atol=2e-5, rtol=3e-4)
    assert info["termination"] in {"converged", "roundoff_limit"}


@pytest.mark.gpu
@pytest.mark.parametrize("narrow", [False, True])
def test_native_cuda_clipped_planes_keep_the_matched_pair(narrow):
    import cupy as cp

    # check-public-imports: allow-private
    from tomojax.core._joseph_cuda_kernels import forward_kernel
    from tomojax.core.joseph_cuda import JosephCUDA
    from tomojax.forward import project_joseph

    grid = Grid(17, 13, 11, 1.2, 0.8, 1.4, vol_center=(0.3, -0.2, 0.7))
    detector = Detector(23, 19, 0.7, 1.1, (0.1, -0.4))
    angles = np.r_[[0, 0.000001, 31, 89.99999, 90, 153], np.linspace(0, 180, 61)]
    geometry = LaminographyGeometry(grid, detector, angles, 51)
    poses = np.asarray([geometry.pose_matrix(i) for i in range(len(angles))], np.float32)
    rng = np.random.default_rng(719)
    x = rng.normal(size=(11, 13, 17)).astype(np.float32)
    y = rng.normal(size=(len(angles), 19, 23)).astype(np.float32)
    operators = JosephCUDA(poses, grid, detector, views_per_batch=4)
    operators.forward_kernel = forward_kernel(narrow=narrow)
    ax, aty = (
        cp.asnumpy(operators.forward(cp.asarray(x))),
        cp.asnumpy(operators.adjoint(cp.asarray(y))),
    )
    expected = np.asarray(project_joseph(jnp.asarray(x), poses, grid, detector, backend="jax"))
    np.testing.assert_allclose(ax, expected, rtol=3e-5, atol=5e-5)
    lhs, rhs = np.vdot(ax.astype(float), y), np.vdot(x.astype(float), aty)
    assert abs(lhs - rhs) < 1e-6 * np.linalg.norm(ax) * np.linalg.norm(y)
    # Solver-private z-fast storage gives the same bits.
    private = JosephCUDA(poses, grid, detector, views_per_batch=4, z_fast=True)
    private.forward_kernel = forward_kernel(narrow=narrow, z_fast=True)
    xt = cp.asarray(np.ascontiguousarray(x.transpose(2, 1, 0)))
    np.testing.assert_array_equal(cp.asnumpy(private.forward(xt)), ax)
    np.testing.assert_array_equal(
        cp.asnumpy(private.adjoint(cp.asarray(y))).transpose(2, 1, 0), aty
    )


@pytest.mark.gpu
def test_native_cuda_inverse_support_keeps_rounded_boundary_weights():
    import cupy as cp

    from tomojax.core.joseph_cuda import JosephCUDA

    grid, detector = Grid(256, 256, 256, 1, 1, 1), Detector(256, 256, 1, 1)
    geometry = LaminographyGeometry(grid, detector, [21.0], 30.0)
    poses = np.asarray([geometry.pose_matrix(0)], np.float32)
    operators = JosephCUDA(poses, grid, detector, views_per_batch=1)
    volume, images = cp.zeros((256,) * 3, cp.float32), cp.zeros((1, 256, 256), cp.float32)
    volume[165, 251, 230], images[0, 84, 178] = 1, 1
    projected = float(operators.forward(volume)[0, 84, 178])
    backprojected = float(operators.adjoint(images)[165, 251, 230])
    assert projected > 1e-6
    np.testing.assert_allclose(backprojected, projected, rtol=2e-6, atol=2e-10)


_SCATTERED_POOL = r"""
import jax, jax.numpy as jnp, numpy as np
import tomojax as tj

# Fill the capped pool with 32 MiB blocks, then free every other one: half the pool is free,
# but no free block holds the data, let alone the solve's working set.
held = []
while True:
    try:
        held.append(jnp.zeros((2**23,), jnp.float32).block_until_ready())
    except Exception:
        break
for block in held[::2]:
    block.delete()
n, views = 48, 1440
grid = tj.Grid(n, n, n, 1, 1, 1)
detector = tj.Detector(192, n, 1, 1)
geometry = tj.ConeGeometry(
    grid, detector, np.linspace(0, 360, views, endpoint=False), tj.ConeBeam(3 * n, 4.5 * n)
)
data = np.random.default_rng(0).random((views, n, 192), dtype=np.float32)
import sys
print("RECONSTRUCT", file=sys.stderr, flush=True)
reconstruction = tj.reconstruct(tj.Scan(data, geometry), "cgls", iterations=3)
print(reconstruction.info["formulation"], bool(np.isfinite(np.asarray(reconstruction.volume)).all()))
"""


@pytest.mark.gpu
@pytest.mark.fresh_gpu
def test_a_scattered_pool_streams_the_data_rather_than_running_out():
    # The free bytes in all exceed the resident solve's need, but no one block holds it.
    import subprocess
    import sys

    env = {**os.environ, "JAX_PLATFORMS": "cuda,cpu", "XLA_CLIENT_MEM_FRACTION": "0.12"}
    run = subprocess.run(
        [sys.executable, "-c", _SCATTERED_POOL],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.split()[-2:] == ["streamed_normal_equations", "True"]
    # The pool's own figures rule the block out: no allocation is tried, so the allocator
    # neither retries nor reports running out.
    assert "bfc_allocator" not in run.stderr.split("RECONSTRUCT")[-1]


@pytest.mark.gpu
def test_cuda_cgls_holds_no_volume_temporaries_beside_its_state():
    # Iterate, direction and two gradients, or the verification's bound and exact
    # gradient: at most five volumes and four sinograms are live at once, so the
    # solve fits where its resident bound is admitted beside another allocator.
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    import cupy as cp

    class Live(cp.cuda.MemoryHook):
        name = "live"

        def __init__(self) -> None:
            self.now = self.peak = 0

        def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id) -> None:
            self.now += mem_size
            self.peak = max(self.peak, self.now)

        def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id) -> None:
            self.now -= mem_size

    # Few views, so the volumes dominate: before, the solve held about six of them.
    n, views = 128, 16
    grid, detector = Grid(n, n, n, 1.0, 1.0, 1.0), Detector(n, n, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, np.linspace(0, 180, views, endpoint=False))
    data = np.random.default_rng(88).normal(size=(views, n, n)).astype(np.float32)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    held = pool.total_bytes()  # other tests' live arrays in this process
    live = Live()
    with live:
        _, info = cgls(
            geometry, grid, detector, data,
            config=CGLSConfig(iterations=20, tolerance=0, backend="cuda"),
        )  # fmt: skip
    assert info["execution_backend"] == "cupy"
    assert live.peak <= 4 * (5 * n**3 + 4 * data.size)
    # The solve's blocks go back to the driver, not into CuPy's cache beside JAX.
    assert pool.total_bytes() <= held


@pytest.mark.gpu
def test_native_cgls_reports_completed_updates():
    import cupy as cp

    from tomojax import progress

    grid = Grid(6, 5, 4, 1, 1, 1)
    detector = Detector(7, 4, 1, 1)
    geometry = ParallelGeometry(grid, detector, [0, 37, 101])
    data = np.random.default_rng(38).normal(size=(3, 4, 7)).astype(np.float32)
    config = CGLSConfig(iterations=2, tolerance=0)
    updates = []

    def observe(event):
        if isinstance(event, progress.Advanced):
            assert stream.done
            updates.append(event.done)

    with cp.cuda.Stream(non_blocking=True) as stream, progress.watch(observe):
        _, info = cgls(geometry, grid, detector, data, config=config)
    assert info["execution_backend"] == "cupy"
    assert updates == [1, 2]
