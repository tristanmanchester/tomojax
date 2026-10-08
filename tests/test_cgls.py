"""Check CGLS against independently assembled dense least-squares systems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry
from tomojax.recon import CGLSConfig, cgls


@pytest.mark.numerical
@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize(
    ("model", "interpolation"), [("ray", "linear"), ("joseph", "linear"), ("joseph", "cubic")]
)
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_underdetermined_cgls_preserves_initial_null_component(
    warm_start, model, interpolation, backend
):
    """A small data residual must not hide a different unmeasured image component."""
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    from tomojax.forward import project_joseph

    shape = (3, 3, 2)
    grid = Grid(*shape, 0.8, 1.2, 1.4)
    detector = Detector(3, 2, 0.9, 1.3, (0.17, -0.23))
    geometry = LaminographyGeometry(grid, detector, [13, 83], tilt_deg=23)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(2)])

    def project(flat):
        volume = flat.reshape(shape)
        if model == "joseph":
            return project_joseph(
                volume, poses, grid, detector, backend="jax", interpolation=interpolation
            ).ravel()
        return jax.vmap(lambda t: forward_project_view_T(t, grid, detector, volume))(poses).ravel()

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
            rtol=1e-7,
            views_per_batch=1,
            projector_backend=backend,
            projector_model=model,
            joseph_interpolation=interpolation,
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
        grid, detector, np.asarray([13, 49, 83, 122, 161]) + offset, tilt_deg=23
    )
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)])

    def project(flat):
        return jax.vmap(
            lambda t: forward_project_view_T(t, grid, detector, flat.reshape((3, 2, 2)))
        )(poses).ravel()

    matrix = np.asarray(jax.jacfwd(project)(jnp.zeros(12))).astype(np.float64)
    return grid, detector, geometry, matrix


@pytest.mark.numerical
@pytest.mark.parametrize("damping", [0.0, 0.7])
@pytest.mark.parametrize("warm_start", [False, True])
def test_cgls_matches_dense_lstsq_with_tail_batches(damping, warm_start):
    grid, detector, geometry, matrix = small_problem()
    rng = np.random.default_rng(842)
    data = rng.normal(size=(5, 4, 5)).astype(np.float32)
    initial = rng.normal(size=(3, 2, 2)).astype(np.float32) if warm_start else None
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
            projector_model="ray",
            iterations=60,
            rtol=1e-7,
            damping=damping,
            views_per_batch=3,
            projector_backend="jax",
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
                projector_model="ray",
                iterations=160,
                rtol=0,
                damping=0.7,
                views_per_batch=3,
                projector_backend="jax",
            ),
        )
        np.testing.assert_allclose(volume.ravel(), expected, rtol=2e-4, atol=3e-5)
        assert info["termination"] in {"converged", "roundoff_limit"}
        assert info["effective_iterations"] < 160


def test_zero_data_and_zero_iteration_limit_preserve_valid_results():
    grid, detector, geometry, _ = small_problem()
    volume, info = cgls(geometry, grid, detector, jnp.zeros((5, 4, 5)))
    np.testing.assert_array_equal(volume, np.zeros((3, 2, 2)))
    assert info["converged"] and info["effective_iterations"] == 0
    initial = jnp.ones((3, 2, 2))
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
    # check-public-imports: allow-private
    from tomojax.recon.cgls import _solve

    _solve.clear_cache()
    outputs = []
    for offset, iterations in [(0.0, 35), (7.0, 50)]:
        grid, detector, geometry, matrix = small_problem(offset)
        truth = np.linspace(-0.3, 1.7 + offset / 10, 12)
        data = (matrix @ truth).reshape((5, 4, 5)).astype(np.float32)
        result, info = cgls(
            geometry,
            grid,
            detector,
            data,
            config=CGLSConfig(
                projector_model="ray",
                iterations=iterations,
                views_per_batch=3,
                projector_backend="jax",
            ),
        )
        np.testing.assert_allclose(result.ravel(), truth, rtol=2e-4, atol=2e-5)
        assert info["converged"]
        outputs.append(np.asarray(result))
    assert _solve._cache_size() == 1
    assert not np.allclose(*outputs)


@pytest.mark.parametrize(
    "bad_config",
    [
        CGLSConfig(iterations=-1),
        CGLSConfig(views_per_batch=0),
        CGLSConfig(rtol=-0.1),
        CGLSConfig(damping=np.inf),
        CGLSConfig(atol=np.nan),
        CGLSConfig(projector_backend="invalid"),
    ],
)
def test_cgls_rejects_invalid_configuration(bad_config):
    grid, detector, geometry, _ = small_problem()
    with pytest.raises(ValueError):
        cgls(geometry, grid, detector, jnp.zeros((5, 4, 5)), config=bad_config)


def test_cgls_rejects_nonfinite_input():
    grid, detector, geometry, _ = small_problem()
    with pytest.raises(ValueError, match="finite"):
        cgls(geometry, grid, detector, jnp.full((5, 4, 5), jnp.nan))


@pytest.mark.gpu
def test_pallas_cgls_matches_dense_damped_solution():
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
            projector_model="ray",
            iterations=60,
            rtol=1e-7,
            damping=damping,
            views_per_batch=3,
            projector_backend="pallas",
        ),
    )
    np.testing.assert_allclose(actual.ravel(), expected, rtol=3e-4, atol=3e-5)
    assert info["projector_backend"] == "pallas"
    normal_residual = (
        matrix.T @ (data.ravel() - matrix @ np.asarray(actual).ravel())
        - damping**2 * np.asarray(actual).ravel()
    )
    assert np.linalg.norm(normal_residual) < 2e-6 * np.linalg.norm(matrix.T @ data.ravel())
    assert info["termination"] in {"converged", "roundoff_limit"}


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("model", ["ray", "joseph"])
def test_bright_region_does_not_hide_weak_region_updates(backend, model):
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    # check-public-imports: allow-private
    from tomojax.core.geometry.views import stack_view_poses
    from tomojax.geometry import ParallelGeometry

    # check-public-imports: allow-private
    from tomojax.recon.cgls import _operators

    grid = Grid(4, 3, 2, 1.0, 1.0, 1.0)
    detector = Detector(9, 2, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, [0.0, 31.0, 67.0, 95.0, 147.0])
    poses = stack_view_poses(geometry, 5)
    initial = np.zeros((4, 3, 2), np.float32)
    initial[:, :, 0] = 1e7
    truth = initial.copy()
    truth[:, :, 1] = 0.01 * np.arange(1, 13).reshape(4, 3)
    # Each detector row sees only one axial slice. The bright slice is already
    # correct and cannot set the precision limit for the independent weak slice.
    project = jax.jit(lambda t, x: _operators(t, grid, detector, None, backend, 5, model)[0](x))
    data = project(poses, jnp.asarray(truth))
    actual, info = cgls(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=CGLSConfig(
            iterations=60,
            rtol=1e-6,
            views_per_batch=5,
            projector_model=model,
            projector_backend=backend,
        ),
    )
    np.testing.assert_array_equal(np.asarray(actual)[:, :, 0], initial[:, :, 0])
    np.testing.assert_allclose(np.asarray(actual)[:, :, 1], truth[:, :, 1], rtol=2e-4, atol=1e-7)
    assert info["converged"]
    assert info["normal_residual_is_recomputed"]
    assert info["residual_recomputations"] >= 1


@pytest.mark.parametrize("damping", [0.0, 0.3])
def test_streamed_cgls_solves_the_same_least_squares_problem(tmp_path, damping):
    grid, detector = Grid(12, 11, 8, 0.8, 1.1, 1.3), Detector(14, 10, 0.9, 1.2, (0.1, -0.2))
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 360, 40, endpoint=False), 30)
    data = np.random.default_rng(0).random((40, 10, 14), dtype=np.float32)
    stored = np.memmap(tmp_path / "views.f32", mode="w+", dtype=np.float32, shape=data.shape)
    stored[:] = data
    results = {}
    for stream, projections in [(False, jnp.asarray(data)), (True, stored)]:
        config = CGLSConfig(
            iterations=300, damping=damping, views_per_batch=7, stream_projections=stream
        )
        results[stream] = cgls(geometry, grid, detector, projections, config=config)
    assert results[True][1]["formulation"] == "streamed_normal_equations"
    assert results[True][1]["converged"] and results[False][1]["converged"]
    np.testing.assert_allclose(results[True][0], results[False][0], rtol=1e-4, atol=1e-5)


def test_streamed_cgls_rejects_nonfinite_views():
    grid, detector = Grid(6, 6, 4, 1, 1, 1), Detector(8, 6, 1, 1)
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 360, 8, endpoint=False), 30)
    data = np.ones((8, 6, 8), np.float32)
    data[3, 2, 2] = np.nan
    with pytest.raises(Exception, match="finite"):
        cgls(geometry, grid, detector, data, config=CGLSConfig(stream_projections=True))
