"""FISTA's owned zero start and first step against an explicit optimizer oracle."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FistaConfig, fista_tv


@pytest.mark.numerical
@pytest.mark.parametrize(("steps", "early"), [(1, False), (7, False), (7, True)])
def test_owned_zero_start_preserves_norm_first_step_and_stopping(steps, early):
    import importlib

    module = importlib.import_module("tomojax.recon.fista_tv")
    grid, detector = Grid(16, 16, 8, 1, 1, 1), Detector(8, 6, 1, 1)
    geometry = LaminographyGeometry(grid, detector, np.linspace(180, 0, 31), tilt=27)
    data = jnp.asarray(np.random.default_rng(1732).uniform(size=(31, 6, 8)), jnp.float32)
    before = np.asarray(data).copy()
    config = FistaConfig(
        iterations=steps,
        tv_weight=0,
        nonnegative=True,
        lipschitz=1000 if early else None,
        tolerance=1e5 if early else None,
        patience=1,
    )
    runtime = module._prepare_fista_runtime(
        geometry, grid, detector, data, init_x=None, config=config, det_grid=None
    )
    assert module.use_owned_fista(grid, data, runtime)
    assert not module.use_owned_fista(grid, data, replace(runtime, x0=jnp.zeros((8, 16, 16))))
    expected, limit, _ = module._run_fista_scan(grid, detector, data, runtime)
    actual, info = fista_tv(geometry, grid, detector, data, config=config)
    np.testing.assert_allclose(actual, expected.x, rtol=3e-5, atol=2e-7)
    np.testing.assert_allclose(info["loss"], expected.loss, rtol=3e-6)
    np.testing.assert_allclose(info["lipschitz"], limit, rtol=1e-6)
    np.testing.assert_array_equal(data, before)
    assert info["effective_iterations"] == int(expected.iters_done)
    assert info["early_stop"] == bool(expected.done)


@pytest.mark.parametrize(
    "changes", [{"tv_weight": 0.1}, {"upper_bound": 1}, {"nonnegative": False}, {"iterations": 0}]
)
def test_owned_zero_start_does_not_change_other_objectives(changes):
    import importlib

    module = importlib.import_module("tomojax.recon.fista_tv")
    grid, detector = Grid(4, 4, 4, 1, 1, 1), Detector(2, 2, 1, 1)
    geometry = LaminographyGeometry(grid, detector, [0, 90], tilt=27)
    data = jnp.ones((2, 2, 2))
    config = FistaConfig(iterations=1, tv_weight=0, nonnegative=True)
    runtime = module._prepare_fista_runtime(
        geometry,
        grid,
        detector,
        data,
        init_x=None,
        config=replace(config, **changes),
        det_grid=None,
    )
    assert not module.use_owned_fista(grid, data, runtime)


@pytest.mark.numerical
@pytest.mark.parametrize(("iterations", "tolerance"), [(0, 0), (1, 0), (4, 0), (10, 0.99)])
def test_cached_first_step_matches_uncached_recurrence(iterations, tolerance):
    grid, detector = Grid(4, 3, 2, 0.8, 1.1, 1.2), Detector(5, 3, 0.7, 1.1)
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 180, 7), tilt=23)
    data = np.random.default_rng(8).random((7, 3, 5), dtype=np.float32)
    cfg = FistaConfig(
        iterations=iterations,
        tv_weight=0.005,
        views_per_batch=3,
        tolerance=tolerance,
        patience=1,
    )
    cached, info = fista_tv(geometry, grid, detector, data, config=cfg)
    uncached, reference = fista_tv(
        geometry, grid, detector, data, config=replace(cfg, lipschitz=info["lipschitz"])
    )
    np.testing.assert_allclose(cached, uncached, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(info["loss"], reference["loss"], rtol=3e-6, atol=3e-7)
    assert info["effective_iterations"] == reference["effective_iterations"]


def test_public_fista_partial_view_batches_run_under_global_x64():
    grid, detector = Grid(1, 1, 1, 1.0, 1.0, 1.0), Detector(1, 1, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, np.zeros(5, np.float32))
    data = np.full((5, 1, 1), 3.0, np.float32)  # five unit rays through one voxel
    config = FistaConfig(
        iterations=1,
        tv_weight=0,
        backend="jax",
        views_per_batch=2,
        power_iterations=3,
    )
    with jax.enable_x64():
        actual, info = fista_tv(geometry, grid, detector, data, config=config)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, 3.0, rtol=3e-6)
    np.testing.assert_allclose(info["lipschitz"], 5.0, rtol=3e-6)
