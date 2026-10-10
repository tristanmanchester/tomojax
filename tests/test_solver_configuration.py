"""Iterative solver controls fail early and explicit step sizes are never discarded."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FistaConfig, SPDHGConfig, fista_tv, spdhg_tv


def _problem(tilted=False):
    grid = Grid(2, 3, 2, 0.8, 1.1, 1.3)
    detector = Detector(4, 3, 0.9, 1.2, (0.17, -0.23))
    angles = np.array([13.0, 68.0, 142.0])
    geometry = (
        LaminographyGeometry(grid, detector, angles, tilt_deg=30)
        if tilted
        else ParallelGeometry(grid, detector, angles)
    )
    data = np.random.default_rng(78).normal(size=(3, 3, 4)).astype(np.float32)
    return grid, detector, geometry, data


@pytest.mark.parametrize("solver", ["fista", "spdhg"])
@pytest.mark.parametrize("field", ["iterations", "views_per_batch", "projector_unroll"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_solvers_reject_invalid_counts_before_building_geometry(monkeypatch, solver, field, value):
    grid, detector, geometry, data = _problem()
    module = importlib.import_module(f"tomojax.recon.{solver}_tv")

    def unexpected_geometry(*args, **kwargs):
        pytest.fail("invalid configuration reached geometry construction")

    monkeypatch.setattr(module, "stack_view_poses", unexpected_geometry)
    config = (FistaConfig if solver == "fista" else SPDHGConfig)(**{field: value})
    with pytest.raises(ValueError, match=field):
        (fista_tv if solver == "fista" else spdhg_tv)(geometry, grid, detector, data, config=config)


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
@pytest.mark.parametrize(
    ("solver", "field"),
    [
        ("fista", "tv_weight"),
        ("fista", "lipschitz"),
        ("fista", "recon_rel_tol"),
        ("spdhg", "tv_weight"),
        ("spdhg", "theta"),
        ("spdhg", "tau"),
        ("spdhg", "sigma_data"),
        ("spdhg", "sigma_tv"),
    ],
)
def test_solvers_reject_invalid_scalars_before_building_geometry(monkeypatch, solver, field, value):
    grid, detector, geometry, data = _problem()
    module = importlib.import_module(f"tomojax.recon.{solver}_tv")

    def unexpected_geometry(*args, **kwargs):
        pytest.fail("invalid configuration reached geometry construction")

    monkeypatch.setattr(module, "stack_view_poses", unexpected_geometry)
    config = (FistaConfig if solver == "fista" else SPDHGConfig)(**{field: value})
    with pytest.raises(ValueError, match=field):
        (fista_tv if solver == "fista" else spdhg_tv)(geometry, grid, detector, data, config=config)


@pytest.mark.parametrize(
    ("solver", "field"),
    [
        ("fista", "views_per_batch"),
        ("fista", "projector_unroll"),
        ("fista", "lipschitz"),
        ("fista", "tv_prox_iterations"),
        ("fista", "power_iterations"),
        ("spdhg", "views_per_batch"),
        ("spdhg", "projector_unroll"),
        ("spdhg", "tau"),
        ("spdhg", "sigma_data"),
        ("spdhg", "sigma_tv"),
    ],
)
def test_solvers_reject_zero_sizes_and_steps(solver, field):
    grid, detector, geometry, data = _problem()
    config = (FistaConfig if solver == "fista" else SPDHGConfig)(**{field: 0})
    with pytest.raises(ValueError, match=field):
        (fista_tv if solver == "fista" else spdhg_tv)(geometry, grid, detector, data, config=config)


@pytest.mark.parametrize("field", ["tv_prox_iterations", "power_iterations", "recon_patience"])
@pytest.mark.parametrize("value", [-1, 1.5])
def test_fista_rejects_invalid_internal_iteration_counts(field, value):
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match=field):
        fista_tv(geometry, grid, detector, data, config=FistaConfig(**{field: value}))


@pytest.mark.parametrize("value", [-1, 1.5])
def test_spdhg_rejects_invalid_logging_interval(value):
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match="log_every"):
        spdhg_tv(geometry, grid, detector, data, config=SPDHGConfig(log_every=value))


def test_fista_rejects_unknown_gradient_mode():
    grid, detector, geometry, data = _problem()
    with pytest.raises(ValueError, match="grad_mode"):
        fista_tv(geometry, grid, detector, data, config=FistaConfig(grad_mode="unknown"))


@pytest.mark.numerical
@pytest.mark.parametrize("solver", ["fista", "spdhg"])
def test_zero_iteration_budget_keeps_initial_volume_and_logs_nothing(solver):
    grid, detector, geometry, data = _problem()
    initial = np.random.default_rng(99).normal(size=(2, 3, 2)).astype(np.float32)
    if solver == "fista":
        config = FistaConfig(iterations=0, tv_weight=0, lipschitz=10, recon_rel_tol=0)
    else:
        config = SPDHGConfig(iterations=0, tv_weight=0, tau=0.1, sigma_data=0.1, log_every=0)
    calls = []
    volume, info = (fista_tv if solver == "fista" else spdhg_tv)(
        geometry,
        grid,
        detector,
        data,
        init_x=initial,
        config=config,
        callback=lambda *args: calls.append(args),
    )
    np.testing.assert_array_equal(volume, initial)
    assert info["loss"] == []
    assert calls == []


@pytest.mark.numerical
@pytest.mark.parametrize("model", ["ray", "joseph"])
@pytest.mark.parametrize("warm_start", [False, True])
def test_fista_auto_step_is_finite_for_zero_operator(model, warm_start):
    grid, detector, geometry, data = _problem()
    initial = np.random.default_rng(99).normal(size=(2, 3, 2)).astype(np.float32)
    volume, info = fista_tv(
        geometry,
        grid,
        detector,
        data,
        init_x=initial if warm_start else None,
        config=FistaConfig(
            iterations=3,
            tv_weight=0,
            support=jnp.zeros((2, 3, 2)),
            projector_model=model,
            projector_backend="jax",
        ),
    )
    np.testing.assert_array_equal(volume, initial if warm_start else np.zeros((2, 3, 2)))
    np.testing.assert_allclose(info["loss"], np.full(3, 0.5 * np.sum(data**2)), rtol=2e-6)
    assert np.isfinite(info["lipschitz"]) and info["lipschitz"] > 0
