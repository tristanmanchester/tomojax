"""Verify that coarse initialization still solves the original fine system."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry
from tomojax.recon import CGLSConfig, cgls, cgls_multires


def problem():
    grid = Grid(5, 3, 3, 0.8, 1.1, 1.3, vol_origin=(-1.7, -0.9, -1.1))
    detector = Detector(9, 7, 0.7, 1.2, (0.17, -0.23))
    geometry = LaminographyGeometry(grid, detector, np.asarray([13, 49, 83, 122, 161]), tilt_deg=23)
    truth = np.random.default_rng(12).uniform(-0.3, 1.7, size=(5, 3, 3)).astype(np.float32)
    return grid, detector, geometry, truth


@pytest.mark.parametrize("explicit_coordinates", [False, True])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_multires_recovers_original_fine_system(explicit_coordinates, backend):
    if backend == "pallas" and (jax.default_backend() != "gpu" or explicit_coordinates):
        pytest.skip("Pallas requires CUDA and canonical coordinates")
    grid, detector, geometry, truth = problem()
    coordinates = None
    if explicit_coordinates:
        u, v = np.meshgrid(
            (np.arange(9) - 4) * 0.7 + 0.17,
            (np.arange(7) - 3) * 1.2 - 0.23,
        )
        # A slightly nonuniform detector exercises coordinates outside the
        # canonical metadata, with exactly the same subset on coarse levels.
        coordinates = (jnp.asarray(u + 0.04 * np.sin(v)).ravel(), jnp.asarray(v).ravel())
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)])
    data = jax.vmap(
        lambda t: forward_project_view_T(t, grid, detector, truth, det_grid=coordinates)
    )(poses)
    result, info = cgls_multires(
        geometry,
        grid,
        detector,
        data,
        factors=(3, 2, 1),
        iters_per_level=(4, 6, 90),
        init_x=jnp.ones_like(truth) * 0.5,
        config=CGLSConfig(
            projector_model="ray", rtol=1e-7, views_per_batch=3, projector_backend=backend
        ),
        det_grid=coordinates,
    )
    np.testing.assert_allclose(result, truth, rtol=3e-4, atol=5e-5)
    assert info["requested_iters"] == 100
    assert info["effective_iters"] == sum(level["effective_iters"] for level in info["levels"])
    assert [level["factor"] for level in info["levels"]] == [3, 2, 1]
    assert info["levels"][-1]["grid"] == grid.to_dict()
    assert info["levels"][-1]["detector"] == detector.to_dict()


def test_single_level_is_the_unmodified_public_solver():
    grid, detector, geometry, initial = problem()
    data = jnp.ones((5, 7, 9))
    config = CGLSConfig(projector_model="ray", iters=4, projector_backend="jax")
    direct, direct_info = cgls(geometry, grid, detector, data, init_x=initial, config=config)
    result, info = cgls_multires(
        geometry,
        grid,
        detector,
        data,
        factors=(1,),
        iters_per_level=(4,),
        init_x=initial,
        config=config,
    )
    # Both JAX scatter and Pallas atomics may sum in a different GPU order.
    np.testing.assert_allclose(result, direct, rtol=1e-5, atol=3e-6)
    assert info["fine_effective_iters"] == direct_info["effective_iters"]
    assert info["normal_residual_norm"] == pytest.approx(
        direct_info["normal_residual_norm"], rel=1e-5
    )


def test_multires_gradient_penalty_solves_the_same_physical_fine_objective():
    grid, detector, geometry, initial = problem()
    data = jnp.asarray(np.random.default_rng(82).normal(size=(5, 7, 9)), jnp.float32)
    config = CGLSConfig(
        projector_model="ray",
        iters=100,
        rtol=1e-7,
        damping=0.2,
        gradient_damping=0.7,
        projector_backend="jax",
    )
    expected, _ = cgls(geometry, grid, detector, data, config=config)
    actual, info = cgls_multires(
        geometry,
        grid,
        detector,
        data,
        factors=(3, 2, 1),
        iters_per_level=(6, 10, 100),
        init_x=initial,
        config=config,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
    assert all(level["gradient_damping"] == 0.7 for level in info["levels"])


@pytest.mark.parametrize(
    ("factors", "budgets"),
    [
        ([], []),
        ([2, 1], [4]),
        ([2], [4]),
        ([1, 2, 1], [4] * 3),
        ([2, 2, 1], [4] * 3),
        ([2, 1], [0, 4]),
    ],
)
def test_invalid_schedule_is_rejected_before_any_solve(factors, budgets, monkeypatch):
    module = importlib.import_module("tomojax.recon.cgls_multires")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid schedules must not start reconstruction")

    monkeypatch.setattr(module, "cgls", unexpected)
    grid, detector, geometry, _ = problem()
    with pytest.raises(ValueError):
        cgls_multires(
            geometry, grid, detector, jnp.zeros((5, 7, 9)), factors=factors, iters_per_level=budgets
        )


def test_breakdown_at_coarse_level_cannot_be_hidden_by_fine_solve(monkeypatch):
    module = importlib.import_module("tomojax.recon.cgls_multires")
    calls = []

    def fail(geometry, grid, *args, **kwargs):
        calls.append(grid)
        return jnp.zeros((grid.nx, grid.ny, grid.nz)), {"termination": "numerical_breakdown"}

    monkeypatch.setattr(module, "cgls", fail)
    grid, detector, geometry, _ = problem()
    with pytest.raises(FloatingPointError, match="factor 2"):
        cgls_multires(geometry, grid, detector, jnp.zeros((5, 7, 9)))
    assert len(calls) == 1
