"""Verify that coarse initialization still solves the original fine system."""

from __future__ import annotations

import importlib
import os

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
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_multires_recovers_original_fine_system(explicit_coordinates, backend):
    if backend == "cuda" and (jax.default_backend() != "gpu" or explicit_coordinates):
        pytest.skip("CUDA requires CUDA and canonical coordinates")
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
        iterations_per_level=(4, 6, 90),
        init_x=jnp.ones_like(truth) * 0.5,
        config=CGLSConfig(
            projector_model="ray", rtol=1e-7, views_per_batch=3, projector_backend=backend
        ),
        det_grid=coordinates,
    )
    np.testing.assert_allclose(result, truth, rtol=3e-4, atol=5e-5)
    assert info["requested_iterations"] == 100
    assert info["effective_iterations"] == sum(
        level["effective_iterations"] for level in info["levels"]
    )
    assert [level["factor"] for level in info["levels"]] == [3, 2, 1]
    assert info["levels"][-1]["grid"] == grid.to_dict()
    assert info["levels"][-1]["detector"] == detector.to_dict()


def test_single_level_is_the_unmodified_public_solver():
    grid, detector, geometry, initial = problem()
    data = jnp.ones((5, 7, 9))
    config = CGLSConfig(projector_model="ray", iterations=4, projector_backend="jax")
    direct, direct_info = cgls(geometry, grid, detector, data, init_x=initial, config=config)
    result, info = cgls_multires(
        geometry,
        grid,
        detector,
        data,
        factors=(1,),
        iterations_per_level=(4,),
        init_x=initial,
        config=config,
    )
    # Both JAX scatter and CUDA atomics may sum in a different GPU order.
    np.testing.assert_allclose(result, direct, rtol=1e-5, atol=3e-6)
    assert info["fine_effective_iterations"] == direct_info["effective_iterations"]
    assert info["normal_residual_norm"] == pytest.approx(
        direct_info["normal_residual_norm"], rel=1e-5
    )


def test_multires_gradient_penalty_solves_the_same_physical_fine_objective():
    grid, detector, geometry, initial = problem()
    data = jnp.asarray(np.random.default_rng(82).normal(size=(5, 7, 9)), jnp.float32)
    config = CGLSConfig(
        projector_model="ray",
        iterations=100,
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
        iterations_per_level=(6, 10, 100),
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
            geometry,
            grid,
            detector,
            jnp.zeros((5, 7, 9)),
            factors=factors,
            iterations_per_level=budgets,
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


@pytest.mark.parametrize("storage", ["array", "memmap"])
def test_host_pyramid_stays_on_the_host_and_streams(storage, tmp_path):
    grid, detector, geometry, truth = problem()
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)])
    data = np.asarray(jax.vmap(lambda t: forward_project_view_T(t, grid, detector, truth))(poses))
    if storage == "memmap":
        mapped = np.memmap(tmp_path / "views.dat", mode="w+", dtype=np.float32, shape=data.shape)
        mapped[:] = data
        data = mapped
    result, info = cgls_multires(
        geometry,
        grid,
        detector,
        data,
        factors=(3, 1),
        iterations_per_level=(6, 120),
        config=CGLSConfig(
            projector_model="ray",
            projector_backend="jax",
            stream_projections=True,
            views_per_batch=3,
            rtol=1e-7,
        ),
    )
    np.testing.assert_allclose(result, truth, rtol=3e-4, atol=5e-5)
    assert all(level["formulation"] == "streamed_normal_equations" for level in info["levels"])


# Its child process needs CUDA, which the suite's own JAX would otherwise hold.
@pytest.mark.gpu
@pytest.mark.fresh_gpu
def test_host_multires_avoids_jax_and_matches_the_device_workflow():
    import subprocess
    import sys

    code = """
import sys
import numpy as np
from tomojax.geometry import Grid, Detector, LaminographyGeometry
from tomojax.recon import CGLSConfig, cgls_multires
rng = np.random.default_rng(27)
grid, det = Grid(9, 7, 5, .8, 1.2, 1.1), Detector(13, 11, .7, .9, (.2, -.1))
geom = LaminographyGeometry(grid, det, [13, 39, 77, 111, 153], 31)
data = rng.normal(size=(5, 11, 13)).astype(np.float32)
kwargs = dict(factors=(3, 1), iterations_per_level=(4, 7),
              config=CGLSConfig(rtol=0, damping=.1, projector_backend='cuda'))
host, info = cgls_multires(geom, grid, det, data, **kwargs)
assert 'jax' not in sys.modules and 'cupyx.scipy.sparse' not in sys.modules
assert isinstance(host, np.ndarray) and info['execution_backend'] == 'cupy'
import jax.numpy as jnp
device, reference = cgls_multires(geom, grid, det, jnp.asarray(data), **kwargs)
np.testing.assert_allclose(host, device, atol=3e-5, rtol=3e-4)
assert info['effective_iterations'] == reference['effective_iterations']
"""
    # The test process may already hold the GPU: the child must not preallocate.
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize("strided", [False, True])
def test_host_multires_rejects_nonfinite_unsampled_rays(strided):
    grid, detector, geometry, _ = problem()
    data = np.ones((5, 7, 9), np.float32)
    if strided:
        data = data[:, :, ::-1]
    data[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cgls_multires(geometry, grid, detector, data, factors=(3, 1), iterations_per_level=(1, 1))
