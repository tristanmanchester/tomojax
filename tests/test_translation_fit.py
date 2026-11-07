"""The joint volume and detector-translation fit behind automatic tj.align."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._translations import EdgeFilter, MotionPrior, fit_translations
from tomojax.core.joseph import forward_project_planes, plane_coefficients, plane_weighted_diagonal
from tomojax.geometry import stack_view_poses


def _laminography(nx=5, ny=4, nz=3, nu=7, nv=6, views=4, tilt=-30.0):
    grid = tj.Grid(nx, ny, nz, 1.0, 1.1, 0.9)
    detector = tj.Detector(nu, nv, 1.0, 1.0)
    angles = np.linspace(0, 360, views, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=tilt)
    return grid, detector, geometry, angles


@pytest.mark.numerical
def test_weighted_diagonal_is_the_explicit_matrix_diagonal():
    grid, detector, geometry, angles = _laminography()
    poses = jnp.asarray(stack_view_poses(geometry, len(angles)))
    cf = plane_coefficients(poses, grid, detector)
    rng = np.random.default_rng(3)
    sw = jnp.asarray(rng.uniform(0.2, 1.5, (len(angles), detector.nv, detector.nu)), jnp.float32)
    columns = []
    for k in range(grid.nx * grid.ny * grid.nz):
        unit = jnp.zeros(grid.nx * grid.ny * grid.nz, jnp.float32).at[k].set(1.0)
        volume = unit.reshape(grid.nz, grid.ny, grid.nx)
        columns.append(forward_project_planes(cf, volume, grid, detector, backend="jax"))
    matrix = jnp.stack(columns, -1)
    expected = jnp.sum((sw[..., None] * matrix) ** 2, axis=(0, 1, 2)).reshape(
        grid.nz, grid.ny, grid.nx
    )
    zeros = jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32)
    got = plane_weighted_diagonal(cf, sw, zeros, grid, detector, backend="jax")
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_weighted_diagonal_gather_matches_the_scatter(dtype):
    grid, detector, geometry, angles = _laminography(nx=19, ny=17, nz=13, nu=23, nv=21, views=7)
    cf = plane_coefficients(jnp.asarray(stack_view_poses(geometry, len(angles))), grid, detector)
    sw = jnp.asarray(np.random.default_rng(4).uniform(0, 1, (len(angles), 21, 23)), jnp.float32)
    start = jnp.full((grid.nz, grid.ny, grid.nx), 0.25, jnp.float32)
    with jax.enable_x64():
        sw, start = sw.astype(dtype), start.astype(dtype)
        cuda = plane_weighted_diagonal(cf, sw, start, grid, detector, backend="cuda")
        scatter = plane_weighted_diagonal(cf, sw, start, grid, detector, backend="jax")
    assert cuda.dtype == jnp.float32
    np.testing.assert_allclose(cuda, scatter, rtol=1e-4, atol=1e-6)


def _dense_constrained(h, constraints, rhs):
    k = constraints.shape[1]
    kkt = np.block([[h, constraints], [constraints.T, np.zeros((k, k))]])
    return np.linalg.solve(kkt, np.concatenate([rhs, np.zeros(k)]))[: len(rhs)]


def test_global_pose_solves_match_dense_constrained_systems():
    views = 37
    rng = np.random.default_rng(5)
    gauge = rng.normal(size=(2 * views, 3))
    with jax.enable_x64(True):
        prior = MotionPrior(views, gauge, knot_spacing=10, delta=0.02)
        a = rng.normal(size=(views, 2, 2))
        blocks = np.einsum("nij,nkj->nik", a, a) + 0.5 * np.eye(2)
        w = rng.uniform(0.1, 2.0, views - 1)
        rhs = rng.normal(size=2 * views)
        f = np.zeros((2 * views, 2 * views))
        for i in range(views):
            f[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = blocks[i]
        diff = np.kron(np.diff(np.eye(views), axis=0), np.eye(2))
        laplacian = diff.T @ np.diag(np.repeat(w, 2)) @ diff
        u, g, c = (np.asarray(m) for m in (prior.U, prior.G, prior.C))
        q = np.eye(2 * views) - u @ u.T
        expected = _dense_constrained(f + q @ laplacian @ q, g, rhs)
        got = prior.solve(prior.factor(jnp.asarray(blocks), jnp.asarray(w)), jnp.asarray(rhs))
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-9)
        expected = _dense_constrained(f + laplacian, c, rhs)
        got = prior.innovation_solve(jnp.asarray(blocks), jnp.asarray(w), jnp.asarray(rhs))
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-9)


def test_edge_filter_transpose_is_exact_with_masks_and_odd_sizes():
    rng = np.random.default_rng(6)
    shape = (3, 13, 17)
    precision = rng.uniform(0, 1, shape).astype(np.float32)
    precision[precision < 0.2] = 0
    precision[1, 4] = 0  # an unobserved row
    a, z = (jnp.asarray(rng.normal(size=shape), jnp.float32) for _ in range(2))
    h = EdgeFilter((0.5, 0.75))
    left = float(jnp.vdot(h.forward(a, precision), z))
    right = float(jnp.vdot(a, h.transpose(z, precision)))
    assert abs(left - right) <= 2e-5 * max(abs(left), 1.0)


def _moving_scan(slip):
    grid = tj.Grid(28, 28, 20, 1.0, 1.0, 1.0)
    detector = tj.Detector(40, 32, 1.0, 1.0)
    views = 90
    geometry = tj.LaminographyGeometry(
        grid, detector, np.linspace(0, 360, views, endpoint=False), tilt_deg=-30
    )
    z, y, x = np.mgrid[:20, :28, :28]
    volume = np.zeros((20, 28, 28), np.float32)
    for cz, cy, cx, r, mu in ((10, 10, 9, 5, 1.0), (8, 17, 18, 4, 0.6), (13, 19, 8, 3, 1.4)):
        volume[(z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r * r] = mu
    shifts = np.zeros((views, 2))
    shifts[45:, 0] = slip
    pars = np.zeros((views, 6))
    pars[:, 3:5] = shifts
    posed = tj.Scan(np.zeros((views, 32, 40), np.float32), geometry).with_poses(pars).geometry
    projections = np.array(tj.project(posed, volume))
    projections += np.random.default_rng(7).normal(0, 0.01, projections.shape).astype(np.float32)
    return projections, np.asarray(stack_view_poses(geometry, views)), grid, detector


@pytest.mark.numerical
@pytest.mark.slow
def test_translation_fit_recovers_a_slip_and_keeps_a_clean_scan_still():
    projections, nominal, grid, detector = _moving_scan(slip=1.5)
    _, _, shifts, record = fit_translations(projections, nominal, grid, detector)
    jumps = np.abs(np.diff(shifts[:, 0]))
    assert abs(jumps[44] - 1.5) < 0.15, record["levels"][-1]["status"]
    assert np.median(np.delete(jumps, 44)) < 0.05
    projections, nominal, grid, detector = _moving_scan(slip=0.0)
    _, _, shifts, _ = fit_translations(projections, nominal, grid, detector)
    assert np.sqrt(np.mean(shifts**2)) < 0.05
