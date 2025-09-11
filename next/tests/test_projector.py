import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax_next.core.geometry import Grid, Detector, ParallelGeometry
from tomojax_next.core.projector import forward_project_view
from tomojax_next.core.operators import adjoint_test_once


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def make_aligned_case(nx=16, ny=16, nz=16):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    # Detector aligned and covering FOV exactly
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = [0.0]
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    # Volume of ones -> line integral equals ny * vy
    vol = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    return grid, det, geom, vol


def test_forward_project_uniform_volume_returns_path_length():
    grid, det, geom, vol = make_aligned_case(16, 16, 16)
    proj = forward_project_view(geom, grid, det, vol, view_index=0)
    expected = grid.ny * grid.vy
    assert np.allclose(np.asarray(proj), expected, atol=1e-4)


def test_adjoint_small_case():
    grid, det, geom, vol = make_aligned_case(8, 8, 8)
    y_like = jnp.ones((det.nv, det.nu), dtype=jnp.float32)
    rel = adjoint_test_once(geom, grid, det, vol, y_like, view_index=0)
    assert rel < 5e-3

