import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fista_tv import fista_tv, grad_data_term
from tomojax.recon.multires import fista_multires


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def make_case(nx=16, ny=16, nz=16, n_views=16):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0, 180, n_views, endpoint=False)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    vol = vol.at[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4].set(1.0)
    projs = []
    for i in range(n_views):
        projs.append(forward_project_view(geom, grid, det, vol, view_index=i))
    projs = jnp.stack(projs, axis=0)
    return grid, det, geom, vol, projs


def test_multires_beats_single_level():
    grid, det, geom, vol, projs = make_case(12, 12, 12, 12)
    # Single-level
    x_single, info_single = fista_tv(geom, grid, det, projs, iters=6, lambda_tv=0.001)
    g_single, loss_single = grad_data_term(geom, grid, det, projs, x_single)

    # Two-level (2 -> 1) with same total iters
    x_multi, info_multi = fista_multires(geom, grid, det, projs, factors=(2, 1), iters_per_level=(3, 3), lambda_tv=0.001)
    g_multi, loss_multi = grad_data_term(geom, grid, det, projs, x_multi)

    assert loss_multi <= loss_single + 1e-3
