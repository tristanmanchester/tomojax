import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fbp import fbp
from tomojax.recon.fista_tv import fista_tv


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def make_simple_case(nx=16, ny=16, nz=16, n_views=16):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0, 180, n_views, endpoint=False)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    # Ground truth: small cube in center
    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    vol = vol.at[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4].set(1.0)
    # Projections
    projs = []
    for i in range(n_views):
        p = forward_project_view(geom, grid, det, vol, view_index=i)
        projs.append(p)
    projs = jnp.stack(projs, axis=0)
    return grid, det, geom, vol, projs


def psnr(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mse = np.mean((x - y) ** 2)
    if mse <= 1e-12:
        return 99.0
    maxv = max(x.max(), y.max())
    return 20.0 * np.log10(maxv) - 10.0 * np.log10(mse)


def test_fbp_basic_psnr():
    grid, det, geom, vol, projs = make_simple_case(16, 16, 16, 24)
    rec = fbp(geom, grid, det, projs, filter_name="ramp")
    # Loose threshold for tiny case
    assert psnr(rec, vol) > 10.0


def test_fista_loss_decreases():
    grid, det, geom, vol, projs = make_simple_case(12, 12, 12, 16)
    x, info = fista_tv(geom, grid, det, projs, iters=5, lambda_tv=0.001)
    loss = info["loss"]
    assert len(loss) == 5
    # Allow some noise, but expect decreasing trend
    assert loss[-1] <= loss[0]


def test_fista_early_stop_triggers():
    grid, det, geom, _, projs = make_simple_case(8, 8, 8, 8)
    zero_projs = jnp.zeros_like(projs)
    _, info = fista_tv(
        geom,
        grid,
        det,
        zero_projs,
        iters=8,
        lambda_tv=0.0,
        recon_rel_tol=1e-6,
        recon_patience=1,
    )
    assert info["early_stop"] is True
    assert info["effective_iters"] <= 2
    assert info["effective_iters"] >= 1
    loss = info["loss"]
    assert len(loss) == 8
    last_active = loss[info["effective_iters"] - 1]
    assert loss[-1] == pytest.approx(last_active)
