import sys
import numpy as np
import jax
import jax.numpy as jnp

from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.recon.fista_tv import grad_data_term


if sys.version_info < (3, 8):
    raise SystemExit


def make_case(n_views=5, nx=12, ny=12, nz=12):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = list(np.linspace(0.0, 180.0, n_views, endpoint=False))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    # random small volume and projections
    key = jax.random.PRNGKey(0)
    vol = jax.random.normal(key, (nx, ny, nz), dtype=jnp.float32) * 0.1
    # Generate projections from this volume for consistency
    # Build them by calling projector through grad_data_term's path (batched)
    # so that both modes see identical y.
    from tomojax.core.projector import forward_project_view
    projs = []
    for i in range(n_views):
        projs.append(forward_project_view(geom, grid, det, vol, view_index=i))
    projs = jnp.stack(projs, axis=0)
    return grid, det, geom, vol, projs


def test_stream_vs_batched_grad_close():
    grid, det, geom, vol, projs = make_case(n_views=4, nx=10, ny=10, nz=10)
    g_b, loss_b = grad_data_term(
        geom,
        grid,
        det,
        projs,
        vol,
        views_per_batch=4,
        checkpoint_projector=True,
        projector_unroll=1,
        gather_dtype="fp32",
        grad_mode="batched",
    )
    g_s, loss_s = grad_data_term(
        geom,
        grid,
        det,
        projs,
        vol,
        views_per_batch=1,  # force streaming via auto
        checkpoint_projector=True,
        projector_unroll=1,
        gather_dtype="fp32",
        grad_mode="stream",
    )
    # Gradients should match closely
    num = jnp.linalg.norm((g_b - g_s).ravel())
    den = jnp.linalg.norm(g_b.ravel()) + 1e-6
    rel = float(num / den)
    assert rel < 5e-3
    # Loss should be effectively identical
    assert float(jnp.abs(loss_b - loss_s)) < 1e-4

