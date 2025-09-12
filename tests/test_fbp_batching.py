import numpy as np
import jax.numpy as jnp
from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fbp import fbp


def make_case(nx=12, ny=12, nz=12, n_views=12):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0, 180, n_views, endpoint=False)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32).at[
        nx // 4 : 3 * nx // 4, ny // 4 : 3 * ny // 4, nz // 4 : 3 * nz // 4
    ].set(1.0)
    projs = jnp.stack(
        [
            forward_project_view(geom, grid, det, vol, view_index=i)
            for i in range(n_views)
        ],
        axis=0,
    )
    return grid, det, geom, vol, projs


def test_fbp_batch_equivalence():
    grid, det, geom, vol, projs = make_case(12, 12, 12, 12)
    rec_all = fbp(geom, grid, det, projs, filter_name="ramp", views_per_batch=0)
    rec_b3 = fbp(
        geom,
        grid,
        det,
        projs,
        filter_name="ramp",
        views_per_batch=3,
    )
    assert np.allclose(np.asarray(rec_all), np.asarray(rec_b3), atol=1e-3)

