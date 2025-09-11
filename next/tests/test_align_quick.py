import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax_next.core.geometry import Grid, Detector, ParallelGeometry
from tomojax_next.core.projector import forward_project_view
from tomojax_next.align.pipeline import align, AlignConfig
from tomojax_next.align.parametrizations import se3_from_5d


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def make_misaligned_case(nx=12, ny=12, nz=12, n_views=8, seed=0):
    rng = np.random.default_rng(seed)
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0, 180, n_views, endpoint=False)
    geom_nom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)

    # Create a simple phantom
    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    vol = vol.at[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4].set(1.0)

    # True per-view small misalignments
    # alpha,beta,phi in radians, dx,dz in pixels
    true_params = np.zeros((n_views, 5), dtype=np.float32)
    true_params[:, 0] = rng.normal(scale=np.deg2rad(0.2), size=n_views)  # alpha
    true_params[:, 1] = rng.normal(scale=np.deg2rad(0.2), size=n_views)  # beta
    true_params[:, 2] = rng.normal(scale=np.deg2rad(0.2), size=n_views)  # phi
    true_params[:, 3] = rng.normal(scale=0.3, size=n_views)  # dx
    true_params[:, 4] = rng.normal(scale=0.3, size=n_views)  # dz

    # Generate projections using augmented pose
    projs = []
    for i in range(n_views):
        class _G:
            def pose_for_view(self, _):
                T_nom = jnp.asarray(geom_nom.pose_for_view(i), dtype=jnp.float32)
                T_al = se3_from_5d(jnp.asarray(true_params[i]))
                return tuple(map(tuple, T_nom @ T_al))

            def rays_for_view(self, _):
                return geom_nom.rays_for_view(i)

        p = forward_project_view(_G(), grid, det, vol, view_index=0)
        projs.append(p)
    projs = jnp.stack(projs, axis=0)

    return grid, det, geom_nom, vol, projs, true_params


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_align_quick_recovers_small_misalignments():
    grid, det, geom, vol, projs, true_params = make_misaligned_case(12, 12, 12, 8, 1)
    x, est_params, info = align(geom, grid, det, projs, cfg=AlignConfig(outer_iters=1, recon_iters=3, lambda_tv=0.001, lr_rot=5e-3, lr_trans=1e-1))

    # Compare rough RMSE (degrees for rotations, pixels for translations)
    rot_rmse_deg = np.rad2deg(rmse(est_params[:, :3], true_params[:, :3]))
    trans_rmse = rmse(est_params[:, 3:], true_params[:, 3:])
    # Loose thresholds for very small run
    assert rot_rmse_deg < 3.0
    assert trans_rmse < 1.5
    # Loss should decrease overall
    assert info["loss"][-1] <= info["loss"][0]
