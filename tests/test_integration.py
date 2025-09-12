import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.data.simulate import SimConfig, simulate
from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fbp import fbp
from tomojax.recon.fista_tv import fista_tv
from tomojax.align.pipeline import align, AlignConfig
from tomojax.align.parametrizations import se3_from_5d


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def psnr(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mse = np.mean((x - y) ** 2)
    if mse <= 1e-12:
        return 99.0
    maxv = max(x.max(), y.max())
    return 20.0 * np.log10(maxv) - 10.0 * np.log10(mse)


def test_integration_parallel_fbp_psnr_from_sim():
    cfg = SimConfig(nx=16, ny=16, nz=16, nu=16, nv=16, n_views=16, phantom="shepp", seed=0)
    data = simulate(cfg)
    grid_d, det_d = data["grid"], data["detector"]
    grid = Grid(nx=grid_d["nx"], ny=grid_d["ny"], nz=grid_d["nz"], vx=grid_d["vx"], vy=grid_d["vy"], vz=grid_d["vz"])
    det = Detector(nu=det_d["nu"], nv=det_d["nv"], du=det_d["du"], dv=det_d["dv"], det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=data["thetas_deg"])
    rec = fbp(geom, grid, det, jnp.asarray(data["projections"]))
    assert psnr(rec, data["volume"]) > 10.0


def test_integration_parallel_fista_decreases_from_sim():
    cfg = SimConfig(nx=12, ny=12, nz=12, nu=12, nv=12, n_views=12, phantom="blobs", seed=1)
    data = simulate(cfg)
    grid_d, det_d = data["grid"], data["detector"]
    grid = Grid(nx=grid_d["nx"], ny=grid_d["ny"], nz=grid_d["nz"], vx=grid_d["vx"], vy=grid_d["vy"], vz=grid_d["vz"])
    det = Detector(nu=det_d["nu"], nv=det_d["nv"], du=det_d["du"], dv=det_d["dv"], det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=data["thetas_deg"])
    x, info = fista_tv(geom, grid, det, jnp.asarray(data["projections"]), iters=5, lambda_tv=0.001)
    assert info["loss"][0] >= info["loss"][-1]


def test_integration_lamino_fista_decreases():
    cfg = SimConfig(nx=12, ny=12, nz=12, nu=12, nv=12, n_views=12, phantom="cube", geometry="lamino", tilt_deg=30.0, seed=2)
    data = simulate(cfg)
    grid_d, det_d = data["grid"], data["detector"]
    grid = Grid(nx=grid_d["nx"], ny=grid_d["ny"], nz=grid_d["nz"], vx=grid_d["vx"], vy=grid_d["vy"], vz=grid_d["vz"])
    det = Detector(nu=det_d["nu"], nv=det_d["nv"], du=det_d["du"], dv=det_d["dv"], det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    from tomojax.core.geometry import LaminographyGeometry
    geom = LaminographyGeometry(grid=grid, detector=det, thetas_deg=data["thetas_deg"], tilt_deg=30.0, tilt_about="x")
    x, info = fista_tv(geom, grid, det, jnp.asarray(data["projections"]), iters=4, lambda_tv=0.001)
    assert info["loss"][0] >= info["loss"][-1]


def test_integration_align_parallel_improves_rmse():
    # Generate clean dataset first
    cfg = SimConfig(nx=12, ny=12, nz=12, nu=12, nv=12, n_views=8, phantom="cube", seed=3)
    data = simulate(cfg)
    grid_d, det_d = data["grid"], data["detector"]
    grid = Grid(nx=grid_d["nx"], ny=grid_d["ny"], nz=grid_d["nz"], vx=grid_d["vx"], vy=grid_d["vy"], vz=grid_d["vz"])
    det = Detector(nu=det_d["nu"], nv=det_d["nv"], du=det_d["du"], dv=det_d["dv"], det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    geom_nom = ParallelGeometry(grid=grid, detector=det, thetas_deg=data["thetas_deg"]) 

    # Apply small random per-view misalignments to generate projections
    rng = np.random.default_rng(0)
    true_params = np.zeros((cfg.n_views, 5), dtype=np.float32)
    true_params[:, :3] = np.deg2rad(rng.normal(scale=0.2, size=(cfg.n_views, 3)))
    true_params[:, 3:] = rng.normal(scale=0.3, size=(cfg.n_views, 2))

    projs = []
    for i in range(cfg.n_views):
        class _G:
            def pose_for_view(self, _):
                T_nom = jnp.asarray(geom_nom.pose_for_view(i), dtype=jnp.float32)
                T_al = se3_from_5d(jnp.asarray(true_params[i]))
                return tuple(map(tuple, T_nom @ T_al))
            def rays_for_view(self, _):
                return geom_nom.rays_for_view(i)
        projs.append(forward_project_view(_G(), grid, det, jnp.asarray(data["volume"]), view_index=0))
    projs = jnp.stack(projs, axis=0)

    # Run quick alignment
    x, est_params, info = align(geom_nom, grid, det, projs, cfg=AlignConfig(outer_iters=1, recon_iters=3, lambda_tv=0.001, lr_rot=5e-3, lr_trans=1e-1))
    # Compute RMSEs
    rot_rmse_deg = np.rad2deg(np.sqrt(np.mean((est_params[:, :3] - true_params[:, :3]) ** 2)))
    trans_rmse = float(np.sqrt(np.mean((est_params[:, 3:] - true_params[:, 3:]) ** 2)))
    assert info["loss"][0] >= info["loss"][-1]
    assert rot_rmse_deg < 3.5
    assert trans_rmse < 1.8
