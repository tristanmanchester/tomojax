import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.spdhg_tv import spdhg_tv, SPDHGConfig
from tomojax.data.simulate import SimConfig, simulate


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def make_simple_case(nx=12, ny=12, nz=12, n_views=12):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0, 180, n_views, endpoint=False)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    vol = vol.at[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4].set(1.0)
    projs = []
    for i in range(n_views):
        p = forward_project_view(geom, grid, det, vol, view_index=i)
        projs.append(p)
    projs = jnp.stack(projs, axis=0)
    return grid, det, geom, vol, projs


def test_spdhg_loss_decreases_basic():
    grid, det, geom, vol, projs = make_simple_case(12, 12, 12, 12)
    cfg = SPDHGConfig(iters=5, lambda_tv=1e-3, views_per_batch=4, log_every=1, seed=0)
    x, info = spdhg_tv(geom, grid, det, projs, config=cfg)
    loss = info["loss"]
    assert len(loss) == 5
    assert np.isfinite(loss[0]) and np.isfinite(loss[-1])
    assert loss[-1] <= loss[0]


def test_spdhg_from_sim_decreases():
    cfg_sim = SimConfig(nx=12, ny=12, nz=12, nu=12, nv=12, n_views=12, phantom="blobs", seed=123)
    data = simulate(cfg_sim)
    grid_d, det_d = data["grid"], data["detector"]
    grid = Grid(nx=grid_d["nx"], ny=grid_d["ny"], nz=grid_d["nz"], vx=grid_d["vx"], vy=grid_d["vy"], vz=grid_d["vz"])
    det = Detector(nu=det_d["nu"], nv=det_d["nv"], du=det_d["du"], dv=det_d["dv"], det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=data["thetas_deg"])
    x, info = spdhg_tv(geom, grid, det, jnp.asarray(data["projections"]), config=SPDHGConfig(iters=4, lambda_tv=1e-3, views_per_batch=4, log_every=1))
    loss = info["loss"]
    assert len(loss) == 4
    assert loss[-1] <= loss[0]


def test_spdhg_keeps_configured_data_dual_step_under_block_sampling():
    grid, det, geom, vol, projs = make_simple_case(8, 8, 8, 12)
    cfg = SPDHGConfig(
        iters=1,
        lambda_tv=1e-3,
        views_per_batch=3,
        tau=0.1,
        sigma_data=0.2,
        sigma_tv=0.1,
        log_every=1,
        seed=0,
    )
    _, info = spdhg_tv(geom, grid, det, projs, config=cfg)
    assert info["num_blocks"] == 4
    assert info["selection_prob"] == pytest.approx(0.25)
    assert info["sigma_data_base"] == pytest.approx(0.2)
    assert info["sigma_data"] == pytest.approx(0.2)


def test_spdhg_callback_exceptions_propagate():
    grid, det, geom, vol, projs = make_simple_case(8, 8, 8, 4)
    cfg = SPDHGConfig(iters=1, lambda_tv=1e-3, views_per_batch=2, log_every=1, seed=0)

    def fail_callback(step: int, loss: float) -> None:
        raise RuntimeError("callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        spdhg_tv(geom, grid, det, projs, config=cfg, callback=fail_callback)


def test_spdhg_callback_reports_first_and_last_logged_losses():
    grid, det, geom, vol, projs = make_simple_case(8, 8, 8, 8)
    cfg = SPDHGConfig(iters=5, lambda_tv=1e-3, views_per_batch=2, log_every=2, seed=0)
    callbacks: list[tuple[int, float]] = []

    def record_callback(step: int, loss: float) -> None:
        callbacks.append((step, loss))

    _, info = spdhg_tv(geom, grid, det, projs, config=cfg, callback=record_callback)

    expected_steps = [1, 3]
    assert [step for step, _ in callbacks] == expected_steps
    assert [value for _, value in callbacks] == pytest.approx(
        [info["loss"][step] for step in expected_steps]
    )
