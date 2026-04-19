import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fbp import fbp
from tomojax.recon.fista_tv import FistaConfig, fista_tv


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


def test_fbp_default_scaling_recovers_reasonable_absolute_intensity():
    n = 32
    n_views = 32
    grid = Grid(nx=n, ny=n, nz=1, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=n, nv=1, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0, 180, n_views, endpoint=False)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)

    x = np.arange(n, dtype=np.float32) - (n / 2.0 - 0.5)
    y = np.arange(n, dtype=np.float32) - (n / 2.0 - 0.5)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    phantom = np.zeros((n, n, 1), dtype=np.float32)
    phantom[xx**2 + yy**2 < 6.0**2] = 1.0
    vol = jnp.asarray(phantom)

    projs = jnp.stack(
        [forward_project_view(geom, grid, det, vol, view_index=i) for i in range(n_views)],
        axis=0,
    )
    rec = np.asarray(fbp(geom, grid, det, projs, filter_name="ramp"))

    roi = (xx**2 + yy**2) < 3.6**2
    roi_mean = float(rec[roi, 0].mean())
    assert roi_mean == pytest.approx(1.0, rel=0.1, abs=0.1)


def test_fbp_rejects_angle_projection_count_mismatch():
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0, 45.0, 90.0])
    projs = jnp.zeros((2, det.nv, det.nu), dtype=jnp.float32)

    with pytest.raises(
        ValueError,
        match=r"expected .*=\(3, 4, 4\).*actual \(2, 4, 4\).*Likely fix",
    ):
        fbp(geom, grid, det, projs)


def test_fbp_rejects_detector_shape_mismatch():
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0, 45.0, 90.0])
    projs = jnp.zeros((3, det.nv + 1, det.nu), dtype=jnp.float32)

    with pytest.raises(
        ValueError,
        match=r"expected .*=\(3, 4, 4\).*actual \(3, 5, 4\).*Likely fix",
    ):
        fbp(geom, grid, det, projs)


def test_fista_rejects_invalid_reconstruction_grid_shape():
    grid = Grid(nx=0, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0])
    projs = jnp.zeros((1, det.nv, det.nu), dtype=jnp.float32)

    with pytest.raises(
        ValueError,
        match=r"positive integer \(nx, ny, nz\).*actual \(0, 4, 4\).*Likely fix",
    ):
        fista_tv(geom, grid, det, projs, config=FistaConfig(iters=1, L=1.0))


def test_fista_loss_decreases():
    grid, det, geom, vol, projs = make_simple_case(12, 12, 12, 16)
    x, info = fista_tv(
        geom,
        grid,
        det,
        projs,
        config=FistaConfig(iters=5, lambda_tv=0.001),
    )
    loss = info["loss"]
    assert len(loss) == 5
    # Allow some noise, but expect decreasing trend
    assert loss[-1] <= loss[0]


def test_fista_runs_with_huber_tv_regulariser():
    grid, det, geom, _, projs = make_simple_case(6, 6, 6, 6)
    x, info = fista_tv(
        geom,
        grid,
        det,
        projs,
        config=FistaConfig(
            iters=3,
            lambda_tv=0.001,
            regulariser="huber_tv",
            huber_delta=0.1,
            L=100.0,
        ),
    )

    assert np.isfinite(np.asarray(x)).all()
    assert np.isfinite(np.asarray(info["loss"])).all()
    assert len(info["loss"]) == 3
    assert info["regulariser"] == "huber_tv"
    assert info["huber_delta"] == pytest.approx(0.1)


def test_fista_early_stop_triggers():
    grid, det, geom, _, projs = make_simple_case(8, 8, 8, 8)
    zero_projs = jnp.zeros_like(projs)
    callbacks: list[tuple[int, float]] = []

    def record_callback(step: int, loss: float) -> None:
        callbacks.append((step, loss))

    _, info = fista_tv(
        geom,
        grid,
        det,
        zero_projs,
        config=FistaConfig(
            iters=8,
            lambda_tv=0.0,
            recon_rel_tol=1e-6,
            recon_patience=1,
        ),
        callback=record_callback,
    )
    assert info["early_stop"] is True
    assert info["effective_iters"] <= 2
    assert info["effective_iters"] >= 1
    loss = info["loss"]
    assert len(loss) == 8
    last_active = loss[info["effective_iters"] - 1]
    assert loss[-1] == pytest.approx(last_active)
    expected_steps = [0]
    if info["effective_iters"] > 1:
        expected_steps.append(info["effective_iters"] - 1)
    assert [step for step, _ in callbacks] == expected_steps
    assert [value for _, value in callbacks] == pytest.approx([loss[step] for step in expected_steps])
