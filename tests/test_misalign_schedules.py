import os
import sys
import json
import numpy as np
import jax.numpy as jnp
import pytest

from tomojax.core.geometry import Grid, Detector
from tomojax.data.io_hdf5 import save_nxtomo, load_nxtomo


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def _make_gt(path, nx=16, ny=16, nz=16, n_views=8):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0.0, 180.0, n_views, endpoint=False)
    # Simple phantom: centered cube
    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    vol = vol.at[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4].set(1.0)
    save_nxtomo(
        path,
        projections=np.zeros((n_views, nz, nx), dtype=np.float32),  # placeholder; misalign will forward project
        thetas_deg=thetas,
        grid=grid,
        detector=det,
        volume=np.asarray(vol),
        geometry_type="parallel",
    )
    return grid, det, thetas


def _run_cli(tmp_path, in_path, out_path, args):
    # Invoke CLI main via module import to avoid subprocess overhead
    from tomojax.cli import misalign as cli
    argv = ["misalign", "--data", in_path, "--out", out_path] + args
    sys_argv_backup = sys.argv[:]
    try:
        sys.argv = argv
        cli.main()
    finally:
        sys.argv = sys_argv_backup


def test_angle_linear_updates_thetas(tmp_path):
    in_path = os.path.join(tmp_path, "gt.nxs")
    out_path = os.path.join(tmp_path, "out_angle_lin.nxs")
    _grid, _det, thetas = _make_gt(in_path, n_views=8)

    _run_cli(tmp_path, in_path, out_path, ["--pert", "angle:linear:delta=5deg"]) 
    data = load_nxtomo(out_path)
    th_out = np.asarray(data["thetas_deg"]).astype(np.float32)
    off = th_out - thetas
    # Expect roughly a 0->5 deg ramp across views
    assert off.min() == pytest.approx(0.0, abs=1e-3)
    assert off.max() == pytest.approx(5.0, abs=5e-2)
    # No random unless requested
    ap = np.asarray(data.get("align_params"))
    assert ap is not None
    assert np.allclose(ap, 0.0, atol=1e-1)


def test_dx_sin_window_peak(tmp_path):
    in_path = os.path.join(tmp_path, "gt.nxs")
    out_path = os.path.join(tmp_path, "out_dx_sin.nxs")
    _grid, det, thetas = _make_gt(in_path, n_views=8)

    _run_cli(tmp_path, in_path, out_path, ["--pert", "dx:sin-window:amp=5px"]) 
    data = load_nxtomo(out_path)
    ap = np.asarray(data["align_params"]).astype(np.float32)
    # dx is column 3, in world units (du=1)
    dx = ap[:, 3]
    assert dx.shape[0] == thetas.shape[0]
    # Peak approximately 5 (sin-window) in pixels * du
    assert dx.max() == pytest.approx(5.0 * float(det.du), abs=0.5)
    # Rotational params should be near zero without randomness
    assert np.allclose(ap[:, :3], 0.0, atol=1e-1)


def test_dx_step_absolute_hold(tmp_path):
    in_path = os.path.join(tmp_path, "gt.nxs")
    out_path = os.path.join(tmp_path, "out_dx_step.nxs")
    _grid, det, thetas = _make_gt(in_path, n_views=8)

    _run_cli(tmp_path, in_path, out_path, ["--pert", "dx:step:at=90deg,to=5px"]) 
    data = load_nxtomo(out_path)
    ap = np.asarray(data["align_params"]).astype(np.float32)
    dx = ap[:, 3]
    # Find index closest to 90 deg
    at = int(np.argmin(np.abs(thetas - 90.0)))
    # Before step ~0, after step ~5px*du
    assert np.allclose(dx[:at], 0.0, atol=0.25)
    assert np.allclose(dx[at:], 5.0 * float(det.du), atol=0.25)

