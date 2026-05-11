from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tomojax.align import AlignConfig, align

# check-public-imports: allow-private
import tomojax.align._pose_stage as pose_stage_module

# check-public-imports: allow-private
import tomojax.align._reconstruction_stage as reconstruction_stage_module

# check-public-imports: allow-private
from tomojax.align._reconstruction_stage import _run_reconstruction_step
from tomojax.align.geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state
from tomojax.align.model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view


def _case(size: int = 5, n_views: int = 3):
    grid = Grid(nx=size, ny=size, nz=size, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=size, nv=size, du=1.0, dv=1.0)
    thetas = np.linspace(0.0, 180.0, n_views, endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)
    volume = jnp.zeros((size, size, size), dtype=jnp.float32)
    volume = volume.at[1:4, 1:4, 1:4].set(1.0)
    projections = jnp.stack(
        [
            forward_project_view(
                geometry,
                grid,
                detector,
                volume,
                i,
                gather_dtype="fp32",
            )
            for i in range(n_views)
        ],
        axis=0,
    )
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(n_views))
    effective = apply_alignment_state(BaseGeometryArrays.from_geometry(geometry, detector), state)
    return grid, detector, geometry, volume, projections, effective.det_grid


def test_huber_fista_core_nonfinite_output_retries_public_streamed_fista(monkeypatch):
    grid, detector, geometry, _volume, projections, det_grid = _case()
    x0 = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    def fake_core_jit(x_in, *_args, **_kwargs):
        return (
            jnp.full_like(x_in, jnp.nan),
            jnp.asarray([jnp.nan], dtype=jnp.float32),
            jnp.asarray(jnp.nan, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
        )

    def fake_public_fista(*_args, init_x, **_kwargs):
        return (
            jnp.ones_like(init_x) * jnp.float32(0.25),
            {
                "loss": [3.0, 2.0],
                "L": 123.0,
                "effective_iters": 2,
                "early_stop": False,
                "regulariser": "huber_tv",
                "huber_delta": 0.01,
            },
        )

    monkeypatch.setattr(reconstruction_stage_module, "_run_huber_fista_core_jit", fake_core_jit)
    monkeypatch.setattr(reconstruction_stage_module, "fista_tv", fake_public_fista)

    x_out, l_next, stat = _run_reconstruction_step(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        det_grid=det_grid,
        params5=jnp.zeros((projections.shape[0], 5), dtype=jnp.float32),
        x=x0,
        cfg=AlignConfig(
            align_profile="lightning",
            outer_iters=1,
            recon_iters=2,
            regulariser="huber_tv",
            lambda_tv=0.0,
            views_per_batch=0,
            projector_backend="jax",
            gather_dtype="fp32",
        ),
        L_prev=None,
        outer_idx=1,
        recon_algo="fista",
    )

    np.testing.assert_allclose(np.asarray(x_out), np.full_like(np.asarray(x_out), 0.25))
    assert l_next == 147.6
    assert stat["recon_retry"] is True
    assert stat["recon_nonfinite_retry"] is True
    assert stat["reconstruction_finite_fraction"] == 1.0
    assert stat.get("reconstruction_failed") is not True
    assert stat["recon_fallback_reason"] == "huber_fista_core_nonfinite_retry_public_stream"


def test_align_stops_before_pose_update_when_reconstruction_remains_nonfinite(monkeypatch):
    grid, detector, geometry, _volume, projections, _det_grid = _case()
    x0 = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    def fake_reconstruction_step(**kwargs):
        x_in = kwargs["x"]
        return (
            jnp.full_like(x_in, jnp.nan),
            None,
            {
                "reconstruction_failed": True,
                "reconstruction_failure_reason": "nonfinite_reconstruction_after_retry",
                "reconstruction_finite_fraction": 0.0,
            },
        )

    monkeypatch.setattr(pose_stage_module, "_run_reconstruction_step", fake_reconstruction_step)

    x_out, params5, info = align(
        geometry,
        grid,
        detector,
        projections,
        init_x=x0,
        cfg=AlignConfig(
            align_profile="lightning",
            outer_iters=3,
            recon_iters=1,
            projector_backend="jax",
            gather_dtype="fp32",
            optimise_dofs=("phi",),
            pose_model="per_view",
            early_stop=False,
        ),
    )

    assert float(jnp.mean(jnp.isfinite(x_out).astype(jnp.float32))) == 0.0
    np.testing.assert_allclose(np.asarray(params5), np.zeros((projections.shape[0], 5)))
    assert info["completed_outer_iters"] == 1
    assert info["loss"] == []
    stat = info["outer_stats"][0]
    assert stat["reconstruction_failed"] is True
    assert stat["reconstruction_failure_reason"] == "nonfinite_reconstruction_after_retry"
    assert "loss_before" not in stat
