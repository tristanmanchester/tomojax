import numpy as np
import pytest
import jax.numpy as jnp

import tomojax.align.pipeline as align_pipeline
from tomojax.align.losses import parse_loss_spec
from tomojax.align.parametrizations import se3_from_5d
from tomojax.align.pipeline import AlignConfig, align
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view


def make_misaligned_case(nx=10, ny=10, nz=10, n_views=5, seed=0):
    rng = np.random.default_rng(seed)
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0.0, 180.0, n_views, endpoint=False)
    geom_nom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)

    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    vol = vol.at[nx // 4:3 * nx // 4, ny // 4:3 * ny // 4, nz // 4:3 * nz // 4].set(1.0)

    true_params = np.zeros((n_views, 5), dtype=np.float32)
    true_params[:, :3] = rng.normal(scale=np.deg2rad(0.15), size=(n_views, 3))
    true_params[:, 3:] = rng.normal(scale=0.2, size=(n_views, 2))

    projs = []
    for i in range(n_views):
        class _G:
            def pose_for_view(self, _):
                T_nom = jnp.asarray(geom_nom.pose_for_view(i), dtype=jnp.float32)
                T_al = se3_from_5d(jnp.asarray(true_params[i]))
                return tuple(map(tuple, T_nom @ T_al))

            def rays_for_view(self, _):
                return geom_nom.rays_for_view(i)

        projs.append(forward_project_view(_G(), grid, det, vol, view_index=0))
    return grid, det, geom_nom, vol, jnp.stack(projs, axis=0)


def _freeze_reconstruction(monkeypatch):
    def fake_fista_tv(*args, **kwargs):
        init_x = kwargs.get("init_x")
        assert init_x is not None
        return jnp.asarray(init_x, dtype=jnp.float32), {"loss": [0.0, 0.0]}

    monkeypatch.setattr(align_pipeline, "fista_tv", fake_fista_tv)


def _run_fixed_volume_alignment(
    monkeypatch,
    *,
    opt_method: str,
    views_per_batch: int,
    loss_name: str,
):
    _freeze_reconstruction(monkeypatch)
    grid, det, geom, vol, projs = make_misaligned_case(seed=7)
    cfg = AlignConfig(
        outer_iters=1,
        recon_iters=1,
        lambda_tv=0.0,
        lr_rot=5e-3,
        lr_trans=5e-2,
        views_per_batch=views_per_batch,
        opt_method=opt_method,
        loss=parse_loss_spec(loss_name),
        early_stop=False,
    )
    _, params5, info = align(geom, grid, det, projs, cfg=cfg, init_x=vol)
    return np.asarray(params5), info


def test_align_gd_chunking_matches_streamed_reference(monkeypatch):
    params_stream, info_stream = _run_fixed_volume_alignment(
        monkeypatch,
        opt_method="gd",
        views_per_batch=1,
        loss_name="l2_otsu",
    )
    params_chunked, info_chunked = _run_fixed_volume_alignment(
        monkeypatch,
        opt_method="gd",
        views_per_batch=4,
        loss_name="l2_otsu",
    )

    np.testing.assert_allclose(params_chunked, params_stream, rtol=2e-4, atol=1e-5)
    assert info_chunked["loss"][-1] == pytest.approx(
        info_stream["loss"][-1], rel=1e-5, abs=1e-6
    )


def test_align_gn_chunking_matches_streamed_reference(monkeypatch):
    params_stream, info_stream = _run_fixed_volume_alignment(
        monkeypatch,
        opt_method="gn",
        views_per_batch=1,
        loss_name="l2",
    )
    params_chunked, info_chunked = _run_fixed_volume_alignment(
        monkeypatch,
        opt_method="gn",
        views_per_batch=4,
        loss_name="l2",
    )

    np.testing.assert_allclose(params_chunked, params_stream, rtol=2e-4, atol=1e-5)
    assert info_chunked["loss"][-1] == pytest.approx(
        info_stream["loss"][-1], rel=1e-5, abs=1e-6
    )
