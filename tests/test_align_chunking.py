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
    init_params5=None,
    **cfg_kwargs,
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
        **cfg_kwargs,
    )
    _, params5, info = align(
        geom,
        grid,
        det,
        projs,
        cfg=cfg,
        init_x=vol,
        init_params5=init_params5,
    )
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


@pytest.mark.parametrize(
    ("opt_method", "loss_name"),
    [
        ("gd", "l2_otsu"),
        ("gn", "l2"),
    ],
)
def test_align_freeze_phi_keeps_initial_values(monkeypatch, opt_method, loss_name):
    _, _, _, _, projs = make_misaligned_case(seed=8)
    init_params5 = np.zeros((projs.shape[0], 5), dtype=np.float32)
    init_params5[:, 2] = np.linspace(-0.03, 0.03, projs.shape[0], dtype=np.float32)

    params5, _ = _run_fixed_volume_alignment(
        monkeypatch,
        opt_method=opt_method,
        views_per_batch=2,
        loss_name=loss_name,
        init_params5=jnp.asarray(init_params5),
        freeze_dofs=("phi",),
    )

    np.testing.assert_array_equal(params5[:, 2], init_params5[:, 2])


@pytest.mark.parametrize(
    ("opt_method", "loss_name"),
    [
        ("gd", "l2_otsu"),
        ("gn", "l2"),
    ],
)
def test_align_optimise_only_dx_dz_keeps_rotations_initial(monkeypatch, opt_method, loss_name):
    _, _, _, _, projs = make_misaligned_case(seed=9)
    init_params5 = np.zeros((projs.shape[0], 5), dtype=np.float32)
    init_params5[:, 0] = np.linspace(-0.02, 0.02, projs.shape[0], dtype=np.float32)
    init_params5[:, 1] = np.linspace(0.01, -0.01, projs.shape[0], dtype=np.float32)
    init_params5[:, 2] = np.linspace(-0.03, 0.03, projs.shape[0], dtype=np.float32)

    params5, _ = _run_fixed_volume_alignment(
        monkeypatch,
        opt_method=opt_method,
        views_per_batch=2,
        loss_name=loss_name,
        init_params5=jnp.asarray(init_params5),
        optimise_dofs=("dx", "dz"),
        bounds={"dx": (-0.01, 0.01), "dz": (-0.02, 0.02)},
    )

    np.testing.assert_array_equal(params5[:, :3], init_params5[:, :3])
    assert np.max(params5[:, 3]) <= 0.01 + 1e-6
    assert np.min(params5[:, 3]) >= -0.01 - 1e-6
    assert np.max(params5[:, 4]) <= 0.02 + 1e-6
    assert np.min(params5[:, 4]) >= -0.02 - 1e-6


@pytest.mark.parametrize(
    ("opt_method", "loss_name"),
    [
        ("gd", "l2_otsu"),
        ("gn", "l2"),
    ],
)
def test_align_bounds_clip_active_translations(monkeypatch, opt_method, loss_name):
    _, _, _, _, projs = make_misaligned_case(seed=10)
    init_params5 = np.zeros((projs.shape[0], 5), dtype=np.float32)
    init_params5[:, 3] = np.linspace(-0.5, 0.5, projs.shape[0], dtype=np.float32)
    init_params5[:, 4] = np.linspace(0.4, -0.4, projs.shape[0], dtype=np.float32)

    params5, _ = _run_fixed_volume_alignment(
        monkeypatch,
        opt_method=opt_method,
        views_per_batch=2,
        loss_name=loss_name,
        init_params5=jnp.asarray(init_params5),
        bounds={"dx": (-0.05, 0.05), "dz": (-0.04, 0.04)},
    )

    assert np.max(params5[:, 3]) <= 0.05 + 1e-6
    assert np.min(params5[:, 3]) >= -0.05 - 1e-6
    assert np.max(params5[:, 4]) <= 0.04 + 1e-6
    assert np.min(params5[:, 4]) >= -0.04 - 1e-6


@pytest.mark.parametrize(
    ("opt_method", "loss_name"),
    [
        ("gd", "l2_otsu"),
        ("gn", "l2"),
    ],
)
def test_align_bounds_do_not_clip_frozen_dofs(monkeypatch, opt_method, loss_name):
    _, _, _, _, projs = make_misaligned_case(seed=11)
    init_params5 = np.zeros((projs.shape[0], 5), dtype=np.float32)
    init_params5[:, 2] = np.linspace(-0.5, 0.5, projs.shape[0], dtype=np.float32)

    params5, _ = _run_fixed_volume_alignment(
        monkeypatch,
        opt_method=opt_method,
        views_per_batch=2,
        loss_name=loss_name,
        init_params5=jnp.asarray(init_params5),
        freeze_dofs=("phi",),
        bounds={"phi": (-0.01, 0.01)},
    )

    np.testing.assert_array_equal(params5[:, 2], init_params5[:, 2])
