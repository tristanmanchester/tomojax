import sys
import numpy as np
import pytest
import jax.numpy as jnp

import tomojax.align.pipeline as align_pipeline
from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.align.losses import loss_spec_name, parse_loss_schedule, parse_loss_spec
from tomojax.align.dofs import resolve_scoped_alignment_dofs
from tomojax.align.geometry_blocks import (
    add_geometry_acquisition_diagnostics,
    summarize_geometry_calibration_stats,
)
from tomojax.align.pipeline import align, align_multires, AlignConfig
from tomojax.align.parametrizations import se3_from_5d


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


def test_scoped_alignment_dofs_resolve_pose_and_geometry_names():
    scoped = resolve_scoped_alignment_dofs(
        optimise_dofs=("det_u_px", "dx", "dz"),
        freeze_dofs=("dz",),
    )

    assert scoped.active_pose_dofs == ("dx",)
    assert scoped.active_geometry_dofs == ("det_u_px",)
    assert scoped.pose_mask == (False, False, False, True, False)
    assert scoped.active_dofs == ("dx", "det_u_px")


def test_legacy_geometry_dofs_merge_into_scoped_alignment_dofs():
    scoped = resolve_scoped_alignment_dofs(
        geometry_dofs=("det_u_px",),
        freeze_dofs=("alpha", "beta", "phi", "dx", "dz"),
    )

    assert scoped.active_pose_dofs == ()
    assert scoped.active_geometry_dofs == ("det_u_px",)
    assert scoped.pose_mask == (False, False, False, False, False)


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
    assert info["recon_algo"] == "fista"
    assert info["outer_stats"][0]["recon_algo"] == "fista"
    assert "fista_first" in info["outer_stats"][0]
    assert "recon_loss_first" in info["outer_stats"][0]


def test_align_runs_with_spdhg_inner_reconstruction():
    grid, det, geom, vol, projs, _ = make_misaligned_case(4, 4, 4, 4, 7)
    cfg = AlignConfig(
        recon_algo="spdhg",
        outer_iters=1,
        recon_iters=2,
        lambda_tv=0.0,
        views_per_batch=2,
        opt_method="gd",
        lr_rot=5e-3,
        lr_trans=1e-1,
        early_stop=False,
    )

    x, est_params, info = align(geom, grid, det, projs, cfg=cfg)

    assert x.shape == vol.shape
    assert est_params.shape == (projs.shape[0], 5)
    assert np.isfinite(np.asarray(x)).all()
    assert np.isfinite(np.asarray(est_params)).all()
    assert info["recon_algo"] == "spdhg"
    assert info["outer_stats"][0]["recon_algo"] == "spdhg"
    assert info["outer_stats"][0]["spdhg_views_per_batch"] == 2
    assert info["outer_stats"][0]["spdhg_num_blocks"] == 2


def test_align_reports_true_relative_improvement_scaling():
    grid, det, geom, vol, projs, _ = make_misaligned_case(6, 6, 6, 6, 2)
    projs = (projs * 1e-3).astype(jnp.float32)
    cfg = AlignConfig(
        outer_iters=2,
        recon_iters=2,
        lambda_tv=0.0,
        lr_rot=5e-3,
        lr_trans=1e-1,
        early_stop=False,
    )
    _, _, info = align(geom, grid, det, projs, cfg=cfg)
    stats = info.get("outer_stats", [])
    assert stats, "align() should report per-outer statistics"

    saw_small_loss = False
    for s in stats:
        loss_before = s.get("loss_before")
        loss_after = s.get("loss_after")
        rel_impr = s.get("rel_impr")
        if loss_before is None or loss_after is None or rel_impr is None:
            continue
        if not np.isfinite(loss_before) or not np.isfinite(loss_after):
            continue
        denom = max(abs(loss_before), 1e-12)
        expected = (loss_before - loss_after) / denom
        assert rel_impr == pytest.approx(expected)
        if abs(loss_before) < 1.0:
            saw_small_loss = True

    assert saw_small_loss, "Test case should exercise small-loss relative scaling"


def test_align_runs_with_cylindrical_volume_mask():
    grid, det, geom, vol, projs, _ = make_misaligned_case(8, 8, 8, 6, 3)
    cfg = AlignConfig(
        outer_iters=1,
        recon_iters=2,
        lambda_tv=0.0,
        lr_rot=5e-3,
        lr_trans=1e-1,
        views_per_batch=4,
        mask_vol="cyl",
        early_stop=False,
    )
    x, est_params, info = align(geom, grid, det, projs, cfg=cfg)

    assert x.shape == vol.shape
    assert est_params.shape == (projs.shape[0], 5)
    assert np.isfinite(np.asarray(x)).all()
    assert np.isfinite(np.asarray(est_params)).all()
    assert np.isfinite(np.asarray(info["loss"])).all()


def test_align_rejects_bad_init_params5_shape_with_expected_and_actual():
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0, 45.0, 90.0])
    projs = jnp.zeros((3, det.nv, det.nu), dtype=jnp.float32)
    bad_params = jnp.zeros((2, 5), dtype=jnp.float32)

    with pytest.raises(
        ValueError,
        match=r"expected \(3, 5\).*actual \(2, 5\).*Likely fix",
    ):
        align(
            geom,
            grid,
            det,
            projs,
            cfg=AlignConfig(outer_iters=1, recon_iters=1, recon_L=1.0),
            init_params5=bad_params,
        )


def test_align_multires_counts_executed_outer_iters_without_observer():
    grid, det, geom, _, projs, _ = make_misaligned_case(6, 6, 6, 6, 4)
    cfg = AlignConfig(
        outer_iters=1,
        recon_iters=1,
        lambda_tv=0.0,
        lr_rot=5e-3,
        lr_trans=1e-1,
        early_stop=False,
    )

    _, _, info = align_multires(geom, grid, det, projs, factors=[2, 1], cfg=cfg)

    assert info["factors"] == [2, 1]
    assert info["total_outer_iters"] == 2


def test_align_multires_geometry_block_estimates_detector_center_without_pose_dofs():
    size = 8
    n_views = 12
    grid = Grid(nx=size, ny=size, nz=size, vx=1.0, vy=1.0, vz=1.0)
    det_nom = Detector(nu=size, nv=size, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    det_true = Detector(nu=size, nv=size, du=1.0, dv=1.0, det_center=(1.0, 0.0))
    thetas = np.linspace(0, 180, n_views, endpoint=False, dtype=np.float32)
    geom_nom = ParallelGeometry(grid=grid, detector=det_nom, thetas_deg=thetas)
    geom_true = ParallelGeometry(grid=grid, detector=det_true, thetas_deg=thetas)
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[2:5, 1:6, 2:4] = 1.0
    vol[5:7, 2:4, 5:7] = 0.7
    volume = jnp.asarray(vol)
    projs = jnp.stack(
        [
            forward_project_view(
                geom_true,
                grid,
                det_true,
                volume,
                i,
                gather_dtype="fp32",
            )
            for i in range(n_views)
        ],
        axis=0,
    )

    checkpoints = []
    _, params5, info = align_multires(
        geom_nom,
        grid,
        det_nom,
        projs,
        factors=[2, 1],
        cfg=AlignConfig(
            outer_iters=2,
            recon_iters=2,
            lambda_tv=0.0,
            geometry_dofs=("det_u_px",),
            freeze_dofs=("alpha", "beta", "phi", "dx", "dz"),
            early_stop=False,
            gather_dtype="fp32",
            checkpoint_projector=False,
            views_per_batch=1,
            gn_damping=1e-3,
        ),
        checkpoint_callback=checkpoints.append,
    )

    det_state = info["geometry_calibration_state"]["detector"]
    det_u = next(v for v in det_state if v["name"] == "det_u_px")
    assert det_u["status"] == "estimated"
    assert float(det_u["value"]) == pytest.approx(1.0, abs=0.65)
    assert checkpoints[-1].geometry_calibration_state is not None
    checkpoint_det_u = next(
        v
        for v in checkpoints[-1].geometry_calibration_state["detector"]
        if v["name"] == "det_u_px"
    )
    assert float(checkpoint_det_u["value"]) == pytest.approx(float(det_u["value"]))
    assert np.asarray(params5).shape == (n_views, 5)
    assert np.allclose(np.asarray(params5), 0.0)
    assert any(stat.get("geometry_block") == "detector_center" for stat in info["outer_stats"])
    geom_stats = [stat for stat in info["outer_stats"] if stat.get("geometry_block")]
    assert geom_stats
    assert {stat.get("loss_kind") for stat in geom_stats} == {"l2_otsu"}
    assert {stat.get("geometry_loss_kind") for stat in geom_stats} == {"l2_otsu"}
    assert {stat.get("geometry_objective") for stat in geom_stats} == {"heldout_reprojection"}
    assert info["wall_time_total"] > 0.0
    diagnostics = info["geometry_calibration_diagnostics"]
    assert diagnostics["schema_version"] == 1
    assert diagnostics["blocks"]
    center = diagnostics["blocks"][0]
    assert center["geometry_block"] == "detector_center"
    assert center["geometry_objective"] == "heldout_reprojection"
    assert center["accepted_updates"] >= 1
    assert center["status"] in {"converged", "underconverged", "ill_conditioned"}


def test_align_multires_rejects_detector_center_with_active_pose_translations():
    grid, det, geom, _, projs, _ = make_misaligned_case(6, 6, 6, 6, 11)

    with pytest.raises(ValueError, match="Gauge-coupled alignment DOFs"):
        align_multires(
            geom,
            grid,
            det,
            projs,
            factors=[2],
            cfg=AlignConfig(
                outer_iters=1,
                recon_iters=1,
                optimise_dofs=("det_u_px", "dx"),
                freeze_dofs=("alpha", "beta", "phi", "dz"),
                early_stop=False,
            ),
        )


def test_geometry_calibration_diagnostics_classify_underconverged_and_ill_conditioned():
    underconverged = summarize_geometry_calibration_stats(
        [
            {
                "geometry_block": "detector_roll",
                "geometry_active_dofs": "detector_roll_deg",
                "geometry_loss_before": 1.0,
                "geometry_loss_after": 0.8,
                "geometry_accepted": True,
                "geometry_step_norm": 0.2,
                "geometry_gradient_norm": 0.01,
                "geometry_max_step": 2.0,
            },
            {
                "geometry_block": "detector_roll",
                "geometry_active_dofs": "detector_roll_deg",
                "geometry_loss_before": 0.8,
                "geometry_loss_after": 0.7,
                "geometry_accepted": True,
                "geometry_step_norm": 0.12,
                "geometry_gradient_norm": 0.008,
                "geometry_max_step": 2.0,
            },
        ]
    )
    ill_conditioned = summarize_geometry_calibration_stats(
        [
            {
                "geometry_block": "axis_direction",
                "geometry_active_dofs": "axis_rot_x_deg",
                "geometry_loss_before": 1.0,
                "geometry_loss_after": 1.0,
                "geometry_accepted": False,
                "geometry_step_norm": 0.0,
                "geometry_gradient_norm": 0.0,
                "geometry_max_step": 2.0,
            }
        ]
    )

    assert underconverged["blocks"][0]["status"] == "underconverged"
    assert underconverged["overall_status"] == "underconverged"
    assert ill_conditioned["blocks"][0]["status"] == "ill_conditioned"
    assert ill_conditioned["overall_status"] == "ill_conditioned"


def test_axis_direction_diagnostics_mark_180_degree_acquisition_ill_conditioned():
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=4, nv=4, du=1.0, dv=1.0)
    geom = ParallelGeometry(
        grid=grid,
        detector=det,
        thetas_deg=np.linspace(0.0, 180.0, 12, endpoint=False, dtype=np.float32),
    )
    diagnostics = {
        "schema_version": 1,
        "overall_status": "converged",
        "blocks": [
            {
                "geometry_block": "axis_direction",
                "geometry_active_dofs": "axis_rot_x_deg",
                "status": "converged",
            }
        ],
    }

    annotated = add_geometry_acquisition_diagnostics(
        diagnostics,
        geom,
        ("axis_rot_x_deg",),
    )

    block = annotated["blocks"][0]
    assert annotated["overall_status"] == "ill_conditioned"
    assert annotated["warnings"] == ["axis_direction_sub_full_rotation_acquisition"]
    assert block["status"] == "ill_conditioned"
    assert block["theta_span_deg"] == pytest.approx(180.0)


def test_align_multires_rejects_non_integral_factors_before_truncating():
    grid, det, geom, _, projs, _ = make_misaligned_case(6, 6, 6, 3, 5)

    with pytest.raises(ValueError, match="integer >= 1"):
        align_multires(
            geom,
            grid,
            det,
            projs,
            factors=[1.5],
            cfg=AlignConfig(outer_iters=0, recon_iters=1, early_stop=False),
        )


def test_align_multires_uses_scheduled_loss_by_level(monkeypatch):
    grid, det, geom, _, projs, _ = make_misaligned_case(8, 8, 8, 4, 6)
    observed_loss_names: list[str] = []

    def fake_align(
        geometry,
        level_grid,
        level_detector,
        level_projections,
        *,
        cfg,
        init_x=None,
        init_params5=None,
        observer=None,
        resume_state=None,
        checkpoint_callback=None,
    ):
        del geometry, observer, resume_state, checkpoint_callback
        loss_name = loss_spec_name(cfg.loss)
        observed_loss_names.append(loss_name)
        params = (
            jnp.zeros((level_projections.shape[0], 5), dtype=jnp.float32)
            if init_params5 is None
            else jnp.asarray(init_params5, dtype=jnp.float32)
        )
        x = (
            jnp.zeros(
                (level_grid.nx, level_grid.ny, level_grid.nz),
                dtype=jnp.float32,
            )
            if init_x is None
            else jnp.asarray(init_x, dtype=jnp.float32)
        )
        info = {
            "loss": [float(len(observed_loss_names))],
            "loss_kind": loss_name,
            "outer_stats": [
                {
                    "outer_idx": 1,
                    "loss_kind": loss_name,
                    "loss_after": float(len(observed_loss_names)),
                }
            ],
            "stopped_by_observer": False,
            "observer_action": "continue",
            "wall_time_total": 0.0,
            "pose_model": "per_view",
            "pose_model_variables": int(level_projections.shape[0] * 5),
            "per_view_variables": 5,
            "pose_model_basis_shape": [int(level_projections.shape[0]), 1],
            "active_dofs": ["alpha", "beta", "phi", "dx", "dz"],
            "completed_outer_iters": 1,
            "small_impr_streak": 0,
            "motion_coeffs": None,
            "L": None,
        }
        return x, params, info

    monkeypatch.setattr(align_pipeline, "align", fake_align)
    cfg = AlignConfig(
        outer_iters=1,
        recon_iters=1,
        loss=parse_loss_schedule(
            "4:phasecorr,2:ssim,1:l2_otsu",
            default=parse_loss_spec("l2"),
        ),
        early_stop=False,
    )

    _, _, info = align_pipeline.align_multires(
        geom,
        grid,
        det,
        projs,
        factors=[4, 2, 1],
        cfg=cfg,
    )

    assert observed_loss_names == ["phasecorr", "ssim", "l2_otsu"]
    assert [stat["loss_kind"] for stat in info["outer_stats"]] == [
        "phasecorr",
        "ssim",
        "l2_otsu",
    ]


def test_align_multires_recovers_from_expected_loss_eval_failure(monkeypatch):
    grid, det, geom, _, projs, _ = make_misaligned_case(6, 6, 6, 6, 5)
    cfg = AlignConfig(
        outer_iters=1,
        recon_iters=1,
        lambda_tv=0.0,
        lr_rot=5e-3,
        lr_trans=1e-1,
        early_stop=False,
    )
    original = align_pipeline._evaluate_align_loss
    injected = {"done": False}

    def flaky_align_loss_eval(eval_loss, *, fallback, context):
        if (not injected["done"]) and context == "Using fallback for final alignment loss bookkeeping":
            injected["done"] = True

            def raise_expected_failure():
                raise FloatingPointError("nan in align loss eval")

            return original(raise_expected_failure, fallback=fallback, context=context)
        return original(eval_loss, fallback=fallback, context=context)

    monkeypatch.setattr(align_pipeline, "_evaluate_align_loss", flaky_align_loss_eval)

    _, _, info = align_multires(geom, grid, det, projs, factors=[2, 1], cfg=cfg)

    assert injected["done"] is True
    assert info["total_outer_iters"] == 2
    assert len(info["loss"]) == 2
    assert np.isfinite(np.asarray(info["loss"])).all()
