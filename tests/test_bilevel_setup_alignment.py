from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import tomojax.align.geometry.detector_center as detector_center
import tomojax.align.geometry.geometry_blocks as geometry_blocks
import tomojax.align.pipeline as pipeline
from tomojax.align.model.dof_specs import ActiveParameterView
from tomojax.align.objectives.folds import FoldSpec
from tomojax.align.geometry.geometry_applier import BaseGeometryArrays
from tomojax.align.objectives.losses import L2OtsuLossSpec, build_loss_adapter
from tomojax.align.model.schedules import schedule_preset
from tomojax.align.model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.align.pipeline import AlignConfig, align_multires
from tomojax.align.objectives.validation_residuals import (
    accumulate_validation_normals,
    score_validation_fixed_volume,
)
from tomojax.calibration.detector_grid import detector_grid_from_calibration
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view


def _detector_grid_case(size: int = 6, n_views: int = 6, *, det_u_px: float = 0.0):
    grid = Grid(nx=size, ny=size, nz=size, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=size, nv=size, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.linspace(0.0, 180.0, n_views, endpoint=False, dtype=np.float32),
    )
    volume_np = np.zeros((size, size, size), dtype=np.float32)
    volume_np[1:4, 2:5, 2:4] = 1.0
    volume_np[4:5, 1:3, 4:5] = 0.6
    volume = jnp.asarray(volume_np)
    true_grid = detector_grid_from_calibration(detector, det_u_px=det_u_px)
    projections = jnp.stack(
        [
            forward_project_view(
                geometry,
                grid,
                detector,
                volume,
                i,
                gather_dtype="fp32",
                det_grid=true_grid,
            )
            for i in range(n_views)
        ],
        axis=0,
    )
    return grid, detector, geometry, volume, projections


def _true_volume_validation_context(detector, geometry, projections):
    base = BaseGeometryArrays.from_geometry(geometry, detector)
    folds = FoldSpec(n_folds=3).build(projections.shape[0])
    loss_adapter = build_loss_adapter(L2OtsuLossSpec(), projections)
    return base, folds, loss_adapter


def test_product_path_has_no_detector_center_candidate_solver_symbols():
    assert not hasattr(detector_center, "calibrate_detector_u_heldout")
    assert not hasattr(geometry_blocks, "optimize_geometry_blocks_for_level")
    assert not hasattr(pipeline, "BilevelCVProjectionObjective")


def test_default_setup_presets_use_bilevel_cv_not_all_data_or_fixed_volume_discovery():
    for name in ("cor", "detector_roll", "axis_direction", "lamino_tilt", "setup_safe"):
        schedule = schedule_preset(name)
        setup_stages = [
            stage
            for stage in schedule.stages
            if any(dof.endswith("_deg") or dof.endswith("_px") for dof in stage.active_dofs)
        ]
        assert setup_stages
        assert all(stage.objective_kind == "bilevel_cv" for stage in setup_stages)


def test_bilevel_cv_detector_center_objective_prefers_hidden_offset_without_candidates():
    grid, detector, geometry, volume, projections = _detector_grid_case(det_u_px=1.5)
    base, folds, loss_adapter = _true_volume_validation_context(detector, geometry, projections)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    corrected = state.replace(setup=state.setup.replace(det_u_px=jnp.asarray(1.5)))
    view = ActiveParameterView.from_dofs(("det_u_px",))

    nominal_score = score_validation_fixed_volume(
        frozen_state=state,
        active_view=view,
        z=view.pack(state),
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_adapter=loss_adapter,
        fold_volume=volume,
        val_idx=folds.val_idx[0],
        val_mask=folds.val_mask[0],
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=False,
        gather_dtype="fp32",
    )
    corrected_score = score_validation_fixed_volume(
        frozen_state=state,
        active_view=view,
        z=view.pack(corrected),
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_adapter=loss_adapter,
        fold_volume=volume,
        val_idx=folds.val_idx[0],
        val_mask=folds.val_mask[0],
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=False,
        gather_dtype="fp32",
    )

    assert float(corrected_score) < float(nominal_score)


def test_bilevel_cv_setup_gradient_is_finite_for_detector_center_and_roll():
    grid, detector, geometry, volume, projections = _detector_grid_case(det_u_px=0.75)
    base, folds, loss_adapter = _true_volume_validation_context(detector, geometry, projections)
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(projections.shape[0]))
    view = ActiveParameterView.from_dofs(("det_u_px", "detector_roll_deg"))
    normals = accumulate_validation_normals(
        frozen_state=state,
        active_view=view,
        z=view.pack(state),
        base=base,
        grid=grid,
        detector=detector,
        projections=projections,
        loss_adapter=loss_adapter,
        fold_volume=volume,
        val_idx=folds.val_idx[0],
        val_mask=folds.val_mask[0],
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=False,
        gather_dtype="fp32",
    )

    assert jnp.isfinite(normals.loss)
    assert normals.grad.shape == (2,)
    assert normals.hess.shape == (2, 2)
    assert jnp.all(jnp.isfinite(normals.grad))
    assert jnp.all(jnp.isfinite(normals.hess))


def test_product_setup_path_uses_validation_lm_not_active_lbfgs(monkeypatch):
    grid, detector, geometry, _volume, projections = _detector_grid_case(
        size=5,
        n_views=4,
        det_u_px=0.5,
    )

    def fail_active_lbfgs(*args, **kwargs):
        raise AssertionError("setup product path must not call active L-BFGS")

    monkeypatch.setattr(pipeline, "run_active_lbfgs", fail_active_lbfgs, raising=False)
    _x, _params, info = align_multires(
        geometry,
        grid,
        detector,
        projections,
        factors=(1,),
        cfg=AlignConfig(
            outer_iters=1,
            recon_iters=1,
            tv_prox_iters=1,
            optimise_dofs=("det_u_px",),
            views_per_batch=1,
            checkpoint_projector=False,
            gather_dtype="fp32",
            recon_positivity=False,
            early_stop=False,
        ),
    )

    setup_stats = [
        stat for stat in info["outer_stats"] if stat.get("optimizer_kind") == "validation_lm"
    ]
    assert setup_stats
    assert setup_stats[0]["active_gradient_mode"] == "validation_residual_jvp"
    assert setup_stats[0]["schedule_stage_name"] == "direct_setup"
