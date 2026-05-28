from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from tomojax.recon.fista_tv import FistaConfig, fista_tv

from ._geometry.detector_center import projection_pair_det_u_seed
from ._geometry.geometry_applier import (
    BaseGeometryArrays,
    apply_setup_to_detector_grid,
    materialize_setup_geometry,
)
from ._model.diagnostics import validate_active_gauge_policy
from ._model.dof_specs import ActiveParameterView
from ._model.state import AlignmentState, PoseState, SetupGeometryState
from ._objectives.fold_recon import FoldReconstructionConfig, reconstruct_train_fold_nograd
from ._objectives.folds import FoldSpec
from ._objectives.loss_adapters import build_loss_adapter
from ._objectives.validation_residuals import (
    accumulate_validation_normals,
    score_validation_fixed_volume,
)
from .optimizers import ValidationLmConfig, run_active_validation_lm

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from ._model.schedules import ResolvedAlignmentStage
    from ._objectives.folds import FoldArrays
    from ._objectives.loss_adapters import LossAdapter
    from ._objectives.loss_specs import AlignmentLossSpec
    from ._observer import OuterStat


FoldEvaluation = tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, object]]


@dataclass(frozen=True)
class SetupValidationObjectiveResult:
    opt_result: Any
    total_loss: float
    residual_count: int
    fold_cache: list[FoldEvaluation]


@dataclass(frozen=True)
class SetupStageResult:
    x: jnp.ndarray
    state: AlignmentState
    losses: list[float]
    public_outer_stats: list[OuterStat]
    checkpoint_outer_stats: list[OuterStat]
    diagnostics: dict[str, object]


def _geometry_with_setup_state(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    setup: SetupGeometryState,
) -> Geometry:
    return materialize_setup_geometry(geometry, grid, detector, setup)


def _geometry_calibration_payload(
    state: AlignmentState,
    active_geometry_dofs: Iterable[str],
) -> dict[str, object]:
    return state.to_calibration_state(active_dofs=active_geometry_dofs).to_dict()


def _run_setup_validation_objective(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    setup_state: AlignmentState,
    active_view: ActiveParameterView,
    base: BaseGeometryArrays,
    folds: FoldArrays,
    loss_adapter: LossAdapter,
    fold_recon_cfg: FoldReconstructionConfig,
    cfg: object,
    init_x: jnp.ndarray | None,
    factor: int,
) -> SetupValidationObjectiveResult:
    stage_start = time.monotonic()
    z_current = active_view.pack(setup_state)
    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_grad = jnp.zeros_like(z_current)
    total_hess = jnp.zeros((int(z_current.size), int(z_current.size)), dtype=jnp.float32)
    residual_count = 0
    fold_cache: list[FoldEvaluation] = []
    for fold in range(folds.n_folds):
        logging.info(
            "Setup validation-LM fold %d/%d: reconstructing train fold for DOFs=%s",
            fold + 1,
            folds.n_folds,
            ",".join(active_view.dofs),
        )
        train_idx = folds.train_idx[fold]
        train_mask = folds.train_mask[fold]
        val_idx = folds.val_idx[fold]
        val_mask = folds.val_mask[fold]
        fold_volume, fold_recon_info = reconstruct_train_fold_nograd(
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            state=setup_state,
            train_idx=train_idx,
            train_mask=train_mask,
            init_x=init_x,
            level_factor=int(factor),
            cfg=fold_recon_cfg,
        )
        normals = accumulate_validation_normals(
            frozen_state=setup_state,
            active_view=active_view,
            z=z_current,
            base=base,
            grid=grid,
            detector=detector,
            projections=projections,
            loss_adapter=loss_adapter,
            fold_volume=fold_volume,
            val_idx=val_idx,
            val_mask=val_mask,
            views_per_batch=max(1, int(cfg.views_per_batch)),
            projector_unroll=int(cfg.projector_unroll),
            checkpoint_projector=bool(cfg.checkpoint_projector),
            gather_dtype=str(cfg.gather_dtype),
        )
        total_loss = total_loss + normals.loss
        total_grad = total_grad + normals.grad
        total_hess = total_hess + normals.hess
        residual_count += int(normals.residual_count)
        fold_cache.append((fold, fold_volume, val_idx, val_mask, fold_recon_info))
        logging.info(
            "Setup validation-LM fold %d/%d: residuals=%d cumulative_loss=%.6g elapsed=%.1fs",
            fold + 1,
            folds.n_folds,
            residual_count,
            float(total_loss),
            time.monotonic() - stage_start,
        )

    def score_candidate(z_candidate: jnp.ndarray) -> float:
        score = jnp.asarray(0.0, dtype=jnp.float32)
        for _fold, fold_volume, val_idx, val_mask, _fold_info in fold_cache:
            score = score + score_validation_fixed_volume(
                frozen_state=setup_state,
                active_view=active_view,
                z=z_candidate,
                base=base,
                grid=grid,
                detector=detector,
                projections=projections,
                loss_adapter=loss_adapter,
                fold_volume=fold_volume,
                val_idx=val_idx,
                val_mask=val_mask,
                views_per_batch=max(1, int(cfg.views_per_batch)),
                projector_unroll=int(cfg.projector_unroll),
                checkpoint_projector=bool(cfg.checkpoint_projector),
                gather_dtype=str(cfg.gather_dtype),
            )
        return float(score)

    opt_result = run_active_validation_lm(
        state=setup_state,
        view=active_view,
        loss=float(total_loss),
        grad=total_grad,
        hess=total_hess,
        score_fn=score_candidate,
        bounds=cfg.bounds,
        cfg=ValidationLmConfig(damping=max(float(cfg.gn_damping), 1e-6)),
    )
    logging.info(
        "Setup validation-LM candidates: loss %.6g -> %.6g accepted=%s elapsed=%.1fs",
        float(total_loss),
        float(opt_result.loss),
        bool(opt_result.accepted),
        time.monotonic() - stage_start,
    )
    return SetupValidationObjectiveResult(
        opt_result=opt_result,
        total_loss=float(total_loss),
        residual_count=residual_count,
        fold_cache=fold_cache,
    )


def _build_geometry_stage_stat(
    *,
    objective_result: SetupValidationObjectiveResult,
    active_view: ActiveParameterView,
    cfg: object,
    fold_recon_cfg: FoldReconstructionConfig,
    folds: FoldArrays,
    loss_name: str,
    schedule_name: str | None,
    stage: ResolvedAlignmentStage | None,
    outer_idx: int,
    init_x: jnp.ndarray | None,
    seed_diagnostics: dict[str, object] | None = None,
) -> OuterStat:
    opt_result = objective_result.opt_result
    stat = dict(opt_result.stats)
    gauge_decision = (
        stage.gauge_decision
        if stage is not None
        else validate_active_gauge_policy(
            active_view.dofs,
            policy=cfg.gauge_policy,
            priors=cfg.gauge_priors,
        )
    )
    stat.update(
        {
            "geometry_block": "setup_validation_lm",
            "geometry_active_dofs": ",".join(active_view.dofs),
            "geometry_objective": "bilevel_cv",
            "geometry_optimizer": "validation_lm",
            "geometry_loss_kind": loss_name,
            "geometry_loss_before": float(objective_result.total_loss),
            "geometry_loss_after": float(opt_result.loss),
            "geometry_accepted": bool(opt_result.accepted),
            "geometry_step_norm": float(stat.get("step_norm_whitened", 0.0) or 0.0),
            "geometry_gradient_norm": float(stat.get("grad_norm_whitened", 0.0) or 0.0),
            "geometry_max_step": 1.0,
            "geometry_status": "converged" if opt_result.accepted else "underconverged",
            "geometry_outer_idx": int(outer_idx),
            "schedule_name": schedule_name,
            "schedule_stage_index": int(stage.index) if stage is not None else None,
            "schedule_stage_name": stage.name if stage is not None else None,
            "schedule_stage_active_dofs": (
                ",".join(stage.active_dofs) if stage is not None else ",".join(active_view.dofs)
            ),
            "gauge_policy": stage.gauge_policy if stage is not None else cfg.gauge_policy,
            "gauge_status": gauge_decision.status,
            "gauge_decision": gauge_decision.to_dict(),
            "objective_kind": "bilevel_cv",
            "objective_provenance": {
                "outer_loss_source": "AlignmentLossSpec",
                "outer_loss_kind": str(loss_name),
                "inner_data_term": "l2_projection",
                "inner_regulariser": str(fold_recon_cfg.regulariser),
                "validation_split": "interleaved_kfold",
                "differentiation_mode": "none",
                "initialization_policy": "current_level_volume" if init_x is not None else "zeros",
            },
            "optimizer_kind": "validation_lm",
            "outer_loss_kind": str(loss_name),
            "recon_sensitivity": "stopped",
            "train_reconstruction_gradient": False,
            "views_per_batch": max(1, int(cfg.views_per_batch)),
            "n_folds": int(folds.n_folds),
            "fold_eval_mode": "stopped_train_recon_validation_lm",
            "folds_used": ",".join(str(item[0]) for item in objective_result.fold_cache),
            "num_train_reconstructions": len(objective_result.fold_cache),
            "validation_residual_count": int(objective_result.residual_count),
            "recon_projection_chunked": True,
            "validation_projection_chunked": True,
            "active_gradient_mode": "validation_residual_jvp",
        }
    )
    if seed_diagnostics is not None and outer_idx == 1:
        stat.update(seed_diagnostics)
    return stat


def _refresh_setup_reconstruction(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    setup_state: AlignmentState,
    init_x: jnp.ndarray | None,
    factor: int,
    cfg: object,
) -> jnp.ndarray:
    geom = _geometry_with_setup_state(geometry, grid, detector, setup_state.setup)
    det_grid = apply_setup_to_detector_grid(
        detector,
        setup_state.setup,
        level_factor=int(factor),
    )
    x_next, _ = fista_tv(
        geom,
        grid,
        detector,
        projections,
        init_x=init_x,
        config=FistaConfig(
            iters=max(1, int(cfg.recon_iters)),
            lambda_tv=float(cfg.lambda_tv),
            regulariser=cfg.regulariser,
            huber_delta=float(cfg.huber_delta),
            tv_prox_iters=int(cfg.tv_prox_iters),
            L=cfg.recon_L,
            views_per_batch=max(1, int(cfg.views_per_batch)),
            projector_unroll=int(cfg.projector_unroll),
            checkpoint_projector=bool(cfg.checkpoint_projector),
            gather_dtype=str(cfg.gather_dtype),
            positivity=bool(cfg.recon_positivity),
        ),
        det_grid=det_grid,
    )
    return x_next


def _build_setup_stage_result(
    *,
    x: jnp.ndarray,
    state: AlignmentState,
    setup_stats: list[OuterStat],
) -> SetupStageResult:
    losses = [
        float(stat["geometry_loss_after"])
        for stat in setup_stats
        if stat.get("geometry_loss_after") is not None
    ]
    return SetupStageResult(
        x=x,
        state=state.replace(volume=x),
        losses=losses,
        public_outer_stats=[dict(stat) for stat in setup_stats],
        checkpoint_outer_stats=[dict(stat) for stat in setup_stats],
        diagnostics={},
    )


def _validate_setup_stage_execution_contract(stage: ResolvedAlignmentStage | None) -> None:
    if stage is None:
        return
    if stage.objective_kind != "bilevel_cv":
        raise ValueError(
            f"Setup alignment stage {stage.name!r} uses unsupported objective "
            f"{stage.objective_kind!r}; setup execution currently supports only "
            "'bilevel_cv'"
        )
    if stage.optimizer_kind != "validation_lm":
        raise ValueError(
            f"Setup alignment stage {stage.name!r} uses unsupported optimizer "
            f"{stage.optimizer_kind!r}; setup execution currently supports only "
            "'validation_lm'"
        )


def _detector_center_seed_diagnostics(
    *,
    geometry: Geometry,
    projections: jnp.ndarray,
    setup_state: AlignmentState,
    active_view: ActiveParameterView,
    schedule_name: str | None,
    stage: ResolvedAlignmentStage | None,
) -> dict[str, object] | None:
    """Return COR seed diagnostics when a detector-u-only setup stage can be seeded."""
    stage_name = stage.name if stage is not None else schedule_name
    if tuple(active_view.dofs) != ("det_u_px",):
        return None
    if stage_name != "cor" and schedule_name != "cor":
        return None
    current = float(setup_state.setup.det_u_px)
    if abs(current) > 1e-6:
        return None
    seed = projection_pair_det_u_seed(projections, geometry)
    if not str(seed.status).startswith("ok"):
        return {
            "detector_center_seed_status": seed.status,
            "detector_center_seed_method": "opposite_angle_projection_pair",
            "detector_center_seed_applied": False,
        }
    return {
        "detector_center_seed_status": seed.status,
        "detector_center_seed_method": "opposite_angle_projection_pair",
        "detector_center_seed_applied": True,
        "detector_center_seed_det_u_px": float(seed.det_u_px),
        "detector_center_seed_pair_lag_px": float(seed.amplitude_px),
    }


def _optimize_setup_geometry_bilevel_for_level(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    init_x: jnp.ndarray | None,
    init_params5: jnp.ndarray | None,
    state: AlignmentState,
    active_geometry_dofs: Iterable[str],
    factor: int,
    cfg: object,
    loss_spec: AlignmentLossSpec,
    loss_name: str,
    schedule_name: str | None = None,
    stage: ResolvedAlignmentStage | None = None,
) -> SetupStageResult:
    _validate_setup_stage_execution_contract(stage)
    base = BaseGeometryArrays.from_geometry(geometry, detector, level_factor=int(factor))
    active_view = ActiveParameterView.from_dofs(active_geometry_dofs)
    alignment_state = state.replace(
        setup=state.setup.replace(nominal_axis_unit=base.nominal_axis_unit),
        pose=PoseState(
            jnp.zeros((int(projections.shape[0]), 5), dtype=jnp.float32)
            if init_params5 is None
            else jnp.asarray(init_params5, dtype=jnp.float32)
        ),
        volume=init_x,
    )
    n_views = int(projections.shape[0])
    n_folds = min(2, n_views)
    folds = FoldSpec(n_folds=n_folds).build(n_views)
    loss_adapter = build_loss_adapter(loss_spec, projections)
    if not bool(loss_adapter.supports_setup_validation_lm):
        stage_name = schedule_name or (stage.name if stage is not None else "setup")
        raise ValueError(
            "Setup validation-LM requires a setup-compatible weighted least-squares loss; "
            f"level {int(factor)} stage {stage_name!r} resolved loss {loss_name!r}"
        )
    fold_recon_cfg = FoldReconstructionConfig(
        iters=max(1, int(cfg.recon_iters)),
        lambda_tv=float(cfg.lambda_tv),
        regulariser=str(cfg.regulariser),
        huber_delta=float(cfg.huber_delta),
        tv_prox_iters=int(cfg.tv_prox_iters),
        L=cfg.recon_L,
        positivity=bool(cfg.recon_positivity),
        views_per_batch=max(1, int(cfg.views_per_batch)),
        projector_unroll=int(cfg.projector_unroll),
        checkpoint_projector=bool(cfg.checkpoint_projector),
        gather_dtype=str(cfg.gather_dtype),
    )

    setup_state = alignment_state
    seed_diagnostics = _detector_center_seed_diagnostics(
        geometry=geometry,
        projections=projections,
        setup_state=setup_state,
        active_view=active_view,
        schedule_name=schedule_name,
        stage=stage,
    )
    if seed_diagnostics is not None and bool(seed_diagnostics.get("detector_center_seed_applied")):
        logging.info(
            "Setup detector-centre seed: det_u_px=%.3f from %s",
            float(seed_diagnostics["detector_center_seed_det_u_px"]),
            seed_diagnostics.get("detector_center_seed_method"),
        )
        setup_state = setup_state.replace(
            setup=setup_state.setup.replace(
                det_u_px=jnp.asarray(
                    float(seed_diagnostics["detector_center_seed_det_u_px"]),
                    dtype=jnp.float32,
                )
            )
        )
    setup_stats: list[OuterStat] = []
    last_loss = math.inf
    outer_limit = max(1, int(stage.maxiter if stage is not None else cfg.outer_iters))
    for outer_idx in range(1, outer_limit + 1):
        stage_name = stage.name if stage is not None else (schedule_name or "setup")
        logging.info(
            "Setup stage %s outer %d/%d level=%d DOFs=%s",
            stage_name,
            outer_idx,
            outer_limit,
            int(factor),
            ",".join(active_view.dofs),
        )
        objective_result = _run_setup_validation_objective(
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            setup_state=setup_state,
            active_view=active_view,
            base=base,
            folds=folds,
            loss_adapter=loss_adapter,
            fold_recon_cfg=fold_recon_cfg,
            cfg=cfg,
            init_x=init_x,
            factor=factor,
        )
        opt_result = objective_result.opt_result
        setup_state = opt_result.state
        last_loss = float(opt_result.loss)
        stat = _build_geometry_stage_stat(
            objective_result=objective_result,
            active_view=active_view,
            cfg=cfg,
            fold_recon_cfg=fold_recon_cfg,
            folds=folds,
            loss_name=loss_name,
            schedule_name=schedule_name,
            stage=stage,
            outer_idx=outer_idx,
            init_x=init_x,
            seed_diagnostics=seed_diagnostics,
        )
        setup_stats.append(stat)
        logging.info(
            "Setup stage %s outer %d/%d: loss %.6g -> %.6g accepted=%s step_norm=%.3g",
            stage_name,
            outer_idx,
            outer_limit,
            float(stat.get("geometry_loss_before", math.nan)),
            float(stat.get("geometry_loss_after", math.nan)),
            bool(stat.get("geometry_accepted", False)),
            float(stat.get("geometry_step_norm", 0.0) or 0.0),
        )
        if bool(cfg.early_stop) and outer_idx > 1:
            prev = float(setup_stats[-2].get("geometry_loss_after", math.inf))
            impr = (prev - last_loss) / max(abs(prev), 1e-6)
            if impr < float(cfg.early_stop_rel_impr):
                break

    x_next = _refresh_setup_reconstruction(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        setup_state=setup_state,
        init_x=init_x,
        factor=factor,
        cfg=cfg,
    )
    return _build_setup_stage_result(x=x_next, state=setup_state, setup_stats=setup_stats)
