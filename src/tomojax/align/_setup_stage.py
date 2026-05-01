from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import jax.numpy as jnp

from ..core.geometry.base import Detector, Geometry, Grid
from ..recon.fista_tv import FistaConfig, fista_tv
from ._observer import OuterStat
from .model.diagnostics import validate_active_gauge_policy
from .model.dof_specs import ActiveParameterView
from .objectives.fold_recon import FoldReconstructionConfig, reconstruct_train_fold_nograd
from .objectives.folds import FoldSpec
from .geometry.geometry_applier import (
    BaseGeometryArrays,
    apply_setup_to_detector_grid,
    materialize_setup_geometry,
)
from .objectives.loss_adapters import build_loss_adapter
from .early_stop import (
    EarlyStopState,
    annotate_stat_with_early_stop,
    evaluate_early_stop,
    resolve_early_stop_policy,
    setup_evidence_from_stat,
)
from .optimizers import ValidationLmConfig, run_active_validation_lm
from .model.schedules import ResolvedAlignmentStage
from .model.state import AlignmentState, PoseState, SetupGeometryState
from .objectives.validation_residuals import (
    accumulate_validation_normals,
    score_validation_fixed_volume,
)


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
    folds,
    loss_adapter,
    fold_recon_cfg: FoldReconstructionConfig,
    cfg: object,
    init_x: jnp.ndarray | None,
    factor: int,
) -> SetupValidationObjectiveResult:
    z_current = active_view.pack(setup_state)
    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_grad = jnp.zeros_like(z_current)
    total_hess = jnp.zeros((int(z_current.size), int(z_current.size)), dtype=jnp.float32)
    residual_count = 0
    fold_cache: list[FoldEvaluation] = []
    for fold in range(folds.n_folds):
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
    folds,
    loss_name: str,
    schedule_name: str | None,
    stage: ResolvedAlignmentStage | None,
    outer_idx: int,
    init_x: jnp.ndarray | None,
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
            "num_train_reconstructions": int(len(objective_result.fold_cache)),
            "validation_residual_count": int(objective_result.residual_count),
            "recon_projection_chunked": True,
            "validation_projection_chunked": True,
            "active_gradient_mode": "validation_residual_jvp",
        }
    )
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
    loss_spec,
    loss_name: str,
    schedule_name: str | None = None,
    stage: ResolvedAlignmentStage | None = None,
) -> SetupStageResult:
    _validate_setup_stage_execution_contract(stage)
    base = BaseGeometryArrays.from_geometry(geometry, detector, level_factor=int(factor))
    active_view = ActiveParameterView.from_dofs(active_geometry_dofs, geometry=geometry)
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
    setup_stats: list[OuterStat] = []
    prev_loss_after: float | None = None
    early_stop_state = EarlyStopState()
    early_stop_policy = resolve_early_stop_policy(
        enabled=bool(cfg.early_stop),
        profile=getattr(cfg, "early_stop_profile", "compute_saving"),
        rel_impr_threshold=float(cfg.early_stop_rel_impr),
        patience=int(cfg.early_stop_patience),
    )
    outer_limit = max(1, int(stage.maxiter if stage is not None else cfg.outer_iters))
    for outer_idx in range(1, outer_limit + 1):
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
        )
        evidence = setup_evidence_from_stat(
            stat,
            active_dofs=tuple(active_geometry_dofs),
            prev_loss_after=prev_loss_after,
        )
        decision = evaluate_early_stop(
            evidence=evidence,
            policy=early_stop_policy,
            state=early_stop_state,
            outer_idx=outer_idx,
        )
        annotate_stat_with_early_stop(stat, decision)
        early_stop_state = decision.state
        prev_loss_after = evidence.loss_after
        setup_stats.append(stat)
        if decision.should_stop:
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
