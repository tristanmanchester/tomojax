from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
import math
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from tomojax.align._config import AlignConfig, _active_dof_mask_for_cfg, _active_dofs_for_cfg
from tomojax.align._model.dofs import bounds_vectors
from tomojax.align._model.gauge import (
    active_gauge_dofs,
    normalize_gauge_fix,
    validate_alignment_gauge_feasible,
)
from tomojax.align._objectives.fixed_volume import (
    ObjectiveProvenance,
    alignment_projector_backend_provenance,
)
from tomojax.align._objectives.loss_adapters import build_loss_adapter
from tomojax.align._objectives.loss_specs import loss_spec_name, resolve_loss_for_level
from tomojax.align._observer import (
    ObserverAction,
    ObserverCallback,
    OuterStat,
    _normalize_observer_action,
    adapt_observer_callback,
)
from tomojax.align._profiles import profile_policy_from_config
from tomojax.align._results import AlignCheckpointCallback, AlignInfo, AlignResumeState
from tomojax.align._stages._reconstruction_stage import _run_reconstruction_step
from tomojax.core import format_duration, progress_iter
from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import get_detector_grid_device
from tomojax.core.validation import (
    validate_grid,
    validate_optional_same_shape,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)

from ._pose_candidates import _second_difference_gram
from ._pose_context import (
    AlignmentRuntimeContext,
    PoseConstraintContext,
    PoseMotionContext,
    _AlignSetupState,
    _build_alignment_volume_mask,
)
from ._pose_objective import PoseObjectiveBundle, _build_pose_objective_bundle
from ._pose_steps import (
    AlignmentStepConstraints,
    AlignmentStepGauge,
    AlignmentStepMotion,
    AlignmentStepObjective,
    AlignmentStepOptimizer,
    AlignmentStepSmoothing,
    _run_alignment_step,
)
from ._pose_summary import _format_outer_summary_lines

_AlignmentStepContexts = tuple[
    AlignmentStepOptimizer,
    AlignmentStepObjective,
    AlignmentStepConstraints,
    AlignmentStepMotion,
    AlignmentStepSmoothing,
    AlignmentStepGauge,
]


def _prepare_align_setup(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    cfg: AlignConfig | None,
    init_x: jnp.ndarray | None,
    init_params5: jnp.ndarray | None,
    observer: ObserverCallback | None,
    resume_state: AlignResumeState | None,
) -> _AlignSetupState:
    if cfg is None:
        cfg = AlignConfig()
    observer_fn = adapt_observer_callback(observer) if observer is not None else None
    validate_grid(grid, "align grid")
    n_views, _, _ = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="align projections",
    )
    if resume_state is not None:
        init_x = resume_state.x
        init_params5 = resume_state.params5
    if init_x is not None:
        validate_volume(init_x, grid, context="align init_x", name="init_x")
    validate_optional_same_shape(
        init_params5,
        (n_views, 5),
        context="align init_params5",
        name="init_params5",
        fix="pass one 5-parameter alignment row per projection view.",
    )
    x = (
        jnp.asarray(init_x, dtype=jnp.float32)
        if init_x is not None
        else jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    )
    params5 = (
        jnp.asarray(init_params5, dtype=jnp.float32)
        if init_params5 is not None
        else jnp.zeros((n_views, 5), dtype=jnp.float32)
    )
    frozen_params5 = params5
    active_mask_tuple = _active_dof_mask_for_cfg(cfg)
    active_mask_bool = jnp.asarray(active_mask_tuple, dtype=bool)
    active_col_indices_np = np.asarray(
        [idx for idx, is_active in enumerate(active_mask_tuple) if is_active],
        dtype=np.int32,
    )
    active_names = _active_dofs_for_cfg(cfg)
    active_mask = active_mask_bool.astype(jnp.float32)
    bounds_lower, bounds_upper = bounds_vectors(cfg.bounds)
    gauge_fix = normalize_gauge_fix(cfg.gauge_fix)
    gauge_dofs = active_gauge_dofs(mode=gauge_fix, active_mask=active_mask_tuple)
    validate_alignment_gauge_feasible(
        mode=gauge_fix,
        active_mask=active_mask_tuple,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
    )
    return _AlignSetupState(
        cfg=cfg,
        observer_fn=observer_fn,
        n_views=n_views,
        x=x,
        params5=params5,
        frozen_params5=frozen_params5,
        active_mask_tuple=active_mask_tuple,
        active_mask_bool=active_mask_bool,
        active_col_indices_np=active_col_indices_np,
        active_names=active_names,
        active_mask=active_mask,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        gauge_fix=gauge_fix,
        gauge_dofs=gauge_dofs,
    )


def _build_alignment_runtime_context(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    n_views: int,
    active_mask: jnp.ndarray,
    det_grid_override: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> AlignmentRuntimeContext:
    pose_stack = stack_view_poses(geometry, n_views)
    validate_pose_stack(pose_stack, n_views, context="align geometry")
    det_grid = (
        get_detector_grid_device(detector) if det_grid_override is None else det_grid_override
    )

    smoothness_weights = (
        jnp.array(
            [cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans],
            dtype=jnp.float32,
        )
        * active_mask
    )
    smoothness_gram = _second_difference_gram(n_views)
    smoothness_weights_sq = smoothness_weights * smoothness_weights
    medium_smoothness_weights_sq = smoothness_weights_sq * jnp.float32(0.4)
    trans_only_smoothness_weights_sq = smoothness_weights_sq.at[:3].set(0.0)
    light_smoothness_weights_sq = smoothness_weights_sq * jnp.float32(0.25)

    active_loss_spec = resolve_loss_for_level(cfg.loss, level_factor=1)
    active_loss_name = loss_spec_name(active_loss_spec)
    loss_adapter = build_loss_adapter(active_loss_spec, projections)
    loss_mask = getattr(loss_adapter.state, "mask", None)
    chunk_size = max(1, int(cfg.views_per_batch))
    chunk_size = min(chunk_size, n_views)
    nv = int(projections.shape[1])
    nu = int(projections.shape[2])

    return AlignmentRuntimeContext(
        pose_stack=pose_stack,
        det_grid=det_grid,
        smoothness_weights=smoothness_weights,
        smoothness_gram=smoothness_gram,
        smoothness_weights_sq=smoothness_weights_sq,
        medium_smoothness_weights_sq=medium_smoothness_weights_sq,
        trans_only_smoothness_weights_sq=trans_only_smoothness_weights_sq,
        light_smoothness_weights_sq=light_smoothness_weights_sq,
        volume_mask=_build_alignment_volume_mask(
            grid,
            detector,
            mask_vol=str(getattr(cfg, "mask_vol", "off")),
        ),
        active_loss_name=active_loss_name,
        loss_adapter=loss_adapter,
        loss_mask=loss_mask,
        has_loss_mask=loss_mask is not None,
        supports_gauss_newton=loss_adapter.supports_gauss_newton,
        objective_provenance=ObjectiveProvenance(
            outer_loss_source="AlignmentLossSpec",
            outer_loss_kind=active_loss_name,
            inner_data_term="current_reconstruction",
            inner_regulariser="none",
            validation_split="none",
            differentiation_mode="none",
            initialization_policy="current_level_volume",
        ).to_dict(),
        nv=nv,
        nu=nu,
        chunk_size=chunk_size,
        num_chunks=(n_views + chunk_size - 1) // chunk_size,
        empty_loss_mask_chunk=jnp.zeros((chunk_size, nv, nu), dtype=jnp.float32),
    )


def _build_alignment_step_contexts(
    *,
    opt_mode: str,
    runtime: AlignmentRuntimeContext,
    objective: PoseObjectiveBundle,
    setup: _AlignSetupState,
    motion_ctx: PoseMotionContext,
    motion_model: Any,
    constraint_ctx: PoseConstraintContext,
    motion_loss_and_grad: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
) -> _AlignmentStepContexts:
    return (
        AlignmentStepOptimizer(opt_mode=opt_mode),
        AlignmentStepObjective(
            active_loss_name=runtime.active_loss_name,
            ls_like=runtime.supports_gauss_newton,
            align_loss=objective.align_loss,
            align_loss_jit=objective.align_loss_jit,
            loss_and_grad_manual=objective.loss_and_grad_manual,
            gn_update_all=objective.gn_update_all,
        ),
        AlignmentStepConstraints(
            active_mask=setup.active_mask,
            active_col_indices_np=setup.active_col_indices_np,
            frozen_params5=setup.frozen_params5,
            bounds_lower=constraint_ctx.bounds_lower,
            bounds_upper=constraint_ctx.bounds_upper,
            apply_full_constraints=constraint_ctx.apply_full_constraints,
            apply_full_constraints_with_stats=constraint_ctx.apply_full_constraints_with_stats,
        ),
        AlignmentStepMotion(
            active_coeff_indices=motion_ctx.active_coeff_indices,
            project_params_to_smooth=motion_ctx.project_params_to_smooth,
            coeffs_to_constrained_params=motion_ctx.coeffs_to_constrained_params,
            use_smooth_pose_model=motion_ctx.use_smooth_pose_model,
            motion_loss_and_grad=motion_loss_and_grad,
            motion_model=motion_model,
        ),
        AlignmentStepSmoothing(
            smoothness_gram=runtime.smoothness_gram,
            light_smoothness_weights_sq=runtime.light_smoothness_weights_sq,
            medium_smoothness_weights_sq=runtime.medium_smoothness_weights_sq,
            smoothness_weights_sq=runtime.smoothness_weights_sq,
            trans_only_smoothness_weights_sq=runtime.trans_only_smoothness_weights_sq,
        ),
        AlignmentStepGauge(
            gauge_fix=constraint_ctx.gauge_fix,
            gauge_dofs=tuple(constraint_ctx.gauge_dofs),
        ),
    )


@dataclass
class _AlignLoopState:
    x: jnp.ndarray
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None
    final_gauge_stats: dict[str, float | str | list[str]]
    l_prev: float | None
    small_impr_streak: int
    loss_hist: list[float]
    outer_stats: list[OuterStat]
    stopped_by_observer: bool
    observer_action: ObserverAction


def _emit_alignment_checkpoint(
    *,
    checkpoint_callback: AlignCheckpointCallback | None,
    state: _AlignLoopState,
    wall_start: float,
) -> None:
    if checkpoint_callback is None:
        return
    checkpoint_callback(
        AlignResumeState(
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            start_outer_iter=len(state.outer_stats),
            loss=list(state.loss_hist),
            outer_stats=[dict(stat) for stat in state.outer_stats],
            L=(float(state.l_prev) if state.l_prev is not None else None),
            small_impr_streak=int(state.small_impr_streak),
            elapsed_offset=float(time.perf_counter() - wall_start),
        )
    )


def _initial_outer_stat(
    *,
    outer_idx: int,
    active_loss_name: str,
    recon_algo: str,
    objective_provenance: Mapping[str, object],
    backend_provenance: Mapping[str, object],
    cfg: AlignConfig,
    profile_policy: Mapping[str, object],
) -> OuterStat:
    return {
        "outer_idx": outer_idx,
        "loss_kind": active_loss_name,
        "recon_algo": recon_algo,
        "objective_kind": "fixed_volume",
        "objective_provenance": dict(objective_provenance),
        "backend_provenance": dict(backend_provenance),
        "outer_loss_kind": active_loss_name,
        "align_profile": str(cfg.align_profile),
        "quality_tier": str(cfg.quality_tier),
        "fallback_policy": str(cfg.fallback_policy),
        "profile_policy": dict(profile_policy),
    }


def _observer_break_decision(
    *,
    observer_fn: ObserverCallback | None,
    state: _AlignLoopState,
    stat: OuterStat,
) -> bool:
    if observer_fn is None:
        return False
    observer_action = _normalize_observer_action(observer_fn(state.x, state.params5, dict(stat)))
    state.observer_action = observer_action
    stat["observer_action"] = observer_action
    stat["observer_stop"] = observer_action != "continue"
    if observer_action == "continue":
        return False
    state.stopped_by_observer = observer_action == "stop_run"
    return True


def _early_stop_break_decision(
    *,
    cfg: AlignConfig,
    state: _AlignLoopState,
    stat: OuterStat,
    rel_impr: float | None,
    outer_idx: int,
) -> bool:
    if not cfg.early_stop:
        return False
    if rel_impr is None:
        state.small_impr_streak = 0
        return False
    rel_for_patience = rel_impr
    if (not math.isfinite(rel_for_patience)) or (rel_for_patience < 0.0):
        rel_for_patience = 0.0
    if rel_for_patience < float(cfg.early_stop_rel_impr):
        state.small_impr_streak += 1
    else:
        state.small_impr_streak = 0
    if state.small_impr_streak < int(cfg.early_stop_patience):
        return False
    if cfg.log_summary:
        logging.info(
            "Early stop after %d outer iters (%s elapsed): "
            "rel_impr=%.3e < %.3e for %d consecutive outers",
            outer_idx,
            format_duration(stat.get("cumulative_time")),
            float(rel_impr),
            float(cfg.early_stop_rel_impr),
            int(cfg.early_stop_patience),
        )
    return True


def _run_align_outer_iteration(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    cfg: AlignConfig,
    state: _AlignLoopState,
    step_contexts: _AlignmentStepContexts,
    outer_idx: int,
    recon_algo: str,
    active_loss_name: str,
    objective_provenance: Mapping[str, object],
    backend_provenance: Mapping[str, object],
    profile_policy: Mapping[str, object],
    wall_start: float,
) -> tuple[OuterStat, float | None]:
    stat = _initial_outer_stat(
        outer_idx=outer_idx,
        active_loss_name=active_loss_name,
        recon_algo=recon_algo,
        objective_provenance=objective_provenance,
        backend_provenance=backend_provenance,
        cfg=cfg,
        profile_policy=profile_policy,
    )
    outer_start = time.perf_counter()
    state.x, state.l_prev, recon_stat = _run_reconstruction_step(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        det_grid=det_grid,
        params5=state.params5,
        x=state.x,
        cfg=cfg,
        L_prev=state.l_prev,
        outer_idx=outer_idx,
        recon_algo=recon_algo,
    )
    stat.update(recon_stat)
    if bool(stat.get("reconstruction_failed", False)):
        stat["outer_time"] = time.perf_counter() - outer_start
        stat["cumulative_time"] = time.perf_counter() - wall_start
        state.outer_stats.append(stat)
        return stat, None
    optimizer, objective, constraints, motion, smoothing, gauge = step_contexts
    (
        state.params5,
        state.motion_coeffs,
        state.final_gauge_stats,
        total_loss,
        rel_impr,
        align_stat,
    ) = _run_alignment_step(
        cfg=cfg,
        optimizer=optimizer,
        objective=objective,
        constraints=constraints,
        motion=motion,
        smoothing=smoothing,
        gauge=gauge,
        params5_in=state.params5,
        motion_coeffs_in=state.motion_coeffs,
        vol=state.x,
        loss_hist=state.loss_hist,
    )
    state.loss_hist.append(total_loss)
    stat.update(align_stat)
    stat["outer_time"] = time.perf_counter() - outer_start
    stat["cumulative_time"] = time.perf_counter() - wall_start
    state.outer_stats.append(stat)
    return stat, rel_impr


def _log_align_outer_summary(
    *,
    stat: OuterStat,
    cfg: AlignConfig,
    recon_algo: str,
) -> None:
    for line in _format_outer_summary_lines(stat, cfg=cfg, recon_algo=recon_algo):
        logging.info(line)


def _log_alignment_completion(
    *,
    cfg: AlignConfig,
    outer_stats: list[OuterStat],
    wall_start: float,
) -> None:
    if not (cfg.log_summary and outer_stats):
        return
    recon_total = sum(
        float(s.get("recon_time", 0.0)) for s in outer_stats if s.get("recon_time") is not None
    )
    align_total = sum(
        float(s.get("align_time", 0.0)) for s in outer_stats if s.get("align_time") is not None
    )
    wall_total = time.perf_counter() - wall_start
    logging.info(
        "Alignment completed in %s (recon %s, align %s over %d outer iters)",
        format_duration(wall_total),
        format_duration(recon_total),
        format_duration(align_total),
        len(outer_stats),
    )
    first_loss = outer_stats[0].get("loss_before")
    final_loss = outer_stats[-1].get("loss_after")
    if (first_loss is not None) and (final_loss is not None):
        total_delta = final_loss - first_loss
        rel_pct = (total_delta / first_loss) * 100.0 if abs(first_loss) > 1e-12 else None
        rel_str = f", {rel_pct:+.2f}%" if rel_pct is not None else ""
        logging.info(
            "  Loss %s -> %s (Δ %s%s)",
            f"{first_loss:.3e}",
            f"{final_loss:.3e}",
            f"{total_delta:+.3e}",
            rel_str,
        )
    best_loss = min(
        (s.get("loss_after") for s in outer_stats if s.get("loss_after") is not None),
        default=None,
    )
    if best_loss is not None and final_loss is not None and best_loss < final_loss:
        logging.info("  Best loss observed: %.3e", best_loss)


def _final_pose_align_info(
    *,
    state: _AlignLoopState,
    active_loss_name: str,
    recon_algo: str,
    l_prev: float | None,
    wall_total: float,
    profile_policy: Mapping[str, object],
    cfg: AlignConfig,
    motion_model: Any,
    constraint_ctx: PoseConstraintContext,
) -> AlignInfo:
    outer_stats = state.outer_stats
    return {
        "loss": state.loss_hist,
        "loss_kind": active_loss_name,
        "recon_algo": recon_algo,
        "L": (float(l_prev) if l_prev is not None else None),
        "outer_stats": outer_stats,
        "stopped_by_observer": state.stopped_by_observer,
        "observer_action": state.observer_action,
        "wall_time_total": float(wall_total),
        "align_profile": str(cfg.align_profile),
        "profile_policy": dict(profile_policy),
        "quality_tier": str(cfg.quality_tier),
        "fallback_policy": str(cfg.fallback_policy),
        "pose_model": motion_model.name,
        "pose_model_variables": int(motion_model.variable_count),
        "per_view_variables": int(motion_model.per_view_variable_count),
        "pose_model_basis_shape": [
            int(motion_model.basis.shape[0]),
            int(motion_model.basis.shape[1]),
        ],
        "active_dofs": list(motion_model.active_names),
        "active_pose_dofs": list(motion_model.active_names),
        "active_geometry_dofs": [],
        "objective_kind": "fixed_volume",
        "objective_kinds": ["fixed_volume"] if outer_stats else [],
        "objective_provenance": (
            dict(outer_stats[-1].get("objective_provenance", {}))
            if outer_stats and isinstance(outer_stats[-1].get("objective_provenance"), Mapping)
            else None
        ),
        "backend_provenance": (
            dict(outer_stats[-1].get("backend_provenance", {}))
            if outer_stats and isinstance(outer_stats[-1].get("backend_provenance"), Mapping)
            else None
        ),
        "optimizer_kind": str(outer_stats[-1].get("optimizer_kind"))
        if outer_stats and outer_stats[-1].get("optimizer_kind") is not None
        else str(cfg.opt_method),
        "completed_outer_iters": len(outer_stats),
        "small_impr_streak": int(state.small_impr_streak),
        "motion_coeffs": state.motion_coeffs,
        "gauge_fix": constraint_ctx.gauge_fix,
        "gauge_fix_dofs": list(constraint_ctx.gauge_dofs),
        "gauge_fix_final": dict(state.final_gauge_stats),
    }


def align(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,  # (n_views, nv, nu)
    *,
    cfg: AlignConfig | None = None,
    init_x: jnp.ndarray | None = None,
    init_params5: jnp.ndarray | None = None,
    observer: ObserverCallback | None = None,
    resume_state: AlignResumeState | None = None,
    checkpoint_callback: AlignCheckpointCallback | None = None,
    det_grid_override: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignInfo]:
    """Alternating reconstruction + per-view alignment (5-DOF) on small cases.

    Returns (x, params5, info) with loss history and optional metrics.
    """
    setup = _prepare_align_setup(
        geometry,
        grid,
        detector,
        projections,
        cfg=cfg,
        init_x=init_x,
        init_params5=init_params5,
        observer=observer,
        resume_state=resume_state,
    )
    cfg = setup.cfg
    observer_fn = setup.observer_fn
    n_views = setup.n_views
    x = setup.x
    params5 = setup.params5
    active_names = setup.active_names
    active_mask = setup.active_mask
    constraint_ctx = PoseConstraintContext.from_setup(setup)

    params5, initial_gauge_stats = constraint_ctx.apply_full_constraints_with_stats(params5)
    logging.info("Alignment gauge fix: %s", constraint_ctx.description())
    final_gauge_stats = dict(initial_gauge_stats)

    motion_ctx = PoseMotionContext.build(
        geometry=geometry,
        cfg=cfg,
        n_views=n_views,
        active_names=active_names,
        params5=params5,
        resume_state=resume_state,
        constraint_ctx=constraint_ctx,
    )
    motion_model = motion_ctx.motion_model
    params5 = motion_ctx.params5
    motion_coeffs = motion_ctx.motion_coeffs
    if motion_ctx.use_smooth_pose_model:
        _, initial_gauge_stats = constraint_ctx.apply_full_constraints_with_stats(params5)
        final_gauge_stats = dict(initial_gauge_stats)

    start_outer_iter = int(resume_state.start_outer_iter) if resume_state is not None else 0
    if start_outer_iter < 0 or start_outer_iter > int(cfg.outer_iters):
        raise ValueError(
            "align resume_state start_outer_iter must be between 0 and cfg.outer_iters; "
            f"got {start_outer_iter} for outer_iters={int(cfg.outer_iters)}"
        )
    runtime = _build_alignment_runtime_context(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        cfg=cfg,
        n_views=n_views,
        active_mask=active_mask,
        det_grid_override=det_grid_override,
    )

    objective = _build_pose_objective_bundle(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        cfg=cfg,
        n_views=n_views,
        active_mask=active_mask,
        runtime=runtime,
    )

    motion_loss_and_grad = motion_ctx.loss_and_grad_for(objective.align_loss)

    opt_mode = str(cfg.opt_method).lower()
    elapsed_offset = float(resume_state.elapsed_offset) if resume_state is not None else 0.0
    wall_start = time.perf_counter() - elapsed_offset
    recon_algo = str(cfg.recon_algo)
    loop_state = _AlignLoopState(
        x=x,
        params5=params5,
        motion_coeffs=motion_coeffs,
        final_gauge_stats=final_gauge_stats,
        l_prev=resume_state.L if resume_state is not None else cfg.recon_L,
        small_impr_streak=int(resume_state.small_impr_streak) if resume_state is not None else 0,
        loss_hist=list(resume_state.loss) if resume_state is not None else [],
        outer_stats=(
            [dict(stat) for stat in resume_state.outer_stats] if resume_state is not None else []
        ),
        stopped_by_observer=False,
        observer_action="continue",
    )

    step_contexts = _build_alignment_step_contexts(
        opt_mode=opt_mode,
        runtime=runtime,
        objective=objective,
        setup=setup,
        motion_ctx=motion_ctx,
        motion_loss_and_grad=motion_loss_and_grad,
        motion_model=motion_model,
        constraint_ctx=constraint_ctx,
    )

    backend_provenance = alignment_projector_backend_provenance(
        pose_stack=runtime.pose_stack,
        grid=grid,
        detector=detector,
        volume=loop_state.x,
        det_grid=runtime.det_grid,
        projector_backend=cfg.projector_backend,
        require_differentiable_projector=True,
        api_surface="alignment.pose_objective",
        gather_dtype=cfg.gather_dtype,
    ).to_dict()
    profile_policy = profile_policy_from_config(cfg).to_dict()

    iter_range = range(start_outer_iter, int(cfg.outer_iters))
    for it in progress_iter(
        iter_range,
        total=max(0, int(cfg.outer_iters) - start_outer_iter),
        desc="Align: outer iters",
    ):
        outer_idx = it + 1
        stat, rel_impr = _run_align_outer_iteration(
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            det_grid=runtime.det_grid,
            cfg=cfg,
            state=loop_state,
            step_contexts=step_contexts,
            outer_idx=outer_idx,
            recon_algo=recon_algo,
            active_loss_name=runtime.active_loss_name,
            objective_provenance=runtime.objective_provenance,
            backend_provenance=backend_provenance,
            profile_policy=profile_policy,
            wall_start=wall_start,
        )
        if cfg.log_summary:
            _log_align_outer_summary(stat=stat, cfg=cfg, recon_algo=recon_algo)
        should_break = _observer_break_decision(
            observer_fn=observer_fn,
            state=loop_state,
            stat=stat,
        )
        should_break = should_break or bool(stat.get("reconstruction_failed", False))
        should_break = should_break or _early_stop_break_decision(
            cfg=cfg,
            state=loop_state,
            stat=stat,
            rel_impr=rel_impr,
            outer_idx=outer_idx,
        )
        _emit_alignment_checkpoint(
            checkpoint_callback=checkpoint_callback,
            state=loop_state,
            wall_start=wall_start,
        )
        if should_break:
            break

    _log_alignment_completion(cfg=cfg, outer_stats=loop_state.outer_stats, wall_start=wall_start)
    wall_total = time.perf_counter() - wall_start
    info = _final_pose_align_info(
        state=loop_state,
        active_loss_name=runtime.active_loss_name,
        recon_algo=recon_algo,
        l_prev=loop_state.l_prev,
        wall_total=wall_total,
        profile_policy=profile_policy,
        cfg=cfg,
        motion_model=motion_model,
        constraint_ctx=constraint_ctx,
    )
    return loop_state.x, loop_state.params5, info
