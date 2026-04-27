from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import math
import time
from typing import Callable, Iterable, Literal, Mapping, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from ..core.geometry.base import Geometry, Grid, Detector
from ..core.geometry.views import stack_view_poses
from ..core.projector import forward_project_view_T, get_detector_grid_device
from ..core.validation import (
    validate_grid,
    validate_optional_same_shape,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)
from ..recon.fista_tv import FistaConfig, fista_tv
from ..recon._tv_ops import Regulariser
from ..utils.logging import progress_iter, format_duration
from .parametrizations import se3_from_5d
from .dofs import (
    DofBounds,
    ScopedAlignmentDofs,
    bounds_vectors,
    normalize_alignment_dofs,
    normalize_bounds,
)
from .diagnostics import GaugePolicy, validate_active_gauge_policy
from ._loss_adapters import build_loss_adapter
from ._loss_specs import (
    L2OtsuLossSpec,
    AlignmentLossConfig,
    loss_spec_name,
    resolve_loss_for_level,
    validate_loss_schedule_levels,
)
from .motion_models import (
    build_pose_motion_model,
    expand_motion_coefficients,
    fit_motion_coefficients,
    scan_coordinate_from_geometry,
)
from .gauge import (
    GaugeFixMode,
    active_gauge_dofs,
    apply_alignment_gauge,
    gauge_stats_to_python,
    normalize_gauge_fix,
    validate_alignment_gauge_feasible,
)
from .optimizers import (
    PoseLbfgsConfig,
    PoseOptimizationContext,
    run_pose_lbfgs,
)
from .objectives import ObjectiveProvenance, project_and_score_stack
from .dof_specs import ActiveParameterView
from .geometry_applier import BaseGeometryArrays, apply_setup_to_detector_grid, setup_axis_unit
from .schedules import (
    AlignmentSchedule,
    ResolvedAlignmentSchedule,
    ResolvedAlignmentStage,
    resolve_alignment_schedule,
)
from .state import AlignmentState, PoseState, SetupGeometryState, alignment_state_from_checkpoint
from .geometry_blocks import (
    add_geometry_acquisition_diagnostics,
    normalize_geometry_dofs,
    summarize_geometry_calibration_stats,
)
from ..utils.fov import cylindrical_mask_xy
from ._observer import (
    LegacyObserverCallback,
    ObserverAction,
    ObserverCallback,
    OuterStat,
    _normalize_observer_action,
    adapt_legacy_observer,
)
from ._results import (
    AlignInfo,
    AlignMultiresInfo,
    AlignMultiresResumeState,
    AlignResumeState,
    enrich_multires_stage_stat as _enrich_multires_stage_stat,
    record_reconstruction_info as _record_reconstruction_info,
)
from ._pose_stage import (
    _evaluate_align_loss,
    _is_expected_align_eval_failure,
    _second_difference_gram,
    _select_gn_candidate,
    _should_prefer_gn_candidate,
    _smooth_gn_candidate,
)
from ._reconstruction_stage import _run_reconstruction_step
from ._setup_stage import (
    _geometry_calibration_payload,
    _geometry_with_setup_state,
    _optimize_setup_geometry_bilevel_for_level,
)
from ._config import (
    AlignConfig,
    _active_dof_mask_for_cfg,
    _active_dofs_for_cfg,
    _resolved_schedule_for_cfg,
)






def _build_alignment_volume_mask(
    grid: Grid,
    detector: Detector,
    *,
    mask_vol: str,
) -> jnp.ndarray | None:
    mask_mode = str(mask_vol).lower()
    if mask_mode in ("off", "none", ""):
        return None
    if mask_mode not in ("cyl", "cylindrical"):
        raise ValueError("align mask_vol must be one of 'off' or 'cyl'")
    try:
        m_xy = cylindrical_mask_xy(grid, detector)
        return jnp.asarray(m_xy, dtype=jnp.float32)[:, :, None]
    except Exception as exc:
        raise ValueError(f"Failed to apply requested mask_vol={mask_mode!r}") from exc




@dataclass(frozen=True)
class _AlignSetupState:
    cfg: AlignConfig
    observer_fn: ObserverCallback | None
    n_views: int
    x: jnp.ndarray
    params5: jnp.ndarray
    frozen_params5: jnp.ndarray
    active_mask_tuple: tuple[bool, bool, bool, bool, bool]
    active_mask_bool: jnp.ndarray
    active_col_indices_np: np.ndarray
    active_names: tuple[str, ...]
    active_mask: jnp.ndarray
    bounds_lower: jnp.ndarray
    bounds_upper: jnp.ndarray
    gauge_fix: GaugeFixMode
    gauge_dofs: tuple[str, ...]


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
    observer_fn = adapt_legacy_observer(observer) if observer is not None else None
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
    frozen_params5 = setup.frozen_params5
    active_mask_tuple = setup.active_mask_tuple
    active_mask_bool = setup.active_mask_bool
    active_col_indices_np = setup.active_col_indices_np
    active_names = setup.active_names
    active_mask = setup.active_mask
    bounds_lower = setup.bounds_lower
    bounds_upper = setup.bounds_upper
    gauge_fix = setup.gauge_fix
    gauge_dofs = setup.gauge_dofs

    def _apply_param_constraints(candidate: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(candidate, bounds_lower, bounds_upper)
        return jnp.where(active_mask_bool, clipped, frozen_params5)

    def _apply_full_constraints(candidate: jnp.ndarray) -> jnp.ndarray:
        constrained = _apply_param_constraints(candidate)
        gauged, _ = apply_alignment_gauge(
            constrained,
            mode=gauge_fix,
            active_mask=active_mask_tuple,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
        )
        return _apply_param_constraints(gauged)

    def _apply_full_constraints_with_stats(
        candidate: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, float | str | list[str]]]:
        constrained = _apply_param_constraints(candidate)
        gauged, stats = apply_alignment_gauge(
            constrained,
            mode=gauge_fix,
            active_mask=active_mask_tuple,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
        )
        gauged = _apply_param_constraints(gauged)
        final_gauged, final_stats = apply_alignment_gauge(
            gauged,
            mode=gauge_fix,
            active_mask=active_mask_tuple,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
        )
        final_gauged = _apply_param_constraints(final_gauged)
        stats_py = gauge_stats_to_python(stats)
        final_py = gauge_stats_to_python(final_stats)
        stats_py["dx_mean_after"] = final_py["dx_mean_after"]
        stats_py["dz_mean_after"] = final_py["dz_mean_after"]
        return final_gauged, stats_py

    params5, initial_gauge_stats = _apply_full_constraints_with_stats(params5)
    gauge_dofs_label = ",".join(gauge_dofs) if gauge_dofs else "no translation DOFs"
    gauge_desc = (
        "none"
        if gauge_fix == "none"
        else f"{gauge_fix} over active {gauge_dofs_label}"
    )
    logging.info("Alignment gauge fix: %s", gauge_desc)
    final_gauge_stats = dict(initial_gauge_stats)

    scan_coordinate = scan_coordinate_from_geometry(geometry, n_views)
    motion_model = build_pose_motion_model(
        pose_model=str(cfg.pose_model),
        n_views=n_views,
        active_dofs=active_names,
        frozen_params5=frozen_params5,
        scan_coordinate=scan_coordinate,
        knot_spacing=int(cfg.knot_spacing),
        degree=int(cfg.degree),
    )
    use_smooth_pose_model = motion_model.name != "per_view"
    motion_coeffs = None
    active_coeff_indices = jnp.asarray(motion_model.active_indices, dtype=jnp.int32)

    def _coeffs_to_constrained_params(coeffs: jnp.ndarray) -> jnp.ndarray:
        return _apply_full_constraints(expand_motion_coefficients(motion_model, coeffs))

    def _project_params_to_smooth(candidate: jnp.ndarray) -> jnp.ndarray:
        constrained = _apply_full_constraints(candidate)
        coeffs = fit_motion_coefficients(motion_model, constrained)
        return _coeffs_to_constrained_params(coeffs)

    if use_smooth_pose_model:
        motion_coeffs = fit_motion_coefficients(motion_model, params5)
        params5 = _coeffs_to_constrained_params(motion_coeffs)
        motion_coeffs = fit_motion_coefficients(motion_model, params5)
        if resume_state is not None and resume_state.motion_coeffs is not None:
            resume_coeffs = jnp.asarray(resume_state.motion_coeffs, dtype=jnp.float32)
            if tuple(resume_coeffs.shape) != tuple(motion_coeffs.shape):
                raise ValueError(
                    "align resume_state motion_coeffs shape mismatch: "
                    f"expected {tuple(motion_coeffs.shape)}, got {tuple(resume_coeffs.shape)}"
                )
            motion_coeffs = resume_coeffs
            params5 = _coeffs_to_constrained_params(motion_coeffs)
        _, initial_gauge_stats = _apply_full_constraints_with_stats(params5)
        final_gauge_stats = dict(initial_gauge_stats)

    start_outer_iter = int(resume_state.start_outer_iter) if resume_state is not None else 0
    if start_outer_iter < 0 or start_outer_iter > int(cfg.outer_iters):
        raise ValueError(
            "align resume_state start_outer_iter must be between 0 and cfg.outer_iters; "
            f"got {start_outer_iter} for outer_iters={int(cfg.outer_iters)}"
        )
    loss_hist = list(resume_state.loss) if resume_state is not None else []
    stopped_by_observer = False
    observer_action: ObserverAction = "continue"

    # Precompute nominal poses once
    T_nom_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_nom_all, n_views, context="align geometry")

    # Precompute detector grid once (device arrays) to avoid repeated transfers/logging
    det_grid = get_detector_grid_device(detector) if det_grid_override is None else det_grid_override

    # Static smoothness weights to avoid rebuilding inside jitted loss
    W_weights = (
        jnp.array([cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans], dtype=jnp.float32)
        * active_mask
    )
    smoothness_gram = _second_difference_gram(n_views)
    smoothness_weights_sq = W_weights * W_weights
    medium_smoothness_weights_sq = smoothness_weights_sq * jnp.float32(0.4)
    trans_only_smoothness_weights_sq = smoothness_weights_sq.at[:3].set(0.0)
    light_smoothness_weights_sq = smoothness_weights_sq * jnp.float32(0.25)

    # Optional static volume mask before projection
    vol_mask = _build_alignment_volume_mask(
        grid,
        detector,
        mask_vol=str(getattr(cfg, "mask_vol", "off")),
    )

    # Build per-view loss once (may precompute masks on targets)
    active_loss_spec = resolve_loss_for_level(cfg.loss, level_factor=1)
    active_loss_name = loss_spec_name(active_loss_spec)
    loss_adapter = build_loss_adapter(active_loss_spec, projections)
    per_view_loss_fn = loss_adapter.per_view_loss
    loss_state = loss_adapter.state
    active_objective_provenance = ObjectiveProvenance(
        outer_loss_source="AlignmentLossSpec",
        outer_loss_kind=active_loss_name,
        inner_data_term="current_reconstruction",
        inner_regulariser="none",
        validation_split="none",
        differentiation_mode="none",
        initialization_policy="current_level_volume",
    ).to_dict()

    nv = int(projections.shape[1])
    nu = int(projections.shape[2])
    chunk_size = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n_views
    chunk_size = min(chunk_size, n_views)
    num_chunks = (n_views + chunk_size - 1) // chunk_size
    loss_mask = getattr(loss_state, "mask", None)
    has_loss_mask = loss_mask is not None
    ls_like = loss_adapter.supports_gauss_newton

    def _chunk_schedule(i: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        i = jnp.asarray(i, dtype=jnp.int32)
        start = i * jnp.int32(chunk_size)
        remaining = jnp.maximum(0, jnp.int32(n_views) - start)
        valid = jnp.minimum(jnp.int32(chunk_size), remaining)
        shift = jnp.int32(chunk_size) - valid
        start_shifted = jnp.maximum(0, start - shift)
        idx = jnp.arange(chunk_size, dtype=jnp.int32)
        vmask = (idx >= (jnp.int32(chunk_size) - valid)).astype(jnp.float32)
        return start_shifted, vmask, start_shifted + idx

    def _apply_vol_mask(vol: jnp.ndarray) -> jnp.ndarray:
        return vol * vol_mask if vol_mask is not None else vol

    def align_loss(params5, vol):
        # Compose augmented poses
        # Current convention: per-view misalignment parameters act in the object
        # frame and are post-multiplied: T_world_from_obj_aug = T_nom @ T_delta.
        # This is consistent across parallel CT and laminography sample-frame.
        T_aug = T_nom_all @ jax.vmap(se3_from_5d)(params5)  # (n_views, 4, 4)
        loss_tot = project_and_score_stack(
            pose_stack=T_aug,
            grid=grid,
            detector=detector,
            volume=_apply_vol_mask(vol),
            det_grid=det_grid,
            targets=projections,
            loss_adapter=loss_adapter,
            views_per_batch=chunk_size,
            projector_unroll=int(cfg.projector_unroll),
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            view_indices=jnp.arange(n_views, dtype=jnp.int32),
        )

        # Smoothness prior across views (2nd difference)
        loss = loss_tot
        if int(params5.shape[0]) >= 3:
            d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
            loss = loss + jnp.sum((d2 * W_weights) ** 2)
        return loss

    # Value function for whole batch (forward only) kept for logging and line search
    align_loss_jit = jax.jit(align_loss)

    if use_smooth_pose_model:

        def motion_align_loss(coeffs, vol):
            return align_loss(_coeffs_to_constrained_params(coeffs), vol)

        motion_loss_and_grad = jax.jit(jax.value_and_grad(motion_align_loss))
    else:
        motion_loss_and_grad = None

    empty_loss_mask_chunk = jnp.zeros((chunk_size, nv, nu), dtype=jnp.float32)

    def _loss_mask_chunk(start_shifted: jnp.ndarray) -> jnp.ndarray:
        if has_loss_mask:
            return jax.lax.dynamic_slice(
                loss_mask,
                (start_shifted, 0, 0),
                (chunk_size, nv, nu),
            )
        return empty_loss_mask_chunk

    def _loss_mask_arg(mask_i: jnp.ndarray) -> jnp.ndarray | None:
        return mask_i[None, ...] if has_loss_mask else None

    def _one_view_loss(p5_i, T_nom_i, y_i, masked_vol, mask_i, view_idx):
        T_i = T_nom_i @ se3_from_5d(p5_i)
        pred_i = forward_project_view_T(
            T_i,
            grid,
            detector,
            masked_vol,
            use_checkpoint=cfg.checkpoint_projector,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
            det_grid=det_grid,
        )
        view_indices = jnp.expand_dims(jnp.asarray(view_idx, dtype=jnp.int32), axis=0)
        lvec = per_view_loss_fn(
            pred_i[None, ...],
            y_i[None, ...],
            _loss_mask_arg(mask_i),
            view_indices=view_indices,
        )
        return lvec[0]

    one_view_val_and_grad_batch = jax.jit(
        jax.vmap(
            jax.value_and_grad(_one_view_loss),
            in_axes=(0, 0, 0, None, 0, 0),
        )
    )

    def _apply_smoothness_gradient(
        params5_in: jnp.ndarray,
        total: jnp.ndarray,
        grad: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if int(params5_in.shape[0]) < 3:
            return total, grad
        d2 = params5_in[:-2] - 2.0 * params5_in[1:-1] + params5_in[2:]
        w = (
            jnp.array(
                [cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans],
                jnp.float32,
            )
            * active_mask
        )
        total = total + jnp.sum((d2 * w) ** 2)
        ww = (w**2) * 2.0
        grad = grad.at[1:-1].add(-2.0 * d2 * ww)
        grad = grad.at[0:-2].add(1.0 * d2 * ww)
        grad = grad.at[2:].add(1.0 * d2 * ww)
        return total, grad

    def loss_and_grad_manual(params5, vol):
        masked_vol = _apply_vol_mask(vol)

        def body(carry, i):
            total, g = carry
            start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
            params_chunk = jax.lax.dynamic_slice(
                params5,
                (start_shifted, 0),
                (chunk_size, params5.shape[1]),
            )
            T_nom_chunk = jax.lax.dynamic_slice(
                T_nom_all,
                (start_shifted, 0, 0),
                (chunk_size, 4, 4),
            )
            y_chunk = jax.lax.dynamic_slice(
                projections,
                (start_shifted, 0, 0),
                (chunk_size, nv, nu),
            )
            lvec, g_chunk = one_view_val_and_grad_batch(
                params_chunk,
                T_nom_chunk,
                y_chunk,
                masked_vol,
                _loss_mask_chunk(start_shifted),
                view_idx_chunk,
            )
            total = total + jnp.sum(lvec * vmask)
            g = g.at[view_idx_chunk].add(g_chunk * vmask[:, None])
            return (total, g), None

        init = (jnp.float32(0.0), jnp.zeros_like(params5))
        (total, g), _ = jax.lax.scan(body, init, jnp.arange(num_chunks, dtype=jnp.int32))
        return _apply_smoothness_gradient(params5, total, g)

    loss_and_grad_manual = jax.jit(loss_and_grad_manual)

    # Gauss–Newton (Levenberg–Marquardt) single-view update
    def _pred_flat(T_i, masked_vol):
        return forward_project_view_T(
            T_i,
            grid,
            detector,
            masked_vol,
            use_checkpoint=cfg.checkpoint_projector,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
            det_grid=det_grid,
        ).ravel()

    def _gn_update_one(p5_i, T_nom_i, y_i, vol, w_i):
        def f(p5):
            T_i = T_nom_i @ se3_from_5d(p5)
            r = _pred_flat(T_i, vol) - y_i.ravel()
            return w_i.ravel() * r

        # J^T r
        r = f(p5_i)
        _, vjp = jax.vjp(f, p5_i)
        g = vjp(r)[0]
        # J^T J via 5 JVPs
        eye5 = jnp.eye(5, dtype=jnp.float32)

        def jvp_col(v):
            return jax.jvp(f, (p5_i,), (v,))[1]

        cols = jax.vmap(jvp_col)(eye5)
        H = cols @ cols.T
        lam = jnp.float32(cfg.gn_damping)
        active = active_mask.astype(H.dtype)
        inactive = jnp.float32(1.0) - active
        H_active = H * active[:, None] * active[None, :]
        system = H_active + lam * jnp.diag(active) + jnp.diag(inactive)
        rhs = -g * active
        dp = jnp.linalg.solve(system, rhs)
        return dp * active

    _gn_update_batch = jax.jit(jax.vmap(_gn_update_one, in_axes=(0, 0, 0, None, 0)))

    def _ls_weight_chunk(y_chunk: jnp.ndarray, mask_chunk: jnp.ndarray) -> jnp.ndarray:
        return loss_adapter.gauss_newton_weights(y_chunk, mask_chunk if has_loss_mask else None)

    def _gn_update_all(params5, vol):
        masked_vol = _apply_vol_mask(vol)

        def body(dp_acc, i):
            start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
            params_chunk = jax.lax.dynamic_slice(
                params5,
                (start_shifted, 0),
                (chunk_size, params5.shape[1]),
            )
            T_chunk = jax.lax.dynamic_slice(
                T_nom_all,
                (start_shifted, 0, 0),
                (chunk_size, 4, 4),
            )
            y_chunk = jax.lax.dynamic_slice(
                projections,
                (start_shifted, 0, 0),
                (chunk_size, nv, nu),
            )
            dp_chunk = _gn_update_batch(
                params_chunk,
                T_chunk,
                y_chunk,
                masked_vol,
                _ls_weight_chunk(y_chunk, _loss_mask_chunk(start_shifted)),
            )
            dp_acc = dp_acc.at[view_idx_chunk].add(dp_chunk * vmask[:, None])
            return dp_acc, None

        dp0 = jnp.zeros_like(params5)
        dp_all, _ = jax.lax.scan(body, dp0, jnp.arange(num_chunks, dtype=jnp.int32))
        return dp_all

    _gn_update_all = jax.jit(_gn_update_all)

    # Reuse measured Lipschitz across outer iterations to avoid repeated power-method
    L_prev = resume_state.L if resume_state is not None else cfg.recon_L
    small_impr_streak = int(resume_state.small_impr_streak) if resume_state is not None else 0
    opt_mode = str(cfg.opt_method).lower()
    outer_stats: list[OuterStat] = (
        [dict(stat) for stat in resume_state.outer_stats] if resume_state is not None else []
    )
    elapsed_offset = float(resume_state.elapsed_offset) if resume_state is not None else 0.0
    wall_start = time.perf_counter() - elapsed_offset
    recon_algo = str(cfg.recon_algo)

    def _emit_checkpoint_state() -> None:
        if checkpoint_callback is None:
            return
        checkpoint_callback(
            AlignResumeState(
                x=x,
                params5=params5,
                motion_coeffs=motion_coeffs,
                start_outer_iter=len(outer_stats),
                loss=list(loss_hist),
                outer_stats=[dict(stat) for stat in outer_stats],
                L=(float(L_prev) if L_prev is not None else None),
                small_impr_streak=int(small_impr_streak),
                elapsed_offset=float(time.perf_counter() - wall_start),
            )
        )

    def _log_outer_summary(stat: OuterStat) -> None:
        outer_idx = int(stat.get("outer_idx", 0))
        total_iters = int(cfg.outer_iters)
        total_time = format_duration(stat.get("outer_time"))
        elapsed = format_duration(stat.get("cumulative_time"))
        solver_label = str(stat.get("recon_algo") or recon_algo).upper()
        if cfg.log_compact:
            # Build compact one-liner with key fields
            parts: list[str] = [f"Outer {outer_idx}/{total_iters}"]
            # Recon summary
            rbits: list[str] = []
            rt = stat.get("recon_time")
            if rt is not None:
                rbits.append(f"{format_duration(rt)}")
            if stat.get("recon_retry"):
                rbits.append("retry")
            lm = stat.get("L_meas")
            ln = stat.get("L_next")
            if (lm is not None) and (ln is not None):
                rbits.append(f"L {lm:.2e}->{ln:.2e}")
            ff = stat.get("recon_loss_first")
            fl = stat.get("recon_loss_last")
            fm = stat.get("recon_loss_min")
            if (ff is not None) and (fl is not None):
                if fm is not None:
                    rbits.append(f"loss {ff:.2e}->{fl:.2e} (min {fm:.2e})")
                else:
                    rbits.append(f"loss {ff:.2e}->{fl:.2e}")
            if rbits:
                parts.append("recon " + solver_label.lower() + " " + " ".join(rbits))
            # Align summary
            abits: list[str] = []
            at = stat.get("align_time")
            if at is not None:
                abits.append(f"{format_duration(at)}")
            sk = stat.get("step_kind")
            if sk == "gn":
                rm = stat.get("rot_mean")
                tm = stat.get("trans_mean")
                if rm is not None:
                    abits.append(f"|drot| {rm:.2e}")
                if tm is not None:
                    abits.append(f"|dtrans| {tm:.2e}")
            elif sk == "lbfgs":
                status = "accepted" if stat.get("lbfgs_accepted") else "rejected"
                if stat.get("lbfgs_fallback_to_gd"):
                    status = "fallback->gd"
                abits.append(status)
                nit = stat.get("lbfgs_nit")
                nfev = stat.get("lbfgs_nfev")
                if nit is not None:
                    abits.append(f"nit {int(nit)}")
                if nfev is not None:
                    abits.append(f"nfev {int(nfev)}")
                best = stat.get("lbfgs_best_loss")
                if best is not None:
                    abits.append(f"best {best:.2e}")
                rm = stat.get("rot_mean")
                tm = stat.get("trans_mean")
                if rm is not None:
                    abits.append(f"|drot| {rm:.2e}")
                if tm is not None:
                    abits.append(f"|dtrans| {tm:.2e}")
            elif sk == "gd":
                if stat.get("lbfgs_fallback_to_gd"):
                    abits.append("lbfgs fallback")
                rr = stat.get("rot_rms")
                tr = stat.get("trans_rms")
                if rr is not None:
                    abits.append(f"rotRMS {rr:.2e}")
                if tr is not None:
                    abits.append(f"transRMS {tr:.2e}")
            lb = stat.get("loss_before")
            la = stat.get("loss_after")
            ld = stat.get("loss_delta")
            rp = stat.get("loss_rel_pct")
            if (lb is not None) and (la is not None):
                rel = f" {rp:+.2f}%" if rp is not None else ""
                abits.append(f"loss {lb:.2e}->{la:.2e} (Δ {ld:+.2e}{rel})")
            if stat.get("gauge_fix") == "mean_translation":
                dxm = stat.get("dx_mean_before_gauge")
                dzm = stat.get("dz_mean_before_gauge")
                if dxm is not None and dzm is not None:
                    abits.append(f"gauge mean dx,dz {dxm:+.2e},{dzm:+.2e}->0")
            elif stat.get("gauge_fix") == "none":
                abits.append("gauge none")
            if abits:
                parts.append("align " + " ".join(abits))
            parts.append(f"elapsed {elapsed}")
            logging.info(" | ".join(parts))
            return
        logging.info(
            "Outer %d/%d | total %s | elapsed %s",
            outer_idx,
            total_iters,
            total_time,
            elapsed,
        )

        recon_parts: list[str] = []
        recon_time = stat.get("recon_time")
        if recon_time is not None:
            recon_parts.append(f"time {format_duration(recon_time)}")
        if stat.get("recon_retry"):
            recon_parts.append("fallback retry")
        l_meas = stat.get("L_meas")
        l_next = stat.get("L_next")
        if (l_meas is not None) and (l_next is not None):
            recon_parts.append(f"L {l_meas:.3e}->{l_next:.3e}")
        f_first = stat.get("recon_loss_first")
        f_last = stat.get("recon_loss_last")
        f_min = stat.get("recon_loss_min")
        if (f_first is not None) and (f_last is not None):
            if f_min is not None:
                recon_parts.append(f"loss {f_first:.3e}->{f_last:.3e} (min {f_min:.3e})")
            else:
                recon_parts.append(f"loss {f_first:.3e}->{f_last:.3e}")
        logging.info(
            "  Recon (%s) | %s",
            solver_label,
            " | ".join(recon_parts) if recon_parts else "-",
        )

        align_parts: list[str] = []
        align_time = stat.get("align_time")
        if align_time is not None:
            align_parts.append(f"time {format_duration(align_time)}")
        step_kind = stat.get("step_kind")
        if step_kind == "gn":
            rot_mean = stat.get("rot_mean")
            trans_mean = stat.get("trans_mean")
            if rot_mean is not None:
                align_parts.append(f"|drot|_mean {rot_mean:.3e} rad")
            if trans_mean is not None:
                align_parts.append(f"|dtrans|_mean {trans_mean:.3e}")
        elif step_kind == "lbfgs":
            status = "accepted" if stat.get("lbfgs_accepted") else "rejected"
            if stat.get("lbfgs_fallback_to_gd"):
                status = "fallback to GD"
            align_parts.append(f"L-BFGS {status}")
            for src, label in (
                ("lbfgs_initial_loss", "initial"),
                ("lbfgs_final_loss", "final"),
                ("lbfgs_best_loss", "best"),
            ):
                value = stat.get(src)
                if value is not None:
                    align_parts.append(f"{label} {value:.3e}")
            nit = stat.get("lbfgs_nit")
            nfev = stat.get("lbfgs_nfev")
            if nit is not None:
                align_parts.append(f"nit {int(nit)}")
            if nfev is not None:
                align_parts.append(f"nfev {int(nfev)}")
            message = stat.get("lbfgs_message")
            if message:
                align_parts.append(str(message))
        elif step_kind == "gd":
            if stat.get("lbfgs_fallback_to_gd"):
                message = stat.get("lbfgs_message")
                align_parts.append("L-BFGS fallback to GD" + (f": {message}" if message else ""))
            rot_rms = stat.get("rot_rms")
            trans_rms = stat.get("trans_rms")
            if rot_rms is not None:
                align_parts.append(f"rot RMS {rot_rms:.3e}")
            if trans_rms is not None:
                align_parts.append(f"trans RMS {trans_rms:.3e}")
        loss_before = stat.get("loss_before")
        loss_after = stat.get("loss_after")
        loss_delta = stat.get("loss_delta")
        rel_pct = stat.get("loss_rel_pct")
        if (loss_before is not None) and (loss_after is not None):
            rel_str = f", {rel_pct:+.2f}%" if rel_pct is not None else ""
            align_parts.append(
                f"loss {loss_before:.3e}->{loss_after:.3e} (Δ {loss_delta:+.3e}{rel_str})"
            )
        if stat.get("gauge_fix") == "mean_translation":
            dxm = stat.get("dx_mean_before_gauge")
            dzm = stat.get("dz_mean_before_gauge")
            dxa = stat.get("dx_mean_after_gauge")
            dza = stat.get("dz_mean_after_gauge")
            if dxm is not None and dzm is not None and dxa is not None and dza is not None:
                align_parts.append(
                    f"gauge mean dx,dz {dxm:+.3e},{dzm:+.3e}->{dxa:+.3e},{dza:+.3e}"
                )
        elif stat.get("gauge_fix") == "none":
            align_parts.append("gauge none")
        logging.info("  Align | %s", " | ".join(align_parts) if align_parts else "-")

    def _run_gd_alignment_step(
        params5_in: jnp.ndarray,
        motion_coeffs_in: jnp.ndarray | None,
        vol: jnp.ndarray,
        loss_before_value: float | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, float | None, jnp.ndarray]:
        scales = jnp.array(
            [cfg.lr_rot, cfg.lr_rot, cfg.lr_rot, cfg.lr_trans, cfg.lr_trans],
            dtype=jnp.float32,
        )
        if use_smooth_pose_model:
            if motion_coeffs_in is None or motion_loss_and_grad is None:
                raise RuntimeError("smooth pose model coefficients were not initialized")
            coeffs_in = motion_coeffs_in
            _, g_coeffs = motion_loss_and_grad(coeffs_in, vol)
            active_scales = scales[active_coeff_indices]
            rms_active = jnp.sqrt(jnp.mean(jnp.square(g_coeffs), axis=0)) + 1e-6
            eff_scales = active_scales / rms_active
            best_coeffs = coeffs_in - g_coeffs * eff_scales[None, :]
            best_params = _coeffs_to_constrained_params(best_coeffs)
            best_loss = _evaluate_align_loss(
                lambda: align_loss_jit(best_params, vol),
                fallback=math.inf,
                context="Treating GD base candidate as rejected during alignment loss evaluation",
            )
            cand_coeffs = coeffs_in - 2.0 * g_coeffs * eff_scales[None, :]
            cand_params = _coeffs_to_constrained_params(cand_coeffs)
            cand_loss = _evaluate_align_loss(
                lambda: align_loss_jit(cand_params, vol),
                fallback=math.inf,
                context="Treating GD doubled-step candidate as rejected during alignment loss evaluation",
            )
            best_loss_f = float(best_loss) if best_loss is not None else math.inf
            cand_loss_f = float(cand_loss) if cand_loss is not None else math.inf
            if not math.isfinite(best_loss_f) and not math.isfinite(cand_loss_f):
                params5_out = _coeffs_to_constrained_params(coeffs_in)
                loss_after_value = loss_before_value
            else:
                params5_out = cand_params if cand_loss_f < best_loss_f else best_params
                chosen_loss = min(best_loss_f, cand_loss_f)
                loss_after_value = (
                    float(chosen_loss) if math.isfinite(chosen_loss) else loss_before_value
                )
            motion_coeffs_out = fit_motion_coefficients(motion_model, params5_out)
            params5_out = _coeffs_to_constrained_params(motion_coeffs_out)
            rms = jnp.zeros((5,), dtype=jnp.float32).at[active_coeff_indices].set(rms_active)
            return params5_out, motion_coeffs_out, loss_after_value, rms

        p5_in = params5_in
        _, g_params = loss_and_grad_manual(params5_in, vol)
        g_params = g_params * active_mask
        rms = jnp.sqrt(jnp.mean(jnp.square(g_params), axis=0)) + 1e-6
        eff_scales = scales / rms
        best_params = _apply_full_constraints(p5_in - g_params * eff_scales)
        best_loss = _evaluate_align_loss(
            lambda: align_loss_jit(best_params, vol),
            fallback=math.inf,
            context="Treating GD base candidate as rejected during alignment loss evaluation",
        )
        cand_params = _apply_full_constraints(p5_in - 2.0 * g_params * eff_scales)
        cand_loss = _evaluate_align_loss(
            lambda: align_loss_jit(cand_params, vol),
            fallback=math.inf,
            context="Treating GD doubled-step candidate as rejected during alignment loss evaluation",
        )
        best_loss_f = float(best_loss) if best_loss is not None else math.inf
        cand_loss_f = float(cand_loss) if cand_loss is not None else math.inf
        if not math.isfinite(best_loss_f) and not math.isfinite(cand_loss_f):
            params5_out = p5_in
            loss_after_value = loss_before_value
        else:
            params5_out = cand_params if cand_loss_f < best_loss_f else best_params
            chosen_loss = min(best_loss_f, cand_loss_f)
            loss_after_value = (
                float(chosen_loss) if math.isfinite(chosen_loss) else loss_before_value
            )
        return params5_out, motion_coeffs_in, loss_after_value, rms

    def _run_lbfgs_alignment_step(
        params5_in: jnp.ndarray,
        motion_coeffs_in: jnp.ndarray | None,
        vol: jnp.ndarray,
        loss_before_value: float | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, float | None, OuterStat]:
        result = run_pose_lbfgs(
            params5_in=params5_in,
            motion_coeffs_in=motion_coeffs_in,
            loss_before_value=loss_before_value,
            objective_fn=lambda candidate: align_loss(candidate, vol),
            eval_loss_fn=lambda candidate, label: _evaluate_align_loss(
                lambda: align_loss_jit(candidate, vol),
                fallback=math.inf,
                context=f"Treating L-BFGS {label} candidate as rejected "
                "during alignment loss evaluation",
            ),
            is_expected_failure=_is_expected_align_eval_failure,
            cfg=PoseLbfgsConfig(
                maxiter=int(cfg.lbfgs_maxiter),
                ftol=float(cfg.lbfgs_ftol),
                gtol=float(cfg.lbfgs_gtol),
                maxls=int(cfg.lbfgs_maxls),
                memory_size=int(cfg.lbfgs_memory_size),
            ),
            context=PoseOptimizationContext(
                active_cols=active_col_indices_np,
                frozen_params5=frozen_params5,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                apply_param_constraints=_apply_full_constraints,
                motion_model=motion_model if use_smooth_pose_model else None,
            ),
        )
        if result.stats.get("lbfgs_fallback_to_gd"):
            logging.warning(
                "%s; falling back to GD for this alignment step",
                result.stats.get("lbfgs_message"),
            )
        return result.params5, result.motion_coeffs, result.loss, result.stats

    def _run_alignment_step(
        params5_in: jnp.ndarray,
        motion_coeffs_in: jnp.ndarray | None,
        vol: jnp.ndarray,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray | None,
        dict[str, float | str | list[str]],
        float,
        float | None,
        OuterStat,
    ]:
        align_start = time.perf_counter()
        stat: OuterStat = {}
        loss_before = _evaluate_align_loss(
            lambda: align_loss_jit(params5_in, vol),
            fallback=None,
            context="Skipping pre-step alignment loss evaluation",
        )
        stat["loss_before"] = loss_before

        if opt_mode == "gn" and ls_like:
            step_kind = "gn"
        elif opt_mode == "gn":
            logging.warning(
                "Gauss-Newton is incompatible with loss=%s; falling back to GD for this step",
                active_loss_name,
            )
            stat["optimizer_fallback"] = "gn->gd"
            step_kind = "gd"
        elif opt_mode == "lbfgs":
            step_kind = "lbfgs"
        else:
            step_kind = "gd"

        params5_out = params5_in
        motion_coeffs_out = motion_coeffs_in
        loss_after = None
        if step_kind == "gn":
            params5_prev = params5_in
            dp_all = _gn_update_all(params5_prev, vol) * active_mask
            constrain_candidate = (
                _project_params_to_smooth if use_smooth_pose_model else _apply_full_constraints
            )
            if cfg.gn_accept_only_improving and (loss_before is not None):
                smooth_candidate = None
                if int(params5_in.shape[0]) >= 3:
                    smooth_candidate = lambda candidate, weights: _smooth_gn_candidate(
                        constrain_candidate(candidate),
                        smoothness_gram,
                        weights,
                    )
                params5_out, loss_after = _select_gn_candidate(
                    params5_prev,
                    dp_all,
                    loss_before=loss_before,
                    eval_loss=lambda candidate: float(
                        _evaluate_align_loss(
                            lambda: align_loss_jit(candidate, vol),
                            fallback=math.inf,
                            context="Treating GN candidate as rejected during alignment loss evaluation",
                        )
                    ),
                    gn_accept_tol=cfg.gn_accept_tol,
                    constrain_candidate=constrain_candidate,
                    smooth_candidate=smooth_candidate,
                    light_smoothness_weights_sq=light_smoothness_weights_sq,
                    medium_smoothness_weights_sq=medium_smoothness_weights_sq,
                    smoothness_weights_sq=smoothness_weights_sq,
                    trans_only_smoothness_weights_sq=trans_only_smoothness_weights_sq,
                )
                params5_out = constrain_candidate(params5_out)
            else:
                params5_out = constrain_candidate(params5_prev + dp_all)
                candidate_loss = _evaluate_align_loss(
                    lambda: align_loss_jit(params5_out, vol),
                    fallback=math.inf,
                    context="Treating GN step as rejected during alignment loss evaluation",
                )
                if candidate_loss is not None and math.isfinite(candidate_loss):
                    loss_after = candidate_loss
                else:
                    params5_out = params5_prev
                    loss_after = loss_before
            if use_smooth_pose_model:
                motion_coeffs_out = fit_motion_coefficients(motion_model, params5_out)
                params5_out = _coeffs_to_constrained_params(motion_coeffs_out)
            try:
                stat["rot_mean"] = float(jnp.mean(jnp.abs(dp_all[:, :3])))
                stat["trans_mean"] = float(jnp.mean(jnp.abs(dp_all[:, 3:])))
            except Exception:
                pass

        elif step_kind == "lbfgs":
            params5_out, motion_coeffs_out, loss_after, lbfgs_stats = _run_lbfgs_alignment_step(
                params5_in,
                motion_coeffs_in,
                vol,
                loss_before,
            )
            stat.update(lbfgs_stats)
            if stat.get("lbfgs_fallback_to_gd"):
                step_kind = "gd"
                params5_out, motion_coeffs_out, loss_after, rms = _run_gd_alignment_step(
                    params5_out,
                    motion_coeffs_out,
                    vol,
                    loss_before,
                )
                try:
                    stat["rot_rms"] = float(jnp.mean(rms[:3]))
                    stat["trans_rms"] = float(jnp.mean(rms[3:]))
                except Exception:
                    pass
        else:
            params5_out, motion_coeffs_out, loss_after, rms = _run_gd_alignment_step(
                params5_in,
                motion_coeffs_in,
                vol,
                loss_before,
            )
            try:
                stat["rot_rms"] = float(jnp.mean(rms[:3]))
                stat["trans_rms"] = float(jnp.mean(rms[3:]))
            except Exception:
                pass

        stat["step_kind"] = step_kind
        stat["optimizer_kind"] = step_kind
        stat["loss_after_step"] = loss_after

        params5_out, gauge_stats = _apply_full_constraints_with_stats(params5_out)
        if use_smooth_pose_model:
            motion_coeffs_out = fit_motion_coefficients(motion_model, params5_out)
            params5_out, gauge_stats = _apply_full_constraints_with_stats(
                expand_motion_coefficients(motion_model, motion_coeffs_out)
            )
            motion_coeffs_out = fit_motion_coefficients(motion_model, params5_out)

        stat["gauge_fix"] = gauge_fix
        stat["gauge_fix_dofs"] = ",".join(gauge_dofs)
        if gauge_fix == "mean_translation":
            stat["dx_mean_before_gauge"] = float(gauge_stats["dx_mean_before"])
            stat["dz_mean_before_gauge"] = float(gauge_stats["dz_mean_before"])
            stat["dx_mean_after_gauge"] = float(gauge_stats["dx_mean_after"])
            stat["dz_mean_after_gauge"] = float(gauge_stats["dz_mean_after"])

        jax.block_until_ready(params5_out)
        stat["align_time"] = time.perf_counter() - align_start

        final_loss_fallback = loss_after
        if final_loss_fallback is None:
            final_loss_fallback = loss_before
        if final_loss_fallback is None and loss_hist:
            final_loss_fallback = loss_hist[-1]
        total_loss_eval = _evaluate_align_loss(
            lambda: align_loss_jit(params5_out, vol),
            fallback=final_loss_fallback,
            context="Using fallback for final alignment loss bookkeeping",
        )
        total_loss = float(total_loss_eval) if total_loss_eval is not None else math.nan
        stat["loss_after"] = total_loss

        if loss_before is not None:
            delta = total_loss - loss_before
            stat["loss_delta"] = delta
            if math.isfinite(loss_before) and abs(loss_before) > 1e-12:
                stat["loss_rel_pct"] = (delta / loss_before) * 100.0
            else:
                stat["loss_rel_pct"] = None
            if math.isfinite(loss_before) and math.isfinite(total_loss):
                rel_impr = (loss_before - total_loss) / max(abs(loss_before), 1e-12)
            else:
                rel_impr = None
        else:
            stat["loss_delta"] = None
            stat["loss_rel_pct"] = None
            rel_impr = None
        stat["rel_impr"] = rel_impr
        return params5_out, motion_coeffs_out, gauge_stats, total_loss, rel_impr, stat

    iter_range = range(start_outer_iter, int(cfg.outer_iters))
    for it in progress_iter(
        iter_range,
        total=max(0, int(cfg.outer_iters) - start_outer_iter),
        desc="Align: outer iters",
    ):
        outer_idx = it + 1
        stat: OuterStat = {
            "outer_idx": outer_idx,
            "loss_kind": active_loss_name,
            "recon_algo": recon_algo,
            "objective_kind": "fixed_volume",
            "objective_provenance": dict(active_objective_provenance),
            "outer_loss_kind": active_loss_name,
        }
        outer_start = time.perf_counter()

        x, L_prev, recon_stat = _run_reconstruction_step(
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            det_grid=det_grid,
            params5=params5,
            x=x,
            cfg=cfg,
            L_prev=L_prev,
            outer_idx=outer_idx,
            recon_algo=recon_algo,
        )
        stat.update(recon_stat)

        params5, motion_coeffs, final_gauge_stats, total_loss, rel_impr, align_stat = (
            _run_alignment_step(params5, motion_coeffs, x)
        )
        loss_hist.append(total_loss)
        stat.update(align_stat)

        outer_time = time.perf_counter() - outer_start
        stat["outer_time"] = outer_time
        stat["cumulative_time"] = time.perf_counter() - wall_start
        outer_stats.append(stat)

        if cfg.log_summary:
            _log_outer_summary(stat)

        should_break = False
        if observer_fn is not None:
            observer_action = _normalize_observer_action(observer_fn(x, params5, dict(stat)))
            stat["observer_action"] = observer_action
            stat["observer_stop"] = observer_action != "continue"
            if observer_action != "continue":
                stopped_by_observer = observer_action == "stop_run"
                should_break = True

        # Early stopping based on alignment improvement during GN/GD step
        if cfg.early_stop and (rel_impr is not None):
            rel_for_patience = rel_impr
            if (not math.isfinite(rel_for_patience)) or (rel_for_patience < 0.0):
                rel_for_patience = 0.0
            if rel_for_patience < float(cfg.early_stop_rel_impr):
                small_impr_streak += 1
            else:
                small_impr_streak = 0
            if small_impr_streak >= int(cfg.early_stop_patience):
                if cfg.log_summary:
                    logging.info(
                        "Early stop after %d outer iters (%s elapsed): rel_impr=%.3e < %.3e for %d consecutive outers",
                        outer_idx,
                        format_duration(stat.get("cumulative_time")),
                        float(rel_impr),
                        float(cfg.early_stop_rel_impr),
                        int(cfg.early_stop_patience),
                    )
                should_break = True
        elif cfg.early_stop:
            small_impr_streak = 0

        _emit_checkpoint_state()
        if should_break:
            break

    if cfg.log_summary and outer_stats:
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
        first_loss = outer_stats[0].get("loss_before") if outer_stats else None
        final_loss = outer_stats[-1].get("loss_after") if outer_stats else None
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

    # Provide last measured/reused L for potential reuse across levels
    wall_total = time.perf_counter() - wall_start
    info = {
        "loss": loss_hist,
        "loss_kind": active_loss_name,
        "recon_algo": recon_algo,
        "L": (float(L_prev) if L_prev is not None else None),
        "outer_stats": outer_stats,
        "stopped_by_observer": stopped_by_observer,
        "observer_action": observer_action,
        "wall_time_total": float(wall_total),
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
        "optimizer_kind": str(outer_stats[-1].get("optimizer_kind"))
        if outer_stats and outer_stats[-1].get("optimizer_kind") is not None
        else str(cfg.opt_method),
        "completed_outer_iters": len(outer_stats),
        "small_impr_streak": int(small_impr_streak),
        "motion_coeffs": motion_coeffs,
        "gauge_fix": gauge_fix,
        "gauge_fix_dofs": list(gauge_dofs),
        "gauge_fix_final": dict(final_gauge_stats),
    }
    return x, params5, info




def align_multires(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    factors: Iterable[int] = (2, 1),
    cfg: AlignConfig | None = None,
    observer: ObserverCallback | None = None,
    resume_state: AlignMultiresResumeState | None = None,
    checkpoint_callback: AlignMultiresCheckpointCallback | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignMultiresInfo]:
    """Coarse-to-fine alignment using simple binning for speed and robustness.

    Carries alignment parameters across levels and downsamples/upsamples volume.
    """
    from ..core.multires import (
        bin_projections,
        scale_detector,
        scale_grid,
        upsample_volume,
        validate_scale_factor,
    )

    if cfg is None:
        cfg = AlignConfig()
    observer_fn = adapt_legacy_observer(observer) if observer is not None else None
    resolved_schedule = _resolved_schedule_for_cfg(cfg, geometry=geometry)
    active_mask_tuple = resolved_schedule.pose_mask
    setup_base = BaseGeometryArrays.from_geometry(geometry, detector)
    setup_alignment_state = alignment_state_from_checkpoint(
        resume_state.geometry_calibration_state if resume_state is not None else None,
        n_views=int(projections.shape[0]),
        volume=resume_state.x if resume_state is not None else None,
    )
    setup_alignment_state = setup_alignment_state.replace(
        setup=setup_alignment_state.setup.replace(
            nominal_axis_unit=setup_base.nominal_axis_unit,
        )
    )
    active_geometry_dofs = resolved_schedule.active_geometry_dofs

    validate_grid(grid, "align_multires grid")
    validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="align_multires projections",
    )

    factors_list = [validate_scale_factor(f) for f in factors]
    validate_loss_schedule_levels(cfg.loss, factors_list)
    levels: list[MultiresLevel] = []
    for f in factors_list:
        g = scale_grid(grid, f)
        d = scale_detector(detector, f)
        y = bin_projections(projections, f)
        validate_projection_stack(
            y,
            d,
            geometry=geometry,
            context=f"align_multires level factor {f} projections",
        )
        levels.append(
            {
                "factor": f,
                "grid": g,
                "detector": d,
                "projections": y,
            }
        )

    x_init = resume_state.x if resume_state is not None and resume_state.level_complete else None
    params5 = (
        resume_state.params5 if resume_state is not None and resume_state.level_complete else None
    )
    prev_factor: int | None = (
        int(resume_state.level_factor)
        if resume_state is not None and resume_state.level_complete
        else None
    )
    loss_hist: list[float] = list(resume_state.loss) if resume_state is not None else []
    global_outer_stats: list[OuterStat] = (
        [dict(stat) for stat in resume_state.outer_stats] if resume_state is not None else []
    )
    stopped_by_observer = False
    final_observer_action: ObserverAction = "continue"
    global_outer_idx = (
        int(resume_state.global_outer_iters_completed) if resume_state is not None else 0
    )
    global_elapsed_offset = float(resume_state.elapsed_offset) if resume_state is not None else 0.0
    executed_outer_iters = int(global_outer_idx)
    final_pose_model_variables: int | None = None
    final_per_view_variables: int | None = None
    final_pose_model_basis_shape: list[int] | None = None
    final_loss_kind: str | None = None
    final_gauge_fix = normalize_gauge_fix(cfg.gauge_fix)
    final_gauge_fix_dofs = list(
        active_gauge_dofs(mode=final_gauge_fix, active_mask=active_mask_tuple)
    )
    final_gauge_fix_stats: dict[str, float | str | list[str]] | None = None
    last_level_index_processed: int | None = None

    if resume_state is not None and resume_state.run_complete:
        levels_to_run: list[tuple[int, MultiresLevel]] = []
        x_init = resume_state.x
        params5 = resume_state.params5
        prev_factor = 1
    else:
        start_level = 0
        if resume_state is not None:
            start_level = int(resume_state.level_index) + (1 if resume_state.level_complete else 0)
        levels_to_run = list(enumerate(levels))[start_level:]

    for li, lvl in levels_to_run:
        g = lvl["grid"]
        d = lvl["detector"]
        y = lvl["projections"]
        active_loss_spec = resolve_loss_for_level(cfg.loss, int(lvl["factor"]))
        active_loss_name = loss_spec_name(active_loss_spec)
        final_loss_kind = active_loss_name
        logging.info(
            "Alignment level %d/%d factor=%d using loss=%s",
            int(li) + 1,
            len(levels),
            int(lvl["factor"]),
            active_loss_name,
        )
        resuming_this_level = (
            resume_state is not None
            and not resume_state.level_complete
            and int(resume_state.level_index) == int(li)
        )
        if resuming_this_level:
            x0 = resume_state.x
        elif x_init is not None and prev_factor is not None:
            # Upsample previous x to current level as init
            f_up = prev_factor // lvl["factor"]
            x0 = upsample_volume(x_init, f_up, (g.nx, g.ny, g.nz))
        else:
            x0 = None

        # Optional translation seeding at the coarsest level via phase correlation
        params0 = resume_state.params5 if resuming_this_level else params5
        if li == 0 and cfg.seed_translations:
            # quick seed recon to project nominal poses
            seed_cfg = FistaConfig(
                iters=max(3, cfg.recon_iters // 2),
                lambda_tv=cfg.lambda_tv,
                regulariser=cfg.regulariser,
                huber_delta=cfg.huber_delta,
                projector_unroll=int(cfg.projector_unroll),
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=cfg.gather_dtype,
                recon_rel_tol=cfg.recon_rel_tol,
                recon_patience=(int(cfg.recon_patience) if cfg.recon_patience is not None else 0),
            )
            x_seed, _ = fista_tv(
                geometry,
                g,
                d,
                y,
                init_x=x0,
                config=seed_cfg,
            )
            T_nom = stack_view_poses(geometry, y.shape[0])
            from ..utils.phasecorr import phase_corr_shift

            vm_pred = jax.vmap(
                lambda T: forward_project_view_T(
                    T,
                    g,
                    d,
                    x_seed,
                    use_checkpoint=cfg.checkpoint_projector,
                    gather_dtype=cfg.gather_dtype,
                ),
                in_axes=0,
            )
            preds = vm_pred(T_nom)
            shift_uv = jax.vmap(phase_corr_shift)(preds, y)  # returns (du, dv)
            shifts = jnp.stack(shift_uv, axis=1).astype(jnp.float32)  # (n, 2)
            # Convert pixel shifts to world units using detector spacing
            dx = shifts[:, 0] * jnp.float32(d.du)
            dz = shifts[:, 1] * jnp.float32(d.dv)
            seed_params = (
                jnp.zeros((y.shape[0], 5), dtype=jnp.float32)
                if params0 is None
                else jnp.asarray(params0, dtype=jnp.float32)
            )
            if active_mask_tuple[3]:
                seed_params = seed_params.at[:, 3].set(dx)
            if active_mask_tuple[4]:
                seed_params = seed_params.at[:, 4].set(dz)
            params0 = seed_params

        level_completed_before = (
            int(resume_state.completed_outer_iters_in_level) if resuming_this_level else 0
        )
        current_level_stats = [
            dict(stat) for stat in global_outer_stats if stat.get("level_index") == int(li)
        ]
        stats_before_level = [
            dict(stat) for stat in global_outer_stats if stat.get("level_index") != int(li)
        ]
        loss_before_level = (
            list(loss_hist[:-level_completed_before])
            if resuming_this_level and level_completed_before > 0
            else list(loss_hist)
        )
        global_before_level = (
            int(executed_outer_iters) - level_completed_before
            if resuming_this_level
            else int(executed_outer_iters)
        )
        resume_stage_index = int(resume_state.stage_index) if resuming_this_level else 0
        resume_stage_completed = (
            bool(resume_state.stage_completed) if resuming_this_level else False
        )
        resume_stage_iters = (
            int(resume_state.completed_outer_iters_in_stage) if resuming_this_level else 0
        )

        def _stat_stage_index(stat: Mapping[str, object]) -> int | None:
            try:
                return int(stat["schedule_stage_index"])
            except Exception:
                return None

        if resuming_this_level:
            preserved_level_stats = [
                dict(stat)
                for stat in current_level_stats
                if (
                    (stage_idx := _stat_stage_index(stat)) is not None
                    and (
                        stage_idx < resume_stage_index
                        or (stage_idx == resume_stage_index and resume_stage_completed)
                    )
                )
            ]
            resume_stage_stats = [
                dict(stat)
                for stat in current_level_stats
                if _stat_stage_index(stat) == resume_stage_index and not resume_stage_completed
            ]
            level_history = (
                list(loss_hist[-level_completed_before:])
                if level_completed_before > 0
                else []
            )
            preserved_loss_count = min(len(preserved_level_stats), len(level_history))
            preserved_level_losses = level_history[:preserved_loss_count]
            resume_stage_losses = level_history[preserved_loss_count:]
            if len(resume_stage_losses) > resume_stage_iters:
                resume_stage_losses = resume_stage_losses[-resume_stage_iters:]
        else:
            preserved_level_stats = []
            resume_stage_stats = []
            preserved_level_losses = []
            resume_stage_losses = []
        level_stats: list[OuterStat] = [dict(stat) for stat in preserved_level_stats]
        level_losses: list[float] = [float(value) for value in preserved_level_losses]
        level_wall_time = 0.0
        level_action: ObserverAction = "continue"
        x_lvl = x0 if x0 is not None else jnp.zeros((g.nx, g.ny, g.nz), dtype=jnp.float32)
        params5 = (
            params0
            if params0 is not None
            else jnp.zeros((y.shape[0], 5), dtype=jnp.float32)
        )
        info: dict[str, object] = {
            "loss": [],
            "loss_kind": active_loss_name,
            "recon_algo": str(cfg.recon_algo),
            "L": None,
            "outer_stats": [],
            "stopped_by_observer": False,
            "observer_action": "continue",
            "wall_time_total": 0.0,
            "pose_model": str(cfg.pose_model),
            "pose_model_variables": 0,
            "per_view_variables": 0,
            "pose_model_basis_shape": [],
            "active_dofs": [],
            "completed_outer_iters": 0,
            "small_impr_streak": 0,
            "motion_coeffs": None,
            "gauge_fix": final_gauge_fix,
            "gauge_fix_dofs": final_gauge_fix_dofs,
            "gauge_fix_final": final_gauge_fix_stats or {},
        }

        def _enrich_level_stats(
            local_stats: list[OuterStat],
            *,
            stage: ResolvedAlignmentStage | None = None,
            global_start: int | None = None,
        ) -> list[OuterStat]:
            enriched_stats: list[OuterStat] = []
            for idx, stat in enumerate(local_stats, start=1):
                enriched = _enrich_multires_stage_stat(
                    stat,
                    level_factor=int(lvl["factor"]),
                    level_index=int(li),
                    global_outer_idx=int(
                        (global_before_level if global_start is None else global_start) + idx
                    ),
                    elapsed_offset=float(global_elapsed_offset),
                    loss_name=active_loss_name,
                    schedule_name=resolved_schedule.name,
                    stage=stage,
                )
                enriched_stats.append(enriched)
            return enriched_stats

        def _emit_multires_checkpoint(
            state: AlignResumeState,
            *,
            level_complete: bool,
            stage: ResolvedAlignmentStage,
            global_start: int,
        ) -> None:
            if checkpoint_callback is None:
                return
            enriched_stats = _enrich_level_stats(
                [dict(stat) for stat in state.outer_stats],
                stage=stage,
                global_start=global_start,
            )
            checkpoint_callback(
                AlignMultiresResumeState(
                    x=state.x,
                    params5=state.params5,
                    motion_coeffs=state.motion_coeffs,
                    level_index=int(li),
                    level_factor=int(lvl["factor"]),
                    completed_outer_iters_in_level=int(
                        len(level_stats) + state.start_outer_iter
                    ),
                    global_outer_iters_completed=int(
                        global_before_level + len(level_stats) + state.start_outer_iter
                    ),
                    prev_factor=prev_factor,
                    loss=loss_before_level + level_losses + list(state.loss),
                    outer_stats=stats_before_level + level_stats + enriched_stats,
                    L=state.L,
                    small_impr_streak=int(state.small_impr_streak),
                    elapsed_offset=float(global_elapsed_offset + state.elapsed_offset),
                    level_complete=bool(level_complete),
                    run_complete=False,
                    geometry_calibration_state=_geometry_calibration_payload(
                        setup_alignment_state,
                        active_geometry_dofs,
                    ),
                    stage_index=int(stage.index),
                    stage_name=stage.name,
                    stage_completed=bool(
                        level_complete or int(state.start_outer_iter) >= int(stage.maxiter)
                    ),
                    completed_outer_iters_in_stage=int(state.start_outer_iter),
                )
            )

        def _stage_observer(
            stage: ResolvedAlignmentStage,
            global_start: int,
            x_obs,
            params_obs,
            stat_obs,
        ):
            nonlocal stopped_by_observer
            enriched = _enrich_multires_stage_stat(
                stat_obs,
                level_factor=int(lvl["factor"]),
                level_index=int(li),
                global_outer_idx=int(global_start + int(stat_obs["outer_idx"])),
                elapsed_offset=float(global_elapsed_offset),
                loss_name=active_loss_name,
                schedule_name=resolved_schedule.name,
                stage=stage,
            )
            if observer_fn is None:
                return "continue"
            return observer_fn(x_obs, params_obs, enriched)

        stage_resume_consumed = False
        for stage in resolved_schedule.stages:
            if resuming_this_level:
                if int(stage.index) < resume_stage_index:
                    continue
                if int(stage.index) == resume_stage_index and resume_stage_completed:
                    continue
            stage_global_start = global_before_level + len(level_stats)
            if stage.active_geometry_dofs:
                cfg_stage = replace(
                    cfg,
                    schedule=None,
                    optimise_dofs=stage.active_geometry_dofs,
                    geometry_dofs=(),
                    outer_iters=int(stage.maxiter),
                    early_stop=bool(stage.early_stop),
                )
                geometry_start = time.perf_counter()
                x_lvl, setup_alignment_state, raw_geometry_stats = (
                    _optimize_setup_geometry_bilevel_for_level(
                        geometry=geometry,
                        grid=g,
                        detector=d,
                        projections=y,
                        init_x=x_lvl,
                        init_params5=params5,
                        state=setup_alignment_state,
                        active_geometry_dofs=stage.active_geometry_dofs,
                        factor=int(lvl["factor"]),
                        cfg=cfg_stage,
                        loss_spec=active_loss_spec,
                        loss_name=active_loss_name,
                        schedule_name=resolved_schedule.name,
                        stage=stage,
                    )
                )
                stage_wall = time.perf_counter() - geometry_start
                level_wall_time += stage_wall
                enriched = _enrich_level_stats(
                    [dict(stat) for stat in raw_geometry_stats],
                    stage=stage,
                    global_start=stage_global_start,
                )
                level_stats.extend(enriched)
                level_losses.extend(
                    float(stat["geometry_loss_after"])
                    for stat in enriched
                    if stat.get("geometry_loss_after") is not None
                )
                info["wall_time_total"] = float(level_wall_time)
                info["completed_outer_iters"] = len(level_stats)
                continue

            if stage.active_pose_dofs:
                pose_optimizer = (
                    stage.optimizer_kind
                    if stage.optimizer_kind in {"gd", "gn", "lbfgs"}
                    else cfg.opt_method
                )
                pose_gauge_fix = (
                    "mean_translation"
                    if stage.gauge_policy == "anchor_mean"
                    else cfg.gauge_fix
                )
                cfg_stage = replace(
                    cfg,
                    schedule=None,
                    optimise_dofs=stage.active_pose_dofs,
                    geometry_dofs=(),
                    opt_method=str(pose_optimizer),
                    outer_iters=int(stage.maxiter),
                    early_stop=bool(stage.early_stop),
                    recon_L=None,
                    loss=active_loss_spec,
                    gauge_fix=pose_gauge_fix,
                )
                align_kwargs = {}
                if active_geometry_dofs:
                    geometry_for_align = _geometry_with_setup_state(
                        geometry,
                        g,
                        d,
                        setup_alignment_state.setup,
                    )
                    det_grid_for_align = apply_setup_to_detector_grid(
                        d,
                        setup_alignment_state.setup,
                        level_factor=int(lvl["factor"]),
                    )
                    align_kwargs["det_grid_override"] = det_grid_for_align
                else:
                    geometry_for_align = geometry
                align_resume_state = None
                if (
                    resuming_this_level
                    and int(stage.index) == resume_stage_index
                    and not resume_stage_completed
                    and not stage_resume_consumed
                ):
                    align_resume_state = AlignResumeState(
                        x=resume_state.x,
                        params5=resume_state.params5,
                        motion_coeffs=resume_state.motion_coeffs,
                        start_outer_iter=resume_stage_iters,
                        loss=list(resume_stage_losses),
                        outer_stats=[dict(stat) for stat in resume_stage_stats],
                        L=resume_state.L,
                        small_impr_streak=int(resume_state.small_impr_streak),
                        elapsed_offset=float(
                            resume_state.elapsed_offset - global_elapsed_offset
                        ),
                    )
                    stage_resume_consumed = True
                x_lvl, params5, info = align(
                    geometry_for_align,
                    g,
                    d,
                    y,
                    cfg=cfg_stage,
                    init_x=x_lvl,
                    init_params5=params5,
                    observer=(
                        (lambda x_obs, params_obs, stat_obs, _stage=stage, _start=stage_global_start: _stage_observer(_stage, _start, x_obs, params_obs, stat_obs))
                        if observer_fn is not None
                        else None
                    ),
                    resume_state=align_resume_state,
                    checkpoint_callback=(
                        (
                            lambda state, _stage=stage, _start=stage_global_start: _emit_multires_checkpoint(
                                state,
                                level_complete=False,
                                stage=_stage,
                                global_start=_start,
                            )
                        )
                        if checkpoint_callback is not None
                        else None
                    ),
                    **align_kwargs,
                )
                setup_alignment_state = setup_alignment_state.replace(
                    pose=PoseState(
                        params5,
                        info.get("motion_coeffs"),  # type: ignore[arg-type]
                    ),
                    volume=x_lvl,
                )
                enriched = _enrich_level_stats(
                    [dict(stat) for stat in info.get("outer_stats", [])],
                    stage=stage,
                    global_start=stage_global_start,
                )
                level_stats.extend(enriched)
                level_losses.extend(float(v) for v in info.get("loss", []))
                try:
                    level_wall_time += float(info.get("wall_time_total") or 0.0)
                except Exception:
                    pass
                level_action = _normalize_observer_action(info.get("observer_action"))
                final_gauge_fix = str(info.get("gauge_fix", final_gauge_fix))
                final_gauge_fix_dofs = list(info.get("gauge_fix_dofs", final_gauge_fix_dofs))
                final_gauge_fix_stats = dict(info.get("gauge_fix_final", {}) or {})
                if level_action != "continue":
                    break

        info["loss"] = level_losses
        info["outer_stats"] = [
            dict(stat)
            for stat in level_stats
            if not str(stat.get("geometry_block") or "").startswith("setup_")
        ]
        info["completed_outer_iters"] = len(level_stats)
        info["wall_time_total"] = float(level_wall_time)
        info["observer_action"] = level_action
        level_completed_after = len(level_stats)
        level_complete = (
            level_completed_after >= sum(int(stage.maxiter) for stage in resolved_schedule.stages)
            or level_action == "advance_level"
            or not bool(info.get("stopped_by_observer", False))
        )
        loss_hist = loss_before_level + level_losses
        global_outer_stats = stats_before_level + level_stats
        global_outer_idx = global_before_level + level_completed_after
        executed_outer_iters = int(global_outer_idx)
        final_pose_model_variables = int(info.get("pose_model_variables") or 0)
        final_per_view_variables = int(info.get("per_view_variables") or 0)
        final_pose_model_basis_shape = list(info.get("pose_model_basis_shape") or [])
        final_gauge_fix = str(info.get("gauge_fix", final_gauge_fix))
        final_gauge_fix_dofs = list(info.get("gauge_fix_dofs", final_gauge_fix_dofs))
        final_gauge_fix_stats = dict(info.get("gauge_fix_final", {}) or {})
        x_init = x_lvl
        prev_factor = lvl["factor"]
        last_level_index_processed = int(li)
        global_elapsed_offset += float(level_wall_time)
        if checkpoint_callback is not None:
            last_stage = resolved_schedule.stages[-1]
            last_stage_iters = sum(
                1
                for stat in level_stats
                if stat.get("schedule_stage_index") == int(last_stage.index)
            )
            checkpoint_callback(
                AlignMultiresResumeState(
                    x=x_lvl,
                    params5=params5,
                    motion_coeffs=info.get("motion_coeffs"),
                    level_index=int(li),
                    level_factor=int(lvl["factor"]),
                    completed_outer_iters_in_level=level_completed_after,
                    global_outer_iters_completed=int(global_outer_idx),
                    prev_factor=prev_factor,
                    loss=list(loss_hist),
                    outer_stats=[dict(stat) for stat in global_outer_stats],
                    L=info.get("L"),
                    small_impr_streak=int(info.get("small_impr_streak", 0)),
                    elapsed_offset=float(global_elapsed_offset),
                    level_complete=bool(level_complete),
                    run_complete=False,
                    geometry_calibration_state=_geometry_calibration_payload(
                        setup_alignment_state,
                        active_geometry_dofs,
                    ),
                    stage_index=int(last_stage.index),
                    stage_name=last_stage.name,
                    stage_completed=bool(level_complete),
                    completed_outer_iters_in_stage=last_stage_iters,
                )
            )
        final_observer_action = level_action
        if level_action == "stop_run":
            stopped_by_observer = True
            break
        if level_action == "advance_level":
            continue

    # Always return a full-resolution-compatible final volume.
    if x_init is None:
        x_final = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    elif prev_factor is not None and prev_factor != 1:
        x_final = upsample_volume(x_init, prev_factor, (grid.nx, grid.ny, grid.nz))
    else:
        x_final = x_init

    run_complete = (
        params5 is not None
        and not stopped_by_observer
        and (
            (resume_state is not None and resume_state.run_complete)
            or last_level_index_processed == len(levels) - 1
            or not levels
        )
    )
    if checkpoint_callback is not None and params5 is not None and run_complete:
        checkpoint_callback(
            AlignMultiresResumeState(
                x=x_final,
                params5=params5,
                motion_coeffs=None,
                level_index=max(0, len(levels) - 1),
                level_factor=1,
                completed_outer_iters_in_level=0,
                global_outer_iters_completed=int(executed_outer_iters),
                prev_factor=1,
                loss=list(loss_hist),
                outer_stats=[dict(stat) for stat in global_outer_stats],
                L=None,
                small_impr_streak=0,
                elapsed_offset=float(global_elapsed_offset),
                level_complete=True,
                run_complete=True,
                geometry_calibration_state=_geometry_calibration_payload(
                    setup_alignment_state,
                    active_geometry_dofs,
                ),
                stage_index=int(resolved_schedule.stages[-1].index),
                stage_name=resolved_schedule.stages[-1].name,
                stage_completed=True,
                completed_outer_iters_in_stage=0,
            )
        )

    geometry_calibration_diagnostics = add_geometry_acquisition_diagnostics(
        summarize_geometry_calibration_stats(global_outer_stats),
        geometry,
        active_geometry_dofs,
    )
    objective_kinds = [
        str(stat.get("objective_kind") or stat.get("geometry_objective"))
        for stat in global_outer_stats
        if isinstance(stat, Mapping)
        and (stat.get("objective_kind") is not None or stat.get("geometry_objective") is not None)
    ]
    objective_provenance = next(
        (
            dict(stat["objective_provenance"])
            for stat in reversed(global_outer_stats)
            if isinstance(stat, Mapping) and isinstance(stat.get("objective_provenance"), Mapping)
        ),
        None,
    )

    return (
        x_final,
        params5 if params5 is not None else jnp.zeros((projections.shape[0], 5), jnp.float32),
        {
            "loss": loss_hist,
            "factors": factors_list,
            "loss_kind": final_loss_kind,
            "recon_algo": str(cfg.recon_algo),
            "stopped_by_observer": stopped_by_observer,
            "observer_action": final_observer_action,
            "total_outer_iters": int(executed_outer_iters),
            "wall_time_total": float(global_elapsed_offset),
            "outer_stats": global_outer_stats,
            "schedule": resolved_schedule.to_dict(),
            "schedule_name": resolved_schedule.name,
            "schedule_stages": [stage.to_dict() for stage in resolved_schedule.stages],
            "gauge_policy": cfg.gauge_policy,
            "gauge_decision": resolved_schedule.gauge_decision.to_dict(),
            "objective_kind": objective_kinds[-1] if objective_kinds else None,
            "objective_kinds": objective_kinds,
            "objective_provenance": objective_provenance,
            "pose_model": str(cfg.pose_model),
            "pose_model_variables": final_pose_model_variables,
            "per_view_variables": final_per_view_variables,
            "pose_model_basis_shape": final_pose_model_basis_shape,
            "active_dofs": list(resolved_schedule.active_dofs),
            "active_pose_dofs": list(resolved_schedule.active_pose_dofs),
            "active_geometry_dofs": list(active_geometry_dofs),
            "gauge_fix": final_gauge_fix,
            "gauge_fix_dofs": final_gauge_fix_dofs,
            "gauge_fix_final": final_gauge_fix_stats,
            "geometry_dofs": list(active_geometry_dofs),
            "geometry_calibration_state": (
                _geometry_calibration_payload(setup_alignment_state, active_geometry_dofs)
                if active_geometry_dofs
                else None
            ),
            "geometry_calibration_diagnostics": geometry_calibration_diagnostics,
        },
    )
