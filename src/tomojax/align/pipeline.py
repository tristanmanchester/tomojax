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
from ..recon.spdhg_tv import SPDHGConfig, spdhg_tv
from ..utils.logging import progress_iter, format_duration
from .parametrizations import se3_from_5d
from .dofs import (
    DofBounds,
    bounds_vectors,
    normalize_alignment_dofs,
    normalize_bounds,
    resolve_scoped_alignment_dofs,
)
from .losses import (
    L2OtsuLossSpec,
    AlignmentLossConfig,
    build_loss_adapter,
    loss_spec_name,
    loss_is_within_relative_tolerance,
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
from .optimizers import PoseLbfgsConfig, run_pose_lbfgs
from .geometry_blocks import (
    GeometryCalibrationState,
    add_geometry_acquisition_diagnostics,
    geometry_with_axis_state,
    level_detector_grid,
    normalize_geometry_dofs,
    optimize_geometry_blocks_for_level,
    summarize_geometry_calibration_stats,
)
from ..utils.fov import cylindrical_mask_xy


ObserverAction = Literal["continue", "advance_level", "stop_run"]
type OuterStatValue = float | int | bool | str | None
type OuterStat = dict[str, OuterStatValue]
ObserverCallback = Callable[[jnp.ndarray, jnp.ndarray, OuterStat], ObserverAction | bool]


def _active_dof_mask_for_cfg(cfg: "AlignConfig") -> tuple[bool, bool, bool, bool, bool]:
    return _scoped_dofs_for_cfg(cfg).pose_mask


def _active_dofs_for_cfg(cfg: "AlignConfig") -> tuple[str, ...]:
    return _scoped_dofs_for_cfg(cfg).active_pose_dofs


def _active_geometry_dofs_for_cfg(
    cfg: "AlignConfig",
    geometry: Geometry | None = None,
) -> tuple[str, ...]:
    return _scoped_dofs_for_cfg(cfg, geometry=geometry).active_geometry_dofs


def _scoped_dofs_for_cfg(
    cfg: "AlignConfig",
    *,
    geometry: Geometry | None = None,
):
    return resolve_scoped_alignment_dofs(
        optimise_dofs=cfg.optimise_dofs,
        freeze_dofs=cfg.freeze_dofs,
        geometry_dofs=cfg.geometry_dofs,
        geometry=geometry,
    )


def _validate_scoped_geometry_pose_gauges(scoped_dofs) -> None:
    pose = set(scoped_dofs.active_pose_dofs)
    geometry = set(scoped_dofs.active_geometry_dofs)
    conflicts: list[str] = []
    if "det_u_px" in geometry and ({"dx", "dz"} & pose):
        conflicts.append("det_u_px cannot be estimated with active per-view dx/dz")
    if "det_v_px" in geometry and ({"dx", "dz"} & pose):
        conflicts.append("det_v_px cannot be estimated with active per-view dx/dz")
    if conflicts:
        raise ValueError(
            "Gauge-coupled alignment DOFs are underdetermined: "
            + "; ".join(conflicts)
            + ". Freeze the pose translations for detector-centre calibration, "
            "or supply a corrected detector centre before pose alignment."
        )


class AlignInfo(TypedDict):
    loss: list[float]
    loss_kind: str
    recon_algo: str
    L: float | None
    outer_stats: list[OuterStat]
    stopped_by_observer: bool
    observer_action: ObserverAction
    wall_time_total: float
    pose_model: str
    pose_model_variables: int
    per_view_variables: int
    pose_model_basis_shape: list[int]
    active_dofs: list[str]
    completed_outer_iters: int
    small_impr_streak: int
    motion_coeffs: jnp.ndarray | None
    gauge_fix: str
    gauge_fix_dofs: list[str]
    gauge_fix_final: dict[str, float | str | list[str]]


class AlignMultiresInfo(TypedDict):
    loss: list[float]
    factors: list[int]
    loss_kind: str | None
    recon_algo: str
    outer_stats: list[OuterStat]
    stopped_by_observer: bool
    observer_action: ObserverAction
    total_outer_iters: int
    wall_time_total: float
    pose_model: str
    pose_model_variables: int | None
    per_view_variables: int | None
    pose_model_basis_shape: list[int] | None
    active_dofs: list[str]
    gauge_fix: str
    gauge_fix_dofs: list[str]
    gauge_fix_final: dict[str, float | str | list[str]] | None
    geometry_dofs: list[str]
    geometry_calibration_state: dict[str, object] | None


class MultiresLevel(TypedDict):
    factor: int
    grid: Grid
    detector: Detector
    projections: jnp.ndarray


@dataclass
class AlignResumeState:
    x: jnp.ndarray
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None = None
    start_outer_iter: int = 0
    loss: list[float] = field(default_factory=list)
    outer_stats: list[OuterStat] = field(default_factory=list)
    L: float | None = None
    small_impr_streak: int = 0
    elapsed_offset: float = 0.0


@dataclass
class AlignMultiresResumeState:
    x: jnp.ndarray
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None = None
    level_index: int = 0
    level_factor: int = 1
    completed_outer_iters_in_level: int = 0
    global_outer_iters_completed: int = 0
    prev_factor: int | None = None
    loss: list[float] = field(default_factory=list)
    outer_stats: list[OuterStat] = field(default_factory=list)
    L: float | None = None
    small_impr_streak: int = 0
    elapsed_offset: float = 0.0
    level_complete: bool = False
    run_complete: bool = False
    geometry_calibration_state: dict[str, object] | None = None


AlignCheckpointCallback = Callable[[AlignResumeState], None]
AlignMultiresCheckpointCallback = Callable[[AlignMultiresResumeState], None]


def _normalize_observer_action(
    action: ObserverAction | str | bool | None,
) -> ObserverAction:
    if action is False or action is None:
        return "continue"
    if action is True:
        return "stop_run"
    if isinstance(action, str):
        lowered = action.strip().lower()
        if lowered in {"continue", "advance_level", "stop_run"}:
            return lowered  # type: ignore[return-value]
    raise ValueError(f"Unsupported observer action: {action!r}")


def _should_prefer_gn_candidate(
    loss_before: float,
    current_loss: float,
    candidate_loss: float,
    rel_tol: float,
) -> bool:
    """Accept tolerated GN candidates only when they improve the current best step."""
    candidate_ok = candidate_loss < loss_before or loss_is_within_relative_tolerance(
        loss_before, candidate_loss, rel_tol
    )
    return candidate_ok and candidate_loss < current_loss


def _second_difference_gram(n: int) -> jnp.ndarray:
    if n < 3:
        return jnp.zeros((n, n), dtype=jnp.float32)
    d2 = jnp.zeros((n - 2, n), dtype=jnp.float32)
    rows = jnp.arange(n - 2, dtype=jnp.int32)
    d2 = d2.at[rows, rows].set(1.0)
    d2 = d2.at[rows, rows + 1].set(-2.0)
    d2 = d2.at[rows, rows + 2].set(1.0)
    return d2.T @ d2


def _smooth_gn_candidate(
    params5: jnp.ndarray,
    smoothness_gram: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Project a per-view GN candidate through the quadratic curvature prior."""
    n_views = int(params5.shape[0])
    if n_views < 3:
        return params5

    eye = jnp.eye(n_views, dtype=jnp.float32)

    def solve_one_dim(rhs: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.cond(
            weight > 0.0,
            lambda _: jnp.linalg.solve(eye + 2.0 * weight * smoothness_gram, rhs),
            lambda _: rhs,
            operand=None,
        )

    return jax.vmap(solve_one_dim, in_axes=(1, 0), out_axes=1)(params5, weights)


def _select_gn_candidate(
    params5_prev: jnp.ndarray,
    dp_all: jnp.ndarray,
    *,
    loss_before: float,
    eval_loss: Callable[[jnp.ndarray], float],
    gn_accept_tol: float,
    constrain_candidate: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    smooth_candidate: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    light_smoothness_weights_sq: jnp.ndarray | None = None,
    medium_smoothness_weights_sq: jnp.ndarray | None = None,
    smoothness_weights_sq: jnp.ndarray | None = None,
    trans_only_smoothness_weights_sq: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, float]:
    """Pick a GN candidate using a small hierarchical full-loss search."""

    def _accepts(candidate_loss: float, current_best_loss: float = float("inf")) -> bool:
        return _should_prefer_gn_candidate(
            loss_before,
            current_best_loss,
            candidate_loss,
            gn_accept_tol,
        )

    def _constrain(candidate: jnp.ndarray) -> jnp.ndarray:
        if constrain_candidate is None:
            return candidate
        return constrain_candidate(candidate)

    raw_params = _constrain(params5_prev + dp_all)
    raw_loss = eval_loss(raw_params)
    if _accepts(raw_loss):
        return raw_params, raw_loss

    half_params = _constrain(params5_prev + jnp.float32(0.5) * dp_all)
    half_loss = eval_loss(half_params)
    if _accepts(half_loss):
        return half_params, half_loss

    base_params = raw_params if raw_loss <= half_loss else half_params

    def _has_active_weights(weights: jnp.ndarray | None) -> bool:
        return weights is not None and bool(jnp.any(weights > 0.0))

    if smooth_candidate is None:
        return params5_prev, loss_before

    smooth_weights = []
    for weights in (
        light_smoothness_weights_sq,
        medium_smoothness_weights_sq,
        smoothness_weights_sq,
        trans_only_smoothness_weights_sq,
    ):
        if _has_active_weights(weights):
            smooth_weights.append(weights)

    if not smooth_weights:
        return params5_prev, loss_before

    best_params = params5_prev
    best_loss = float("inf")
    accepted = False
    for weights in smooth_weights:
        candidate_params = _constrain(smooth_candidate(base_params, weights))
        candidate_loss = eval_loss(candidate_params)
        if _accepts(candidate_loss, best_loss):
            best_params = candidate_params
            best_loss = candidate_loss
            accepted = True

    if accepted:
        return best_params, best_loss
    return params5_prev, loss_before


_EXPECTED_ALIGN_EVAL_FAILURE_SNIPPETS = (
    "allocator",
    "cholesky",
    "failed to converge",
    "inf",
    "nan",
    "non-finite",
    "not positive definite",
    "out of memory",
    "resource_exhausted",
    "singular",
    "svd",
)


def _is_expected_align_eval_failure(exc: Exception) -> bool:
    if isinstance(exc, FloatingPointError):
        return True
    msg = str(exc).lower()
    return any(snippet in msg for snippet in _EXPECTED_ALIGN_EVAL_FAILURE_SNIPPETS)


def _evaluate_align_loss(
    eval_loss: Callable[[], float | jnp.ndarray],
    *,
    fallback: float | None,
    context: str,
) -> float | None:
    try:
        return float(eval_loss())
    except Exception as exc:
        if _is_expected_align_eval_failure(exc):
            logging.warning("%s after expected numeric failure: %s", context, exc)
            return fallback
        raise


@dataclass
class AlignConfig:
    outer_iters: int = 5
    recon_iters: int = 10
    lambda_tv: float = 0.005
    regulariser: Regulariser = "tv"
    huber_delta: float = 1e-2
    tv_prox_iters: int = 10
    recon_algo: Literal["fista", "spdhg"] = "fista"
    recon_positivity: bool = True
    spdhg_seed: int = 0
    # Reconstruction stopping criteria
    recon_rel_tol: float | None = None
    recon_patience: int = 2
    # Alignment step sizes
    lr_rot: float = 1e-3  # radians
    lr_trans: float = 1e-1  # world units
    # Memory/throughput knobs
    views_per_batch: int = 1  # stream one view at a time
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    # Solver and regularization
    opt_method: str = "gn"
    gn_damping: float = 1e-6
    lbfgs_maxiter: int = 20
    lbfgs_ftol: float = 1e-6
    lbfgs_gtol: float = 1e-5
    lbfgs_maxls: int = 20
    lbfgs_memory_size: int = 10
    w_rot: float = 0.0
    w_trans: float = 0.0
    optimise_dofs: tuple[str, ...] | None = None
    freeze_dofs: tuple[str, ...] = field(default_factory=tuple)
    geometry_dofs: tuple[str, ...] = field(default_factory=tuple)
    bounds: DofBounds | str | Mapping[str, object] = field(default_factory=tuple)
    pose_model: Literal["per_view", "polynomial", "spline"] = "per_view"
    knot_spacing: int = 8
    degree: int = 3
    gauge_fix: GaugeFixMode = "mean_translation"
    seed_translations: bool = False
    # Volume masking before forward projection (modeling for ROI/truncation)
    # Options: "off" (default), "cyl" (cylindrical mask in x–y broadcast along z)
    mask_vol: str = "off"
    # Logging
    log_summary: bool = False
    log_compact: bool = True  # print one compact line per outer when log_summary is enabled
    # Reconstruction Lipschitz (optional override to skip power-method)
    recon_L: float | None = None
    # Early stopping across outers (alignment phase)
    early_stop: bool = True
    early_stop_rel_impr: float = 1e-3  # stop if (before-after)/before < this
    early_stop_patience: int = 2
    # Accept GN steps only when they improve the loss, up to gn_accept_tol.
    gn_accept_only_improving: bool = True
    gn_accept_tol: float = 0.0  # allow tiny increases if >0 (as fraction of before)
    # Data term / similarity
    loss: AlignmentLossConfig = field(default_factory=L2OtsuLossSpec)

    def __post_init__(self) -> None:
        recon_algo = str(self.recon_algo).strip().lower().replace("-", "_")
        if recon_algo in {"fista_tv"}:
            recon_algo = "fista"
        elif recon_algo in {"spdhg_tv"}:
            recon_algo = "spdhg"
        self.recon_algo = recon_algo  # type: ignore[assignment]
        if self.recon_algo not in {"fista", "spdhg"}:
            raise ValueError("recon_algo must be one of 'fista' or 'spdhg'")
        opt_method = str(self.opt_method).strip().lower().replace("-", "_")
        if opt_method in {"lbfgsb", "l_bfgs", "l_bfgs_b"}:
            opt_method = "lbfgs"
        self.opt_method = opt_method
        if self.opt_method not in {"gd", "gn", "lbfgs"}:
            raise ValueError("opt_method must be one of 'gd', 'gn', or 'lbfgs'")
        if int(self.lbfgs_maxiter) < 1:
            raise ValueError("lbfgs_maxiter must be >= 1")
        if int(self.lbfgs_maxls) < 1:
            raise ValueError("lbfgs_maxls must be >= 1")
        if int(self.lbfgs_memory_size) < 1:
            raise ValueError("lbfgs_memory_size must be >= 1")
        if float(self.lbfgs_ftol) < 0.0:
            raise ValueError("lbfgs_ftol must be >= 0")
        if float(self.lbfgs_gtol) < 0.0:
            raise ValueError("lbfgs_gtol must be >= 0")
        if self.optimise_dofs is not None:
            self.optimise_dofs = normalize_alignment_dofs(
                self.optimise_dofs,
                option_name="optimise_dofs",
            )
        self.freeze_dofs = normalize_alignment_dofs(self.freeze_dofs, option_name="freeze_dofs")
        self.geometry_dofs = normalize_geometry_dofs(
            self.geometry_dofs,
            geometry=None,
        )
        _active_dof_mask_for_cfg(self)
        self.bounds = normalize_bounds(self.bounds, option_name="bounds")
        pose_model = str(self.pose_model).strip().lower().replace("-", "_")
        self.pose_model = pose_model  # type: ignore[assignment]
        if self.pose_model not in {"per_view", "polynomial", "spline"}:
            raise ValueError("pose_model must be one of 'per_view', 'polynomial', or 'spline'")
        if self.pose_model == "polynomial" and int(self.degree) < 0:
            raise ValueError("degree must be >= 0 for polynomial pose_model")
        if self.pose_model == "spline":
            if int(self.knot_spacing) < 1:
                raise ValueError("knot_spacing must be >= 1 for spline pose_model")
            if int(self.degree) not in (1, 2, 3):
                raise ValueError("degree must be one of 1, 2, or 3 for spline pose_model")
        self.gauge_fix = normalize_gauge_fix(self.gauge_fix)
        if self.gauge_fix == "mean_translation":
            bounds_lower, bounds_upper = bounds_vectors(self.bounds)
            active_mask_for_gauge = _active_dof_mask_for_cfg(self)
            validate_alignment_gauge_feasible(
                mode=self.gauge_fix,
                active_mask=active_mask_for_gauge,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
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
    if cfg is None:
        cfg = AlignConfig()
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
    # Initialize volume and params
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

    # Vmapped projector across views (pose-aware). Closure captures unroll as a static constant.
    def _project_batch(T_batch, vol):
        f = lambda T: forward_project_view_T(
            T,
            grid,
            detector,
            vol,
            use_checkpoint=cfg.checkpoint_projector,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
            det_grid=det_grid,
        )
        return jax.vmap(f, in_axes=0)(T_batch)

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
    vol_mask = None
    try:
        if str(getattr(cfg, "mask_vol", "off")).lower() in ("cyl", "cylindrical"):
            m_xy = cylindrical_mask_xy(grid, detector)
            vol_mask = jnp.asarray(m_xy, dtype=jnp.float32)[:, :, None]
    except Exception:
        vol_mask = None

    # Build per-view loss once (may precompute masks on targets)
    active_loss_spec = resolve_loss_for_level(cfg.loss, level_factor=1)
    active_loss_name = loss_spec_name(active_loss_spec)
    loss_adapter = build_loss_adapter(active_loss_spec, projections)
    per_view_loss_fn = loss_adapter.per_view_loss
    loss_state = loss_adapter.state

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

    if has_loss_mask:

        def _loss_chunk_values(
            pred: jnp.ndarray,
            y_chunk: jnp.ndarray,
            start_shifted: jnp.ndarray,
            view_idx_chunk: jnp.ndarray,
        ) -> jnp.ndarray:
            mask_chunk = jax.lax.dynamic_slice(
                loss_mask, (start_shifted, 0, 0), (chunk_size, nv, nu)
            )
            return per_view_loss_fn(
                pred,
                y_chunk,
                mask_chunk,
                view_indices=view_idx_chunk,
            )
    else:

        def _loss_chunk_values(
            pred: jnp.ndarray,
            y_chunk: jnp.ndarray,
            start_shifted: jnp.ndarray,
            view_idx_chunk: jnp.ndarray,
        ) -> jnp.ndarray:
            del start_shifted
            return per_view_loss_fn(
                pred,
                y_chunk,
                None,
                view_indices=view_idx_chunk,
            )

    def align_loss(params5, vol):
        # Compose augmented poses
        # Current convention: per-view misalignment parameters act in the object
        # frame and are post-multiplied: T_world_from_obj_aug = T_nom @ T_delta.
        # This is consistent across parallel CT and laminography sample-frame.
        T_aug = T_nom_all @ jax.vmap(se3_from_5d)(params5)  # (n_views, 4, 4)
        masked_vol = _apply_vol_mask(vol)

        def body(loss_acc, i):
            start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
            T_chunk = jax.lax.dynamic_slice(T_aug, (start_shifted, 0, 0), (chunk_size, 4, 4))
            y_chunk = jax.lax.dynamic_slice(
                projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
            )
            pred = _project_batch(T_chunk, masked_vol)
            lvec = _loss_chunk_values(pred, y_chunk, start_shifted, view_idx_chunk)
            loss_batch = jnp.sum(lvec * vmask)
            return (loss_acc + loss_batch, None)

        loss0 = jnp.float32(0.0)
        loss_tot, _ = jax.lax.scan(body, loss0, jnp.arange(num_chunks, dtype=jnp.int32))

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

    # Memory-safe gradient over fixed-size chunks.
    if has_loss_mask:

        def _one_view_loss_masked(p5_i, T_nom_i, y_i, masked_vol, mask_i, view_idx):
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
                mask_i[None, ...],
                view_indices=view_indices,
            )
            return lvec[0]

        one_view_val_and_grad_batch = jax.jit(
            jax.vmap(
                jax.value_and_grad(_one_view_loss_masked),
                in_axes=(0, 0, 0, None, 0, 0),
            )
        )

        def loss_and_grad_manual(params5, vol):
            masked_vol = _apply_vol_mask(vol)

            def body(carry, i):
                total, g = carry
                start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
                params_chunk = jax.lax.dynamic_slice(
                    params5, (start_shifted, 0), (chunk_size, params5.shape[1])
                )
                T_nom_chunk = jax.lax.dynamic_slice(
                    T_nom_all, (start_shifted, 0, 0), (chunk_size, 4, 4)
                )
                y_chunk = jax.lax.dynamic_slice(
                    projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                mask_chunk = jax.lax.dynamic_slice(
                    loss_mask, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                lvec, g_chunk = one_view_val_and_grad_batch(
                    params_chunk,
                    T_nom_chunk,
                    y_chunk,
                    masked_vol,
                    mask_chunk,
                    view_idx_chunk,
                )
                total = total + jnp.sum(lvec * vmask)
                g = g.at[view_idx_chunk].add(g_chunk * vmask[:, None])
                return (total, g), None

            init = (jnp.float32(0.0), jnp.zeros_like(params5))
            (total, g), _ = jax.lax.scan(body, init, jnp.arange(num_chunks, dtype=jnp.int32))
            if int(params5.shape[0]) >= 3:
                d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
                w = (
                    jnp.array(
                        [cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans],
                        jnp.float32,
                    )
                    * active_mask
                )
                total = total + jnp.sum((d2 * w) ** 2)
                ww = (w**2) * 2.0
                g = g.at[1:-1].add(-2.0 * d2 * ww)
                g = g.at[0:-2].add(1.0 * d2 * ww)
                g = g.at[2:].add(1.0 * d2 * ww)
            return total, g
    else:

        def _one_view_loss_unmasked(p5_i, T_nom_i, y_i, masked_vol, view_idx):
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
                None,
                view_indices=view_indices,
            )
            return lvec[0]

        one_view_val_and_grad_batch = jax.jit(
            jax.vmap(
                jax.value_and_grad(_one_view_loss_unmasked),
                in_axes=(0, 0, 0, None, 0),
            )
        )

        def loss_and_grad_manual(params5, vol):
            masked_vol = _apply_vol_mask(vol)

            def body(carry, i):
                total, g = carry
                start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
                params_chunk = jax.lax.dynamic_slice(
                    params5, (start_shifted, 0), (chunk_size, params5.shape[1])
                )
                T_nom_chunk = jax.lax.dynamic_slice(
                    T_nom_all, (start_shifted, 0, 0), (chunk_size, 4, 4)
                )
                y_chunk = jax.lax.dynamic_slice(
                    projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                lvec, g_chunk = one_view_val_and_grad_batch(
                    params_chunk,
                    T_nom_chunk,
                    y_chunk,
                    masked_vol,
                    view_idx_chunk,
                )
                total = total + jnp.sum(lvec * vmask)
                g = g.at[view_idx_chunk].add(g_chunk * vmask[:, None])
                return (total, g), None

            init = (jnp.float32(0.0), jnp.zeros_like(params5))
            (total, g), _ = jax.lax.scan(body, init, jnp.arange(num_chunks, dtype=jnp.int32))
            if int(params5.shape[0]) >= 3:
                d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
                w = (
                    jnp.array(
                        [cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans],
                        jnp.float32,
                    )
                    * active_mask
                )
                total = total + jnp.sum((d2 * w) ** 2)
                ww = (w**2) * 2.0
                g = g.at[1:-1].add(-2.0 * d2 * ww)
                g = g.at[0:-2].add(1.0 * d2 * ww)
                g = g.at[2:].add(1.0 * d2 * ww)
            return total, g

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

    if has_loss_mask:

        def _ls_weight_chunk(y_chunk: jnp.ndarray, mask_chunk: jnp.ndarray) -> jnp.ndarray:
            return loss_adapter.gauss_newton_weights(y_chunk, mask_chunk)

        def _gn_update_all(params5, vol):
            masked_vol = _apply_vol_mask(vol)

            def body(dp_acc, i):
                start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
                params_chunk = jax.lax.dynamic_slice(
                    params5, (start_shifted, 0), (chunk_size, params5.shape[1])
                )
                T_chunk = jax.lax.dynamic_slice(
                    T_nom_all, (start_shifted, 0, 0), (chunk_size, 4, 4)
                )
                y_chunk = jax.lax.dynamic_slice(
                    projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                mask_chunk = jax.lax.dynamic_slice(
                    loss_mask, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                w_chunk = _ls_weight_chunk(y_chunk, mask_chunk)
                dp_chunk = _gn_update_batch(
                    params_chunk,
                    T_chunk,
                    y_chunk,
                    masked_vol,
                    w_chunk,
                )
                dp_acc = dp_acc.at[view_idx_chunk].add(dp_chunk * vmask[:, None])
                return dp_acc, None

            dp0 = jnp.zeros_like(params5)
            dp_all, _ = jax.lax.scan(body, dp0, jnp.arange(num_chunks, dtype=jnp.int32))
            return dp_all
    else:

        def _ls_weight_chunk(y_chunk: jnp.ndarray) -> jnp.ndarray:
            return loss_adapter.gauss_newton_weights(y_chunk, None)

        def _gn_update_all(params5, vol):
            masked_vol = _apply_vol_mask(vol)

            def body(dp_acc, i):
                start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
                params_chunk = jax.lax.dynamic_slice(
                    params5, (start_shifted, 0), (chunk_size, params5.shape[1])
                )
                T_chunk = jax.lax.dynamic_slice(
                    T_nom_all, (start_shifted, 0, 0), (chunk_size, 4, 4)
                )
                y_chunk = jax.lax.dynamic_slice(
                    projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                w_chunk = _ls_weight_chunk(y_chunk)
                dp_chunk = _gn_update_batch(
                    params_chunk,
                    T_chunk,
                    y_chunk,
                    masked_vol,
                    w_chunk,
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
            frozen_params5=frozen_params5,
            active_cols=active_col_indices_np,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            loss_before_value=loss_before_value,
            objective_fn=lambda candidate: align_loss(candidate, vol),
            eval_loss_fn=lambda candidate, label: _evaluate_align_loss(
                lambda: align_loss_jit(candidate, vol),
                fallback=math.inf,
                context=f"Treating L-BFGS {label} candidate as rejected "
                "during alignment loss evaluation",
            ),
            apply_param_constraints=_apply_full_constraints,
            is_expected_failure=_is_expected_align_eval_failure,
            cfg=PoseLbfgsConfig(
                maxiter=int(cfg.lbfgs_maxiter),
                ftol=float(cfg.lbfgs_ftol),
                gtol=float(cfg.lbfgs_gtol),
                maxls=int(cfg.lbfgs_maxls),
                memory_size=int(cfg.lbfgs_memory_size),
            ),
            motion_model=motion_model if use_smooth_pose_model else None,
        )
        if result.stats.get("lbfgs_fallback_to_gd"):
            logging.warning(
                "%s; falling back to GD for this alignment step",
                result.stats.get("lbfgs_message"),
            )
        return result.params5, result.motion_coeffs, result.loss, result.stats

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
        }
        outer_start = time.perf_counter()

        # Reconstruction step
        class _GAll:
            def pose_for_view(self, i):
                T_nom = jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32)
                T_al = se3_from_5d(params5[i])
                return tuple(map(tuple, T_nom @ T_al))

            def rays_for_view(self, i):
                return geometry.rays_for_view(i)

        def _run_fista_safe(vpb: int | None, unroll: int, gather: str, gm: str):
            fista_cfg = FistaConfig(
                iters=cfg.recon_iters,
                lambda_tv=cfg.lambda_tv,
                regulariser=cfg.regulariser,
                huber_delta=cfg.huber_delta,
                L=L_prev,
                views_per_batch=vpb,
                projector_unroll=int(unroll),
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=gather,
                grad_mode=gm,
                tv_prox_iters=int(cfg.tv_prox_iters),
                recon_rel_tol=cfg.recon_rel_tol,
                recon_patience=(int(cfg.recon_patience) if cfg.recon_patience is not None else 0),
            )
            return fista_tv(
                _GAll(),
                grid,
                detector,
                projections,
                init_x=x,
                config=fista_cfg,
                det_grid=det_grid,
            )

        def _run_spdhg():
            spdhg_cfg = SPDHGConfig(
                iters=int(cfg.recon_iters),
                lambda_tv=float(cfg.lambda_tv),
                regulariser=cfg.regulariser,
                huber_delta=float(cfg.huber_delta),
                views_per_batch=max(1, int(cfg.views_per_batch)),
                seed=int(cfg.spdhg_seed) + int(outer_idx) - 1,
                projector_unroll=int(cfg.projector_unroll),
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=cfg.gather_dtype,
                positivity=bool(cfg.recon_positivity),
                log_every=1,
            )
            return spdhg_tv(
                _GAll(),
                grid,
                detector,
                projections,
                init_x=x,
                config=spdhg_cfg,
                det_grid=det_grid,
            )

        vpb0 = cfg.views_per_batch if cfg.views_per_batch > 0 else None
        recon_retry = False
        recon_start = time.perf_counter()
        if recon_algo == "fista":
            try:
                x, info_rec = _run_fista_safe(
                    vpb0,
                    int(cfg.projector_unroll),
                    cfg.gather_dtype,
                    "auto",
                )
            except Exception as e:
                msg = str(e)
                is_oom = (
                    ("RESOURCE_EXHAUSTED" in msg)
                    or ("Out of memory" in msg)
                    or ("Allocator" in msg)
                )
                if is_oom:
                    logging.warning(
                        "FISTA OOM detected; retrying with safer settings (vpb=1, unroll=1, stream)"
                    )
                    try:
                        recon_retry = True
                        x, info_rec = _run_fista_safe(1, 1, cfg.gather_dtype, "stream")
                    except Exception as e2:
                        msg2 = str(e2)
                        if (
                            ("RESOURCE_EXHAUSTED" in msg2)
                            or ("Out of memory" in msg2)
                            or ("Allocator" in msg2)
                        ):
                            logging.error(
                                "FISTA still OOM at finest level. Reduce memory pressure "
                                "(smaller problem size or lower internal batching), or "
                                "provide --recon-L to skip power-method."
                            )
                        raise
                else:
                    raise
        else:
            x, info_rec = _run_spdhg()
        # Ensure device work is finished before timing recon.
        jax.block_until_ready(x)
        recon_time = time.perf_counter() - recon_start
        stat["recon_time"] = recon_time
        stat["recon_retry"] = recon_retry
        # Capture and reuse measured L next iteration (with small safety margin)
        if recon_algo == "fista":
            try:
                L_meas = float(info_rec.get("L", 0.0))
                if L_meas > 0.0:
                    L_prev = 1.2 * L_meas
                    stat["L_meas"] = L_meas
                    stat["L_next"] = L_prev
            except Exception:
                pass
        if info_rec and "loss" in info_rec and info_rec["loss"]:
            try:
                lhist = info_rec["loss"]
                stat["recon_loss_first"] = float(lhist[0])
                stat["recon_loss_last"] = float(lhist[-1])
                stat["recon_loss_min"] = float(min(lhist))
                if recon_algo == "fista":
                    stat["fista_first"] = float(lhist[0])
                    stat["fista_last"] = float(lhist[-1])
                    stat["fista_min"] = float(min(lhist))
            except Exception:
                pass
        if recon_algo == "spdhg" and info_rec:
            for src, dst in (
                ("tau", "spdhg_tau"),
                ("sigma_data", "spdhg_sigma_data"),
                ("sigma_tv", "spdhg_sigma_tv"),
                ("views_per_batch", "spdhg_views_per_batch"),
                ("num_blocks", "spdhg_num_blocks"),
                ("A_norm", "spdhg_A_norm"),
            ):
                value = info_rec.get(src)
                if value is not None:
                    stat[dst] = (
                        int(value)
                        if dst in {"spdhg_views_per_batch", "spdhg_num_blocks"}
                        else float(value)
                    )
            stat["spdhg_seed"] = int(cfg.spdhg_seed) + int(outer_idx) - 1

        # Alignment step: Gauss–Newton, LBFGS, or gradient descent
        # Evaluate alignment loss before update (needed for GN acceptance / early stop)
        align_start = time.perf_counter()
        loss_before = _evaluate_align_loss(
            lambda: align_loss_jit(params5, x),
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
        loss_after = None
        if step_kind == "gn":
            params5_prev = params5
            dp_all = _gn_update_all(params5_prev, x) * active_mask
            constrain_candidate = (
                _project_params_to_smooth if use_smooth_pose_model else _apply_full_constraints
            )
            if cfg.gn_accept_only_improving and (loss_before is not None):
                smooth_candidate = None
                if int(params5.shape[0]) >= 3:
                    smooth_candidate = lambda candidate, weights: _smooth_gn_candidate(
                        constrain_candidate(candidate),
                        smoothness_gram,
                        weights,
                    )
                params5, loss_after = _select_gn_candidate(
                    params5_prev,
                    dp_all,
                    loss_before=loss_before,
                    eval_loss=lambda candidate: float(
                        _evaluate_align_loss(
                            lambda: align_loss_jit(candidate, x),
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
                params5 = constrain_candidate(params5)
            else:
                params5 = constrain_candidate(params5_prev + dp_all)
                candidate_loss = _evaluate_align_loss(
                    lambda: align_loss_jit(params5, x),
                    fallback=math.inf,
                    context="Treating GN step as rejected during alignment loss evaluation",
                )
                if candidate_loss is not None and math.isfinite(candidate_loss):
                    loss_after = candidate_loss
                else:
                    params5 = params5_prev
                    loss_after = loss_before
            if use_smooth_pose_model:
                motion_coeffs = fit_motion_coefficients(motion_model, params5)
                params5 = _coeffs_to_constrained_params(motion_coeffs)
            # Log step stats
            try:
                stat["rot_mean"] = float(jnp.mean(jnp.abs(dp_all[:, :3])))
                stat["trans_mean"] = float(jnp.mean(jnp.abs(dp_all[:, 3:])))
            except Exception:
                pass

        elif step_kind == "lbfgs":
            params5, motion_coeffs, loss_after, lbfgs_stats = _run_lbfgs_alignment_step(
                params5,
                motion_coeffs,
                x,
                loss_before,
            )
            stat.update(lbfgs_stats)
            if stat.get("lbfgs_fallback_to_gd"):
                step_kind = "gd"
                params5, motion_coeffs, loss_after, rms = _run_gd_alignment_step(
                    params5,
                    motion_coeffs,
                    x,
                    loss_before,
                )
                try:
                    stat["rot_rms"] = float(jnp.mean(rms[:3]))
                    stat["trans_rms"] = float(jnp.mean(rms[3:]))
                except Exception:
                    pass
        else:
            params5, motion_coeffs, loss_after, rms = _run_gd_alignment_step(
                params5,
                motion_coeffs,
                x,
                loss_before,
            )
            try:
                stat["rot_rms"] = float(jnp.mean(rms[:3]))
                stat["trans_rms"] = float(jnp.mean(rms[3:]))
            except Exception:
                pass
        stat["step_kind"] = step_kind
        stat["loss_after_step"] = loss_after
        params5, final_gauge_stats = _apply_full_constraints_with_stats(params5)
        if use_smooth_pose_model:
            motion_coeffs = fit_motion_coefficients(motion_model, params5)
            params5, final_gauge_stats = _apply_full_constraints_with_stats(
                expand_motion_coefficients(motion_model, motion_coeffs)
            )
            motion_coeffs = fit_motion_coefficients(motion_model, params5)
        stat["gauge_fix"] = gauge_fix
        stat["gauge_fix_dofs"] = ",".join(gauge_dofs)
        if gauge_fix == "mean_translation":
            stat["dx_mean_before_gauge"] = float(final_gauge_stats["dx_mean_before"])
            stat["dz_mean_before_gauge"] = float(final_gauge_stats["dz_mean_before"])
            stat["dx_mean_after_gauge"] = float(final_gauge_stats["dx_mean_after"])
            stat["dz_mean_after_gauge"] = float(final_gauge_stats["dz_mean_after"])
        # Ensure device work from alignment step is finished before timing.
        jax.block_until_ready(params5)
        stat["align_time"] = time.perf_counter() - align_start

        # Track overall data loss
        final_loss_fallback = loss_after
        if final_loss_fallback is None:
            final_loss_fallback = loss_before
        if final_loss_fallback is None and loss_hist:
            final_loss_fallback = loss_hist[-1]
        total_loss_eval = _evaluate_align_loss(
            lambda: align_loss_jit(params5, x),
            fallback=final_loss_fallback,
            context="Using fallback for final alignment loss bookkeeping",
        )
        total_loss = float(total_loss_eval) if total_loss_eval is not None else math.nan
        loss_hist.append(total_loss)
        stat["loss_after"] = total_loss
        if loss_before is not None:
            delta = total_loss - loss_before
            stat["loss_delta"] = delta
            if math.isfinite(loss_before) and abs(loss_before) > 1e-12:
                stat["loss_rel_pct"] = (delta / loss_before) * 100.0
            else:
                stat["loss_rel_pct"] = None
            if math.isfinite(loss_before) and math.isfinite(total_loss):
                denom = max(abs(loss_before), 1e-12)
                rel_impr = (loss_before - total_loss) / denom
            else:
                rel_impr = None
        else:
            stat["loss_delta"] = None
            stat["loss_rel_pct"] = None
            rel_impr = None
        stat["rel_impr"] = rel_impr

        outer_time = time.perf_counter() - outer_start
        stat["outer_time"] = outer_time
        stat["cumulative_time"] = time.perf_counter() - wall_start
        outer_stats.append(stat)

        if cfg.log_summary:
            _log_outer_summary(stat)

        should_break = False
        if observer is not None:
            observer_action = _normalize_observer_action(observer(x, params5, dict(stat)))
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
    from ..recon.multires import (
        _validated_scale_factor,
        bin_projections,
        scale_detector,
        scale_grid,
        upsample_volume,
    )

    if cfg is None:
        cfg = AlignConfig()
    scoped_dofs = _scoped_dofs_for_cfg(cfg, geometry=geometry)
    _validate_scoped_geometry_pose_gauges(scoped_dofs)
    active_mask_tuple = scoped_dofs.pose_mask
    geometry_state = GeometryCalibrationState.from_checkpoint(
        resume_state.geometry_calibration_state if resume_state is not None else None,
        geometry,
        active_geometry_dofs=scoped_dofs.active_geometry_dofs,
    )

    validate_grid(grid, "align_multires grid")
    validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="align_multires projections",
    )

    factors_list = [_validated_scale_factor(f) for f in factors]
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
        loss_adapter = build_loss_adapter(active_loss_spec, y)
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

        geometry_stats: list[OuterStat] = []
        geometry_for_align = geometry
        detector_for_align = d
        det_grid_for_align = None
        geometry_completed_outer_iters = 0
        geometry_wall_time = 0.0
        if geometry_state.active_geometry_dofs:
            geometry_start = time.perf_counter()
            detector_u_heldout_only = (
                tuple(geometry_state.active_geometry_dofs) == ("det_u_px",)
                and not any(active_mask_tuple)
            )
            geometry_outer_iters = (
                1
                if any(active_mask_tuple) or detector_u_heldout_only
                else int(cfg.outer_iters)
            )
            raw_geometry_stats = []
            for geometry_outer_idx in range(1, int(geometry_outer_iters) + 1):
                x0, geometry_state, step_stats = optimize_geometry_blocks_for_level(
                    geometry=geometry,
                    grid=g,
                    detector=d,
                    projections=y,
                    init_x=x0,
                    state=geometry_state,
                    factor=int(lvl["factor"]),
                    recon_iters=int(cfg.recon_iters),
                    lambda_tv=float(cfg.lambda_tv),
                    regulariser=str(cfg.regulariser),
                    huber_delta=float(cfg.huber_delta),
                    tv_prox_iters=int(cfg.tv_prox_iters),
                    views_per_batch=max(1, int(cfg.views_per_batch)),
                    projector_unroll=int(cfg.projector_unroll),
                    checkpoint_projector=bool(cfg.checkpoint_projector),
                    gather_dtype=str(cfg.gather_dtype),
                    gn_damping=float(cfg.gn_damping),
                    outer_iters=1,
                    loss_adapter=loss_adapter,
                    loss_spec=active_loss_spec,
                )
                for stat in step_stats:
                    stat["geometry_outer_idx"] = int(geometry_outer_idx)
                raw_geometry_stats.extend(step_stats)
            geometry_wall_time = time.perf_counter() - geometry_start
            geometry_completed_outer_iters = max(
                (
                    int(stat.get("geometry_outer_idx", 0))
                    for stat in raw_geometry_stats
                    if isinstance(stat, Mapping)
                ),
                default=0,
            )
            geometry_stats = [
                {
                    **dict(stat),
                    "level_factor": int(lvl["factor"]),
                    "level_index": int(li),
                    "global_outer_idx": int(
                        global_outer_idx + int(stat.get("geometry_outer_idx", 0))
                    ),
                    "loss_kind": active_loss_name,
                }
                for stat in raw_geometry_stats
            ]
            geometry_for_align = geometry_with_axis_state(geometry, g, d, geometry_state)
            det_grid_for_align = level_detector_grid(
                d,
                state=geometry_state,
                factor=int(lvl["factor"]),
            )
        else:
            det_grid_for_align = None

        # Run alignment at this level
        # Re-estimate L at each level using a fresh (streamed) power-method for stability
        cfg_level = replace(cfg, recon_L=None, loss=active_loss_spec)
        level_completed_before = (
            int(resume_state.completed_outer_iters_in_level) if resuming_this_level else 0
        )
        stats_before_level = [
            dict(stat) for stat in global_outer_stats if stat.get("level_index") != int(li)
        ]
        loss_before_level = (
            list(loss_hist[:-level_completed_before])
            if resuming_this_level and level_completed_before > 0
            else list(loss_hist)
        )
        global_before_level = int(executed_outer_iters)

        def _enrich_level_stats(local_stats: list[OuterStat]) -> list[OuterStat]:
            enriched_stats: list[OuterStat] = []
            for idx, stat in enumerate(local_stats, start=1):
                enriched = dict(stat)
                enriched["level_factor"] = int(lvl["factor"])
                enriched["level_index"] = int(li)
                enriched["global_outer_idx"] = int(global_before_level + idx)
                enriched["loss_kind"] = str(enriched.get("loss_kind") or active_loss_name)
                level_elapsed = stat.get("cumulative_time")
                try:
                    level_elapsed_f = float(level_elapsed) if level_elapsed is not None else None
                except Exception:
                    level_elapsed_f = None
                enriched["level_elapsed_seconds"] = level_elapsed_f
                enriched["global_elapsed_seconds"] = (
                    float(global_elapsed_offset + level_elapsed_f)
                    if level_elapsed_f is not None
                    else None
                )
                enriched_stats.append(enriched)
            return enriched_stats

        def _emit_multires_checkpoint(state: AlignResumeState, *, level_complete: bool) -> None:
            if checkpoint_callback is None:
                return
            enriched_stats = _enrich_level_stats([dict(stat) for stat in state.outer_stats])
            checkpoint_callback(
                AlignMultiresResumeState(
                    x=state.x,
                    params5=state.params5,
                    motion_coeffs=state.motion_coeffs,
                    level_index=int(li),
                    level_factor=int(lvl["factor"]),
                    completed_outer_iters_in_level=int(state.start_outer_iter),
                    global_outer_iters_completed=int(global_before_level + state.start_outer_iter),
                    prev_factor=prev_factor,
                    loss=loss_before_level + list(state.loss),
                    outer_stats=stats_before_level + enriched_stats,
                    L=state.L,
                    small_impr_streak=int(state.small_impr_streak),
                    elapsed_offset=float(global_elapsed_offset + state.elapsed_offset),
                    level_complete=bool(level_complete),
                    run_complete=False,
                    geometry_calibration_state=geometry_state.to_calibration_state().to_dict(),
                )
            )

        current_level_stats = [
            dict(stat) for stat in global_outer_stats if stat.get("level_index") == int(li)
        ]
        align_resume_state = None
        if resuming_this_level:
            align_resume_state = AlignResumeState(
                x=resume_state.x,
                params5=resume_state.params5,
                motion_coeffs=resume_state.motion_coeffs,
                start_outer_iter=level_completed_before,
                loss=list(loss_hist[-level_completed_before:])
                if level_completed_before > 0
                else [],
                outer_stats=current_level_stats,
                L=resume_state.L,
                small_impr_streak=int(resume_state.small_impr_streak),
                elapsed_offset=float(resume_state.elapsed_offset - global_elapsed_offset),
            )

        def _level_observer(x_obs, params_obs, stat_obs):
            nonlocal stopped_by_observer
            enriched = dict(stat_obs)
            enriched["level_factor"] = int(lvl["factor"])
            enriched["level_index"] = int(li)
            enriched["global_outer_idx"] = int(global_before_level + int(stat_obs["outer_idx"]))
            enriched["loss_kind"] = str(enriched.get("loss_kind") or active_loss_name)
            level_elapsed = stat_obs.get("cumulative_time")
            try:
                level_elapsed_f = float(level_elapsed) if level_elapsed is not None else None
            except Exception:
                level_elapsed_f = None
            enriched["level_elapsed_seconds"] = level_elapsed_f
            enriched["global_elapsed_seconds"] = (
                float(global_elapsed_offset + level_elapsed_f)
                if level_elapsed_f is not None
                else None
            )
            if observer is None:
                return "continue"
            return _normalize_observer_action(observer(x_obs, params_obs, enriched))

        align_kwargs = {}
        if det_grid_for_align is not None:
            align_kwargs["det_grid_override"] = det_grid_for_align
        if any(active_mask_tuple):
            x_lvl, params5, info = align(
                geometry_for_align,
                g,
                detector_for_align,
                y,
                cfg=cfg_level,
                init_x=x0,
                init_params5=params0,
                observer=_level_observer if observer is not None else None,
                resume_state=align_resume_state,
                checkpoint_callback=lambda state: _emit_multires_checkpoint(
                    state,
                    level_complete=False,
                ),
                **align_kwargs,
            )
        else:
            x_lvl = (
                x0
                if x0 is not None
                else jnp.zeros((g.nx, g.ny, g.nz), dtype=jnp.float32)
            )
            params5 = (
                params0
                if params0 is not None
                else jnp.zeros((y.shape[0], 5), dtype=jnp.float32)
            )
            info = {
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
            if geometry_completed_outer_iters:
                info["completed_outer_iters"] = int(geometry_completed_outer_iters)
                info["loss"] = [
                    float(stat["geometry_loss_after"])
                    for stat in geometry_stats
                    if stat.get("geometry_loss_after") is not None
                ]
                info["wall_time_total"] = float(geometry_wall_time)
        level_completed_after = int(
            info.get("completed_outer_iters", len(info.get("outer_stats", [])))
        )
        level_action = _normalize_observer_action(info.get("observer_action"))
        level_complete = (
            level_completed_after >= int(cfg.outer_iters)
            or level_action == "advance_level"
            or not bool(info.get("stopped_by_observer", False))
        )
        loss_hist = loss_before_level + list(info.get("loss", []))
        global_outer_stats = (
            stats_before_level
            + geometry_stats
            + _enrich_level_stats(
            [dict(stat) for stat in info.get("outer_stats", [])]
            )
        )
        global_outer_idx = global_before_level + level_completed_after
        executed_outer_iters = int(global_outer_idx)
        final_pose_model_variables = int(info["pose_model_variables"])
        final_per_view_variables = int(info["per_view_variables"])
        final_pose_model_basis_shape = list(info["pose_model_basis_shape"])
        final_gauge_fix = str(info.get("gauge_fix", final_gauge_fix))
        final_gauge_fix_dofs = list(info.get("gauge_fix_dofs", final_gauge_fix_dofs))
        final_gauge_fix_stats = dict(info.get("gauge_fix_final", {}) or {})
        x_init = x_lvl
        prev_factor = lvl["factor"]
        last_level_index_processed = int(li)
        try:
            elapsed_increment = float(info.get("wall_time_total") or 0.0)
            if any(active_mask_tuple):
                elapsed_increment += float(geometry_wall_time)
            global_elapsed_offset += elapsed_increment
        except Exception:
            pass
        if checkpoint_callback is not None:
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
                    geometry_calibration_state=geometry_state.to_calibration_state().to_dict(),
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
                geometry_calibration_state=geometry_state.to_calibration_state().to_dict(),
            )
        )

    geometry_calibration_diagnostics = add_geometry_acquisition_diagnostics(
        summarize_geometry_calibration_stats(global_outer_stats),
        geometry,
        geometry_state.active_geometry_dofs,
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
            "pose_model": str(cfg.pose_model),
            "pose_model_variables": final_pose_model_variables,
            "per_view_variables": final_per_view_variables,
            "pose_model_basis_shape": final_pose_model_basis_shape,
            "active_dofs": list(
                _scoped_dofs_for_cfg(cfg, geometry=geometry).active_dofs
            ),
            "active_pose_dofs": list(_active_dofs_for_cfg(cfg)),
            "active_geometry_dofs": list(geometry_state.active_geometry_dofs),
            "gauge_fix": final_gauge_fix,
            "gauge_fix_dofs": final_gauge_fix_dofs,
            "gauge_fix_final": final_gauge_fix_stats,
            "geometry_dofs": list(geometry_state.active_geometry_dofs),
            "geometry_calibration_state": (
                geometry_state.to_calibration_state().to_dict()
                if geometry_state.active_geometry_dofs
                else None
            ),
            "geometry_calibration_diagnostics": geometry_calibration_diagnostics,
        },
    )
