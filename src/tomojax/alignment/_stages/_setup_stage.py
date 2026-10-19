from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import logging
import math
import time
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import numpy as np

from tomojax.alignment._geometry.geometry_applier import (
    BaseGeometryArrays,
    apply_setup_to_detector_grid,
    materialize_setup_geometry,
    setup_moved_detector,
)
from tomojax.alignment._geometry.initializers import (
    common_line_setup_seed,
    reprojection_det_u_seed,
    sinogram_det_u_seed,
)
from tomojax.alignment._model.diagnostics import validate_active_gauge_policy
from tomojax.alignment._model.dof_specs import ActiveParameterView
from tomojax.alignment._model.dofs import POSE_WIDTH
from tomojax.alignment._model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.alignment._objectives.fold_recon import (
    FoldReconstructionConfig,
    reconstruct_train_fold_nograd,
)
from tomojax.alignment._objectives.folds import FoldSpec
from tomojax.alignment._objectives.loss_adapters import build_loss_adapter
from tomojax.alignment._objectives.validation_residuals import (
    FoldValidation,
    accumulate_validation_normals,
    validation_scorer,
)
from tomojax.alignment._quality_policy import (
    reconstruction_quality_policy,
    scaled_reconstruction_iterations,
)
from tomojax.alignment.optimizers import ValidationLmConfig, run_active_validation_lm
from tomojax.recon.fista_tv import FistaConfig, fista_tv

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from tomojax.alignment._config import AlignConfig
    from tomojax.alignment._model.dofs import DofBounds
    from tomojax.alignment._model.schedules import ResolvedAlignmentStage
    from tomojax.alignment._objectives.folds import FoldArrays
    from tomojax.alignment._objectives.loss_adapters import LossAdapter
    from tomojax.alignment._objectives.loss_specs import AlignmentLossSpec
    from tomojax.alignment._observer import OuterStat
    from tomojax.core.geometry.base import Detector, Geometry, Grid


FoldEvaluation = tuple[int, FoldValidation, dict[str, object]]


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
    cfg: AlignConfig,
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
        validation = FoldValidation(
            frozen_state=setup_state,
            active_view=active_view,
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
            ray_integrator=cfg.ray_integrator,
        )
        normals = accumulate_validation_normals(validation, z_current)
        total_loss = total_loss + normals.loss
        total_grad = total_grad + normals.grad
        total_hess = total_hess + normals.hess
        residual_count += int(normals.residual_count)
        fold_cache.append((fold, validation, fold_recon_info))
        logging.info(
            "Setup validation-LM fold %d/%d: residuals=%d cumulative_loss=%.6g elapsed=%.1fs",
            fold + 1,
            folds.n_folds,
            residual_count,
            float(total_loss),
            time.monotonic() - stage_start,
        )

    score_candidate = _candidate_scorer(fold_cache)

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


def _candidate_scorer(folds: list[FoldEvaluation]) -> Callable[[jnp.ndarray], float]:
    """Own each frozen fold's compiled loss for the candidate search."""
    scorers = tuple(validation_scorer(validation) for _, validation, _ in folds)

    def score(z_candidate: jnp.ndarray) -> float:
        return float(sum(scorer(z_candidate) for scorer in scorers))

    return score


def _build_geometry_stage_stat(
    *,
    objective_result: SetupValidationObjectiveResult,
    active_view: ActiveParameterView,
    cfg: AlignConfig,
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
            "quality_tier": str(getattr(cfg, "stage_quality_tier", "")),
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
            "train_reconstruction_iterations": int(fold_recon_cfg.iterations),
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
    cfg: AlignConfig,
) -> jnp.ndarray:
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "stage_quality_tier", "fast")))
    moved = setup_moved_detector(detector, setup_state.setup, level_factor=int(factor))
    if moved is not None:  # the batched (CUDA) operators, as in the folds
        x_next, _ = fista_tv(
            _geometry_with_setup_state(geometry, grid, moved, setup_state.setup),
            grid,
            moved,
            projections,
            init_x=init_x,
            config=FistaConfig(
                iterations=scaled_reconstruction_iterations(cfg.iterations, quality_policy),
                tv_weight=float(cfg.tv_weight),
                regulariser=cfg.regulariser,
                huber_delta=float(cfg.huber_delta),
                tv_prox_iterations=int(cfg.tv_prox_iterations),
                lipschitz=cfg.lipschitz,
                nonnegative=bool(cfg.nonnegative),
                ray_integrator=cfg.ray_integrator,
                projector_backend="jax" if cfg.projector_backend == "jax" else "auto",
                views_per_batch=int(cfg.views_per_batch) or None,
            ),
        )
        return x_next
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
            projector_model="ray",
            projector_backend="jax",
            iterations=scaled_reconstruction_iterations(cfg.iterations, quality_policy),
            tv_weight=float(cfg.tv_weight),
            regulariser=cfg.regulariser,
            huber_delta=float(cfg.huber_delta),
            tv_prox_iterations=int(cfg.tv_prox_iterations),
            lipschitz=cfg.lipschitz,
            views_per_batch=max(1, int(cfg.views_per_batch)),
            projector_unroll=int(cfg.projector_unroll),
            checkpoint_projector=bool(cfg.checkpoint_projector),
            gather_dtype=str(cfg.gather_dtype),
            nonnegative=bool(cfg.nonnegative),
            ray_integrator=cfg.ray_integrator,
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


_SLAB_ROWS = 8
# Device bytes per validation view and voxel plane crossing each pixel, as
# measured for the linearised Joseph projection (with room to spare).
_VALIDATION_BYTES_PER_PLANE_PIXEL = 36


def _validation_views_per_batch(grid: Grid, detector: Detector) -> int:
    """Views the validation residuals take at once: what fits in 60% of free device memory."""
    from tomojax.backends import device_free_memory_bytes

    free = device_free_memory_bytes()
    if free is None:
        return 1
    planes = max(grid.nx, grid.ny, grid.nz)
    per_view = _VALIDATION_BYTES_PER_PLANE_PIXEL * planes * detector.nv * detector.nu
    return max(1, min(32, int(0.6 * free) // max(1, per_view)))


def _validation_config(cfg: AlignConfig, grid: Grid, detector: Detector) -> AlignConfig:
    """Size validation batches without resetting the stage's non-init quality tier."""
    if int(cfg.views_per_batch) > 0:
        return cfg
    validation = copy.copy(cfg)
    validation.views_per_batch = _validation_views_per_batch(grid, detector)
    return validation


@dataclass(frozen=True)
class _FitInputs:
    """What the setup fit reconstructs and compares: the whole scan, or a slab of it."""

    geometry: Geometry
    grid: Grid
    detector: Detector
    projections: jnp.ndarray
    base: BaseGeometryArrays
    loss_adapter: LossAdapter
    init_x: jnp.ndarray | None


def _fit_inputs(
    whole: _FitInputs, *, dofs: Iterable[str], loss_spec: AlignmentLossSpec, factor: int
) -> _FitInputs:
    """``whole``, or for a parallel scan's axis offset alone, a slab of its central rows."""
    slab = _parallel_row_slab(whole.geometry, whole.grid, whole.detector, whole.projections, dofs)
    if slab is None:
        return whole
    geometry, grid, detector, projections = slab
    base = BaseGeometryArrays.from_geometry(geometry, detector, level_factor=factor)
    adapter = build_loss_adapter(loss_spec, projections)
    return _FitInputs(geometry, grid, detector, projections, base, adapter, None)


def _parallel_row_slab(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    dofs: Iterable[str],
) -> tuple[Geometry, Grid, Detector, jnp.ndarray] | None:
    """A slab of central detector rows, and the grid over it, for fitting ``det_u_px`` alone.

    In a parallel beam every detector row sees the same rotation axis, so its
    offset is found as well from a few rows as from all: the fit's
    reconstructions then cost a slab, not the volume. None for other
    geometries or parameters, or when the detector has few rows already.
    """
    from tomojax.core.geometry import Grid as GridType, ParallelGeometry

    if type(geometry) is not ParallelGeometry or tuple(dofs) != ("det_u_px",):
        return None
    nv = int(projections.shape[1])
    if nv <= 2 * _SLAB_ROWS:
        return None
    r0 = (nv - _SLAB_ROWS) // 2
    r1 = r0 + _SLAB_ROWS
    u, v = detector.center
    v_mid = v + (r0 + r1 - nv) / 2 * detector.dv  # the slab's centre, in lab z
    slab_detector = replace(detector, nv=_SLAB_ROWS, center=(u, v_mid))
    origin = grid.vol_origin
    if origin is not None:
        cx, cy = (
            origin[i] + (n - 1) / 2 * d
            for i, (n, d) in enumerate(((grid.nx, grid.vx), (grid.ny, grid.vy)))
        )
    else:
        cx, cy = (grid.vol_center or (0.0, 0.0, 0.0))[:2]
    nz = max(1, math.ceil(_SLAB_ROWS * detector.dv / grid.vz))
    slab_grid = GridType(
        grid.nx, grid.ny, nz, grid.vx, grid.vy, grid.vz, vol_center=(cx, cy, v_mid)
    )
    slab_geometry = replace(geometry, grid=slab_grid, detector=slab_detector)
    return slab_geometry, slab_grid, slab_detector, projections[:, r0:r1, :]


def _seeded(
    state: AlignmentState,
    seed: dict[str, object] | None,
    *,
    view: ActiveParameterView,
    cfg: AlignConfig,
) -> AlignmentState:
    """``state`` with the detector-centre seed's offset, when one was found."""
    if seed is None:
        return state
    setup = state.setup
    if seed.get("detector_center_seed_applied"):
        offset = float(cast("float", seed["detector_center_seed_det_u_px"]))
        logging.info("Setup detector-centre seed: det_u_px=%.3f", offset)
        setup = setup.replace(det_u_px=jnp.asarray(offset, dtype=jnp.float32))
    if seed.get("axis_seed_applied"):
        x, y = cast("tuple[float, float]", seed["axis_seed_rotations_deg"])
        setup = setup.replace(axis_rot_x_rad=jnp.deg2rad(x), axis_rot_y_rad=jnp.deg2rad(y))
    candidate = state.replace(setup=setup)
    # An infeasible seed can have a better residual than every feasible LM
    # candidate. Bound it before reconstructing, not only during line search.
    lower, upper = view.bounds_whitened(candidate, bounds=cast("DofBounds", cfg.bounds))
    values = view.pack(candidate)
    bounded = jnp.clip(values, lower, upper)
    seed["setup_seed_clipped_to_bounds"] = bool(np.any(np.asarray(values != bounded)))
    return view.unpack(candidate, bounded)


def _setup_seed_diagnostics(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    factor: int,
    projections: jnp.ndarray,
    setup_state: AlignmentState,
    active_view: ActiveParameterView,
    schedule_name: str | None,
    stage: ResolvedAlignmentStage | None,
    cfg: AlignConfig,
) -> dict[str, object] | None:
    """Seed setup before pose fitting, retaining the axial/short-arc COR fallback."""
    if np.any(np.asarray(setup_state.pose.pose_params)):
        return None
    seed_info = common_line_setup_seed(
        projections,
        geometry,
        detector=detector,
        setup=setup_state.setup,
        active_dofs=active_view.dofs,
        bounds=cast("DofBounds", cfg.bounds),
        factor=factor,
    )
    # The legacy sinogram/reprojection searches assume an unrolled detector.
    # Without common-line centre evidence, let the rolled setup objective refine
    # the supplied origin rather than seeding it from a different ray geometry.
    center_ready = seed_info is not None and seed_info.get("detector_center_seed_applied")
    if center_ready or float(setup_state.setup.detector_roll_rad) != 0.0:
        return seed_info
    stage_name = stage.name if stage is not None else schedule_name
    if "det_u_px" not in active_view.dofs:
        return seed_info
    if stage_name not in {"cor", "setup"} and schedule_name not in {"cor", "setup_safe"}:
        return seed_info
    current = float(setup_state.setup.det_u_px)
    if abs(current) > 1e-6:
        return seed_info
    # A parallel scan over a half turn: Vo's sinogram method, which uses every
    # view and is not misled by an object larger than the field of view.
    # Otherwise a search for the offset whose FBP reprojects most
    # consistently, which suits any scan geometry.
    from tomojax.core.geometry import ParallelGeometry

    seed, method = None, "fbp_reprojection_residual_search"
    if type(geometry) is ParallelGeometry:
        seed = sinogram_det_u_seed(projections, np.asarray(geometry.angles))
        method = "sinogram_vo_2014"
    if seed is None:
        seed = reprojection_det_u_seed(projections, geometry, grid, detector)
        method = "fbp_reprojection_residual_search"
    det_u_level_px = float(seed.det_u_px)
    if method == "sinogram_vo_2014":
        # The sinogram locates the absolute detector centre, whereas setup
        # offsets are increments from this level's existing geometry. That
        # centre includes any physical shift introduced by detector binning.
        det_u_level_px -= float(detector.center[0]) / float(detector.du)
    return {
        **(seed_info or {}),
        "detector_center_seed_status": seed.status,
        "detector_center_seed_method": method,
        "detector_center_seed_applied": True,
        # Level pixels to native pixels.
        "detector_center_seed_det_u_px": det_u_level_px * int(factor),
        "detector_center_seed_residual": float(seed.amplitude_px),
    }


def initialize_setup_geometry_for_level(
    *,
    geometry: Geometry,
    detector: Detector,
    projections: jnp.ndarray,
    state: AlignmentState,
    active_geometry_dofs: tuple[str, ...],
    factor: int,
    cfg: AlignConfig,
) -> SetupStageResult:
    """Fit a projection-domain seed without validation or a volume update.

    This stage precedes motion fitting. At subsequent levels, nonzero poses
    leave the calibrated setup untouched rather than fitting uncorrected data.
    """
    base = BaseGeometryArrays.from_geometry(geometry, detector, level_factor=factor)
    state = state.replace(setup=state.setup.replace(nominal_axis_unit=base.nominal_axis_unit))
    view = ActiveParameterView.from_dofs(active_geometry_dofs)
    seed = None
    reason = "nonzero_pose_corrections"
    if not np.any(np.asarray(state.pose.pose_params)):
        seed = common_line_setup_seed(
            projections,
            geometry,
            detector=detector,
            setup=state.setup,
            active_dofs=view.dofs,
            bounds=cast("DofBounds", cfg.bounds),
            factor=factor,
        )
        if (
            seed is not None
            and not seed.get("detector_center_seed_applied")
            and float(cast("float", seed.get("axis_seed_correlation_after", 0.0))) <= 0.0
        ):
            seed = None
        reason = "unavailable_projection_evidence" if seed is None else "projection_evidence"
    state = _seeded(state, seed, view=view, cfg=cfg)
    stat = cast(
        "OuterStat",
        {
            **(seed or {}),
            "geometry_block": "setup_spectral_initializer",
            "geometry_active_dofs": ",".join(view.dofs),
            "geometry_objective": "projection_consistency",
            "geometry_optimizer": "spectral",
            "geometry_status": "unvalidated_seed" if seed is not None else "seed_declined",
            "geometry_seed_reason": reason,
            "geometry_validation_performed": False,
            "num_train_reconstructions": 0,
            "validation_residual_count": 0,
        },
    )
    return _build_setup_stage_result(
        x=cast("jnp.ndarray", state.volume), state=state, setup_stats=[stat]
    )


def _optimize_setup_geometry_bilevel_for_level(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    init_x: jnp.ndarray | None,
    init_pose_params: jnp.ndarray | None,
    state: AlignmentState,
    active_geometry_dofs: Iterable[str],
    factor: int,
    cfg: AlignConfig,
    loss_spec: AlignmentLossSpec,
    loss_name: str,
    schedule_name: str | None = None,
    stage: ResolvedAlignmentStage | None = None,
) -> SetupStageResult:
    _validate_setup_stage_execution_contract(stage)
    base = BaseGeometryArrays.from_geometry(geometry, detector, level_factor=int(factor))
    active_view = ActiveParameterView.from_dofs(active_geometry_dofs)
    setup_state = state.replace(
        setup=state.setup.replace(nominal_axis_unit=base.nominal_axis_unit),
        pose=PoseState(
            jnp.zeros((int(projections.shape[0]), POSE_WIDTH), dtype=jnp.float32)
            if init_pose_params is None
            else jnp.asarray(init_pose_params, dtype=jnp.float32),
            translation_frame=cfg.pose_translation_frame,
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
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "stage_quality_tier", "fast")))
    fold_recon_cfg = FoldReconstructionConfig(
        iterations=scaled_reconstruction_iterations(cfg.iterations, quality_policy),
        tv_weight=float(cfg.tv_weight),
        regulariser=str(cfg.regulariser),
        huber_delta=float(cfg.huber_delta),
        tv_prox_iterations=int(cfg.tv_prox_iterations),
        lipschitz=cfg.lipschitz,
        nonnegative=bool(cfg.nonnegative),
        views_per_batch=int(cfg.views_per_batch) or None,
        projector_unroll=int(cfg.projector_unroll),
        checkpoint_projector=bool(cfg.checkpoint_projector),
        gather_dtype=str(cfg.gather_dtype),
        ray_integrator=cfg.ray_integrator,
        projector_backend="jax" if cfg.projector_backend == "jax" else "auto",
    )
    seed_diagnostics = _setup_seed_diagnostics(
        geometry=geometry,
        grid=grid,
        detector=detector,
        factor=int(factor),
        projections=projections,
        setup_state=setup_state,
        active_view=active_view,
        schedule_name=schedule_name,
        stage=stage,
        cfg=cfg,
    )
    setup_state = _seeded(setup_state, seed_diagnostics, view=active_view, cfg=cfg)
    setup_stats: list[OuterStat] = []
    outer_limit = max(1, int(stage.maxiter if stage is not None else cfg.outer_iterations))
    fit = _fit_inputs(
        _FitInputs(geometry, grid, detector, projections, base, loss_adapter, init_x),
        dofs=active_view.dofs,
        loss_spec=loss_spec,
        factor=int(factor),
    )
    validation_cfg = _validation_config(cfg, fit.grid, fit.detector)
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
            geometry=fit.geometry,
            grid=fit.grid,
            detector=fit.detector,
            projections=fit.projections,
            setup_state=setup_state,
            active_view=active_view,
            base=fit.base,
            folds=folds,
            loss_adapter=fit.loss_adapter,
            fold_recon_cfg=fold_recon_cfg,
            cfg=validation_cfg,
            init_x=fit.init_x,
            factor=factor,
        )
        opt_result = objective_result.opt_result
        setup_state = opt_result.state
        last_loss = float(opt_result.loss)
        stat = _build_geometry_stage_stat(
            objective_result=objective_result,
            active_view=active_view,
            cfg=validation_cfg,
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
        if bool(cfg.early_stop):
            # This round's own improvement: a seeded start may need no more.
            before = float(stat.get("geometry_loss_before", math.inf))
            impr = (before - last_loss) / max(abs(before), 1e-6)
            if math.isfinite(before) and impr < float(cfg.early_stop_rel_impr):
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
