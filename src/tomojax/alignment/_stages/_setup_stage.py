from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import logging
import math
import time
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np

from tomojax.alignment._geometry.geometry_applier import (
    BaseGeometryArrays,
)
from tomojax.alignment._geometry.initializers import (
    common_line_setup_seed,
    reprojection_det_u_seed,
    sinogram_det_u_seed,
)
from tomojax.alignment._model.dof_specs import ActiveParameterView
from tomojax.alignment._model.dofs import POSE_WIDTH
from tomojax.alignment._model.state import AlignmentState, PoseState
from tomojax.alignment._objectives.fold_recon import (
    reconstruct_train_fold_nograd,
)
from tomojax.alignment._objectives.folds import FoldSpec
from tomojax.alignment._objectives.loss_adapters import build_loss_adapter
from tomojax.alignment._objectives.validation_residuals import (
    FoldValidation,
    accumulate_validation_normals,
    validation_scorer,
)
from tomojax.alignment._profiles import (
    reconstruction_quality_policy,
    scaled_reconstruction_iterations,
)
from tomojax.alignment._reconstruction_config import fista_config
from tomojax.alignment.optimizers import ValidationLmConfig, run_active_validation_lm
from tomojax.recon.fista_tv import FistaConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tomojax.alignment._config import AlignConfig
    from tomojax.alignment._model.dofs import DofBounds
    from tomojax.alignment._model.schedules import ResolvedAlignmentStage
    from tomojax.alignment._objectives.folds import FoldArrays
    from tomojax.alignment._objectives.loss_adapters import LossAdapter
    from tomojax.alignment._objectives.loss_specs import AlignmentLossSpec
    from tomojax.alignment._observer import OuterStat
    from tomojax.core.geometry.base import Detector, Geometry, Grid


@dataclass(frozen=True)
class SetupStageResult:
    x: jnp.ndarray
    state: AlignmentState
    losses: list[float]
    checkpoint_outer_stats: list[OuterStat]


def _geometry_calibration_payload(
    state: AlignmentState,
    active_geometry_dofs: Iterable[str],
) -> dict[str, object]:
    return state.to_calibration_state(active_dofs=active_geometry_dofs).to_dict()


# What every validation-LM round records about its method.
_VALIDATION_LM_RECORD: dict[str, object] = {
    "geometry_block": "setup_validation_lm",
    "geometry_objective": "bilevel_cv",
    "geometry_optimizer": "validation_lm",
    "geometry_max_step": 1.0,
    "objective_kind": "bilevel_cv",
    "optimizer_kind": "validation_lm",
    "recon_sensitivity": "stopped",
    "train_reconstruction_gradient": False,
    "fold_eval_mode": "stopped_train_recon_validation_lm",
    "recon_projection_chunked": True,
    "validation_projection_chunked": True,
    "active_gradient_mode": "validation_residual_jvp",
}


def _validation_lm_round(
    fit: _FitInputs,
    *,
    state: AlignmentState,
    view: ActiveParameterView,
    folds: FoldArrays,
    recon_cfg: FistaConfig,
    cfg: AlignConfig,
    factor: int,
) -> tuple[AlignmentState, dict[str, object]]:
    """One bilevel round: reconstruct each train fold, then step on its held-out views."""
    start = time.monotonic()
    z = view.pack(state)
    loss, grad = jnp.float32(0), jnp.zeros_like(z)
    hess = jnp.zeros((z.size, z.size), dtype=jnp.float32)
    residuals, validations = 0, []
    for fold in range(folds.n_folds):
        logging.info(
            "Setup validation-LM fold %d/%d: reconstructing train fold for DOFs=%s",
            fold + 1,
            folds.n_folds,
            ",".join(view.dofs),
        )
        fold_volume, _ = reconstruct_train_fold_nograd(
            geometry=fit.geometry,
            grid=fit.grid,
            detector=fit.detector,
            projections=fit.projections,
            state=state,
            train_idx=folds.train_idx[fold],
            train_mask=folds.train_mask[fold],
            init_x=fit.init_x,
            level_factor=int(factor),
            cfg=recon_cfg,
        )
        validation = FoldValidation(
            frozen_state=state,
            active_view=view,
            base=fit.base,
            grid=fit.grid,
            detector=fit.detector,
            projections=fit.projections,
            loss_adapter=fit.loss_adapter,
            fold_volume=fold_volume,
            val_idx=folds.val_idx[fold],
            val_mask=folds.val_mask[fold],
            views_per_batch=max(1, int(cfg.views_per_batch)),
            projector_unroll=1,
            checkpoint_projector=bool(cfg.checkpoint_projector),
            gather_dtype=str(cfg.gather_dtype),
            ray_integrator=cfg.ray_integrator,
        )
        normals = accumulate_validation_normals(validation, z)
        loss, grad, hess = loss + normals.loss, grad + normals.grad, hess + normals.hess
        residuals += int(normals.residual_count)
        validations.append(validation)
        logging.info(
            "Setup validation-LM fold %d/%d: residuals=%d cumulative_loss=%.6g elapsed=%.1fs",
            fold + 1,
            folds.n_folds,
            residuals,
            float(loss),
            time.monotonic() - start,
        )
    # Each frozen fold keeps its compiled loss for the candidate search; built
    # once every fold is reconstructed, so no scorer is alive during one.
    scorers = [validation_scorer(validation) for validation in validations]
    result = run_active_validation_lm(
        state=state,
        view=view,
        loss=float(loss),
        grad=grad,
        hess=hess,
        score_fn=lambda candidate: float(sum(scorer(candidate) for scorer in scorers)),
        bounds=cfg.bounds,
        cfg=ValidationLmConfig(damping=max(float(cfg.gn_damping), 1e-6)),
    )
    logging.info(
        "Setup validation-LM candidates: loss %.6g -> %.6g accepted=%s elapsed=%.1fs",
        float(loss),
        float(result.loss),
        bool(result.accepted),
        time.monotonic() - start,
    )
    stats = result.stats
    return result.state, {
        **stats,
        **_VALIDATION_LM_RECORD,
        "geometry_loss_before": float(loss),
        "geometry_loss_after": float(result.loss),
        "geometry_accepted": bool(result.accepted),
        "geometry_step_norm": float(stats.get("step_norm_whitened", 0.0) or 0.0),
        "geometry_gradient_norm": float(stats.get("grad_norm_whitened", 0.0) or 0.0),
        "geometry_status": "converged" if result.accepted else "underconverged",
        "validation_residual_count": residuals,
    }


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
        checkpoint_outer_stats=[dict(stat) for stat in setup_stats],
    )


_SLAB_ROWS = 8
# Device bytes per validation view and voxel plane crossing each pixel, as
# measured for the linearised Joseph projection (with room to spare).
_VALIDATION_BYTES_PER_PLANE_PIXEL = 36


def _validation_config(cfg: AlignConfig, grid: Grid, detector: Detector) -> AlignConfig:
    """Size validation batches without resetting the stage's non-init quality tier.

    Unless the configuration fixes it, the validation residuals take as many views
    at once as fit in 60% of free device memory.
    """
    from tomojax.backends import device_free_memory_bytes

    if int(cfg.views_per_batch) > 0:
        return cfg
    validation = copy.copy(cfg)
    free = device_free_memory_bytes()
    planes = max(grid.nx, grid.ny, grid.nz)
    per_view = _VALIDATION_BYTES_PER_PLANE_PIXEL * planes * detector.nv * detector.nu
    validation.views_per_batch = (
        1 if free is None else max(1, min(32, int(0.6 * free) // max(1, per_view)))
    )
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
    """``whole``, or for a parallel scan's axis offset alone, a slab of its central rows.

    In a parallel beam every detector row sees the same rotation axis, so its
    offset is found as well from a few rows as from all: the fit's
    reconstructions then cost a slab, not the volume. Other geometries or
    parameters, and detectors with few rows already, use the whole scan.
    """
    from tomojax.core.geometry import Grid as GridType, ParallelGeometry

    geometry, grid, detector = whole.geometry, whole.grid, whole.detector
    nv = int(whole.projections.shape[1])
    if type(geometry) is not ParallelGeometry or tuple(dofs) != ("det_u_px",):
        return whole
    if nv <= 2 * _SLAB_ROWS:  # few rows already
        return whole
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
    projections = whole.projections[:, r0:r1, :]
    base = BaseGeometryArrays.from_geometry(slab_geometry, slab_detector, level_factor=factor)
    adapter = build_loss_adapter(loss_spec, projections)
    return _FitInputs(slab_geometry, slab_grid, slab_detector, projections, base, adapter, None)


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
    whole: _FitInputs,
    setup_state: AlignmentState,
    active_view: ActiveParameterView,
    stage: ResolvedAlignmentStage,
    *,
    schedule_name: str | None,
    cfg: AlignConfig,
    factor: int,
) -> dict[str, object] | None:
    """Seed setup before pose fitting, retaining the axial/short-arc COR fallback."""
    geometry, grid, detector, projections = (
        whole.geometry,
        whole.grid,
        whole.detector,
        whole.projections,
    )
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
    if "det_u_px" not in active_view.dofs:
        return seed_info
    if stage.name not in {"cor", "setup"} and schedule_name not in {"cor", "setup_safe"}:
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
    factor: int,
    cfg: AlignConfig,
    loss_spec: AlignmentLossSpec,
    loss_name: str,
    schedule_name: str | None,
    stage: ResolvedAlignmentStage,
) -> SetupStageResult:
    if (stage.objective_kind, stage.optimizer_kind) != ("bilevel_cv", "validation_lm"):
        raise ValueError(
            f"Setup alignment stage {stage.name!r} uses objective {stage.objective_kind!r} "
            f"with optimizer {stage.optimizer_kind!r}; setup execution supports only "
            "'bilevel_cv' with 'validation_lm'"
        )
    base = BaseGeometryArrays.from_geometry(geometry, detector, level_factor=int(factor))
    n_views = int(projections.shape[0])
    active_view = ActiveParameterView.from_dofs(stage.active_geometry_dofs)
    setup_state = state.replace(
        setup=state.setup.replace(nominal_axis_unit=base.nominal_axis_unit),
        pose=PoseState(
            jnp.zeros((n_views, POSE_WIDTH), dtype=jnp.float32)
            if init_pose_params is None
            else jnp.asarray(init_pose_params, dtype=jnp.float32),
            translation_frame=cfg.pose_translation_frame,
        ),
        volume=init_x,
    )
    folds = FoldSpec(n_folds=min(2, n_views)).build(n_views)
    loss_adapter = build_loss_adapter(loss_spec, projections)
    if not bool(loss_adapter.supports_setup_validation_lm):
        raise ValueError(
            "Setup validation-LM requires a setup-compatible weighted least-squares loss; "
            f"level {int(factor)} stage {schedule_name or stage.name!r} resolved loss {loss_name!r}"
        )
    quality_policy = reconstruction_quality_policy(str(getattr(cfg, "stage_quality_tier", "fast")))
    fold_recon_cfg = fista_config(
        cfg,
        iterations=scaled_reconstruction_iterations(cfg.iterations, quality_policy),
        views_per_batch=int(cfg.views_per_batch) or None,
        checkpoint_projector=bool(cfg.checkpoint_projector),
        gather_dtype=str(cfg.gather_dtype),
        backend="jax" if cfg.backend == "jax" else "auto",
    )
    whole = _FitInputs(geometry, grid, detector, projections, base, loss_adapter, init_x)
    seed_diagnostics = _setup_seed_diagnostics(
        whole, setup_state, active_view, stage, schedule_name=schedule_name, cfg=cfg, factor=factor
    )
    setup_state = _seeded(setup_state, seed_diagnostics, view=active_view, cfg=cfg)
    fit = _fit_inputs(whole, dofs=active_view.dofs, loss_spec=loss_spec, factor=int(factor))
    validation_cfg = _validation_config(cfg, fit.grid, fit.detector)
    gauge = stage.gauge_decision
    stage_record = {
        "geometry_active_dofs": ",".join(active_view.dofs),
        "geometry_loss_kind": loss_name,
        "quality_tier": str(getattr(cfg, "stage_quality_tier", "")),
        "schedule_name": schedule_name,
        "schedule_stage_index": int(stage.index),
        "schedule_stage_name": stage.name,
        "schedule_stage_active_dofs": ",".join(stage.active_dofs),
        "gauge_policy": stage.gauge_policy,
        "gauge_status": gauge.status,
        "gauge_decision": gauge.to_dict(),
        "objective_provenance": {
            "outer_loss_source": "AlignmentLossSpec",
            "outer_loss_kind": str(loss_name),
            "inner_data_term": "l2_projection",
            "inner_regulariser": "none",
            "validation_split": "interleaved_kfold",
            "differentiation_mode": "none",
            "initialization_policy": "current_level_volume" if init_x is not None else "zeros",
        },
        "outer_loss_kind": str(loss_name),
        "train_reconstruction_iterations": int(fold_recon_cfg.iterations),
        "views_per_batch": max(1, int(validation_cfg.views_per_batch)),
        "n_folds": int(folds.n_folds),
        "folds_used": ",".join(str(fold) for fold in range(folds.n_folds)),
        "num_train_reconstructions": int(folds.n_folds),
    }
    setup_stats: list[OuterStat] = []
    outer_limit = max(1, int(stage.maxiter))
    for outer_idx in range(1, outer_limit + 1):
        logging.info(
            "Setup stage %s outer %d/%d level=%d", stage.name, outer_idx, outer_limit, factor
        )
        setup_state, round_stat = _validation_lm_round(
            fit,
            state=setup_state,
            view=active_view,
            folds=folds,
            recon_cfg=fold_recon_cfg,
            cfg=validation_cfg,
            factor=factor,
        )
        stat = {**round_stat, **stage_record, "geometry_outer_idx": outer_idx}
        if seed_diagnostics is not None and outer_idx == 1:
            stat.update(seed_diagnostics)
        setup_stats.append(cast("OuterStat", stat))
        # This round's own improvement: a seeded start may need no more.
        before, after = (cast("float", stat[f"geometry_loss_{k}"]) for k in ("before", "after"))
        impr = (before - after) / max(abs(before), 1e-6)
        if cfg.early_stop and math.isfinite(before) and impr < float(cfg.early_stop_rel_impr):
            break

    # The level's volume: the same reconstruction over every view, poses included.
    views = jnp.arange(n_views)
    x_next, _ = reconstruct_train_fold_nograd(
        geometry=geometry,
        grid=grid,
        detector=detector,
        projections=projections,
        state=setup_state,
        train_idx=views,
        train_mask=jnp.ones_like(views),
        init_x=init_x,
        level_factor=int(factor),
        cfg=fold_recon_cfg,
    )
    return _build_setup_stage_result(x=x_next, state=setup_state, setup_stats=setup_stats)
