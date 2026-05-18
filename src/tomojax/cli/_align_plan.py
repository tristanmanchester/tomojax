from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import jax.numpy as jnp
import numpy as np

from tomojax.align.api import (
    AlignConfig,
    AlignMultiresResumeState,
    AlignResumeState,
    CheckpointError,
    align,
    align_multires,
    normalize_alignment_profile,
    profile_policy_from_config,
    resolve_alignment_schedule,
    resolve_profiled_cli_defaults,
    validate_loss_schedule_levels,
)
from tomojax.cli._runtime import transfer_guard_context
from tomojax.geometry import (
    Detector,
    Grid,
    compute_roi,
    grid_from_detector_fov,
    grid_from_detector_fov_slices,
)
from tomojax.io import build_geometry_from_dataset_metadata, load_projection_payload

from ._align_checkpoint import (
    checkpoint_metadata,
    metadata_float,
    metadata_int,
    resume_state_from_checkpoint,
)
from ._align_command import (
    AlignmentMode,
    align_command_from_args,
    parse_dof_args,
    parse_loss_config,
)
from ._align_types import AlignCliExecutionResult, AlignCliRunPlan

if TYPE_CHECKING:
    from tomojax.align.api import FallbackPolicy, GaugeFixMode, GaugePolicy, QualityTier
    from tomojax.recon.types import Regulariser


def init_jax_compilation_cache() -> None:
    """Enable JAX persistent compilation cache for faster re-runs.

    Directory precedence:
    - TOMOJAX_JAX_CACHE_DIR if set
    - ${XDG_CACHE_HOME:-~/.cache}/tomojax/jax_cache
    """
    try:
        import jax

        cache_dir_text = os.environ.get("TOMOJAX_JAX_CACHE_DIR")
        if cache_dir_text:
            cache_dir = Path(cache_dir_text)
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            cache_dir = base / "tomojax" / "jax_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        logging.info("JAX compilation cache: %s", cache_dir)
    except Exception:
        # Best-effort; skip on any failure silently
        pass


def _resolve_recon_grid_and_mask(
    grid: Grid,
    detector: Detector,
    *,
    is_parallel: bool = True,
    roi_mode: str,
    grid_override: tuple[int, int, int] | list[int] | None,
) -> tuple[Grid, bool]:
    recon_grid = grid
    apply_cyl_mask = False

    if roi_mode != "off":
        try:
            roi = compute_roi(grid, detector, crop_y_to_u=is_parallel)
            full_half_x = ((grid.nx / 2.0) - 0.5) * float(grid.vx)
            full_half_y = ((grid.ny / 2.0) - 0.5) * float(grid.vy)
            full_half_z = ((grid.nz / 2.0) - 0.5) * float(grid.vz)
            det_smaller = (
                (roi.r_u + 1e-6) < full_half_x
                or (is_parallel and (roi.r_u + 1e-6) < full_half_y)
                or (roi.r_v + 1e-6) < full_half_z
            )
            if roi_mode == "cube" or (roi_mode == "auto" and det_smaller):
                if roi_mode == "auto" and not is_parallel:
                    recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=False)
                else:
                    recon_grid = grid_from_detector_fov_slices(
                        grid,
                        detector,
                        crop_y_to_u=is_parallel,
                    )
            elif roi_mode == "bbox":
                recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=is_parallel)
            elif roi_mode == "cyl":
                recon_grid = grid_from_detector_fov_slices(
                    grid,
                    detector,
                    crop_y_to_u=is_parallel,
                )
                apply_cyl_mask = True
        except Exception as exc:
            if roi_mode == "auto":
                logging.warning(
                    "--roi=auto could not be applied; continuing without ROI crop: %s",
                    exc,
                    exc_info=True,
                )
            else:
                raise ValueError(f"Failed to apply requested --roi={roi_mode!r}") from exc

    # Explicit grid overrides take full precedence over ROI-derived masking.
    if grid_override is not None:
        NX, NY, NZ = map(int, grid_override)
        recon_grid = replace(recon_grid, nx=NX, ny=NY, nz=NZ)
        apply_cyl_mask = False

    return recon_grid, apply_cyl_mask


def _schedule_for_public_mode(mode: AlignmentMode, *, align_profile: str) -> str:
    """Resolve the product-facing alignment mode to an internal schedule."""
    if mode == "cor":
        return "cor"
    if mode == "pose":
        return "lightning_pose" if align_profile == "lightning" else "tortoise_pose"
    if mode == "max":
        return "setup_safe"
    return "setup_safe"


def build_align_cli_run_plan(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_metadata: dict[str, Any],
) -> AlignCliRunPlan:
    loss_config, loss_params = parse_loss_config(args, parser)
    optimise_dofs, freeze_dofs = parse_dof_args(args, parser)
    level_args = cast("list[int] | None", args.levels)
    levels = (
        [int(v) for v in level_args] if level_args is not None and len(level_args) > 0 else None
    )
    try:
        validate_loss_schedule_levels(loss_config, levels if levels is not None else [1])
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.align_profile = normalize_alignment_profile(cast("str", args.align_profile))
    except ValueError as exc:
        parser.error(str(exc))
    configured_keys = set(cast("list[str]", config_metadata.get("explicit_cli_keys", []))) | set(
        cast("dict[str, object]", config_metadata.get("config_file_values", {})).keys()
    )
    if (
        cast("str", args.mode) == "max"
        and "align_profile" not in configured_keys
        and "quality" not in configured_keys
    ):
        args.align_profile = "tortoise"
    profile_options = resolve_profiled_cli_defaults(
        align_profile=cast("str", args.align_profile),
        current={
            "projector_backend": cast("str", args.projector_backend),
            "gather_dtype": cast("str", args.gather_dtype),
            "regulariser": cast("str", args.regulariser),
            "recon_algo": cast("str", args.recon_algo),
            "views_per_batch": cast("int", args.views_per_batch),
            "checkpoint_projector": cast("bool", args.checkpoint_projector),
            "pose_model": cast("str", args.pose_model),
        },
        configured_keys=configured_keys,
    )
    args.projector_backend = str(profile_options["projector_backend"])
    args.gather_dtype = str(profile_options["gather_dtype"])
    args.regulariser = str(profile_options["regulariser"])
    args.recon_algo = str(profile_options["recon_algo"])
    args.views_per_batch = metadata_int(profile_options["views_per_batch"], 0)
    args.checkpoint_projector = bool(profile_options["checkpoint_projector"])
    args.pose_model = str(profile_options["pose_model"])
    args.quality_tier = str(profile_options["quality_tier"])
    args.fallback_policy = str(profile_options["fallback_policy"])
    if "schedule" not in configured_keys and optimise_dofs is None:
        args.schedule = _schedule_for_public_mode(
            cast("AlignmentMode", args.mode),
            align_profile=cast("str", args.align_profile),
        )
    config_metadata["profile_options"] = dict(profile_options)
    effective_options = config_metadata.get("effective_options")
    if isinstance(effective_options, dict):
        for key in (
            "mode",
            "align_profile",
            "projector_backend",
            "gather_dtype",
            "regulariser",
            "recon_algo",
            "views_per_batch",
            "checkpoint_projector",
            "pose_model",
            "quality_tier",
            "fallback_policy",
        ):
            effective_options[key] = getattr(args, key)

    command = align_command_from_args(args)
    meta = load_projection_payload(command.data)
    geometry_meta = meta.geometry_inputs()
    initial_grid_override = (
        command.grid if (meta.grid is None and command.grid is not None) else None
    )
    grid, detector, geom = build_geometry_from_dataset_metadata(
        geometry_meta,
        grid_override=initial_grid_override,
        apply_saved_alignment=False,
    )
    projections = jnp.asarray(meta.projections, dtype=np.float32)
    try:
        resolved_schedule = resolve_alignment_schedule(
            schedule=command.schedule,
            optimise_dofs=optimise_dofs,
            freeze_dofs=freeze_dofs,
            geometry_dofs=(),
            geometry=geom,
            gauge_policy=cast("GaugePolicy", command.gauge_policy),
            opt_method=command.opt_method,
            outer_iters=command.outer_iters,
            early_stop=command.early_stop,
        )
        geometry_dofs = resolved_schedule.active_geometry_dofs
        schedule_metadata: dict[str, object] | None = resolved_schedule.to_dict()
    except ValueError as exc:
        parser.error(str(exc))

    from tomojax.backends import default_gather_dtype as _default_gather_dtype

    gather_dtype = command.requested_gather_dtype
    if gather_dtype == "auto":
        gather_dtype = _default_gather_dtype()

    cfg = AlignConfig(
        align_profile=command.align_profile,
        outer_iters=command.outer_iters,
        recon_iters=command.recon_iters,
        recon_algo=cast(
            "Literal['fista', 'spdhg', 'fista_tv', 'spdhg_tv', 'fista-tv', 'spdhg-tv']",
            command.recon_algo,
        ),
        lambda_tv=command.lambda_tv,
        regulariser=cast("Regulariser", command.regulariser),
        huber_delta=command.huber_delta,
        tv_prox_iters=command.tv_prox_iters,
        recon_positivity=command.recon_positivity,
        spdhg_seed=command.spdhg_seed,
        lr_rot=command.lr_rot,
        lr_trans=command.lr_trans,
        views_per_batch=command.views_per_batch,
        projector_unroll=command.projector_unroll,
        projector_backend=command.projector_backend,
        quality_tier=cast("QualityTier", command.quality_tier),
        fallback_policy=cast("FallbackPolicy", command.fallback_policy),
        checkpoint_projector=command.checkpoint_projector,
        gather_dtype=gather_dtype,
        opt_method=command.opt_method,
        gn_damping=command.gn_damping,
        lbfgs_maxiter=command.lbfgs_maxiter,
        lbfgs_ftol=command.lbfgs_ftol,
        lbfgs_gtol=command.lbfgs_gtol,
        lbfgs_maxls=command.lbfgs_maxls,
        lbfgs_memory_size=command.lbfgs_memory_size,
        w_rot=command.w_rot,
        w_trans=command.w_trans,
        schedule=command.schedule,
        optimise_dofs=optimise_dofs,
        freeze_dofs=freeze_dofs,
        geometry_dofs=(),
        bounds=() if command.bounds is None else command.bounds,
        gauge_policy=cast(
            "GaugePolicy | Literal['anchor-mean', 'prior-required', 'diagnose-only']",
            command.gauge_policy,
        ),
        pose_model=cast(
            "Literal['per_view', 'per-view', 'polynomial', 'spline']",
            command.pose_model,
        ),
        knot_spacing=command.knot_spacing,
        degree=command.degree,
        gauge_fix=cast("GaugeFixMode", command.gauge_fix),
        loss=loss_config,
        seed_translations=command.seed_translations,
        log_summary=command.log_summary,
        log_compact=command.log_compact,
        recon_L=command.recon_l,
        early_stop=command.early_stop,
        early_stop_rel_impr=(
            command.early_stop_rel if command.early_stop_rel is not None else 1e-3
        ),
        early_stop_patience=(
            command.early_stop_patience if command.early_stop_patience is not None else 2
        ),
        mask_vol=command.mask_vol,
    )
    schedule_metadata["profile_policy"] = profile_policy_from_config(cfg).to_dict()
    recon_grid, apply_cyl_mask = _resolve_recon_grid_and_mask(
        grid,
        detector,
        is_parallel=meta.geometry_type == "parallel",
        roi_mode=command.roi.lower(),
        grid_override=command.grid,
    )
    if recon_grid is not grid:
        _, _, geom = build_geometry_from_dataset_metadata(
            geometry_meta,
            grid_override=recon_grid,
            apply_saved_alignment=False,
        )

    checkpoint_path = command.checkpoint or command.resume
    checkpoint_every = command.checkpoint_every
    if checkpoint_path is not None and checkpoint_every is None:
        checkpoint_every = 1
    if checkpoint_every is not None and int(checkpoint_every) < 1:
        parser.error("--checkpoint-every must be an integer >= 1")

    run_levels = levels
    if run_levels is None and (command.schedule is not None or bool(geometry_dofs)):
        run_levels = [1]

    expected_checkpoint_metadata = checkpoint_metadata(
        meta=meta,
        projections=projections,
        cfg=cfg,
        command=command,
        recon_grid=recon_grid,
        detector=detector,
        state_grid=recon_grid,
        state_detector=detector,
        gather_dtype=gather_dtype,
        levels=run_levels,
        level_index=0,
        level_factor=1,
        completed_outer_iters_in_level=0,
        global_outer_iters_completed=0,
        prev_factor=None,
        L_prev=None,
        small_impr_streak=0,
        elapsed_offset=0.0,
        level_complete=False,
        run_complete=False,
        schedule_metadata=schedule_metadata,
    )
    resume_state = None
    if command.resume is not None:
        try:
            resume_state = resume_state_from_checkpoint(
                command.resume,
                expected_metadata=expected_checkpoint_metadata,
                used_multires=run_levels is not None,
            )
        except CheckpointError as exc:
            raise SystemExit(f"tomojax align: {exc}") from exc
        logging.info("Resuming alignment from checkpoint %s", command.resume)

    return AlignCliRunPlan(
        command=command,
        cli_args=args,
        config_metadata=config_metadata,
        loss_config=loss_config,
        loss_params=loss_params,
        levels=levels,
        run_levels=run_levels,
        meta=meta,
        geometry_meta=geometry_meta,
        grid=grid,
        recon_grid=recon_grid,
        detector=detector,
        geometry=geom,
        projections=projections,
        cfg=cfg,
        gather_dtype=gather_dtype,
        geometry_dofs=tuple(geometry_dofs),
        schedule_metadata=schedule_metadata,
        checkpoint_path=checkpoint_path,
        checkpoint_every=None if checkpoint_every is None else int(checkpoint_every),
        resume_state=resume_state,
        apply_cyl_mask=apply_cyl_mask,
    )


def execute_alignment_plan(
    plan: AlignCliRunPlan,
    *,
    single_checkpoint_callback: Callable[..., None],
    multires_checkpoint_callback: Callable[[AlignMultiresResumeState], None],
) -> AlignCliExecutionResult:
    command = plan.command
    if plan.run_levels is not None and len(plan.run_levels) > 0:
        with transfer_guard_context(command.transfer_guard):
            x, params5, info = align_multires(
                plan.geometry,
                plan.recon_grid,
                plan.detector,
                plan.projections,
                factors=plan.run_levels,
                cfg=plan.cfg,
                resume_state=(
                    plan.resume_state
                    if isinstance(plan.resume_state, AlignMultiresResumeState)
                    else None
                ),
                checkpoint_callback=multires_checkpoint_callback
                if plan.checkpoint_path is not None
                else None,
            )
        return AlignCliExecutionResult(x=x, params5=params5, info=dict(info))

    with transfer_guard_context(command.transfer_guard):
        x, params5, info = align(
            plan.geometry,
            plan.recon_grid,
            plan.detector,
            plan.projections,
            cfg=plan.cfg,
            resume_state=plan.resume_state
            if isinstance(plan.resume_state, AlignResumeState)
            else None,
            checkpoint_callback=single_checkpoint_callback
            if plan.checkpoint_path is not None
            else None,
        )
    info_dict = dict(info)
    if plan.checkpoint_path is not None:
        motion_coeffs = info_dict.get("motion_coeffs")
        motion_coeffs_array = (
            jnp.asarray(motion_coeffs, dtype=np.float32)
            if isinstance(motion_coeffs, np.ndarray)
            else None
        )
        outer_stats_raw = info_dict.get("outer_stats", [])
        if not isinstance(outer_stats_raw, list):
            outer_stats_raw = []
        completed_outer_iters = info_dict.get("completed_outer_iters", len(outer_stats_raw))
        loss_raw = info_dict.get("loss", [])
        if not isinstance(loss_raw, list):
            loss_raw = []
        l_value = info_dict.get("L")
        small_impr_streak = info_dict.get("small_impr_streak", 0)
        wall_time_total = info_dict.get("wall_time_total", 0.0)
        single_checkpoint_callback(
            AlignResumeState(
                x=x,
                params5=params5,
                motion_coeffs=motion_coeffs_array,
                start_outer_iter=metadata_int(completed_outer_iters, len(outer_stats_raw)),
                loss=list(loss_raw),
                outer_stats=[dict(stat) for stat in outer_stats_raw if isinstance(stat, dict)],
                L=float(l_value) if isinstance(l_value, int | float) else None,
                small_impr_streak=metadata_int(small_impr_streak, 0),
                elapsed_offset=metadata_float(wall_time_total, 0.0),
            ),
            run_complete=True,
        )
    return AlignCliExecutionResult(x=x, params5=params5, info=info_dict)
