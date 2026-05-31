from __future__ import annotations

# ruff: noqa: D100,D103,TC003
import argparse
from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from tomojax._typed_arrays import (
    jax_float32_array,
    object_list,
    update_jax_config,
)
from tomojax.align.api import (
    AlignConfig,
    AlignmentLossConfig,
    AlignMultiresResumeState,
    AlignResumeState,
    CheckpointError,
    align,
    align_multires,
    load_alignment_checkpoint,
    normalize_alignment_profile,
    profile_policy_from_config,
    resolve_alignment_schedule,
    resolve_profiled_cli_defaults,
    validate_loss_schedule_levels,
)
from tomojax.cli._reconstruction_region import resolve_reconstruction_region
from tomojax.cli._runtime import transfer_guard_context
from tomojax.io import build_geometry_from_dataset_metadata, load_projection_payload

from .checkpoint import (
    AlignCliCheckpointMetadataContext,
    initial_checkpoint_metadata,
    metadata_int,
    resume_state_from_checkpoint,
)
from .command import (
    AlignCommand,
    AlignmentMode,
    align_command_from_args,
    parse_dof_args,
    parse_loss_config,
)
from .types import AlignCliExecutionResult, AlignCliRunPlan

if TYPE_CHECKING:
    import jax.numpy as jnp

    from tomojax.align.api import (
        AlignInfo,
        FallbackPolicy,
        GaugeFixMode,
        GaugePolicy,
        QualityTier,
    )
    from tomojax.geometry import Detector, Geometry, Grid
    from tomojax.io import ProjectionDataset
    from tomojax.recon.types import Regulariser


@dataclass(frozen=True, slots=True)
class _ParsedAlignOptions:
    loss_config: AlignmentLossConfig
    loss_params: dict[str, float]
    optimise_dofs: tuple[str, ...] | None
    freeze_dofs: tuple[str, ...]
    levels: list[int] | None


@dataclass(frozen=True, slots=True)
class _LoadedAlignInputs:
    meta: ProjectionDataset
    geometry_meta: dict[str, Any]
    grid: Grid
    detector: Detector
    geometry: Geometry
    projections: jnp.ndarray


@dataclass(frozen=True, slots=True)
class _ResolvedAlignConfig:
    cfg: AlignConfig
    gather_dtype: str
    schedule_metadata: dict[str, object]


def init_jax_compilation_cache() -> None:
    """Enable JAX persistent compilation cache for faster re-runs.

    Directory precedence:
    - TOMOJAX_JAX_CACHE_DIR if set
    - ${XDG_CACHE_HOME:-~/.cache}/tomojax/jax_cache
    """
    try:
        cache_dir_text = os.environ.get("TOMOJAX_JAX_CACHE_DIR")
        if cache_dir_text:
            cache_dir = Path(cache_dir_text)
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            cache_dir = base / "tomojax" / "jax_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        update_jax_config("jax_compilation_cache_dir", str(cache_dir))
        update_jax_config("jax_persistent_cache_min_entry_size_bytes", -1)
        update_jax_config("jax_persistent_cache_min_compile_time_secs", 0)
        update_jax_config(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        logging.info("JAX compilation cache: %s", cache_dir)
    except Exception:
        # Best-effort; skip on any failure silently
        pass


def _schedule_for_public_mode(mode: AlignmentMode, *, align_profile: str) -> str:
    """Resolve the product-facing alignment mode to an internal schedule."""
    if mode == "cor":
        return "cor"
    if mode == "pose":
        return "lightning_pose" if align_profile == "lightning" else "tortoise_pose"
    if mode == "cor_then_pose":
        return "cor_then_pose"
    if mode in {"auto", "max"}:
        return "setup_safe"
    return "setup_safe"


def _default_levels_for_public_mode(mode: AlignmentMode) -> list[int]:
    """Return the implicit multires pyramid for product modes."""
    if mode in {"auto", "max", "cor_then_pose"}:
        return [4, 2, 1]
    return [1]


def _completed_single_resume_state(
    *,
    x: jnp.ndarray,
    params5: jnp.ndarray,
    info: AlignInfo,
) -> AlignResumeState:
    return AlignResumeState(
        x=x,
        params5=params5,
        motion_coeffs=info["motion_coeffs"],
        start_outer_iter=int(info["completed_outer_iters"]),
        loss=list(info["loss"]),
        outer_stats=list(info["outer_stats"]),
        L=info["L"],
        small_impr_streak=int(info["small_impr_streak"]),
        elapsed_offset=float(info["wall_time_total"]),
    )


def _restore_resume_schedule_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    configured_keys: set[str],
) -> list[str]:
    resume_path = cast("str | None", getattr(args, "resume", None))
    if resume_path is None:
        return []
    try:
        checkpoint = load_alignment_checkpoint(resume_path)
    except CheckpointError as exc:
        raise SystemExit(f"tomojax align: {exc}") from exc
    cli_options = checkpoint.metadata.get("cli_options")
    if not isinstance(cli_options, dict):
        return []
    cli_options = cast("dict[str, object]", cli_options)

    restored_keys: list[str] = []
    for key in ("optimise_dofs", "freeze_dofs", "schedule"):
        if key in configured_keys or key not in cli_options:
            continue
        value = cli_options[key]
        if key in {"optimise_dofs", "freeze_dofs"}:
            if value is None:
                setattr(args, key, None)
            elif isinstance(value, list):
                setattr(args, key, [str(item) for item in object_list(cast("object", value))])
            else:
                parser.error(f"checkpoint cli option {key!r} must be a list or null")
        elif value is None or isinstance(value, str):
            setattr(args, key, value)
        else:
            parser.error("checkpoint cli option 'schedule' must be a string or null")
        restored_keys.append(key)
    return restored_keys


def _configured_cli_keys(config_metadata: dict[str, Any]) -> set[str]:
    return set(cast("list[str]", config_metadata.get("explicit_cli_keys", []))) | set(
        cast("dict[str, object]", config_metadata.get("config_file_values", {})).keys()
    )


def _update_effective_options(
    config_metadata: dict[str, Any],
    args: argparse.Namespace,
    keys: tuple[str, ...] | list[str],
) -> None:
    effective_options = config_metadata.get("effective_options")
    if isinstance(effective_options, dict):
        for key in keys:
            effective_options[key] = getattr(args, key)


def _parse_align_cli_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> _ParsedAlignOptions:
    loss_config, loss_params = parse_loss_config(args, parser)
    optimise_dofs, freeze_dofs = parse_dof_args(args, parser)
    level_args = cast("list[int] | None", args.levels)
    levels = (
        [int(v) for v in level_args] if level_args is not None and len(level_args) > 0 else None
    )
    return _ParsedAlignOptions(
        loss_config=loss_config,
        loss_params=loss_params,
        optimise_dofs=optimise_dofs,
        freeze_dofs=freeze_dofs,
        levels=levels,
    )


def _resolve_profile_defaults_phase(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_metadata: dict[str, Any],
    *,
    configured_keys: set[str],
    optimise_dofs: tuple[str, ...] | None,
) -> None:
    try:
        args.align_profile = normalize_alignment_profile(cast("str", args.align_profile))
    except ValueError as exc:
        parser.error(str(exc))
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
    _update_effective_options(
        config_metadata,
        args,
        (
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
        ),
    )


def _load_alignment_inputs(command: AlignCommand) -> _LoadedAlignInputs:
    meta = load_projection_payload(command.data)
    geometry_meta = meta.geometry_inputs()
    initial_grid_override = (
        command.grid if (meta.grid is None and command.grid is not None) else None
    )
    grid, detector, geometry = build_geometry_from_dataset_metadata(
        geometry_meta,
        grid_override=initial_grid_override,
        apply_saved_alignment=False,
    )
    return _LoadedAlignInputs(
        meta=meta,
        geometry_meta=geometry_meta,
        grid=grid,
        detector=detector,
        geometry=geometry,
        projections=jax_float32_array(meta.projections),
    )


def _resolve_schedule_and_config(
    parser: argparse.ArgumentParser,
    command: AlignCommand,
    parsed: _ParsedAlignOptions,
) -> _ResolvedAlignConfig:
    try:
        resolved_schedule = resolve_alignment_schedule(
            schedule=command.schedule,
            optimise_dofs=parsed.optimise_dofs,
            freeze_dofs=parsed.freeze_dofs,
            gauge_policy=cast("GaugePolicy", command.gauge_policy),
            opt_method=command.opt_method,
            outer_iters=command.outer_iters,
            early_stop=command.early_stop,
        )
        schedule_metadata: dict[str, object] = resolved_schedule.to_dict()
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
        optimise_dofs=parsed.optimise_dofs,
        freeze_dofs=parsed.freeze_dofs,
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
        loss=parsed.loss_config,
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
    return _ResolvedAlignConfig(
        cfg=cfg,
        gather_dtype=gather_dtype,
        schedule_metadata=schedule_metadata,
    )


def build_align_cli_run_plan(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_metadata: dict[str, Any],
) -> AlignCliRunPlan:
    configured_keys = _configured_cli_keys(config_metadata)
    restored_keys = _restore_resume_schedule_options(parser, args, configured_keys=configured_keys)
    _update_effective_options(config_metadata, args, restored_keys)
    parsed = _parse_align_cli_options(parser, args)
    _resolve_profile_defaults_phase(
        parser,
        args,
        config_metadata,
        configured_keys=configured_keys,
        optimise_dofs=parsed.optimise_dofs,
    )
    command = align_command_from_args(args)
    inputs = _load_alignment_inputs(command)
    resolved = _resolve_schedule_and_config(parser, command, parsed)
    region = resolve_reconstruction_region(
        inputs.grid,
        inputs.detector,
        geometry_type=str(inputs.meta.geometry_type),
        roi_mode=command.roi,
        grid_override=command.grid,
        mask_mode=command.mask_vol,
    )
    recon_grid = region.recon_grid
    geom = inputs.geometry
    if recon_grid is not inputs.grid:
        _, _, geom = build_geometry_from_dataset_metadata(
            inputs.geometry_meta,
            grid_override=recon_grid,
            apply_saved_alignment=False,
        )

    checkpoint_path = command.checkpoint or command.resume
    checkpoint_every = command.checkpoint_every
    if checkpoint_path is not None and checkpoint_every is None:
        checkpoint_every = 1
    if checkpoint_every is not None and int(checkpoint_every) < 1:
        parser.error("--checkpoint-every must be an integer >= 1")

    run_levels = parsed.levels
    has_geometry_dofs = bool(resolved.schedule_metadata.get("active_geometry_dofs", ()))
    if run_levels is None and (command.schedule is not None or has_geometry_dofs):
        run_levels = _default_levels_for_public_mode(command.mode)
    try:
        validate_loss_schedule_levels(
            parsed.loss_config,
            run_levels if run_levels is not None else [1],
        )
    except ValueError as exc:
        parser.error(str(exc))

    expected_checkpoint_metadata = initial_checkpoint_metadata(
        context=AlignCliCheckpointMetadataContext(
            meta=inputs.meta,
            projections=inputs.projections,
            cfg=resolved.cfg,
            command=command,
            recon_grid=recon_grid,
            detector=inputs.detector,
            gather_dtype=resolved.gather_dtype,
            schedule_metadata=resolved.schedule_metadata,
        ),
        levels=run_levels,
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
        loss_config=parsed.loss_config,
        loss_params=parsed.loss_params,
        levels=parsed.levels,
        run_levels=run_levels,
        meta=inputs.meta,
        geometry_meta=inputs.geometry_meta,
        grid=inputs.grid,
        recon_grid=recon_grid,
        detector=inputs.detector,
        geometry=geom,
        projections=inputs.projections,
        cfg=resolved.cfg,
        gather_dtype=resolved.gather_dtype,
        schedule_metadata=resolved.schedule_metadata,
        checkpoint_path=checkpoint_path,
        checkpoint_every=None if checkpoint_every is None else int(checkpoint_every),
        resume_state=resume_state,
        apply_cyl_mask=region.apply_output_mask,
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
                config=plan.cfg,
                resume_state=(
                    plan.resume_state
                    if isinstance(plan.resume_state, AlignMultiresResumeState)
                    else None
                ),
                checkpoint_callback=multires_checkpoint_callback
                if plan.checkpoint_path is not None
                else None,
            )
        return AlignCliExecutionResult(x=x, params5=params5, info=info)

    with transfer_guard_context(command.transfer_guard):
        x, params5, info = align(
            plan.geometry,
            plan.recon_grid,
            plan.detector,
            plan.projections,
            config=plan.cfg,
            resume_state=plan.resume_state
            if isinstance(plan.resume_state, AlignResumeState)
            else None,
            checkpoint_callback=single_checkpoint_callback
            if plan.checkpoint_path is not None
            else None,
        )
    if plan.checkpoint_path is not None:
        single_checkpoint_callback(
            _completed_single_resume_state(
                x=x,
                params5=params5,
                info=info,
            ),
            run_complete=True,
        )
    return AlignCliExecutionResult(x=x, params5=params5, info=info)
