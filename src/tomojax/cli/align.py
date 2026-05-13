"""Run geometry alignment workflows from the public TomoJAX CLI."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, cast

from tomojax.cli._jax_allocator import configure_jax_allocator_defaults

configure_jax_allocator_defaults()

import jax.numpy as jnp
import numpy as np

from tomojax.align.api import (
    AlignConfig,
    AlignMultiresResumeState,
    AlignResumeState,
    align,
    normalize_alignment_profile,
    profile_policy_from_config,
    resolve_profiled_cli_defaults,
)
from tomojax.align.io.checkpoint import (
    AlignmentCheckpointGeometrySnapshot,
    AlignmentCheckpointMetadataInput,
    AlignmentCheckpointProgress,
    AlignmentProjectionIdentity,
    CheckpointError,
    CheckpointMetadata,
    build_alignment_checkpoint_metadata_from_input,
    load_alignment_checkpoint,
    save_alignment_checkpoint,
    validate_alignment_checkpoint,
)
from tomojax.align.io.params_export import save_alignment_params_csv, save_alignment_params_json
from tomojax.align.model.dofs import (
    DofBounds,
    normalize_alignment_dofs,
    normalize_bounds,
)
from tomojax.align.model.schedules import PUBLIC_SCHEDULE_PRESETS, resolve_alignment_schedule
from tomojax.align.objectives.loss_specs import (
    AlignmentLossConfig,
    parse_loss_schedule,
    parse_loss_spec,
    validate_loss_schedule_levels,
)
from tomojax.calibration.manifest import build_calibrated_geometry_metadata_patch
from tomojax.cli._runtime import transfer_guard_context
from tomojax.cli.config import parse_args_with_config
from tomojax.cli.manifest import build_manifest, save_manifest
from tomojax.core import log_jax_env, setup_logging
from tomojax.core.geometry import Detector, Grid  # noqa: TC001
from tomojax.geometry import (
    DISK_VOLUME_AXES,
    compute_roi,
    cylindrical_mask_xy,
    grid_from_detector_fov,
    grid_from_detector_fov_slices,
)
from tomojax.io import (
    build_geometry_from_dataset_metadata,
    load_projection_payload,
    save_projection_payload,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.align.api import FallbackPolicy, QualityTier
    from tomojax.align.model.gauge import GaugeFixMode
    from tomojax.align.model.schedules import GaugePolicy
    from tomojax.recon.types import Regulariser


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive float") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive float")
    return parsed


def _init_jax_compilation_cache() -> None:
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


def _metadata_int(value: object, default: int = 0) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    return default


def _metadata_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    return default


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


def _parse_dof_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    try:
        optimise_dofs = (
            None
            if args.optimise_dofs is None
            else normalize_alignment_dofs(args.optimise_dofs, option_name="--optimise-dofs")
        )
        freeze_dofs = normalize_alignment_dofs(args.freeze_dofs, option_name="--freeze-dofs")
    except ValueError as exc:
        parser.error(str(exc))
    return optimise_dofs, freeze_dofs


def _parse_bounds_arg(value: object) -> DofBounds:
    try:
        return normalize_bounds(value, option_name="--bounds")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_loss_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[AlignmentLossConfig, dict[str, float]]:
    loss_params: dict[str, float] = {}
    for kv in args.loss_param:
        if "=" not in kv:
            parser.error(f"--loss-param must be k=v, got: {kv}")
        k, v = kv.split("=", 1)
        try:
            loss_params[k.strip()] = float(v)
        except ValueError:
            parser.error(f"--loss-param value must be numeric: {kv}")

    try:
        loss_spec = parse_loss_spec(str(args.loss), loss_params if loss_params else None)
        if args.loss_schedule is None:
            return loss_spec, loss_params
        return parse_loss_schedule(args.loss_schedule, default=loss_spec), loss_params
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    raise AssertionError("unreachable")


def _build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    p = argparse.ArgumentParser(description="Joint reconstruction + alignment on dataset (.nxs)")
    p.add_argument("--config", help="Load command defaults from a TOML config file")
    p.add_argument("--data", help="Input .nxs")
    p.add_argument(
        "--align-profile",
        choices=["lightning", "tortoise"],
        default="lightning",
        help=(
            "High-level alignment posture: lightning is the default aggressive path; "
            "tortoise favors JAX/FP32/reference-oriented behavior"
        ),
    )
    p.add_argument("--outer-iters", type=int, default=5)
    p.add_argument("--recon-iters", type=int, default=10)
    p.add_argument(
        "--recon-algo",
        choices=["fista", "spdhg"],
        default="fista",
        help="Inner reconstruction solver used during alignment (default: fista)",
    )
    p.add_argument("--lambda-tv", type=float, default=0.005)
    p.add_argument(
        "--regulariser",
        choices=["tv", "huber_tv"],
        default="tv",
        help="Regulariser for inner reconstruction: tv (default) or huber_tv",
    )
    p.add_argument(
        "--huber-delta",
        type=_positive_float,
        default=1e-2,
        help="Huber-TV transition radius for --regulariser huber_tv",
    )
    p.add_argument(
        "--tv-prox-iters",
        type=int,
        default=10,
        help="Inner iterations for the FISTA TV proximal operator",
    )
    p.add_argument(
        "--views-per-batch",
        type=int,
        default=1,
        help="Projection views per inner reconstruction batch/subset (default: 1)",
    )
    p.add_argument(
        "--projector-unroll",
        type=int,
        default=1,
        help="Projector loop unroll factor for differentiable alignment paths (default: 1)",
    )
    p.add_argument(
        "--projector-backend",
        choices=["jax", "pallas"],
        default="jax",
        help=(
            "Alignment projector backend: jax is the default gradient-safe reference; "
            "pallas requests supported accelerator paths with JAX fallback metadata"
        ),
    )
    p.add_argument(
        "--spdhg-seed",
        type=int,
        default=0,
        help="Base random seed for SPDHG subset order inside alignment",
    )
    rp = p.add_mutually_exclusive_group()
    rp.add_argument(
        "--recon-positivity",
        dest="recon_positivity",
        action="store_true",
        help="Enable positivity projection for SPDHG inner reconstructions (default)",
    )
    rp.add_argument(
        "--no-recon-positivity",
        dest="recon_positivity",
        action="store_false",
        help="Disable positivity projection for SPDHG inner reconstructions",
    )
    p.set_defaults(recon_positivity=True)
    p.add_argument("--lr-rot", type=float, default=1e-3)
    p.add_argument("--lr-trans", type=float, default=1e-1)
    p.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="Optional multires factors, e.g., 4 2 1",
    )
    p.add_argument(
        "--gather-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Projector gather dtype (auto: bf16 on GPU/TPU, else fp32)",
    )
    ck = p.add_mutually_exclusive_group()
    ck.add_argument("--checkpoint-projector", dest="checkpoint_projector", action="store_true")
    ck.add_argument("--no-checkpoint-projector", dest="checkpoint_projector", action="store_false")
    p.set_defaults(checkpoint_projector=True)
    p.add_argument(
        "--opt-method",
        choices=["gd", "gn", "lbfgs"],
        default="gn",
        help=(
            "Alignment optimizer: gd, gn, or lbfgs. GN is supported for L2-like "
            "losses: l2, l2_otsu, edge_l2, pwls."
        ),
    )
    p.add_argument(
        "--gn-damping",
        type=float,
        default=1e-3,
        help="Levenberg-Marquardt damping for GN",
    )
    p.add_argument(
        "--lbfgs-maxiter",
        type=int,
        default=20,
        help="Maximum Optax L-BFGS iterations per alignment outer step",
    )
    p.add_argument(
        "--lbfgs-ftol",
        type=float,
        default=1e-6,
        help="Relative function tolerance for Optax L-BFGS",
    )
    p.add_argument(
        "--lbfgs-gtol",
        type=float,
        default=1e-5,
        help="Gradient-norm tolerance for Optax L-BFGS",
    )
    p.add_argument(
        "--lbfgs-maxls",
        type=int,
        default=20,
        help="Maximum Optax L-BFGS line-search steps per iteration",
    )
    p.add_argument(
        "--lbfgs-memory-size",
        type=int,
        default=10,
        help="Number of previous gradient/step pairs stored by Optax L-BFGS",
    )
    p.add_argument(
        "--optimise-dofs",
        nargs="+",
        default=None,
        metavar="DOF[,DOF]",
        help=(
            "Named alignment DOFs to optimise across pose and geometry: "
            "alpha,beta,phi,dx,dz,det_u_px,det_v_px,detector_roll_deg,"
            "axis_rot_x_deg,axis_rot_y_deg. Example: dx,dz or det_u_px"
        ),
    )
    p.add_argument(
        "--freeze-dofs",
        nargs="+",
        default=None,
        metavar="DOF[,DOF]",
        help="Named alignment DOFs to keep fixed at initial values. Example: phi or det_u_px",
    )
    p.add_argument(
        "--schedule",
        choices=list(PUBLIC_SCHEDULE_PRESETS),
        default=None,
        help=(
            "Executable alignment preset. Setup presets use validation-LM stages; "
            "explicit --optimise-dofs is the lower-level direct surface."
        ),
    )
    p.add_argument(
        "--bounds",
        type=_parse_bounds_arg,
        default=None,
        metavar="DOF=LOWER:UPPER[,DOF=LOWER:UPPER]",
        help=(
            "Finite per-DOF parameter bounds. Pose rotations use radians, translations "
            "use world units, setup *_deg DOFs use degrees, and det_*_px uses native "
            "detector pixels. Example: det_u_px=-8:8,detector_roll_deg=-5:5"
        ),
    )
    p.add_argument(
        "--gauge-policy",
        choices=["reject", "anchor_mean", "prior_required", "diagnose_only"],
        default="reject",
        help=(
            "Policy for gauge-coupled direct/expert DOF sets. Public presets carry "
            "their own stage policies; direct mixed setup+pose defaults to reject."
        ),
    )
    p.add_argument(
        "--pose-model",
        choices=["per_view", "polynomial", "spline"],
        default="per_view",
        help=(
            "Alignment pose parameterization: per_view optimizes one 5-DOF vector per "
            "view; polynomial and spline optimize smooth low-dimensional trajectories"
        ),
    )
    p.add_argument(
        "--knot-spacing",
        type=int,
        default=8,
        help="View spacing between spline knots when --pose-model spline is used",
    )
    p.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial degree or spline degree for smooth pose models",
    )
    p.add_argument(
        "--gauge-fix",
        choices=["mean_translation", "none"],
        default="mean_translation",
        help=(
            "Gauge fixing for alignment parameters: mean_translation subtracts the "
            "scan-wide mean from active dx,dz after updates (default); none preserves "
            "historical unconstrained traces"
        ),
    )
    p.add_argument("--w-rot", type=float, default=1e-3, help="Smoothness weight for rotations")
    p.add_argument("--w-trans", type=float, default=1e-3, help="Smoothness weight for translations")
    p.add_argument(
        "--seed-translations",
        action="store_true",
        help="Phase-correlation init for dx,dz at coarsest level",
    )
    p.add_argument(
        "--log-summary",
        action="store_true",
        help="Print per-outer summaries (FISTA loss, alignment loss before/after)",
    )
    p.add_argument(
        "--log-compact",
        dest="log_compact",
        action="store_true",
        default=True,
        help="Use compact one-line per-outer summary when --log-summary is set (default: on)",
    )
    p.add_argument("--no-log-compact", dest="log_compact", action="store_false")
    p.add_argument(
        "--recon-L",
        type=float,
        default=None,
        help="Fixed Lipschitz constant for FISTA inside alignment (skip power-method)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Write resumable alignment checkpoints to PATH after completed outer iterations.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        metavar="N",
        help="Checkpoint every N completed global outer iterations (default: 1 when enabled).",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Resume alignment from a checkpoint. Defaults future checkpoint writes to this path.",
    )
    # Data term / similarity
    p.add_argument(
        "--loss",
        choices=[
            "l2",
            "charbonnier",
            "huber",
            "cauchy",
            "lorentzian",
            "welsch",
            "leclerc",
            "barron",
            "student_t",
            "correntropy",
            "zncc",
            "ssim",
            "ms-ssim",
            "mi",
            "nmi",
            "renyi_mi",
            "grad_l1",
            "edge_l2",
            "ngf",
            "grad_orient",
            "phasecorr",
            "fft_mag",
            "chamfer_edge",
            "l2_otsu",
            "ssim_otsu",
            "tversky",
            "swd",
            "mind",
            "pwls",
            "poisson",
        ],
        default="l2_otsu",
        help="Data term / similarity to optimize (default: l2_otsu)",
    )
    p.add_argument(
        "--loss-schedule",
        default=None,
        help=(
            "Pyramid-level loss schedule as LEVEL:LOSS entries, e.g. "
            "4:phasecorr,2:ssim,1:l2_otsu. Unspecified levels use --loss."
        ),
    )
    p.add_argument(
        "--loss-param",
        action="append",
        default=[],
        help="Loss parameter as k=v (repeatable), e.g., delta=1.0, eps=1e-3, window=7, temp=0.5",
    )
    # Early stopping controls (alignment phase)
    es = p.add_mutually_exclusive_group()
    es.add_argument(
        "--early-stop",
        dest="early_stop",
        action="store_true",
        help="Enable early stopping across outers (default)",
    )
    es.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="Disable early stopping across outers",
    )
    p.set_defaults(early_stop=True)
    p.add_argument(
        "--early-stop-rel",
        type=float,
        default=None,
        help="Relative improvement threshold for early stop (default 1e-3)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Consecutive outers below threshold before stopping (default 2)",
    )
    p.add_argument(
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help=(
            "JAX transfer guard mode during compute "
            "(default: off; use log/disallow for diagnostics)"
        ),
    )
    p.add_argument("--out", help="Output .nxs with recon and alignment params")
    p.add_argument(
        "--save-params-json",
        default=None,
        help="Optional JSON sidecar for final per-view alignment parameters",
    )
    p.add_argument(
        "--save-params-csv",
        default=None,
        help="Optional CSV sidecar for final per-view alignment parameters",
    )
    p.add_argument(
        "--save-manifest",
        metavar="PATH",
        default=None,
        help="Write a JSON reproducibility manifest for this alignment run.",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars if tqdm is available",
    )
    p.add_argument(
        "--roi",
        choices=["auto", "off", "cube", "bbox", "cyl"],
        default="auto",
        help=(
            "Region to reconstruct: auto: square x-y slices + z from detector height; "
            "off: use full grid; cube: same as auto; bbox: rectangular FOV bbox; "
            "cyl: auto + zero outside cylindrical FOV"
        ),
    )
    p.add_argument(
        "--mask-vol",
        choices=["off", "cyl"],
        default="off",
        help=(
            "Mask the volume before forward projection in alignment: "
            "off (default), or cyl for cylindrical x-y mask broadcast along z."
        ),
    )
    p.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Override reconstruction grid size (nx ny nz). Voxel sizes stay as in input metadata.",
    )
    p.add_argument(
        "--volume-axes",
        choices=["zyx", "xyz"],
        default=DISK_VOLUME_AXES,
        help="On-disk axis order for saved volumes (default: zyx for viewer compatibility).",
    )
    return p


def _checkpoint_cli_options(args: argparse.Namespace, *, gather_dtype: str) -> dict[str, object]:
    return {
        "align_profile": str(args.align_profile),
        "roi": str(args.roi),
        "grid": args.grid,
        "requested_gather_dtype": str(args.gather_dtype),
        "gather_dtype": str(gather_dtype),
        "recon_algo": str(args.recon_algo),
        "views_per_batch": int(args.views_per_batch),
        "spdhg_seed": int(args.spdhg_seed),
        "recon_positivity": bool(args.recon_positivity),
        "projector_unroll": int(args.projector_unroll),
        "projector_backend": str(args.projector_backend),
        "quality_tier": str(getattr(args, "quality_tier", "")),
        "fallback_policy": str(getattr(args, "fallback_policy", "")),
        "checkpoint_projector": bool(args.checkpoint_projector),
        "mask_vol": str(args.mask_vol),
        "gauge_fix": str(args.gauge_fix),
        "gauge_policy": str(args.gauge_policy),
        "optimise_dofs": list(args.optimise_dofs or []),
        "freeze_dofs": list(args.freeze_dofs or []),
        "schedule": args.schedule,
    }


def _checkpoint_metadata(
    *,
    meta: object,
    projections: jnp.ndarray,
    cfg: AlignConfig,
    args: argparse.Namespace,
    recon_grid: Grid,
    detector: Detector,
    state_grid: Grid,
    state_detector: Detector,
    gather_dtype: str,
    levels: list[int] | None,
    level_index: int,
    level_factor: int,
    completed_outer_iters_in_level: int,
    global_outer_iters_completed: int,
    prev_factor: int | None,
    L_prev: float | None,
    small_impr_streak: int,
    elapsed_offset: float,
    level_complete: bool,
    run_complete: bool,
    schedule_metadata: dict[str, object] | None = None,
    geometry_calibration_state: dict[str, object] | None = None,
    schedule_state: dict[str, object] | None = None,
) -> CheckpointMetadata:
    geometry_meta = getattr(getattr(meta, "metadata", meta), "geometry_meta", None)
    geometry_type = getattr(meta, "geometry_type", "parallel")
    return build_alignment_checkpoint_metadata_from_input(
        AlignmentCheckpointMetadataInput(
            projection=AlignmentProjectionIdentity(
                shape=tuple(int(v) for v in projections.shape),
                dtype=str(projections.dtype),
            ),
            geometry=AlignmentCheckpointGeometrySnapshot(
                geometry_type=str(geometry_type),
                geometry_meta=geometry_meta,
                reconstruction_grid=recon_grid.to_dict(),
                detector=detector.to_dict(),
                state_grid=state_grid.to_dict(),
                state_detector=state_detector.to_dict(),
                geometry_calibration_state=geometry_calibration_state,
            ),
            progress=AlignmentCheckpointProgress(
                levels=levels,
                level_index=int(level_index),
                level_factor=int(level_factor),
                completed_outer_iters_in_level=int(completed_outer_iters_in_level),
                global_outer_iters_completed=int(global_outer_iters_completed),
                prev_factor=prev_factor,
                current_inner_iteration=0,
                L_prev=L_prev,
                small_impr_streak=small_impr_streak,
                elapsed_offset=elapsed_offset,
                level_complete=level_complete,
                run_complete=run_complete,
            ),
            config=cfg,
            cli_options=_checkpoint_cli_options(args, gather_dtype=gather_dtype),
            random_state={
                "alignment": None,
                "seed_translations": (
                    "deterministic_phase_correlation" if cfg.seed_translations else None
                ),
            },
            schedule_metadata=schedule_metadata,
            schedule_state=schedule_state,
        )
    )


def _resume_state_from_checkpoint(
    checkpoint_path: str,
    *,
    expected_metadata: CheckpointMetadata,
    used_multires: bool,
) -> AlignResumeState | AlignMultiresResumeState:
    checkpoint = load_alignment_checkpoint(checkpoint_path)
    validate_alignment_checkpoint(checkpoint, expected_metadata)
    metadata = checkpoint.metadata
    if used_multires:
        schedule_state = metadata.get("schedule_state")
        if not isinstance(schedule_state, dict):
            schedule_state = {}
        schedule_state = cast("dict[str, object]", schedule_state)
        prev_factor_value = metadata.get("prev_factor")
        geometry_calibration_state = metadata.get("geometry_calibration_state")
        return AlignMultiresResumeState(
            x=jnp.asarray(checkpoint.x, dtype=jnp.float32),
            params5=jnp.asarray(checkpoint.params5, dtype=jnp.float32),
            motion_coeffs=(
                None
                if checkpoint.motion_coeffs is None
                else jnp.asarray(checkpoint.motion_coeffs, dtype=jnp.float32)
            ),
            level_index=int(metadata.get("level_index", 0)),
            level_factor=int(metadata.get("level_factor", 1)),
            completed_outer_iters_in_level=int(metadata.get("completed_outer_iters_in_level", 0)),
            global_outer_iters_completed=int(metadata.get("global_outer_iters_completed", 0)),
            prev_factor=None if prev_factor_value is None else int(prev_factor_value),
            loss=list(checkpoint.loss_history),
            outer_stats=[dict(stat) for stat in checkpoint.outer_stats],
            L=metadata.get("L_prev"),
            small_impr_streak=int(metadata.get("small_impr_streak", 0)),
            elapsed_offset=float(metadata.get("elapsed_offset", 0.0)),
            level_complete=bool(metadata.get("level_complete", False)),
            run_complete=bool(metadata.get("run_complete", False)),
            geometry_calibration_state=(
                dict(geometry_calibration_state)
                if isinstance(geometry_calibration_state, dict)
                else None
            ),
            stage_index=_metadata_int(schedule_state.get("stage_index"), 0),
            stage_name=(
                str(schedule_state["stage_name"])
                if schedule_state.get("stage_name") is not None
                else None
            ),
            stage_completed=bool(schedule_state.get("stage_completed", False)),
            completed_outer_iters_in_stage=_metadata_int(
                schedule_state.get("completed_outer_iters_in_stage"), 0
            ),
        )
    return AlignResumeState(
        x=jnp.asarray(checkpoint.x, dtype=jnp.float32),
        params5=jnp.asarray(checkpoint.params5, dtype=jnp.float32),
        motion_coeffs=(
            None
            if checkpoint.motion_coeffs is None
            else jnp.asarray(checkpoint.motion_coeffs, dtype=jnp.float32)
        ),
        start_outer_iter=int(metadata.get("completed_outer_iters_in_level", 0)),
        loss=list(checkpoint.loss_history),
        outer_stats=[dict(stat) for stat in checkpoint.outer_stats],
        L=metadata.get("L_prev"),
        small_impr_streak=int(metadata.get("small_impr_streak", 0)),
        elapsed_offset=float(metadata.get("elapsed_offset", 0.0)),
    )


@dataclass(frozen=True, slots=True)
class AlignCliRunPlan:
    """Resolved inputs needed to execute an alignment CLI run."""

    args: argparse.Namespace
    config_metadata: dict[str, Any]
    loss_config: AlignmentLossConfig
    loss_params: dict[str, float]
    levels: list[int] | None
    run_levels: list[int] | None
    meta: Any
    geometry_meta: dict[str, Any]
    grid: Grid
    recon_grid: Grid
    detector: Detector
    geometry: Any
    projections: jnp.ndarray
    cfg: AlignConfig
    gather_dtype: str
    geometry_dofs: tuple[str, ...]
    schedule_metadata: dict[str, object] | None
    checkpoint_path: str | None
    checkpoint_every: int | None
    resume_state: AlignResumeState | AlignMultiresResumeState | None
    apply_cyl_mask: bool


@dataclass(frozen=True, slots=True)
class AlignCliExecutionResult:
    """Alignment result returned by the CLI execution helper."""

    x: jnp.ndarray
    params5: jnp.ndarray
    info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AlignCliCheckpointCallbacks:
    """Checkpoint callbacks for single-level and multires alignment runs."""

    single: Callable[..., None]
    multires: Callable[[AlignMultiresResumeState], None]


def _build_align_cli_run_plan(  # noqa: PLR0912, PLR0915
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_metadata: dict[str, Any],
) -> AlignCliRunPlan:
    loss_config, loss_params = _parse_loss_config(args, parser)
    optimise_dofs, freeze_dofs = _parse_dof_args(args, parser)
    levels = (
        [int(v) for v in args.levels] if args.levels is not None and len(args.levels) > 0 else None
    )
    try:
        validate_loss_schedule_levels(loss_config, levels if levels is not None else [1])
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.align_profile = normalize_alignment_profile(args.align_profile)
    except ValueError as exc:
        parser.error(str(exc))
    configured_keys = set(config_metadata.get("explicit_cli_keys", [])) | set(
        config_metadata.get("config_file_values", {}).keys()
    )
    profile_options = resolve_profiled_cli_defaults(
        align_profile=args.align_profile,
        current={
            "projector_backend": args.projector_backend,
            "gather_dtype": args.gather_dtype,
            "regulariser": args.regulariser,
            "recon_algo": args.recon_algo,
            "views_per_batch": args.views_per_batch,
            "checkpoint_projector": args.checkpoint_projector,
            "pose_model": args.pose_model,
        },
        configured_keys=configured_keys,
    )
    args.projector_backend = str(profile_options["projector_backend"])
    args.gather_dtype = str(profile_options["gather_dtype"])
    args.regulariser = str(profile_options["regulariser"])
    args.recon_algo = str(profile_options["recon_algo"])
    args.views_per_batch = int(cast("Any", profile_options["views_per_batch"]))
    args.checkpoint_projector = bool(profile_options["checkpoint_projector"])
    args.pose_model = str(profile_options["pose_model"])
    args.quality_tier = str(profile_options["quality_tier"])
    args.fallback_policy = str(profile_options["fallback_policy"])
    if "schedule" not in configured_keys and optimise_dofs is None:
        args.schedule = "lightning_pose" if args.align_profile == "lightning" else "tortoise_pose"
    config_metadata["profile_options"] = dict(profile_options)
    effective_options = config_metadata.get("effective_options")
    if isinstance(effective_options, dict):
        for key in (
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

    meta = load_projection_payload(args.data)
    geometry_meta = meta.geometry_inputs()
    initial_grid_override = args.grid if (meta.grid is None and args.grid is not None) else None
    grid, detector, geom = build_geometry_from_dataset_metadata(
        geometry_meta,
        grid_override=initial_grid_override,
        apply_saved_alignment=False,
    )
    projections = jnp.asarray(meta.projections, dtype=jnp.float32)
    try:
        resolved_schedule = resolve_alignment_schedule(
            schedule=args.schedule,
            optimise_dofs=optimise_dofs,
            freeze_dofs=freeze_dofs,
            geometry_dofs=(),
            geometry=geom,
            gauge_policy=cast("GaugePolicy", str(args.gauge_policy)),
            opt_method=str(args.opt_method),
            outer_iters=int(args.outer_iters),
            early_stop=bool(args.early_stop),
        )
        geometry_dofs = resolved_schedule.active_geometry_dofs
        schedule_metadata: dict[str, object] | None = resolved_schedule.to_dict()
    except ValueError as exc:
        parser.error(str(exc))

    from tomojax.backends import default_gather_dtype as _default_gather_dtype

    gather_dtype = str(args.gather_dtype)
    if gather_dtype == "auto":
        gather_dtype = _default_gather_dtype()

    cfg = AlignConfig(
        align_profile=str(args.align_profile),
        outer_iters=args.outer_iters,
        recon_iters=args.recon_iters,
        recon_algo=cast("Any", str(args.recon_algo)),
        lambda_tv=args.lambda_tv,
        regulariser=cast("Regulariser", str(args.regulariser)),
        huber_delta=float(args.huber_delta),
        tv_prox_iters=int(args.tv_prox_iters),
        recon_positivity=bool(args.recon_positivity),
        spdhg_seed=int(args.spdhg_seed),
        lr_rot=args.lr_rot,
        lr_trans=args.lr_trans,
        views_per_batch=int(args.views_per_batch),
        projector_unroll=int(args.projector_unroll),
        projector_backend=str(args.projector_backend),
        quality_tier=cast("QualityTier", str(args.quality_tier)),
        fallback_policy=cast("FallbackPolicy", str(args.fallback_policy)),
        checkpoint_projector=bool(args.checkpoint_projector),
        gather_dtype=gather_dtype,
        opt_method=str(args.opt_method),
        gn_damping=float(args.gn_damping),
        lbfgs_maxiter=int(args.lbfgs_maxiter),
        lbfgs_ftol=float(args.lbfgs_ftol),
        lbfgs_gtol=float(args.lbfgs_gtol),
        lbfgs_maxls=int(args.lbfgs_maxls),
        lbfgs_memory_size=int(args.lbfgs_memory_size),
        w_rot=float(args.w_rot),
        w_trans=float(args.w_trans),
        schedule=args.schedule,
        optimise_dofs=optimise_dofs,
        freeze_dofs=freeze_dofs,
        geometry_dofs=(),
        bounds=args.bounds,
        gauge_policy=cast("Any", str(args.gauge_policy)),
        pose_model=cast("Any", str(args.pose_model)),
        knot_spacing=int(args.knot_spacing),
        degree=int(args.degree),
        gauge_fix=cast("GaugeFixMode", str(args.gauge_fix)),
        loss=loss_config,
        seed_translations=bool(args.seed_translations),
        log_summary=bool(args.log_summary),
        log_compact=bool(args.log_compact),
        recon_L=(float(args.recon_L) if args.recon_L is not None else None),
        early_stop=bool(args.early_stop),
        early_stop_rel_impr=(
            float(args.early_stop_rel) if args.early_stop_rel is not None else 1e-3
        ),
        early_stop_patience=(
            int(args.early_stop_patience) if args.early_stop_patience is not None else 2
        ),
        mask_vol=str(args.mask_vol),
    )
    schedule_metadata["profile_policy"] = profile_policy_from_config(cfg).to_dict()
    recon_grid, apply_cyl_mask = _resolve_recon_grid_and_mask(
        grid,
        detector,
        is_parallel=meta.geometry_type == "parallel",
        roi_mode=str(args.roi).lower(),
        grid_override=args.grid,
    )
    if recon_grid is not grid:
        _, _, geom = build_geometry_from_dataset_metadata(
            geometry_meta,
            grid_override=recon_grid,
            apply_saved_alignment=False,
        )

    checkpoint_path = args.checkpoint or args.resume
    checkpoint_every = args.checkpoint_every
    if checkpoint_path is not None and checkpoint_every is None:
        checkpoint_every = 1
    if checkpoint_every is not None and int(checkpoint_every) < 1:
        parser.error("--checkpoint-every must be an integer >= 1")

    run_levels = levels
    if run_levels is None and (args.schedule is not None or bool(geometry_dofs)):
        run_levels = [1]

    expected_checkpoint_metadata = _checkpoint_metadata(
        meta=meta,
        projections=projections,
        cfg=cfg,
        args=args,
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
    if args.resume is not None:
        try:
            resume_state = _resume_state_from_checkpoint(
                args.resume,
                expected_metadata=expected_checkpoint_metadata,
                used_multires=run_levels is not None,
            )
        except CheckpointError as exc:
            raise SystemExit(f"tomojax align: {exc}") from exc
        logging.info("Resuming alignment from checkpoint %s", args.resume)

    return AlignCliRunPlan(
        args=args,
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


def _state_grid_detector_for_checkpoint(
    plan: AlignCliRunPlan,
    level_factor: int,
    *,
    run_complete: bool,
) -> tuple[Grid, Detector]:
    if plan.run_levels is None or run_complete:
        return plan.recon_grid, plan.detector
    from tomojax.core.multires import scale_detector, scale_grid

    return scale_grid(plan.recon_grid, int(level_factor)), scale_detector(
        plan.detector,
        int(level_factor),
    )


def _make_align_cli_checkpoint_callbacks(plan: AlignCliRunPlan) -> AlignCliCheckpointCallbacks:
    def write_single_checkpoint(
        state: AlignResumeState,
        *,
        run_complete: bool = False,
    ) -> None:
        if plan.checkpoint_path is None:
            return
        completed = int(state.start_outer_iter)
        every = int(plan.checkpoint_every or 1)
        if not run_complete and (completed <= 0 or completed % every != 0):
            return
        metadata = _checkpoint_metadata(
            meta=plan.meta,
            projections=plan.projections,
            cfg=plan.cfg,
            args=plan.args,
            recon_grid=plan.recon_grid,
            detector=plan.detector,
            state_grid=plan.recon_grid,
            state_detector=plan.detector,
            gather_dtype=plan.gather_dtype,
            levels=None,
            level_index=0,
            level_factor=1,
            completed_outer_iters_in_level=completed,
            global_outer_iters_completed=completed,
            prev_factor=None,
            L_prev=state.L,
            small_impr_streak=int(state.small_impr_streak),
            elapsed_offset=float(state.elapsed_offset),
            level_complete=run_complete or completed >= int(plan.cfg.outer_iters),
            run_complete=run_complete,
            schedule_metadata=plan.schedule_metadata,
        )
        save_alignment_checkpoint(
            plan.checkpoint_path,
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            loss_history=state.loss,
            outer_stats=state.outer_stats,
            metadata=metadata,
        )
        logging.info("Saved alignment checkpoint to %s", plan.checkpoint_path)

    def write_multires_checkpoint(state: AlignMultiresResumeState) -> None:
        if plan.checkpoint_path is None:
            return
        completed = int(state.global_outer_iters_completed)
        every = int(plan.checkpoint_every or 1)
        if (
            not state.run_complete
            and not state.level_complete
            and (completed <= 0 or completed % every != 0)
        ):
            return
        state_grid, state_detector = _state_grid_detector_for_checkpoint(
            plan,
            int(state.level_factor),
            run_complete=bool(state.run_complete),
        )
        metadata = _checkpoint_metadata(
            meta=plan.meta,
            projections=plan.projections,
            cfg=plan.cfg,
            args=plan.args,
            recon_grid=plan.recon_grid,
            detector=plan.detector,
            state_grid=state_grid,
            state_detector=state_detector,
            gather_dtype=plan.gather_dtype,
            levels=plan.run_levels,
            level_index=int(state.level_index),
            level_factor=int(state.level_factor),
            completed_outer_iters_in_level=int(state.completed_outer_iters_in_level),
            global_outer_iters_completed=completed,
            prev_factor=state.prev_factor,
            L_prev=state.L,
            small_impr_streak=int(state.small_impr_streak),
            elapsed_offset=float(state.elapsed_offset),
            level_complete=bool(state.level_complete),
            run_complete=bool(state.run_complete),
            schedule_metadata=plan.schedule_metadata,
            schedule_state={
                "stage_index": int(state.stage_index),
                "stage_name": state.stage_name,
                "stage_completed": bool(state.stage_completed),
                "completed_outer_iters_in_stage": int(state.completed_outer_iters_in_stage),
            },
            geometry_calibration_state=state.geometry_calibration_state,
        )
        save_alignment_checkpoint(
            plan.checkpoint_path,
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            loss_history=state.loss,
            outer_stats=state.outer_stats,
            metadata=metadata,
        )
        logging.info("Saved alignment checkpoint to %s", plan.checkpoint_path)

    return AlignCliCheckpointCallbacks(
        single=write_single_checkpoint, multires=write_multires_checkpoint
    )


def _execute_alignment_plan(
    plan: AlignCliRunPlan,
    *,
    single_checkpoint_callback: Callable[..., None],
    multires_checkpoint_callback: Callable[[AlignMultiresResumeState], None],
) -> AlignCliExecutionResult:
    args = plan.args
    if plan.run_levels is not None and len(plan.run_levels) > 0:
        from tomojax.align.pipeline import align_multires

        with transfer_guard_context(args.transfer_guard):
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

    with transfer_guard_context(args.transfer_guard):
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
            jnp.asarray(motion_coeffs, dtype=jnp.float32)
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
                start_outer_iter=_metadata_int(completed_outer_iters, len(outer_stats_raw)),
                loss=list(loss_raw),
                outer_stats=[dict(stat) for stat in outer_stats_raw if isinstance(stat, dict)],
                L=float(l_value) if isinstance(l_value, int | float) else None,
                small_impr_streak=_metadata_int(small_impr_streak, 0),
                elapsed_offset=_metadata_float(wall_time_total, 0.0),
            ),
            run_complete=True,
        )
    return AlignCliExecutionResult(x=x, params5=params5, info=info_dict)


def _apply_alignment_output_mask(plan: AlignCliRunPlan, x: jnp.ndarray) -> jnp.ndarray:
    if not plan.apply_cyl_mask:
        return x
    try:
        m_xy = cylindrical_mask_xy(plan.recon_grid, plan.detector)
        m = jnp.asarray(m_xy, dtype=x.dtype)[:, :, None]
        return x * m
    except Exception:
        m_xy = cylindrical_mask_xy(plan.recon_grid, plan.detector)
        m = np.asarray(m_xy, dtype=np.float32)[:, :, None]
        return jnp.asarray(np.asarray(x) * m)


def _alignment_gauge_metadata(
    plan: AlignCliRunPlan,
    info: dict[str, Any],
) -> dict[str, object]:
    return {
        "mode": str(info.get("gauge_fix", plan.args.gauge_fix)),
        "dofs": list(info.get("gauge_fix_dofs", [])),
        "final": dict(info.get("gauge_fix_final", {}) or {}),
    }


def _write_alignment_result_volume(
    plan: AlignCliRunPlan,
    *,
    x: jnp.ndarray,
    params5_np: np.ndarray,
    gauge_metadata: dict[str, object],
    geometry_calibration_state: object,
) -> Any:
    save_meta = plan.meta.copy_metadata()
    save_meta.grid = plan.recon_grid.to_dict()
    save_meta.volume = np.asarray(x)
    save_meta.align_params = params5_np
    save_meta.align_gauge = gauge_metadata
    if isinstance(geometry_calibration_state, dict):
        calibration_patch = build_calibrated_geometry_metadata_patch(
            calibration_state=geometry_calibration_state,
            detector=plan.detector.to_dict(),
            geometry_meta=save_meta.geometry_meta or {},
        )
        save_meta.detector = calibration_patch["detector"]  # type: ignore[assignment]
        save_meta.geometry_meta = calibration_patch["geometry_meta"]  # type: ignore[assignment]
        save_meta.geometry_calibration = calibration_patch["geometry_calibration"]  # type: ignore[assignment]
    save_meta.frame = str(plan.meta.frame or "sample")
    save_meta.volume_axes_order = str(plan.args.volume_axes)
    save_projection_payload(
        plan.args.out,
        projections=plan.meta.projections,
        metadata=save_meta,
    )
    logging.info("Saved alignment results to %s", plan.args.out)
    return save_meta


def _write_alignment_params_exports(
    plan: AlignCliRunPlan,
    *,
    params5_np: np.ndarray,
    gauge_metadata: dict[str, object],
) -> None:
    if plan.args.save_params_json is not None:
        save_alignment_params_json(
            plan.args.save_params_json,
            params5_np,
            du=float(plan.detector.du),
            dv=float(plan.detector.dv),
            gauge_metadata=gauge_metadata,
        )
        logging.info("Saved alignment parameter JSON to %s", plan.args.save_params_json)
    if plan.args.save_params_csv is not None:
        save_alignment_params_csv(
            plan.args.save_params_csv,
            params5_np,
            du=float(plan.detector.du),
            dv=float(plan.detector.dv),
        )
        logging.info("Saved alignment parameter CSV to %s", plan.args.save_params_csv)


def _build_alignment_manifest_payload_from_result(
    plan: AlignCliRunPlan,
    execution: AlignCliExecutionResult,
    *,
    x: jnp.ndarray,
    params5_np: np.ndarray,
    gauge_metadata: dict[str, object],
    geometry_calibration_state: object,
    save_meta: Any,
) -> dict[str, object]:
    args = plan.args
    info = execution.info
    loss_values = info.get("loss", [])
    return {
        "input_path": args.data,
        "output_path": args.out,
        "save_params_json": args.save_params_json,
        "save_params_csv": args.save_params_csv,
        "manifest_path": args.save_manifest,
        "config_path": plan.config_metadata["config_path"],
        "config_file_values": plan.config_metadata["config_file_values"],
        "explicit_cli_keys": plan.config_metadata["explicit_cli_keys"],
        "effective_options": plan.config_metadata["effective_options"],
        "geometry_type": str(plan.meta.geometry_type),
        "input_projection_shape": list(plan.meta.projections.shape),
        "reconstruction_grid": plan.recon_grid.to_dict(),
        "detector": plan.detector.to_dict(),
        "align_profile": str(args.align_profile),
        "profile_policy": profile_policy_from_config(plan.cfg).to_dict(),
        "roi": {
            "requested": str(args.roi),
            "is_parallel": bool(plan.meta.geometry_type == "parallel"),
            "grid_changed": plan.recon_grid != plan.grid,
            "cylindrical_output_mask": bool(plan.apply_cyl_mask),
        },
        "requested_gather_dtype": str(args.gather_dtype),
        "gather_dtype": plan.gather_dtype,
        "recon_algo": str(args.recon_algo),
        "regulariser": str(args.regulariser),
        "huber_delta": float(args.huber_delta),
        "views_per_batch": int(args.views_per_batch),
        "spdhg_seed": int(args.spdhg_seed),
        "recon_positivity": bool(args.recon_positivity),
        "projector_unroll": int(args.projector_unroll),
        "projector_backend": str(args.projector_backend),
        "quality_tier": str(getattr(args, "quality_tier", "")),
        "fallback_policy": str(getattr(args, "fallback_policy", "")),
        "checkpoint_projector": bool(args.checkpoint_projector),
        "transfer_guard": str(args.transfer_guard),
        "levels": plan.run_levels,
        "schedule": info.get("schedule", plan.schedule_metadata),
        "used_multires": bool(plan.run_levels is not None and len(plan.run_levels) > 0),
        "checkpoint_path": args.checkpoint,
        "checkpoint_every": args.checkpoint_every,
        "resume_path": args.resume,
        "loss_params": plan.loss_params,
        "loss_spec": plan.loss_config,
        "align_config": plan.cfg,
        "objective_kind": info.get("objective_kind"),
        "objective_kinds": list(info.get("objective_kinds", [])),
        "objective_provenance": info.get("objective_provenance"),
        "backend_provenance": info.get("backend_provenance"),
        "gauge_policy": str(args.gauge_policy),
        "gauge_decision": info.get("gauge_decision"),
        "active_dofs": list(info.get("active_dofs", [])),
        "active_pose_dofs": list(info.get("active_pose_dofs", [])),
        "active_geometry_dofs": list(info.get("active_geometry_dofs", [])),
        "geometry_dofs": list(plan.geometry_dofs),
        "geometry_calibration_state": geometry_calibration_state,
        "alignment_params_shape": list(params5_np.shape),
        "alignment_gauge": gauge_metadata,
        "volume_shape": list(np.asarray(x).shape),
        "volume_axes": str(args.volume_axes),
        "frame": str(save_meta.frame),
        "run_info": {
            "loss_count": len(loss_values),
            "final_loss": loss_values[-1] if len(loss_values) else None,
            "loss_kind": info.get("loss_kind"),
            "stopped_by_observer": info.get("stopped_by_observer"),
            "observer_action": info.get("observer_action"),
        },
    }


def _write_alignment_manifest(
    plan: AlignCliRunPlan,
    execution: AlignCliExecutionResult,
    *,
    x: jnp.ndarray,
    params5_np: np.ndarray,
    gauge_metadata: dict[str, object],
    geometry_calibration_state: object,
    save_meta: Any,
) -> None:
    if plan.args.save_manifest is None:
        return
    payload = _build_alignment_manifest_payload_from_result(
        plan,
        execution,
        x=x,
        params5_np=params5_np,
        gauge_metadata=gauge_metadata,
        geometry_calibration_state=geometry_calibration_state,
        save_meta=save_meta,
    )
    manifest = build_manifest("tomojax align", list(sys.argv), plan.args, payload)
    save_manifest(plan.args.save_manifest, manifest)
    logging.info("Saved reproducibility manifest to %s", plan.args.save_manifest)


def _write_alignment_outputs(
    plan: AlignCliRunPlan,
    execution: AlignCliExecutionResult,
) -> None:
    x = _apply_alignment_output_mask(plan, execution.x)
    params5_np = np.asarray(execution.params5)
    gauge_metadata = _alignment_gauge_metadata(plan, execution.info)
    geometry_calibration_state = execution.info.get("geometry_calibration_state")
    save_meta = _write_alignment_result_volume(
        plan,
        x=x,
        params5_np=params5_np,
        gauge_metadata=gauge_metadata,
        geometry_calibration_state=geometry_calibration_state,
    )
    _write_alignment_params_exports(
        plan,
        params5_np=params5_np,
        gauge_metadata=gauge_metadata,
    )
    _write_alignment_manifest(
        plan,
        execution,
        x=x,
        params5_np=params5_np,
        gauge_metadata=gauge_metadata,
        geometry_calibration_state=geometry_calibration_state,
        save_meta=save_meta,
    )


def main() -> None:
    """Run alignment from the public CLI."""
    p = _build_parser()
    args, config_metadata = parse_args_with_config(p, required=("data", "out"))

    setup_logging()
    log_jax_env()
    _init_jax_compilation_cache()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"
    plan = _build_align_cli_run_plan(p, args, config_metadata)
    checkpoint_callbacks = _make_align_cli_checkpoint_callbacks(plan)
    execution = _execute_alignment_plan(
        plan,
        single_checkpoint_callback=checkpoint_callbacks.single,
        multires_checkpoint_callback=checkpoint_callbacks.multires,
    )
    _write_alignment_outputs(plan, execution)


if __name__ == "__main__":  # pragma: no cover
    main()
