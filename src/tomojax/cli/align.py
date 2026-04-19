from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import numpy as np
import jax.numpy as jnp
import os
import sys

from ..data.geometry_meta import build_geometry_from_meta
from ..data.io_hdf5 import NXTomoMetadata, load_nxtomo, save_nxtomo
from ..align.dofs import DofBounds, active_dof_mask, normalize_bounds, normalize_dofs
from ..align.losses import (
    AlignmentLossConfig,
    parse_loss_schedule,
    parse_loss_spec,
    validate_loss_schedule_levels,
)
from ..align.params_export import save_alignment_params_csv, save_alignment_params_json
from ..core.geometry import Grid, Detector
from ..align.checkpoint import (
    CheckpointError,
    build_alignment_checkpoint_metadata,
    load_alignment_checkpoint,
    save_alignment_checkpoint,
    validate_alignment_checkpoint,
)
from ..align.pipeline import (
    align,
    AlignConfig,
    AlignResumeState,
    AlignMultiresResumeState,
)
from ..utils.logging import setup_logging, log_jax_env
from ..utils.axes import DISK_VOLUME_AXES
from ._runtime import transfer_guard_context
from .config import parse_args_with_config
from .manifest import build_manifest, save_manifest

from ..utils.fov import (
    compute_roi,
    grid_from_detector_fov_slices,
    grid_from_detector_fov,
    cylindrical_mask_xy,
)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("value must be a positive float")
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
        # Avoid noisy logs on environments without this feature
        from jax.experimental import compilation_cache as cc  # type: ignore
    except Exception:
        return
    try:
        cache_dir = os.environ.get("TOMOJAX_JAX_CACHE_DIR")
        if not cache_dir:
            base = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
            cache_dir = os.path.join(base, "tomojax", "jax_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cc.initialize_cache(cache_dir)
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
    except Exception:
        det_smaller = False

    recon_grid = grid
    apply_cyl_mask = False
    if roi_mode == "cube" or (roi_mode == "auto" and det_smaller):
        if roi_mode == "auto" and not is_parallel:
            recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=False)
        else:
            recon_grid = grid_from_detector_fov_slices(grid, detector, crop_y_to_u=is_parallel)
    elif roi_mode == "bbox":
        recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=is_parallel)
    elif roi_mode == "cyl":
        recon_grid = grid_from_detector_fov_slices(grid, detector, crop_y_to_u=is_parallel)
        apply_cyl_mask = True

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
            else normalize_dofs(args.optimise_dofs, option_name="--optimise-dofs")
        )
        freeze_dofs = normalize_dofs(args.freeze_dofs, option_name="--freeze-dofs")
        active_dof_mask(optimise_dofs=optimise_dofs, freeze_dofs=freeze_dofs)
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Joint reconstruction + alignment on dataset (.nxs)")
    p.add_argument("--config", help="Load command defaults from a TOML config file")
    p.add_argument("--data", help="Input .nxs")
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
        choices=["gd", "gn"],
        default="gn",
        help="Alignment optimizer: gd or gn (GN supported for L2-like losses: l2, l2_otsu, edge_l2, pwls)",
    )
    p.add_argument(
        "--gn-damping",
        type=float,
        default=1e-3,
        help="Levenberg-Marquardt damping for GN",
    )
    p.add_argument(
        "--optimise-dofs",
        nargs="+",
        default=None,
        metavar="DOF[,DOF]",
        help="Named alignment DOFs to optimise: alpha,beta,phi,dx,dz. Example: dx,dz",
    )
    p.add_argument(
        "--freeze-dofs",
        nargs="+",
        default=None,
        metavar="DOF[,DOF]",
        help="Named alignment DOFs to keep fixed at initial values. Example: phi",
    )
    p.add_argument(
        "--bounds",
        type=_parse_bounds_arg,
        default=None,
        metavar="DOF=LOWER:UPPER[,DOF=LOWER:UPPER]",
        help=(
            "Finite per-DOF parameter bounds. Rotations are radians; translations are "
            "world units. Example: dx=-20:20,dz=-20:20,alpha=-0.05:0.05"
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
        help="JAX transfer guard mode during compute (default: off; use log/disallow when debugging)",
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
            "Region to reconstruct: auto: square x–y slices + z from detector height; "
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
            "off (default), or cyl for cylindrical x–y mask broadcast along z."
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
        "roi": str(args.roi),
        "grid": args.grid,
        "requested_gather_dtype": str(args.gather_dtype),
        "gather_dtype": str(gather_dtype),
        "recon_algo": str(args.recon_algo),
        "views_per_batch": int(args.views_per_batch),
        "spdhg_seed": int(args.spdhg_seed),
        "recon_positivity": bool(args.recon_positivity),
        "projector_unroll": 1,
        "checkpoint_projector": bool(args.checkpoint_projector),
        "mask_vol": str(args.mask_vol),
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
) -> dict[str, object]:
    geometry_meta = getattr(getattr(meta, "metadata", meta), "geometry_meta", None)
    geometry_type = getattr(meta, "geometry_type", "parallel")
    return build_alignment_checkpoint_metadata(
        projections_shape=tuple(int(v) for v in projections.shape),
        projections_dtype=str(projections.dtype),
        geometry_type=str(geometry_type),
        geometry_meta=geometry_meta,
        reconstruction_grid=recon_grid.to_dict(),
        detector=detector.to_dict(),
        state_grid=state_grid.to_dict(),
        state_detector=state_detector.to_dict(),
        levels=levels,
        level_index=int(level_index),
        level_factor=int(level_factor),
        completed_outer_iters_in_level=int(completed_outer_iters_in_level),
        global_outer_iters_completed=int(global_outer_iters_completed),
        config=cfg,
        cli_options=_checkpoint_cli_options(args, gather_dtype=gather_dtype),
        prev_factor=prev_factor,
        current_inner_iteration=0,
        L_prev=L_prev,
        small_impr_streak=small_impr_streak,
        elapsed_offset=elapsed_offset,
        random_state={
            "alignment": None,
            "seed_translations": (
                "deterministic_phase_correlation" if cfg.seed_translations else None
            ),
        },
        level_complete=level_complete,
        run_complete=run_complete,
    )


def _resume_state_from_checkpoint(
    checkpoint_path: str,
    *,
    expected_metadata: dict[str, object],
    used_multires: bool,
) -> AlignResumeState | AlignMultiresResumeState:
    checkpoint = load_alignment_checkpoint(checkpoint_path)
    validate_alignment_checkpoint(checkpoint, expected_metadata)
    metadata = checkpoint.metadata
    if used_multires:
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
            completed_outer_iters_in_level=int(
                metadata.get("completed_outer_iters_in_level", 0)
            ),
            global_outer_iters_completed=int(
                metadata.get("global_outer_iters_completed", 0)
            ),
            prev_factor=(
                None
                if metadata.get("prev_factor") is None
                else int(metadata.get("prev_factor"))
            ),
            loss=list(checkpoint.loss_history),
            outer_stats=[dict(stat) for stat in checkpoint.outer_stats],
            L=metadata.get("L_prev"),
            small_impr_streak=int(metadata.get("small_impr_streak", 0)),
            elapsed_offset=float(metadata.get("elapsed_offset", 0.0)),
            level_complete=bool(metadata.get("level_complete", False)),
            run_complete=bool(metadata.get("run_complete", False)),
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


def main() -> None:
    p = _build_parser()
    args, config_metadata = parse_args_with_config(p, required=("data", "out"))

    setup_logging()
    log_jax_env()
    _init_jax_compilation_cache()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"
    loss_config, loss_params = _parse_loss_config(args, p)
    optimise_dofs, freeze_dofs = _parse_dof_args(args, p)
    levels = (
        [int(v) for v in args.levels]
        if args.levels is not None and len(args.levels) > 0
        else None
    )
    try:
        validate_loss_schedule_levels(loss_config, levels if levels is not None else [1])
    except ValueError as exc:
        p.error(str(exc))

    meta = load_nxtomo(args.data)
    geometry_meta = meta.geometry_inputs()
    initial_grid_override = args.grid if (meta.grid is None and args.grid is not None) else None
    grid, detector, geom = build_geometry_from_meta(
        geometry_meta,
        grid_override=initial_grid_override,
        apply_saved_alignment=False,
    )
    proj = jnp.asarray(meta.projections, dtype=jnp.float32)

    # Resolve default gather dtype lazily at runtime
    from ..utils.memory import default_gather_dtype as _default_gather_dtype

    _gather = str(args.gather_dtype)
    if _gather == "auto":
        _gather = _default_gather_dtype()

    cfg = AlignConfig(
        outer_iters=args.outer_iters,
        recon_iters=args.recon_iters,
        recon_algo=str(args.recon_algo),
        lambda_tv=args.lambda_tv,
        regulariser=str(args.regulariser),
        huber_delta=float(args.huber_delta),
        tv_prox_iters=int(args.tv_prox_iters),
        recon_positivity=bool(args.recon_positivity),
        spdhg_seed=int(args.spdhg_seed),
        lr_rot=args.lr_rot,
        lr_trans=args.lr_trans,
        views_per_batch=int(args.views_per_batch),
        projector_unroll=1,
        checkpoint_projector=bool(args.checkpoint_projector),
        gather_dtype=_gather,
        opt_method=str(args.opt_method),
        gn_damping=float(args.gn_damping),
        w_rot=float(args.w_rot),
        w_trans=float(args.w_trans),
        optimise_dofs=optimise_dofs,
        freeze_dofs=freeze_dofs,
        bounds=args.bounds,
        pose_model=str(args.pose_model),
        knot_spacing=int(args.knot_spacing),
        degree=int(args.degree),
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
    # ROI handling (align on realistic FOV)
    recon_grid, apply_cyl_mask = _resolve_recon_grid_and_mask(
        grid,
        detector,
        is_parallel=meta.geometry_type == "parallel",
        roi_mode=str(args.roi).lower(),
        grid_override=args.grid,
    )

    # Rebuild geometry if grid changed
    if recon_grid is not grid:
        # Once ROI and explicit sizing resolve an effective grid, keep that grid's
        # origin/centre metadata authoritative when rebuilding geometry.
        _, _, geom = build_geometry_from_meta(
            geometry_meta,
            grid_override=recon_grid,
            apply_saved_alignment=False,
        )

    checkpoint_path = args.checkpoint or args.resume
    checkpoint_every = args.checkpoint_every
    if checkpoint_path is not None and checkpoint_every is None:
        checkpoint_every = 1
    if checkpoint_every is not None and int(checkpoint_every) < 1:
        p.error("--checkpoint-every must be an integer >= 1")

    expected_checkpoint_metadata = _checkpoint_metadata(
        meta=meta,
        projections=proj,
        cfg=cfg,
        args=args,
        recon_grid=recon_grid,
        detector=detector,
        state_grid=recon_grid,
        state_detector=detector,
        gather_dtype=_gather,
        levels=levels,
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
    )
    resume_state = None
    if args.resume is not None:
        try:
            resume_state = _resume_state_from_checkpoint(
                args.resume,
                expected_metadata=expected_checkpoint_metadata,
                used_multires=levels is not None,
            )
        except CheckpointError as exc:
            raise SystemExit(f"tomojax-align: {exc}") from exc
        logging.info("Resuming alignment from checkpoint %s", args.resume)

    def _state_grid_detector(
        level_factor: int,
        *,
        run_complete: bool,
    ) -> tuple[Grid, Detector]:
        if levels is None or run_complete:
            return recon_grid, detector
        from ..recon.multires import scale_detector, scale_grid

        return scale_grid(recon_grid, int(level_factor)), scale_detector(detector, int(level_factor))

    def _write_single_checkpoint(
        state: AlignResumeState,
        *,
        run_complete: bool = False,
    ) -> None:
        if checkpoint_path is None:
            return
        completed = int(state.start_outer_iter)
        if not run_complete and (completed <= 0 or completed % int(checkpoint_every or 1) != 0):
            return
        metadata = _checkpoint_metadata(
            meta=meta,
            projections=proj,
            cfg=cfg,
            args=args,
            recon_grid=recon_grid,
            detector=detector,
            state_grid=recon_grid,
            state_detector=detector,
            gather_dtype=_gather,
            levels=None,
            level_index=0,
            level_factor=1,
            completed_outer_iters_in_level=completed,
            global_outer_iters_completed=completed,
            prev_factor=None,
            L_prev=state.L,
            small_impr_streak=int(state.small_impr_streak),
            elapsed_offset=float(state.elapsed_offset),
            level_complete=run_complete or completed >= int(cfg.outer_iters),
            run_complete=run_complete,
        )
        save_alignment_checkpoint(
            checkpoint_path,
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            loss_history=state.loss,
            outer_stats=state.outer_stats,
            metadata=metadata,
        )
        logging.info("Saved alignment checkpoint to %s", checkpoint_path)

    def _write_multires_checkpoint(state: AlignMultiresResumeState) -> None:
        if checkpoint_path is None:
            return
        completed = int(state.global_outer_iters_completed)
        if (
            not state.run_complete
            and not state.level_complete
            and (completed <= 0 or completed % int(checkpoint_every or 1) != 0)
        ):
            return
        state_grid, state_detector = _state_grid_detector(
            int(state.level_factor),
            run_complete=bool(state.run_complete),
        )
        metadata = _checkpoint_metadata(
            meta=meta,
            projections=proj,
            cfg=cfg,
            args=args,
            recon_grid=recon_grid,
            detector=detector,
            state_grid=state_grid,
            state_detector=state_detector,
            gather_dtype=_gather,
            levels=levels,
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
        )
        save_alignment_checkpoint(
            checkpoint_path,
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            loss_history=state.loss,
            outer_stats=state.outer_stats,
            metadata=metadata,
        )
        logging.info("Saved alignment checkpoint to %s", checkpoint_path)

    if args.levels is not None and len(args.levels) > 0:
        from ..align.pipeline import align_multires

        with transfer_guard_context(args.transfer_guard):
            x, params5, info = align_multires(
                geom,
                recon_grid,
                detector,
                proj,
                factors=args.levels,
                cfg=cfg,
                resume_state=(
                    resume_state if isinstance(resume_state, AlignMultiresResumeState) else None
                ),
                checkpoint_callback=_write_multires_checkpoint
                if checkpoint_path is not None
                else None,
            )
    else:
        with transfer_guard_context(args.transfer_guard):
            x, params5, info = align(
                geom,
                recon_grid,
                detector,
                proj,
                cfg=cfg,
                resume_state=resume_state if isinstance(resume_state, AlignResumeState) else None,
                checkpoint_callback=_write_single_checkpoint if checkpoint_path is not None else None,
            )
        if checkpoint_path is not None:
            _write_single_checkpoint(
                AlignResumeState(
                    x=x,
                    params5=params5,
                    motion_coeffs=info.get("motion_coeffs") if isinstance(info, dict) else None,
                    start_outer_iter=int(
                        info.get("completed_outer_iters", len(info.get("outer_stats", [])))
                        if isinstance(info, dict)
                        else int(cfg.outer_iters)
                    ),
                    loss=list(info.get("loss", [])) if isinstance(info, dict) else [],
                    outer_stats=(
                        [dict(stat) for stat in info.get("outer_stats", [])]
                        if isinstance(info, dict)
                        else []
                    ),
                    L=info.get("L") if isinstance(info, dict) else None,
                    small_impr_streak=int(info.get("small_impr_streak", 0))
                    if isinstance(info, dict)
                    else 0,
                    elapsed_offset=float(info.get("wall_time_total", 0.0))
                    if isinstance(info, dict)
                    else 0.0,
                ),
                run_complete=True,
            )

    # Optional cylindrical mask in x–y
    if apply_cyl_mask:
        import numpy as _np

        try:
            m_xy = cylindrical_mask_xy(recon_grid, detector)
            m = jnp.asarray(m_xy, dtype=x.dtype)[:, :, None]
            x = x * m
        except Exception:
            m_xy = cylindrical_mask_xy(recon_grid, detector)
            m = _np.asarray(m_xy, dtype=_np.float32)[:, :, None]
            x = jnp.asarray(_np.asarray(x) * m)

    # Avoid copying projections back from device: reuse host array from metadata
    params5_np = np.asarray(params5)
    save_meta = meta.copy_metadata()
    save_meta.grid = recon_grid.to_dict()
    save_meta.volume = np.asarray(x)
    save_meta.align_params = params5_np
    save_meta.frame = str(meta.frame or "sample")
    save_meta.volume_axes_order = str(args.volume_axes)
    save_nxtomo(
        args.out,
        projections=meta.projections,
        metadata=save_meta,
    )
    logging.info("Saved alignment results to %s", args.out)
    if args.save_params_json is not None:
        save_alignment_params_json(
            args.save_params_json,
            params5_np,
            du=float(detector.du),
            dv=float(detector.dv),
        )
        logging.info("Saved alignment parameter JSON to %s", args.save_params_json)
    if args.save_params_csv is not None:
        save_alignment_params_csv(
            args.save_params_csv,
            params5_np,
            du=float(detector.du),
            dv=float(detector.dv),
        )
        logging.info("Saved alignment parameter CSV to %s", args.save_params_csv)
    if args.save_manifest is not None:
        loss_values = info.get("loss", []) if isinstance(info, dict) else []
        manifest = build_manifest(
            "tomojax-align",
            list(sys.argv),
            args,
            {
                "input_path": args.data,
                "output_path": args.out,
                "save_params_json": args.save_params_json,
                "save_params_csv": args.save_params_csv,
                "manifest_path": args.save_manifest,
                "config_path": config_metadata["config_path"],
                "config_file_values": config_metadata["config_file_values"],
                "explicit_cli_keys": config_metadata["explicit_cli_keys"],
                "effective_options": config_metadata["effective_options"],
                "geometry_type": str(meta.geometry_type),
                "input_projection_shape": list(meta.projections.shape),
                "reconstruction_grid": recon_grid.to_dict(),
                "detector": detector.to_dict(),
                "roi": {
                    "requested": str(args.roi),
                    "is_parallel": bool(meta.geometry_type == "parallel"),
                    "grid_changed": recon_grid != grid,
                    "cylindrical_output_mask": bool(apply_cyl_mask),
                },
                "requested_gather_dtype": str(args.gather_dtype),
                "gather_dtype": _gather,
                "recon_algo": str(args.recon_algo),
                "regulariser": str(args.regulariser),
                "huber_delta": float(args.huber_delta),
                "views_per_batch": int(args.views_per_batch),
                "spdhg_seed": int(args.spdhg_seed),
                "recon_positivity": bool(args.recon_positivity),
                "projector_unroll": 1,
                "checkpoint_projector": bool(args.checkpoint_projector),
                "transfer_guard": str(args.transfer_guard),
                "levels": args.levels,
                "used_multires": bool(args.levels is not None and len(args.levels) > 0),
                "checkpoint_path": args.checkpoint,
                "checkpoint_every": args.checkpoint_every,
                "resume_path": args.resume,
                "loss_params": loss_params,
                "loss_spec": loss_config,
                "align_config": cfg,
                "alignment_params_shape": list(params5_np.shape),
                "volume_shape": list(np.asarray(x).shape),
                "volume_axes": str(args.volume_axes),
                "frame": str(save_meta.frame),
                "run_info": {
                    "loss_count": len(loss_values),
                    "final_loss": loss_values[-1] if len(loss_values) else None,
                    "stopped_by_observer": (
                        info.get("stopped_by_observer") if isinstance(info, dict) else None
                    ),
                    "observer_action": (
                        info.get("observer_action") if isinstance(info, dict) else None
                    ),
                },
            },
        )
        save_manifest(args.save_manifest, manifest)
        logging.info("Saved reproducibility manifest to %s", args.save_manifest)


if __name__ == "__main__":  # pragma: no cover
    main()
