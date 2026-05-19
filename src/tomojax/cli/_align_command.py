from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from typing import Literal, cast

import numpy as np

from tomojax.align.api import (
    PUBLIC_SCHEDULE_PRESETS,
    AlignmentLossConfig,
    DofBounds,
    normalize_alignment_dofs,
    normalize_alignment_profile,
    normalize_bounds,
    parse_loss_schedule,
    parse_loss_spec,
)
from tomojax.geometry import DISK_VOLUME_AXES

type AlignmentMode = Literal["cor", "pose", "auto", "max"]


_PUBLIC_HELP_OPTIONS = frozenset(
    {
        "-h",
        "--help",
        "--config",
        "--data",
        "--mode",
        "--quality",
        "--out",
        "--save-manifest",
        "--progress",
        "--roi",
        "--grid",
        "--volume-axes",
        "--checkpoint",
        "--checkpoint-every",
        "--resume",
    }
)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive float") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive float")
    return parsed


def parse_dof_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    optimise_dofs_arg = cast("list[str] | None", args.optimise_dofs)
    freeze_dofs_arg = cast("list[str] | None", args.freeze_dofs)
    try:
        optimise_dofs = (
            None
            if optimise_dofs_arg is None
            else normalize_alignment_dofs(optimise_dofs_arg, option_name="--optimise-dofs")
        )
        freeze_dofs = normalize_alignment_dofs(freeze_dofs_arg, option_name="--freeze-dofs")
    except ValueError as exc:
        parser.error(str(exc))
    return optimise_dofs, freeze_dofs


def _parse_bounds_arg(value: object) -> DofBounds:
    try:
        return normalize_bounds(value, option_name="--bounds")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_loss_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[AlignmentLossConfig, dict[str, float]]:
    loss_name = cast("str", args.loss)
    loss_schedule = cast("str | None", args.loss_schedule)
    loss_param_items = cast("list[str]", args.loss_param)
    loss_params: dict[str, float] = {}
    for kv in loss_param_items:
        if "=" not in kv:
            parser.error(f"--loss-param must be k=v, got: {kv}")
        k, v = kv.split("=", 1)
        try:
            loss_params[k.strip()] = float(v)
        except ValueError:
            parser.error(f"--loss-param value must be numeric: {kv}")

    try:
        loss_spec = parse_loss_spec(loss_name, loss_params if loss_params else None)
        if loss_schedule is None:
            return loss_spec, loss_params
        return parse_loss_schedule(loss_schedule, default=loss_spec), loss_params
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    raise AssertionError("unreachable")


def _alignment_quality_argument(value: str) -> str:
    """Normalize public quality names to the internal profile policy."""
    try:
        return normalize_alignment_profile(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("quality must be 'fast' or 'reference'") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Joint reconstruction + alignment on dataset (.nxs)")
    _ = p.add_argument("--config", help="Load command defaults from a TOML config file")
    _ = p.add_argument("--data", help="Input .nxs")
    _ = p.add_argument(
        "--mode",
        choices=["cor", "pose", "auto", "max"],
        default="auto",
        help=(
            "High-level alignment mode. cor solves detector centre; pose solves "
            "per-view motion; auto runs the default setup+pose workflow; max "
            "uses the slower reference-quality posture."
        ),
    )
    _ = p.add_argument(
        "--quality",
        dest="align_profile",
        type=_alignment_quality_argument,
        default="lightning",
        metavar="{fast,reference}",
        help="Execution quality posture: fast (default) or reference.",
    )
    _ = p.add_argument(
        "--align-profile",
        dest="align_profile",
        type=_alignment_quality_argument,
        help=argparse.SUPPRESS,
    )
    _ = p.add_argument("--outer-iters", type=int, default=5)
    _ = p.add_argument("--recon-iters", type=int, default=10)
    _ = p.add_argument(
        "--recon-algo",
        choices=["fista", "spdhg"],
        default="fista",
        help="Inner reconstruction solver used during alignment (default: fista)",
    )
    _ = p.add_argument("--lambda-tv", type=float, default=0.005)
    _ = p.add_argument(
        "--regulariser",
        choices=["tv", "huber_tv"],
        default="tv",
        help="Regulariser for inner reconstruction: tv (default) or huber_tv",
    )
    _ = p.add_argument(
        "--huber-delta",
        type=_positive_float,
        default=1e-2,
        help="Huber-TV transition radius for --regulariser huber_tv",
    )
    _ = p.add_argument(
        "--tv-prox-iters",
        type=int,
        default=10,
        help="Inner iterations for the FISTA TV proximal operator",
    )
    _ = p.add_argument(
        "--views-per-batch",
        type=int,
        default=1,
        help="Projection views per inner reconstruction batch/subset (default: 1)",
    )
    _ = p.add_argument(
        "--projector-unroll",
        type=int,
        default=1,
        help="Projector loop unroll factor for differentiable alignment paths (default: 1)",
    )
    _ = p.add_argument(
        "--projector-backend",
        choices=["jax", "pallas"],
        default="jax",
        help=(
            "Alignment projector backend: jax is the default gradient-safe reference; "
            "pallas requests supported accelerator paths with JAX fallback metadata"
        ),
    )
    _ = p.add_argument(
        "--spdhg-seed",
        type=int,
        default=0,
        help="Base random seed for SPDHG subset order inside alignment",
    )
    rp = p.add_mutually_exclusive_group()
    _ = rp.add_argument(
        "--recon-positivity",
        dest="recon_positivity",
        action="store_true",
        help="Enable positivity projection for SPDHG inner reconstructions (default)",
    )
    _ = rp.add_argument(
        "--no-recon-positivity",
        dest="recon_positivity",
        action="store_false",
        help="Disable positivity projection for SPDHG inner reconstructions",
    )
    p.set_defaults(recon_positivity=True)
    _ = p.add_argument("--lr-rot", type=float, default=1e-3)
    _ = p.add_argument("--lr-trans", type=float, default=1e-1)
    _ = p.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="Optional multires factors, e.g., 4 2 1",
    )
    _ = p.add_argument(
        "--gather-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Projector gather dtype (auto: bf16 on GPU/TPU, else fp32)",
    )
    ck = p.add_mutually_exclusive_group()
    _ = ck.add_argument("--checkpoint-projector", dest="checkpoint_projector", action="store_true")
    _ = ck.add_argument(
        "--no-checkpoint-projector",
        dest="checkpoint_projector",
        action="store_false",
    )
    p.set_defaults(checkpoint_projector=True)
    _ = p.add_argument(
        "--opt-method",
        choices=["gd", "gn", "lbfgs"],
        default="gn",
        help=(
            "Alignment optimizer: gd, gn, or lbfgs. GN is supported for L2-like "
            "losses: l2, l2_otsu, edge_l2, pwls."
        ),
    )
    _ = p.add_argument(
        "--gn-damping",
        type=float,
        default=1e-3,
        help="Levenberg-Marquardt damping for GN",
    )
    _ = p.add_argument(
        "--lbfgs-maxiter",
        type=int,
        default=20,
        help="Maximum Optax L-BFGS iterations per alignment outer step",
    )
    _ = p.add_argument(
        "--lbfgs-ftol",
        type=float,
        default=1e-6,
        help="Relative function tolerance for Optax L-BFGS",
    )
    _ = p.add_argument(
        "--lbfgs-gtol",
        type=float,
        default=1e-5,
        help="Gradient-norm tolerance for Optax L-BFGS",
    )
    _ = p.add_argument(
        "--lbfgs-maxls",
        type=int,
        default=20,
        help="Maximum Optax L-BFGS line-search steps per iteration",
    )
    _ = p.add_argument(
        "--lbfgs-memory-size",
        type=int,
        default=10,
        help="Number of previous gradient/step pairs stored by Optax L-BFGS",
    )
    _ = p.add_argument(
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
    _ = p.add_argument(
        "--freeze-dofs",
        nargs="+",
        default=None,
        metavar="DOF[,DOF]",
        help="Named alignment DOFs to keep fixed at initial values. Example: phi or det_u_px",
    )
    _ = p.add_argument(
        "--schedule",
        choices=list(PUBLIC_SCHEDULE_PRESETS),
        default=None,
        help=(
            "Executable alignment preset. Setup presets use validation-LM stages; "
            "explicit --optimise-dofs is the lower-level direct surface."
        ),
    )
    _ = p.add_argument(
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
    _ = p.add_argument(
        "--gauge-policy",
        choices=["reject", "anchor_mean", "prior_required", "diagnose_only"],
        default="reject",
        help=(
            "Policy for gauge-coupled direct/expert DOF sets. Public presets carry "
            "their own stage policies; direct mixed setup+pose defaults to reject."
        ),
    )
    _ = p.add_argument(
        "--pose-model",
        choices=["per_view", "polynomial", "spline"],
        default="per_view",
        help=(
            "Alignment pose parameterization: per_view optimizes one 5-DOF vector per "
            "view; polynomial and spline optimize smooth low-dimensional trajectories"
        ),
    )
    _ = p.add_argument(
        "--knot-spacing",
        type=int,
        default=8,
        help="View spacing between spline knots when --pose-model spline is used",
    )
    _ = p.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial degree or spline degree for smooth pose models",
    )
    _ = p.add_argument(
        "--gauge-fix",
        choices=["mean_translation", "none"],
        default="mean_translation",
        help=(
            "Gauge fixing for alignment parameters: mean_translation subtracts the "
            "scan-wide mean from active dx,dz after updates (default); none preserves "
            "historical unconstrained traces"
        ),
    )
    _ = p.add_argument("--w-rot", type=float, default=1e-3, help="Smoothness weight for rotations")
    _ = p.add_argument(
        "--w-trans",
        type=float,
        default=1e-3,
        help="Smoothness weight for translations",
    )
    _ = p.add_argument(
        "--seed-translations",
        action="store_true",
        help="Phase-correlation init for dx,dz at coarsest level",
    )
    _ = p.add_argument(
        "--log-summary",
        action="store_true",
        help="Print per-outer summaries (FISTA loss, alignment loss before/after)",
    )
    _ = p.add_argument(
        "--log-compact",
        dest="log_compact",
        action="store_true",
        default=True,
        help="Use compact one-line per-outer summary when --log-summary is set (default: on)",
    )
    _ = p.add_argument("--no-log-compact", dest="log_compact", action="store_false")
    _ = p.add_argument(
        "--recon-L",
        type=float,
        default=None,
        help="Fixed Lipschitz constant for FISTA inside alignment (skip power-method)",
    )
    _ = p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Write resumable alignment checkpoints to PATH after completed outer iterations.",
    )
    _ = p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        metavar="N",
        help="Checkpoint every N completed global outer iterations (default: 1 when enabled).",
    )
    _ = p.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Resume alignment from a checkpoint. Defaults future checkpoint writes to this path.",
    )
    # Data term / similarity
    _ = p.add_argument(
        "--loss",
        choices=[
            "l2",
            "charbonnier",
            "huber",
            "cauchy",
            "welsch",
            "barron",
            "student_t",
            "correntropy",
            "zncc",
            "ssim",
            "ms_ssim",
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
    _ = p.add_argument(
        "--loss-schedule",
        default=None,
        help=(
            "Pyramid-level loss schedule as LEVEL:LOSS entries, e.g. "
            "4:phasecorr,2:ssim,1:l2_otsu. Unspecified levels use --loss."
        ),
    )
    _ = p.add_argument(
        "--loss-param",
        action="append",
        default=[],
        help="Loss parameter as k=v (repeatable), e.g., delta=1.0, eps=1e-3, window=7, temp=0.5",
    )
    # Early stopping controls (alignment phase)
    es = p.add_mutually_exclusive_group()
    _ = es.add_argument(
        "--early-stop",
        dest="early_stop",
        action="store_true",
        help="Enable early stopping across outers (default)",
    )
    _ = es.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="Disable early stopping across outers",
    )
    p.set_defaults(early_stop=True)
    _ = p.add_argument(
        "--early-stop-rel",
        type=float,
        default=None,
        help="Relative improvement threshold for early stop (default 1e-3)",
    )
    _ = p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Consecutive outers below threshold before stopping (default 2)",
    )
    _ = p.add_argument(
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help=(
            "JAX transfer guard mode during compute "
            "(default: off; use log/disallow for strict transfer checks)"
        ),
    )
    _ = p.add_argument("--out", help="Output .nxs with recon and alignment params")
    _ = p.add_argument(
        "--save-params-json",
        default=None,
        help="Optional JSON sidecar for final per-view alignment parameters",
    )
    _ = p.add_argument(
        "--save-params-csv",
        default=None,
        help="Optional CSV sidecar for final per-view alignment parameters",
    )
    _ = p.add_argument(
        "--save-manifest",
        metavar="PATH",
        default=None,
        help="Write a JSON reproducibility manifest for this alignment run.",
    )
    _ = p.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars if tqdm is available",
    )
    _ = p.add_argument(
        "--roi",
        choices=["auto", "off", "cube", "bbox", "cyl"],
        default="auto",
        help=(
            "Region to reconstruct: auto: square x-y slices + z from detector height; "
            "off: use full grid; cube: same as auto; bbox: rectangular FOV bbox; "
            "cyl: auto + zero outside cylindrical FOV"
        ),
    )
    _ = p.add_argument(
        "--mask-vol",
        choices=["off", "cyl"],
        default="off",
        help=(
            "Mask the volume before forward projection in alignment: "
            "off (default), or cyl for cylindrical x-y mask broadcast along z."
        ),
    )
    _ = p.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Override reconstruction grid size (nx ny nz). Voxel sizes stay as in input metadata.",
    )
    _ = p.add_argument(
        "--volume-axes",
        choices=["zyx", "xyz"],
        default=DISK_VOLUME_AXES,
        help="On-disk axis order for saved volumes (default: zyx for viewer convention).",
    )
    _hide_expert_alignment_help(p)
    return p


def _hide_expert_alignment_help(parser: argparse.ArgumentParser) -> None:
    """Keep default alignment help product-shaped while accepting expert flags."""
    for action in parser._actions:
        if not action.option_strings:
            continue
        if set(action.option_strings).isdisjoint(_PUBLIC_HELP_OPTIONS):
            action.help = argparse.SUPPRESS


@dataclass(frozen=True, slots=True)
class AlignCommand:
    """Typed command values resolved from the public alignment parser."""

    data: str
    out: str
    mode: AlignmentMode
    align_profile: str
    outer_iters: int
    recon_iters: int
    roi: str
    grid: list[int] | None
    requested_gather_dtype: str
    recon_algo: str
    lambda_tv: float
    regulariser: str
    huber_delta: float
    tv_prox_iters: int
    views_per_batch: int
    spdhg_seed: int
    recon_positivity: bool
    projector_unroll: int
    projector_backend: str
    quality_tier: str
    fallback_policy: str
    checkpoint_projector: bool
    mask_vol: str
    gauge_fix: str
    gauge_policy: str
    opt_method: str
    gn_damping: float
    lbfgs_maxiter: int
    lbfgs_ftol: float
    lbfgs_gtol: float
    lbfgs_maxls: int
    lbfgs_memory_size: int
    lr_rot: float
    lr_trans: float
    w_rot: float
    w_trans: float
    bounds: DofBounds | None
    pose_model: str
    knot_spacing: int
    degree: int
    seed_translations: bool
    log_summary: bool
    log_compact: bool
    recon_l: float | None
    early_stop: bool
    early_stop_rel: float | None
    early_stop_patience: int | None
    optimise_dofs: list[str]
    freeze_dofs: list[str]
    schedule: str | None
    checkpoint: str | None
    checkpoint_every: int | None
    resume: str | None
    transfer_guard: str
    save_params_json: str | None
    save_params_csv: str | None
    save_manifest: str | None
    volume_axes: str


def align_command_from_args(args: argparse.Namespace) -> AlignCommand:
    """Snapshot parser/config output into typed alignment command values."""
    return AlignCommand(
        data=cast("str", args.data),
        out=cast("str", args.out),
        mode=cast("AlignmentMode", args.mode),
        align_profile=cast("str", args.align_profile),
        outer_iters=cast("int", args.outer_iters),
        recon_iters=cast("int", args.recon_iters),
        roi=cast("str", args.roi),
        grid=cast("list[int] | None", args.grid),
        requested_gather_dtype=cast("str", args.gather_dtype),
        recon_algo=cast("str", args.recon_algo),
        lambda_tv=cast("float", args.lambda_tv),
        regulariser=cast("str", args.regulariser),
        huber_delta=cast("float", args.huber_delta),
        tv_prox_iters=cast("int", args.tv_prox_iters),
        views_per_batch=cast("int", args.views_per_batch),
        spdhg_seed=cast("int", args.spdhg_seed),
        recon_positivity=cast("bool", args.recon_positivity),
        projector_unroll=cast("int", args.projector_unroll),
        projector_backend=cast("str", args.projector_backend),
        quality_tier=cast("str", getattr(args, "quality_tier", "")),
        fallback_policy=cast("str", getattr(args, "fallback_policy", "")),
        checkpoint_projector=cast("bool", args.checkpoint_projector),
        mask_vol=cast("str", args.mask_vol),
        gauge_fix=cast("str", args.gauge_fix),
        gauge_policy=cast("str", args.gauge_policy),
        opt_method=cast("str", args.opt_method),
        gn_damping=cast("float", args.gn_damping),
        lbfgs_maxiter=cast("int", args.lbfgs_maxiter),
        lbfgs_ftol=cast("float", args.lbfgs_ftol),
        lbfgs_gtol=cast("float", args.lbfgs_gtol),
        lbfgs_maxls=cast("int", args.lbfgs_maxls),
        lbfgs_memory_size=cast("int", args.lbfgs_memory_size),
        lr_rot=cast("float", args.lr_rot),
        lr_trans=cast("float", args.lr_trans),
        w_rot=cast("float", args.w_rot),
        w_trans=cast("float", args.w_trans),
        bounds=cast("DofBounds | None", args.bounds),
        pose_model=cast("str", args.pose_model),
        knot_spacing=cast("int", args.knot_spacing),
        degree=cast("int", args.degree),
        seed_translations=cast("bool", args.seed_translations),
        log_summary=cast("bool", args.log_summary),
        log_compact=cast("bool", args.log_compact),
        recon_l=cast("float | None", args.recon_L),
        early_stop=cast("bool", args.early_stop),
        early_stop_rel=cast("float | None", args.early_stop_rel),
        early_stop_patience=cast("int | None", args.early_stop_patience),
        optimise_dofs=list(cast("list[str] | None", args.optimise_dofs) or []),
        freeze_dofs=list(cast("list[str] | None", args.freeze_dofs) or []),
        schedule=cast("str | None", args.schedule),
        checkpoint=cast("str | None", args.checkpoint),
        checkpoint_every=cast("int | None", args.checkpoint_every),
        resume=cast("str | None", args.resume),
        transfer_guard=cast("str", args.transfer_guard),
        save_params_json=cast("str | None", args.save_params_json),
        save_params_csv=cast("str | None", args.save_params_csv),
        save_manifest=cast("str | None", args.save_manifest),
        volume_axes=cast("str", args.volume_axes),
    )
