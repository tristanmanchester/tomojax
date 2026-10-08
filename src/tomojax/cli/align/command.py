from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from typing import Literal, cast

import numpy as np

from tomojax.alignment.api import (
    PUBLIC_SCHEDULE_PRESETS,
    AlignmentLossConfig,
    DofBounds,
    normalize_alignment_dofs,
    normalize_bounds,
    parse_loss_schedule,
    parse_loss_spec,
)
from tomojax.cli._options import add_config, add_output, hide_expert

# ruff: noqa: D100,D103
from tomojax.core.validation import option_name
from tomojax.geometry.api import DISK_VOLUME_AXES

type AlignmentMode = Literal["cor", "pose", "auto", "max", "cor_then_pose"]


_PUBLIC_OPTIONS = (
    "--mode",
    "--quality",
    "--freeze",
    "--levels",
    "--pose-solver",
    "--roi",
    "--grid",
    "--volume-axes",
    "--manifest",
    "--dry-run",
    "--checkpoint",
    "--resume",
    "--progress",
)
# Public mode names and the internal schedules' names for them; ``full`` at
# reference quality is the former ``max``.
_MODES = {"pose": "pose", "cor": "cor", "cor-then-pose": "cor_then_pose", "full": "auto"}


def public_mode(mode: str) -> str:
    """The public name of an internal alignment mode."""
    return {v: k for k, v in _MODES.items()}.get(str(mode), str(mode))


def _mode_argument(value: str) -> str:
    key = option_name(value, separator="-")
    if key not in _MODES:
        raise argparse.ArgumentTypeError(
            f"mode must be one of pose, cor, cor-then-pose, full; got {value!r}"
        )
    return _MODES[key]


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
    freeze_arg = cast("list[str] | None", args.freeze)
    try:
        optimise_dofs = (
            None
            if optimise_dofs_arg is None
            else normalize_alignment_dofs(optimise_dofs_arg, option_name="--optimise-dofs")
        )
        freeze = normalize_alignment_dofs(freeze_arg, option_name="--freeze")
    except ValueError as exc:
        parser.error(str(exc))
    return optimise_dofs, freeze


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tomojax align",
        description=(
            "Estimate a scan's geometry corrections (setup geometry and per-view motion) "
            "and reconstruct with them. OUTPUT holds the corrected scan, its poses and "
            "the volume; tomojax recon applies them."
        ),
        epilog=(
            "Modes:\n"
            "  pose           per-view motion: rotations and translations (default)\n"
            "  cor            setup geometry only: detector centre (parallel), or the\n"
            "                 axis offset and detector roll (cone beam)\n"
            "  cor-then-pose  setup geometry, then per-view motion\n"
            "  full           detector centre, roll and axis direction, then motion,\n"
            "                 coarse to fine\n\n"
            "Examples:\n"
            "  tomojax align scan.nxs -o aligned.nxs\n"
            "  tomojax align scan.nxs -o aligned.nxs --mode cor-then-pose --freeze dy\n"
            "  tomojax recon aligned.nxs -o recon.nxs --method cgls"
        ),
    )
    _add_input_mode_options(p)
    _add_reconstruction_options(p)
    _add_projector_runtime_options(p)
    _add_optimizer_options(p)
    _add_dof_schedule_options(p)
    _add_loss_options(p)
    _add_checkpoint_options(p)
    _add_output_options(p)
    hide_expert(p, _PUBLIC_OPTIONS)
    return p


def _add_input_mode_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument("data", metavar="INPUT", help="Scan to align (.nxs, .h5, .npz)")
    add_output(p, "Output .nxs: the corrected scan, its per-view poses and the volume")
    add_config(p)
    _ = p.add_argument(
        "--mode",
        type=_mode_argument,
        default="pose",
        metavar="{pose,cor,cor-then-pose,full}",
        help="What to estimate (default pose; see Modes below)",
    )
    _ = p.add_argument(
        "--quality",
        choices=["fast", "reference"],
        default="fast",
        help="fast (default) or reference: slower, more conservative solver settings",
    )


def _add_reconstruction_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--outer-iterations",
        type=int,
        default=None,
        help="Outer alignment iterations; default 30 for coupled pose stages, else 5",
    )
    _ = p.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Inner reconstruction iterations per outer iteration (default: 10)",
    )
    _ = p.add_argument(
        "--reconstruction",
        choices=["fista", "spdhg"],
        default="fista",
        help="Inner reconstruction solver used during alignment (default: fista)",
    )
    _ = p.add_argument(
        "--tv-weight",
        type=float,
        default=0.005,
        help="TV weight of the inner reconstruction (default: 0.005)",
    )
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
        "--tv-prox-iterations",
        type=int,
        default=10,
        help="Inner iterations for the FISTA TV proximal operator",
    )
    _ = p.add_argument(
        "--views-per-batch",
        type=int,
        default=0,
        help="Projection views per inner reconstruction batch; 0 sizes it from free GPU memory",
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
        "--seed",
        type=int,
        default=0,
        help="Base random seed for SPDHG subset order inside alignment",
    )
    _ = p.add_argument(
        "--nonnegative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep inner reconstruction voxels nonnegative (default: on)",
    )


def _add_projector_runtime_options(p: argparse.ArgumentParser) -> None:
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
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help=(
            "JAX transfer guard mode during compute "
            "(default: off; use log/disallow for strict transfer checks)"
        ),
    )


def _add_optimizer_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--pose-solver",
        choices=["coupled", "alternating"],
        default=None,
        help=(
            "How pose stages solve: coupled fits the volume and the poses together "
            "(Gauss-Newton; default for pose and cor-then-pose); alternating refines the "
            "poses against a fixed volume between volume updates (default for full)"
        ),
    )
    _ = p.add_argument(
        "--ray-integrator",
        choices=["joseph", "joseph_cubic", "exact", "sampled"],
        default=None,
        help=(
            "Forward model: joseph (voxel-centre planes, fastest), joseph_cubic, exact "
            "(trilinear basis integral) or sampled. Default: joseph"
        ),
    )
    _ = p.add_argument("--lr-rot", type=float, default=1e-3)
    _ = p.add_argument("--lr-trans", type=float, default=1e-1)
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


def _add_dof_schedule_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        metavar="FACTOR",
        help="Coarse-to-fine downsampling factors, for example 4 2 1 (default: by mode)",
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
        "--freeze",
        nargs="+",
        default=None,
        metavar="DOF",
        help=(
            "Parameters to keep fixed: alpha, beta, phi, dx, dz, dy (along a cone beam), "
            "or setup ones such as det_u_px"
        ),
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
            "their own stage policies; direct mixed setup+pose defaults to reject. "
            "Use anchor_mean for reconstruction-quality mixed correction."
        ),
    )
    _ = p.add_argument(
        "--pose-model",
        choices=["per_view", "polynomial", "spline"],
        default="per_view",
        help=(
            "Alignment pose parameterization: per_view optimizes one pose vector per "
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
        "--pose-translation-frame",
        choices=["object", "detector"],
        default="detector",
        help=(
            "Frame of the per-view dx,dz translations: detector (default) moves the "
            "sample along lab x,z, so every view can shift in both detector directions "
            "and a centre-of-rotation offset is a constant dx; object moves it along "
            "its own x,z axes, which cannot express horizontal detector shifts at views "
            "near 90 degrees"
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
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Search each view's detector shift globally (reconstruct, reproject, "
            "cross-correlate) before local alignment, extending the translation "
            "capture range. Default: on for the coupled pose solver, off otherwise"
        ),
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
        "--lipschitz",
        type=float,
        default=None,
        help="Fixed Lipschitz constant for FISTA inside alignment (skip power-method)",
    )


def _add_loss_options(p: argparse.ArgumentParser) -> None:
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
        "--early-stop-rel-impr",
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


def _add_checkpoint_options(p: argparse.ArgumentParser) -> None:
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
        help=(
            "Resume alignment from a checkpoint. Restores optimise_dofs, freeze"
            " and schedule from the checkpoint unless those options are set explicitly."
            " Defaults future checkpoint writes to this path."
        ),
    )


def _add_output_options(p: argparse.ArgumentParser) -> None:
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
        "--manifest",
        metavar="JSON",
        default=None,
        help="Also write a JSON record of the run (inputs, settings, versions)",
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
            "Crop the grid to the detector's field of view: auto (default), cube, bbox, "
            "cyl (auto, zeroing outside the cylinder every view sees), or off"
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
        help="Reconstruct NX x NY x NZ voxels of the scan's voxel size",
    )
    _ = p.add_argument(
        "--volume-axes",
        choices=["zyx", "xyz"],
        default=DISK_VOLUME_AXES,
        help="On-disk axis order for saved volumes (default: zyx for viewer convention).",
    )
    _ = p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan as JSON and exit without aligning",
    )


@dataclass(frozen=True, slots=True)
class AlignCommand:
    """Typed command values resolved from the public alignment parser."""

    data: str
    out: str
    mode: AlignmentMode
    quality: str
    outer_iterations: int
    iterations: int
    roi: str
    grid: list[int] | None
    requested_gather_dtype: str
    pose_solver: str
    ray_integrator: str | None
    reconstruction: str
    tv_weight: float
    regulariser: str
    huber_delta: float
    tv_prox_iterations: int
    views_per_batch: int
    seed: int
    nonnegative: bool
    projector_unroll: int
    projector_backend: str
    checkpoint_projector: bool
    mask_vol: str
    pose_translation_frame: str
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
    seed_translations: bool | None
    log_summary: bool
    log_compact: bool
    lipschitz: float | None
    early_stop: bool
    early_stop_rel_impr: float | None
    early_stop_patience: int | None
    optimise_dofs: list[str]
    freeze: list[str]
    schedule: str | None
    dry_run: bool
    checkpoint: str | None
    checkpoint_every: int | None
    resume: str | None
    transfer_guard: str
    save_params_json: str | None
    save_params_csv: str | None
    manifest: str | None
    volume_axes: str


def _pose_solver(args: argparse.Namespace) -> str:
    # As tomojax.alignment.alignment_plan: coupled wherever pose stages follow
    # at most the centre-of-rotation calibration.
    requested = cast("str | None", args.pose_solver)
    if requested is not None:
        return requested
    return "coupled" if cast("str", args.mode) in {"pose", "cor_then_pose"} else "alternating"


def align_command_from_args(args: argparse.Namespace) -> AlignCommand:
    """Snapshot parser/config output into typed alignment command values."""
    return AlignCommand(
        data=cast("str", args.data),
        out=cast("str", args.out),
        mode=cast("AlignmentMode", args.mode),
        quality=cast("str", args.quality),
        outer_iterations=(
            cast("int", args.outer_iterations)
            if cast("int | None", args.outer_iterations) is not None
            else (30 if _pose_solver(args) == "coupled" and cast("str", args.mode) != "cor" else 5)
        ),
        iterations=cast("int", args.iterations),
        roi=cast("str", args.roi),
        grid=cast("list[int] | None", args.grid),
        requested_gather_dtype=cast("str", args.gather_dtype),
        pose_solver=_pose_solver(args),
        ray_integrator=cast("str | None", args.ray_integrator),
        reconstruction=cast("str", args.reconstruction),
        tv_weight=cast("float", args.tv_weight),
        regulariser=cast("str", args.regulariser),
        huber_delta=cast("float", args.huber_delta),
        tv_prox_iterations=cast("int", args.tv_prox_iterations),
        views_per_batch=cast("int", args.views_per_batch),
        seed=cast("int", args.seed),
        nonnegative=cast("bool", args.nonnegative),
        projector_unroll=cast("int", args.projector_unroll),
        projector_backend=cast("str", args.projector_backend),
        checkpoint_projector=cast("bool", args.checkpoint_projector),
        mask_vol=cast("str", args.mask_vol),
        pose_translation_frame=cast("str", args.pose_translation_frame),
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
        seed_translations=cast("bool | None", args.seed_translations),
        log_summary=cast("bool", args.log_summary),
        log_compact=cast("bool", args.log_compact),
        lipschitz=cast("float | None", args.lipschitz),
        early_stop=cast("bool", args.early_stop),
        early_stop_rel_impr=cast("float | None", args.early_stop_rel_impr),
        early_stop_patience=cast("int | None", args.early_stop_patience),
        optimise_dofs=list(cast("list[str] | None", args.optimise_dofs) or []),
        freeze=list(cast("list[str] | None", args.freeze) or []),
        schedule=cast("str | None", args.schedule),
        dry_run=cast("bool", args.dry_run),
        checkpoint=cast("str | None", args.checkpoint),
        checkpoint_every=cast("int | None", args.checkpoint_every),
        resume=cast("str | None", args.resume),
        transfer_guard=cast("str", args.transfer_guard),
        save_params_json=cast("str | None", args.save_params_json),
        save_params_csv=cast("str | None", args.save_params_csv),
        manifest=cast("str | None", args.manifest),
        volume_axes=cast("str", args.volume_axes),
    )
