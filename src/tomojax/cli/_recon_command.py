"""Command parsing contracts for the reconstruction CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from tomojax.cli.config import ConfigValue, parse_args_with_config
from tomojax.geometry import DISK_VOLUME_AXES

if TYPE_CHECKING:
    from collections.abc import Sequence

type ViewsPerBatch = int | Literal["auto"]
type ReconTransferGuardMode = Literal["off", "log", "disallow"]
type ReconAlgorithm = Literal["fbp", "fista", "spdhg"]
type ReconRoiMode = Literal["off", "auto", "cube", "bbox"]
type ReconMaskMode = Literal["off", "cyl"]
type ReconFrame = Literal["sample", "lab"]
type ReconVolumeAxes = Literal["zyx", "xyz"]
type ReconRegulariser = Literal["tv", "huber_tv"]
type ReconWarmStart = Literal["none", "fbp"]


@dataclass(frozen=True)
class ReconCommand:
    """Typed command plan for the public reconstruction workflow."""

    config: str | None
    data: str
    out: str
    algo: ReconAlgorithm
    filter: str
    iters: int
    lambda_tv: float
    regulariser: ReconRegulariser
    huber_delta: float
    tv_prox_iters: int
    lipschitz: float | None
    positivity: bool
    lower_bound: float | None
    upper_bound: float | None
    views_per_batch: ViewsPerBatch | None
    theta: float
    spdhg_seed: int
    spdhg_tau: float | None
    spdhg_sigma_data: float | None
    spdhg_sigma_tv: float | None
    warm_start: ReconWarmStart
    gather_dtype: str
    checkpoint_projector: bool
    quicklook: str | None
    save_manifest: str | None
    roi: ReconRoiMode
    grid: tuple[int, int, int] | None
    frame: ReconFrame
    volume_axes: ReconVolumeAxes
    progress: bool
    transfer_guard: ReconTransferGuardMode
    mask_vol: ReconMaskMode
    apply_saved_alignment: bool
    det_u_px: float | None
    det_v_px: float | None


def _parse_views_per_batch(value: str) -> int | str:
    """Parse ``--views-per-batch`` as a positive/zero integer or ``auto``."""
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--views-per-batch must be 'auto' or an integer") from exc


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive float") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive float")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a finite float") from exc
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite float")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reconstruct volume from dataset (.nxs)")
    _add_input_options(p)
    _add_algorithm_options(p)
    _add_iterative_options(p)
    _add_spdhg_options(p)
    _add_output_options(p)
    _add_geometry_options(p)
    _add_runtime_options(p)
    return p


def _add_input_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument("--config", help="Load command defaults from a TOML config file")
    _ = p.add_argument("--data", help="Input .nxs")


def _add_algorithm_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument("--algo", choices=["fbp", "fista", "spdhg"], default="fbp")
    _ = p.add_argument("--filter", default="ramp", help="FBP filter: ramp|shepp|hann")


def _add_iterative_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Iterations for iterative algos (FISTA/SPDHG)",
    )
    _ = p.add_argument(
        "--lambda-tv",
        type=float,
        default=0.005,
        help="TV regularization weight (FISTA/SPDHG)",
    )
    _ = p.add_argument(
        "--regulariser",
        choices=["tv", "huber_tv"],
        default="tv",
        help="Regulariser for iterative algos: tv (default) or huber_tv",
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
        help="Inner iterations for TV proximal operator (FISTA)",
    )
    _ = p.add_argument(
        "--L",
        type=float,
        default=None,
        help="Fixed Lipschitz constant for FISTA (skip power-method)",
    )
    pos = p.add_mutually_exclusive_group()
    _ = pos.add_argument(
        "--positivity",
        dest="positivity",
        action="store_true",
        help="Enable nonnegative projection for FISTA reconstructions",
    )
    _ = pos.add_argument(
        "--no-positivity",
        dest="positivity",
        action="store_false",
        help="Disable nonnegative projection for FISTA reconstructions",
    )
    p.set_defaults(positivity=False)
    _ = p.add_argument(
        "--lower-bound",
        type=float,
        default=None,
        help="Optional lower voxel bound for FISTA reconstructions",
    )
    _ = p.add_argument(
        "--upper-bound",
        type=float,
        default=None,
        help="Optional upper voxel bound for FISTA reconstructions",
    )


def _add_spdhg_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--views-per-batch",
        type=_parse_views_per_batch,
        default=None,
        help=(
            "Views per projection batch, or 'auto' to estimate from available memory "
            "(default: 1 for FBP/FISTA, 16 for SPDHG)"
        ),
    )
    _ = p.add_argument("--theta", type=float, default=1.0, help="SPDHG: extrapolation for xbar")
    _ = p.add_argument("--spdhg-seed", type=int, default=0, help="SPDHG: RNG seed for block order")
    _ = p.add_argument(
        "--spdhg-tau",
        type=float,
        default=None,
        help="SPDHG: override primal step size (auto if None)",
    )
    _ = p.add_argument(
        "--spdhg-sigma-data",
        type=float,
        default=None,
        help="SPDHG: override data dual step (auto if None)",
    )
    _ = p.add_argument(
        "--spdhg-sigma-tv",
        type=float,
        default=None,
        help="SPDHG: override TV dual step (auto if None)",
    )
    _ = p.add_argument(
        "--warm-start",
        choices=["none", "fbp"],
        default="none",
        help="Initialize iterative algo from this method (spdhg only): none|fbp",
    )


def _add_output_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument("--out", help="Output .nxs containing recon (and copying projections)")
    _ = p.add_argument(
        "--quicklook",
        "--save-preview",
        dest="quicklook",
        metavar="PATH",
        default=None,
        help="Write a percentile-scaled central xy slice PNG preview to PATH.",
    )
    _ = p.add_argument(
        "--save-manifest",
        metavar="PATH",
        default=None,
        help="Write a JSON reproducibility manifest for this reconstruction run.",
    )


def _add_geometry_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--roi",
        choices=["off", "auto", "cube", "bbox"],
        default="auto",
        help=(
            "Optional ROI cropping based on detector FOV (default: auto). "
            "auto: square x-y slices + z from detector height if detector < grid; "
            "cube: force cubic ROI (nx=ny=nz) inside FOV; "
            "bbox: rectangular FOV bbox; off: keep original grid"
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
        "--frame",
        choices=["sample", "lab"],
        default="sample",
        help="Frame to record for saved volume (default: sample).",
    )
    _ = p.add_argument(
        "--volume-axes",
        choices=["zyx", "xyz"],
        default=DISK_VOLUME_AXES,
        help="On-disk axis order for saved volumes (default: zyx for viewer convention).",
    )
    _ = p.add_argument(
        "--det-u-px",
        type=_finite_float,
        default=None,
        help="Override detector centre u offset in detector pixels for COR sweeps.",
    )
    _ = p.add_argument(
        "--det-v-px",
        type=_finite_float,
        default=None,
        help="Override detector centre v offset in detector pixels for COR sweeps.",
    )
    saved = p.add_mutually_exclusive_group()
    _ = saved.add_argument(
        "--apply-saved-alignment",
        dest="apply_saved_alignment",
        action="store_true",
        help="Apply saved per-view alignment parameters from the input metadata.",
    )
    _ = saved.add_argument(
        "--ignore-saved-alignment",
        dest="apply_saved_alignment",
        action="store_false",
        help="Ignore saved per-view alignment parameters from the input metadata (default).",
    )
    p.set_defaults(apply_saved_alignment=False)


def _add_runtime_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--gather-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Projector gather dtype (auto: bf16 on GPU/TPU, else fp32; accumulation stays fp32)",
    )
    ck = p.add_mutually_exclusive_group()
    _ = ck.add_argument(
        "--checkpoint-projector",
        dest="checkpoint_projector",
        action="store_true",
        help="Enable projector checkpointing",
    )
    _ = ck.add_argument(
        "--no-checkpoint-projector",
        dest="checkpoint_projector",
        action="store_false",
        help="Disable projector checkpointing",
    )
    p.set_defaults(checkpoint_projector=True)
    _ = p.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars if tqdm is available",
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
    _ = p.add_argument(
        "--mask-vol",
        choices=["off", "cyl"],
        default="off",
        help=(
            "Mask the volume during forward projection (FISTA) or on output (FBP): "
            "off (default), cyl for cylindrical x-y mask broadcast along z."
        ),
    )


def parse_recon_command(
    argv: Sequence[str] | None = None,
) -> tuple[ReconCommand, dict[str, ConfigValue]]:
    """Parse CLI/config defaults into a typed reconstruction command plan."""
    parser = _build_parser()
    args, config_metadata = parse_args_with_config(parser, argv, required=("data", "out"))
    grid = cast("list[int] | None", args.grid)
    return (
        ReconCommand(
            config=cast("str | None", args.config),
            data=cast("str", args.data),
            out=cast("str", args.out),
            algo=cast("ReconAlgorithm", args.algo),
            filter=cast("str", args.filter),
            iters=cast("int", args.iters),
            lambda_tv=cast("float", args.lambda_tv),
            regulariser=cast("ReconRegulariser", args.regulariser),
            huber_delta=cast("float", args.huber_delta),
            tv_prox_iters=cast("int", args.tv_prox_iters),
            lipschitz=cast("float | None", args.L),
            positivity=cast("bool", args.positivity),
            lower_bound=cast("float | None", args.lower_bound),
            upper_bound=cast("float | None", args.upper_bound),
            views_per_batch=cast("ViewsPerBatch | None", args.views_per_batch),
            theta=cast("float", args.theta),
            spdhg_seed=cast("int", args.spdhg_seed),
            spdhg_tau=cast("float | None", args.spdhg_tau),
            spdhg_sigma_data=cast("float | None", args.spdhg_sigma_data),
            spdhg_sigma_tv=cast("float | None", args.spdhg_sigma_tv),
            warm_start=cast("ReconWarmStart", args.warm_start),
            gather_dtype=cast("str", args.gather_dtype),
            checkpoint_projector=cast("bool", args.checkpoint_projector),
            quicklook=cast("str | None", args.quicklook),
            save_manifest=cast("str | None", args.save_manifest),
            roi=cast("ReconRoiMode", args.roi),
            grid=None if grid is None else (int(grid[0]), int(grid[1]), int(grid[2])),
            frame=cast("ReconFrame", args.frame),
            volume_axes=cast("ReconVolumeAxes", args.volume_axes),
            progress=cast("bool", args.progress),
            transfer_guard=cast("ReconTransferGuardMode", args.transfer_guard),
            mask_vol=cast("ReconMaskMode", args.mask_vol),
            apply_saved_alignment=cast("bool", args.apply_saved_alignment),
            det_u_px=cast("float | None", args.det_u_px),
            det_v_px=cast("float | None", args.det_v_px),
        ),
        config_metadata,
    )


__all__ = ["ReconCommand", "parse_recon_command"]
