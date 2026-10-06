"""Command parsing contracts for the reconstruction CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from tomojax.cli._options import add_config, add_output, check_paths, hide_expert
from tomojax.cli.config import ConfigValue, parse_args_with_config
from tomojax.geometry.api import DISK_VOLUME_AXES

if TYPE_CHECKING:
    from collections.abc import Sequence

type ViewsPerBatch = int | Literal["auto"]
type ReconTransferGuardMode = Literal["off", "log", "disallow"]
type ReconAlgorithm = Literal["fbp", "cgls", "fista", "spdhg"]
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


_PUBLIC = (
    "--method",
    "--filter",
    "--iterations",
    "--tv-weight",
    "--nonnegative",
    "--warm-start",
    "--seed",
    "--grid",
    "--roi",
    "--mask",
    "--ignore-alignment",
    "--preview",
    "--manifest",
    "--volume-axes",
    "--progress",
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tomojax recon",
        description=(
            "Reconstruct a volume from a scan. A saved alignment (from tomojax align) "
            "is applied unless --ignore-alignment."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax recon scan.nxs -o recon.nxs\n"
            "  tomojax recon scan.nxs -o recon.nxs --method cgls --iterations 100\n"
            "  tomojax recon scan.nxs -o recon.nxs --method fista --tv-weight 0.002 --nonnegative"
        ),
    )
    _add_input_options(p)
    _add_algorithm_options(p)
    _add_iterative_options(p)
    _add_spdhg_options(p)
    _add_output_options(p)
    _add_geometry_options(p)
    _add_runtime_options(p)
    hide_expert(p, _PUBLIC)
    return p


def _add_input_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument("data", metavar="INPUT", help="Scan to reconstruct (.nxs, .h5, .npz)")
    add_output(p, "Output .nxs: the volume, with the scan's projections and geometry")
    add_config(p)


def _add_algorithm_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--method",
        choices=["fbp", "cgls", "fista", "spdhg"],
        default="fbp",
        help=(
            "fbp: filtered backprojection, FDK for cone beams (default); cgls: least "
            "squares; fista, spdhg: least squares with total variation"
        ),
    )
    _ = p.add_argument(
        "--filter",
        choices=["ramp", "shepp-logan", "hann"],
        default="ramp",
        help="FBP filter (default ramp)",
    )


def _add_iterative_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--iterations",
        type=int,
        default=50,
        metavar="N",
        help="Iterations of cgls, fista and spdhg (default 50)",
    )
    _ = p.add_argument(
        "--tv-weight",
        type=float,
        default=0.005,
        metavar="W",
        help="Total-variation weight of fista and spdhg (default 0.005)",
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
    _ = p.add_argument(
        "--nonnegative",
        action="store_true",
        help="Keep fista and spdhg voxels nonnegative",
    )
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
    _ = p.add_argument(
        "--seed", type=int, default=0, metavar="N", help="Random seed of spdhg's view order"
    )
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
        action="store_true",
        help="Start fista and spdhg from the FBP reconstruction",
    )


def _add_output_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--preview",
        metavar="PNG",
        default=None,
        help="Also write a PNG of the central x-y slice",
    )
    _ = p.add_argument(
        "--manifest",
        metavar="JSON",
        default=None,
        help="Also write a JSON record of the run (inputs, settings, versions)",
    )


def _add_geometry_options(p: argparse.ArgumentParser) -> None:
    _ = p.add_argument(
        "--roi",
        choices=["off", "auto", "cube", "bbox"],
        default="auto",
        help=(
            "Crop the grid to the detector's field of view: auto (default, when the "
            "detector sees less than the grid), cube, bbox, or off"
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
    _ = p.add_argument(
        "--ignore-alignment",
        dest="apply_saved_alignment",
        action="store_false",
        help="Use the nominal geometry, not the alignment saved in INPUT",
    )


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
        "--mask",
        choices=["off", "cyl"],
        default="off",
        help="cyl: zero the volume outside the cylinder every view sees (default off)",
    )


def parse_recon_command(
    argv: Sequence[str] | None = None,
) -> tuple[ReconCommand, dict[str, ConfigValue]]:
    """Parse CLI/config defaults into a typed reconstruction command plan."""
    parser = _build_parser()
    args, config_metadata = parse_args_with_config(parser, argv)
    check_paths(
        parser,
        inputs=[cast("str", args.data)],
        outputs=[cast("str", args.out)],
        force=bool(cast("bool", args.force)),
    )
    grid = cast("list[int] | None", args.grid)
    return (
        ReconCommand(
            config=cast("str | None", args.config),
            data=cast("str", args.data),
            out=cast("str", args.out),
            algo=cast("ReconAlgorithm", args.method),
            filter=cast("str", args.filter),
            iters=cast("int", args.iterations),
            lambda_tv=cast("float", args.tv_weight),
            regulariser=cast("ReconRegulariser", args.regulariser),
            huber_delta=cast("float", args.huber_delta),
            tv_prox_iters=cast("int", args.tv_prox_iters),
            lipschitz=cast("float | None", args.L),
            positivity=cast("bool", args.nonnegative),
            lower_bound=cast("float | None", args.lower_bound),
            upper_bound=cast("float | None", args.upper_bound),
            views_per_batch=cast("ViewsPerBatch | None", args.views_per_batch),
            theta=cast("float", args.theta),
            spdhg_seed=cast("int", args.seed),
            spdhg_tau=cast("float | None", args.spdhg_tau),
            spdhg_sigma_data=cast("float | None", args.spdhg_sigma_data),
            spdhg_sigma_tv=cast("float | None", args.spdhg_sigma_tv),
            warm_start="fbp" if cast("bool", args.warm_start) else "none",
            gather_dtype=cast("str", args.gather_dtype),
            checkpoint_projector=cast("bool", args.checkpoint_projector),
            quicklook=cast("str | None", args.preview),
            save_manifest=cast("str | None", args.manifest),
            roi=cast("ReconRoiMode", args.roi),
            grid=None if grid is None else (int(grid[0]), int(grid[1]), int(grid[2])),
            frame=cast("ReconFrame", args.frame),
            volume_axes=cast("ReconVolumeAxes", args.volume_axes),
            progress=cast("bool", args.progress),
            transfer_guard=cast("ReconTransferGuardMode", args.transfer_guard),
            mask_vol=cast("ReconMaskMode", args.mask),
            apply_saved_alignment=cast("bool", args.apply_saved_alignment),
            det_u_px=cast("float | None", args.det_u_px),
            det_v_px=cast("float | None", args.det_v_px),
        ),
        config_metadata,
    )


__all__ = ["ReconCommand", "parse_recon_command"]
