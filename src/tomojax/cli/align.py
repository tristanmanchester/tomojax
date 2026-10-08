"""``tomojax align``: :func:`tomojax.align` on the command line.

The command loads INPUT with its saved poses (unless ``--no-poses``), so it
corrects on top of them as Python does; chooses the grid (``--roi``,
``--grid``); applies a ``--config`` file's expert settings, which are
:class:`~tomojax.alignment.AlignConfig` fields, to the configuration the mode
and quality resolve to; aligns with :func:`tomojax.align`; and saves the
result with :func:`tomojax.save`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import json
import logging
import os
import sys
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.cli._options import add_config, add_output, check_paths, hide_expert
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax import Alignment, Scan
    from tomojax.alignment import AlignConfig, AlignmentPlan
    from tomojax.alignment.api import AlignmentLossSpec, AlignmentMode, QualityTier
    from tomojax.geometry import Grid

_PUBLIC = (
    "--mode",
    "--quality",
    "--levels",
    "--freeze",
    "--roi",
    "--grid",
    "--checkpoint",
    "--poses",
    "--manifest",
    "--progress",
    "--dry-run",
)


_EPILOG = (
    "Modes:\n"
    "  pose           per-view motion: rotations and translations (default)\n"
    "  cor            setup geometry only: detector centre (parallel), or the\n"
    "                 axis offset and detector roll (cone beam)\n"
    "  cor-then-pose  setup geometry, then per-view motion\n"
    "  full           detector centre, roll and axis direction, then motion,\n"
    "                 coarse to fine\n\n"
    "Expert settings are tomojax.alignment.AlignConfig fields. Each replaces that\n"
    "field of the configuration the mode and quality give, which --dry-run prints;\n"
    "--config-keys shows the default mode's values.\n\n"
    "Examples:\n"
    "  tomojax align scan.nxs -o aligned.nxs\n"
    "  tomojax align scan.nxs -o aligned.nxs --mode cor-then-pose --freeze dy\n"
    "  tomojax recon aligned.nxs -o recon.nxs --method cgls"
)


def build_parser() -> argparse.ArgumentParser:
    """The ``tomojax align`` parser; its expert settings are ``AlignConfig``'s fields."""
    from tomojax.alignment import MODES

    p = argparse.ArgumentParser(
        prog="tomojax align",
        description=(
            "Estimate a scan's geometry corrections (setup geometry and per-view motion) "
            "and reconstruct with them, on top of any poses saved in INPUT. OUTPUT holds "
            "the corrected scan, its poses and the volume; tomojax recon applies them."
        ),
        epilog=_EPILOG,
    )
    _ = p.add_argument("data", metavar="INPUT", help="Scan to align (.nxs, .h5, .npz)")
    add_output(p, "Output .nxs: the corrected scan, its per-view poses and the volume")
    add_config(p, settings=_settings())
    _ = p.add_argument(
        "--mode",
        choices=MODES,
        default="pose",
        help="What to estimate (default pose; see Modes below)",
    )
    _ = p.add_argument(
        "--quality",
        choices=["fast", "reference"],
        default="fast",
        help="fast (default) or reference: slower, more conservative solver settings",
    )
    _ = p.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        metavar="FACTOR",
        help="Coarse-to-fine binning factors, for example 4 2 1 (default: by mode and grid)",
    )
    _ = p.add_argument(
        "--freeze",
        nargs="+",
        default=[],
        metavar="DOF",
        help=(
            "Parameters to keep fixed: alpha, beta, phi, dx, dz, dy (along a cone beam), "
            "or setup ones such as det_u_px"
        ),
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
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Reconstruct NX x NY x NZ voxels of the scan's voxel size",
    )
    _ = p.add_argument(
        "--checkpoint",
        metavar="PATH",
        default=None,
        help=(
            "Save progress to PATH after each outer iteration; a checkpoint of this "
            "alignment there is resumed"
        ),
    )
    _ = p.add_argument(
        "--poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Correct on top of the poses saved in INPUT (--no-poses: the nominal geometry)",
    )
    _ = p.add_argument(
        "--manifest",
        metavar="JSON",
        default=None,
        help="Also write a JSON record of the run (inputs, settings, versions)",
    )
    _ = p.add_argument(
        "--progress", action="store_true", help="Show progress bars if tqdm is available"
    )
    _ = p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan and grid as JSON and exit without aligning",
    )
    hide_expert(p, _PUBLIC)
    return p


def _settings() -> dict[str, object]:
    """A config file's expert settings: ``AlignConfig``'s fields, as the default mode sets them."""
    from tomojax.alignment import AlignConfig, alignment_plan
    from tomojax.alignment.api import loss_spec_name
    from tomojax.geometry import Grid

    # The mode's configuration does not depend on the grid, only its levels do.
    defaults = alignment_plan("pose", Grid(1, 1, 1, 1.0, 1.0, 1.0)).config
    settings = {f.name: getattr(defaults, f.name) for f in fields(AlignConfig) if f.init}
    settings["loss"] = loss_spec_name(cast("AlignmentLossSpec", defaults.loss))
    return settings


def _loss(value: object) -> object:
    """A ``loss`` setting as ``AlignConfig`` takes it.

    A loss name (``"huber"``), a table of a name and its parameters
    (``{ name = "huber", delta = 1.0 }``), or a level schedule
    (``"4:phasecorr,2:ssim"``, l2_otsu at other levels).
    """
    from tomojax.alignment.api import parse_loss_schedule, parse_loss_spec

    if isinstance(value, str):
        if ":" in value:
            return parse_loss_schedule(value, default=parse_loss_spec("l2_otsu"))
        return parse_loss_spec(value)
    if isinstance(value, dict):
        params = dict(cast("dict[str, object]", value))
        name = params.pop("name", None)
        if isinstance(name, str):
            return parse_loss_spec(name, cast("dict[str, float]", params))
    raise ValueError(
        'loss must be a name, a table such as { name = "huber", delta = 1.0 }, or "LEVEL:LOSS,..."'
    )


def _config(base: AlignConfig, settings: Mapping[str, object]) -> AlignConfig | None:
    """``base`` with the config file's expert ``settings``; None without any."""
    if not settings:
        return None
    values = dict(settings)
    if "loss" in values:
        values["loss"] = _loss(values["loss"])
    if "optimise_dofs" in values:
        # A parameter list replaces the mode's schedule; the two exclude each other.
        _ = values.setdefault("schedule", None)
    return replace(base, **values)


@dataclass(frozen=True)
class _Run:
    """The command's arguments."""

    data: str
    out: str
    mode: AlignmentMode
    quality: QualityTier
    levels: tuple[int, ...] | None
    freeze: tuple[str, ...]
    roi: str
    grid: tuple[int, int, int] | None
    checkpoint: str | None
    poses: bool
    manifest: str | None
    dry_run: bool

    @classmethod
    def of(cls, args: argparse.Namespace) -> _Run:
        """The typed values of ``args``."""
        levels = cast("list[int] | None", args.levels)
        grid = cast("list[int] | None", args.grid)
        return cls(
            data=cast("str", args.data),
            out=cast("str", args.out),
            mode=cast("AlignmentMode", args.mode),
            quality=cast("QualityTier", args.quality),
            levels=None if levels is None else tuple(levels),
            freeze=tuple(cast("list[str]", args.freeze)),
            roi=cast("str", args.roi),
            grid=None if grid is None else (grid[0], grid[1], grid[2]),
            checkpoint=cast("str | None", args.checkpoint),
            poses=cast("bool", args.poses),
            manifest=cast("str | None", args.manifest),
            dry_run=cast("bool", args.dry_run),
        )

    def plan(self, grid: Grid, config: AlignConfig | None = None) -> AlignmentPlan:
        """The plan of this run's mode, quality, levels and frozen parameters on ``grid``."""
        from tomojax.alignment import alignment_plan

        return alignment_plan(
            self.mode,
            grid,
            quality=self.quality,
            levels=self.levels,
            freeze=self.freeze,
            config=config,
        )


def _grid(scan: Scan, run: _Run) -> tuple[Grid, bool]:
    """The grid ``--roi`` and ``--grid`` choose, and whether to zero outside the cylinder."""
    from tomojax.cli._reconstruction_region import resolve_reconstruction_region

    region = resolve_reconstruction_region(
        scan.grid,
        scan.detector,
        geometry_type="parallel" if scan.source is None else scan.source.geometry_type,
        roi_mode=run.roi,
        grid_override=run.grid,
    )
    return region.recon_grid, region.apply_output_mask


def _plan_payload(run: _Run, plan: AlignmentPlan, grid: Grid) -> str:
    """The resolved plan and grid, as ``--dry-run`` prints them."""
    from tomojax.alignment.api import resolved_schedule_for_config
    from tomojax.io.api import normalize_json

    payload: dict[str, object] = {
        "input_path": run.data,
        "output_path": run.out,
        "mode": plan.mode,
        "quality": run.quality,
        "pose_solver": plan.pose_solver,
        "levels": list(plan.levels),
        "roi": run.roi,
        "grid": grid.to_dict(),
        "schedule": resolved_schedule_for_config(plan.config).to_dict(),
        "loss": repr(plan.config.loss),
        "config": plan.config,
    }
    return json.dumps(normalize_json(payload), indent=2, sort_keys=True)


def _write_manifest(
    args: argparse.Namespace,
    metadata: Mapping[str, object],
    scan: Scan,
    result: Alignment,
    *,
    masked: bool,
) -> None:
    """Write ``--manifest``: the run's inputs and settings and the alignment's record."""
    from tomojax.cli.manifest import build_manifest, save_manifest

    run = _Run.of(args)
    config = result.info.get("config")
    payload: dict[str, object] = {
        "input_path": run.data,
        "output_path": run.out,
        "manifest_path": run.manifest,
        "checkpoint_path": run.checkpoint,
        "config_path": metadata["config_path"],
        "config_file_values": metadata["config_file_values"],
        "explicit_cli_keys": metadata["explicit_cli_keys"],
        "effective_options": metadata["effective_options"],
        "settings": metadata["settings"],
        "input_projection_shape": list(np.shape(scan.projections)),
        "input_poses": scan.poses is not None,
        "roi": {
            "requested": run.roi,
            "grid_changed": result.scan.grid != scan.grid,
            "cylindrical_output_mask": masked,
        },
        "reconstruction_grid": result.scan.grid.to_dict(),
        "detector": result.scan.detector.to_dict(),
        "loss": None if config is None else repr(cast("AlignConfig", config).loss),
        "poses_shape": list(result.poses.shape),
        "volume_shape": list(np.shape(result.volume)),
        # Mode, levels, configuration, losses, gauge and any calibrated geometry.
        "alignment": result.info,
    }
    manifest = build_manifest("tomojax align", list(sys.argv), args, payload)
    save_manifest(str(run.manifest), manifest)
    logging.info("Saved reproducibility manifest to %s", run.manifest)


def main() -> None:
    """Run ``tomojax align``."""
    import tomojax as tj
    from tomojax.core import log_jax_env, setup_logging
    from tomojax.geometry import cylindrical_mask_xy

    parser = build_parser()
    args, metadata = parse_args_with_config(parser)
    run = _Run.of(args)
    check_paths(parser, inputs=[run.data], outputs=[run.out], force=cast("bool", args.force))
    setup_logging()
    log_jax_env()
    if cast("bool", args.progress):
        os.environ["TOMOJAX_PROGRESS"] = "1"

    scan = tj.load(run.data, poses=run.poses)
    grid, masked = _grid(scan, run)
    try:
        config = _config(run.plan(grid).config, cast("dict[str, object]", metadata["settings"]))
        plan = run.plan(grid, config)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if run.dry_run:
        print(_plan_payload(run, plan, grid))
        return

    result = tj.align(
        scan,
        mode=run.mode,
        quality=run.quality,
        levels=run.levels,
        freeze=run.freeze,
        grid=grid,
        checkpoint=run.checkpoint,
        config=config,
    )
    if masked:
        inside = np.asarray(cylindrical_mask_xy(grid, scan.detector), np.float32)[:, :, None]
        result = replace(result, volume=np.asarray(result.volume) * inside)
    implied = result.info.get("implied_detector_u_px")
    if implied is not None:
        logging.info("The poses hold a detector-u (centre-of-rotation) offset of %.3f px", implied)
    tj.save(run.out, result)
    logging.info("Saved alignment results to %s", run.out)
    if run.manifest is not None:
        _write_manifest(args, metadata, scan, result, masked=masked)


__all__ = ["build_parser", "main"]
