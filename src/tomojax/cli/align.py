"""``tomojax align``: one geometry fit on a saved scan."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import partial
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, get_args

import numpy as np

from tomojax import progress
from tomojax._parameters import Fit, method_fit
from tomojax._workflow.field_of_view import (
    REGION_CHOICES,
    REGION_HELP,
)
from tomojax.cli._options import (
    EXIT_STATUS,
    add_checkpoint,
    add_config,
    add_force,
    check_paths,
    checkpointed,
)
from tomojax.cli._progress import ProgressParser, add_progress, run_command
from tomojax.cli._watch import watch_directory
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax import Alignment, Scan
    from tomojax._workflow.operations import Region
    from tomojax._workflow.smooth_alignment import CenterSelection, JointSelection
    from tomojax.alignment import AlignConfig
    from tomojax.cli._watch import WatchDirectory
    from tomojax.wire.progress import Result

LOG = logging.getLogger(__name__)


class _Args(argparse.Namespace):
    data: str
    out: str | None
    force: bool
    progress: str
    fit: list[Fit] | None
    method: str | None
    motion: str | None
    iterations: int | None
    image_iterations: int | None
    tolerance: float | None
    interpolation: str | None
    resolution: int | None
    region: Region | None
    grid: list[int] | None
    apply_poses: bool
    manifest: str | None
    checkpoint: str | None
    dry_run: bool
    watch_dir: str | None


# The everyday settings: tj.align's keywords, each a field of AlignConfig.
_SETTINGS = ("motion", "iterations", "image_iterations", "tolerance", "interpolation", "resolution")


def build_parser() -> ProgressParser:
    """The ``tomojax align`` parser, with the same fit names as Python."""
    parser = ProgressParser(
        prog="tomojax align",
        description=(
            "Fit selected geometry parameters jointly with a reconstruction, or estimate "
            "the rotation center from near-opposite views (--method opposite-views). "
            "OUTPUT holds the corrected scan and poses, plus the working image for joint fits."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax align scan.nxs -o aligned.nxs\n"
            "  tomojax align scan.nxs -o aligned.nxs --fit center\n"
            "  tomojax align scan.nxs -o aligned.nxs --fit translations tilt roll\n"
            "  tomojax align scan.nxs -o centered.nxs --method opposite-views\n"
            "  tomojax reconstruct aligned.nxs -o recon.nxs\n\n" + EXIT_STATUS
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument("data", metavar="INPUT", help="Scan to align (.nxs, .h5, .npz)")
    _ = parser.add_argument(
        "-o",
        "--output",
        dest="out",
        default=None,
        metavar="OUTPUT",
        help="Output .nxs: corrected scan and poses, plus working image for joint fits "
        "(optional with --dry-run)",
    )
    add_force(parser)
    add_progress(parser, watch_dir=True)
    _ = parser.add_argument(
        "--fit",
        nargs="+",
        choices=get_args(Fit.__value__),
        default=None,
        metavar="NAME",
        help=(
            "Parameters to fit, the rest held fixed: center, tilt, roll, translations, rotations "
            "(the axis's wobble). Default translations; center for opposite-views"
        ),
    )
    _ = parser.add_argument(
        "--method",
        choices=("joint", "opposite-views"),
        default=None,
        help="By default --fit center uses sinogram focus on dense axial half-turns. "
        "Joint fits an image with the parameters. Opposite-views fits the center from paired rays.",
    )
    _ = parser.add_argument(
        "--motion",
        choices=("independent", "smooth"),
        help="independent (default): each view may differ from its neighbors; smooth: drift "
        "and occasional slips, for a steady stage",
    )
    _ = parser.add_argument(
        "--iterations", type=int, help="Gauss-Newton iterations at the working resolution (8)"
    )
    _ = parser.add_argument(
        "--image-iterations",
        type=int,
        help="Image-solve iterations per Gauss-Newton iteration (16)",
    )
    _ = parser.add_argument(
        "--tolerance",
        type=float,
        help="A level stops when two pose proposals move views less than this, in working "
        "pixels (0.01); 0 runs to the iteration limit",
    )
    _ = parser.add_argument(
        "--interpolation",
        choices=("linear", "cubic"),
        help="The fit's projector interpolation (cubic with an angle on parallel rays, "
        "else linear)",
    )
    _ = parser.add_argument(
        "--resolution", type=int, help="The working detector's longest side, in pixels (320)"
    )
    _ = parser.add_argument("--region", choices=REGION_CHOICES, default=None, help=REGION_HELP)
    _ = parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Resample the selected field of view to NX x NY x NZ voxels",
    )
    _ = parser.add_argument(
        "--apply-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Correct on top of saved poses (--no-apply-poses uses the nominal geometry)",
    )
    _ = parser.add_argument("--manifest", metavar="JSON", help="Also save a JSON record of the run")
    add_checkpoint(parser)
    add_config(parser, settings=_config_settings)
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (levels, grids, estimated memory) and stop; JSON with --progress json",
    )
    return parser


def _config_settings() -> dict[str, object]:
    """A config file's settings: every AlignConfig field the flags do not set."""
    from tomojax.alignment import AlignConfig

    return {k: v for k, v in asdict(AlignConfig()).items() if k not in _SETTINGS}


def _build_align_config(
    args: _Args,
    settings: Mapping[str, object],
) -> AlignConfig:
    """Validate expert TOML settings and construct ``AlignConfig`` before loading data."""
    from tomojax.alignment import AlignConfig

    config = AlignConfig()
    defaults: dict[str, object] = asdict(config)
    coerced: dict[str, object] = {}
    for key, value in settings.items():
        if key == "levels":
            if not isinstance(value, list | tuple) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            ):
                raise ValueError("levels must be a list of positive integers ending at 1")
            coerced[key] = tuple(int(item) for item in value)
            continue
        if key == "image_tv_weight" and value == "coverage":
            coerced[key] = value
            continue
        expected = defaults.get(key)
        if isinstance(expected, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
        elif isinstance(expected, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        elif isinstance(expected, float) and (
            isinstance(value, bool) or not isinstance(value, int | float)
        ):
            raise ValueError(f"{key} must be a number")
        coerced[key] = value
    given: dict[str, object] = vars(args)
    overrides = {name: given[name] for name in _SETTINGS if given[name] is not None}
    return replace(replace(config, **coerced), **overrides)


def _validate_opposite_views(
    args: _Args,
    settings: Mapping[str, object],
) -> None:
    """Reject unsupported --fit and joint-only options for ``--method opposite-views``."""
    if args.fit is not None and args.fit != ["center"]:
        raise ValueError(f"--method opposite-views fits --fit center alone, not {args.fit!r}")
    joint_given = [name for name in _SETTINGS if getattr(args, name) is not None]
    if args.grid is not None:
        joint_given.append("grid")
    if args.region is not None:
        joint_given.append("region")
    joint_given.extend(sorted(settings))
    if joint_given:
        raise ValueError(
            f"--method opposite-views fits no image and takes no joint options "
            f"({', '.join(sorted(set(joint_given)))})"
        )


def _write_manifest(
    args: _Args,
    metadata: Mapping[str, object],
    scan: Scan,
    result: Alignment,
    *,
    fit: tuple[str, ...],
    masked: bool,
    method: str,
) -> None:
    """Save the requested inputs, fitted geometry and solver record."""
    from tomojax.cli.manifest import build_manifest, save_manifest

    payload: dict[str, object] = {
        "input_path": args.data,
        "output_path": args.out,
        "manifest_path": args.manifest,
        "config_path": metadata["config_path"],
        "config_file_values": metadata["config_file_values"],
        "explicit_cli_keys": metadata["explicit_cli_keys"],
        "effective_options": metadata["effective_options"],
        "settings": metadata["settings"],
        "input_projection_shape": list(np.shape(scan.projections)),
        "input_poses": scan.poses is not None,
        "fit": list(fit),
        "method": method,
        "region": {
            "requested": args.region,
            "grid_changed": result.grid is not None and result.grid != scan.grid,
            "cylindrical_output_mask": masked,
        },
        "scan_grid": result.scan.grid.to_dict(),
        "working_grid": None if result.grid is None else result.grid.to_dict(),
        "detector": result.scan.detector.to_dict(),
        "alignment": result.info,
    }
    manifest = build_manifest("tomojax align", list(sys.argv), args, payload)
    save_manifest(str(args.manifest), manifest)
    LOG.info("Saved alignment manifest to %s", args.manifest)


def _validate_align_args(
    parser: ProgressParser,
    args: _Args,
    settings: Mapping[str, object],
) -> AlignConfig | None:
    """Validate CLI and TOML arguments before loading scan data."""
    if args.fit is not None and len(args.fit) != len(set(args.fit)):
        parser.error("--fit names must not repeat")
    if not args.dry_run and not args.out:
        parser.error("the following arguments are required: -o/--output")
    if args.grid is not None and min(args.grid) < 1:
        parser.error("--grid dimensions must be positive integers")
    try:
        if args.method == "opposite-views":
            _validate_opposite_views(args, settings)
            return None
        if not settings and all(getattr(args, name) is None for name in _SETTINGS):
            return None
        return _build_align_config(args, settings)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def _dry_run_plan(
    args: _Args,
    fit: tuple[str, ...],
    scan: Scan,
    work: JointSelection | CenterSelection,
    plan: object,
) -> dict[str, object]:
    """Return the JSON document of ``tomojax align --dry-run``."""
    import msgspec

    from tomojax._workflow.smooth_alignment import JointSelection

    document: dict[str, object] = {
        **msgspec.to_builtins(plan),
        "input_path": args.data,
        "output_path": args.out,
        "fit": list(fit),
        "method": "joint" if isinstance(work, JointSelection) else work.method,
    }
    if isinstance(work, JointSelection):
        document["config"] = asdict(work.config)
        document["grid"] = (scan.grid if work.grid is None else work.grid).to_dict()
        document["bounded_working_grid"] = work.grid is None
    return document


_STOP_LABELS = {
    "step_budget_reached": "step limit reached",
    "line_search_stalled": "no acceptable update",
    "pose_steps_below_tolerance": "pose updates below tolerance",
    "stationary": "stationary poses",
}


def _fit_summary(result: Alignment) -> list[str]:
    """Format fitted corrections and stopping reasons for terminal output."""
    report = result.info.get("report")
    report = report if isinstance(report, dict) else {}
    poses = report.get("poses", {})
    lines = []
    for unit, title in (("native px", "Per-view shifts"), ("deg", "Per-view rotations")):
        stats = [
            f"{axis} {values['rms']:.3g}/{values['max']:.3g}"
            for axis, values in poses.items()
            if values["unit"] == unit
        ]
        if stats:
            lines.append(f"{title} (RMS/max, {unit})  {', '.join(stats)}")
    calibration = result.info.get("calibration", {})
    if isinstance(calibration, dict) and calibration:
        changes = []
        for axis, change in calibration.items():
            values = np.atleast_1d(change["change"])
            size = ", ".join(f"{value:.3g}" for value in values)
            unit = "native px" if axis == "center" else "deg"
            changes.append(f"{axis} {size} {unit}")
        lines.append(f"Shared corrections  {', '.join(changes)}")
    levels = report.get("levels", [])
    stages = []
    for i, level in enumerate(levels, start=1):
        n = level["iterations"]
        count = "1 step" if n == 1 else f"{n} steps"
        stages.append(f"level {i} {_STOP_LABELS[level['ended']]} ({count})")
    refinement = report.get("refinement")
    if refinement in _STOP_LABELS:
        stages.append(f"refinement {_STOP_LABELS[refinement]}")
    if stages:
        lines.append(f"Stopped  {', '.join(stages)}")
    return lines


def main() -> int:
    """Run ``tomojax align``."""
    parser = build_parser()
    args, metadata = parse_args_with_config(parser, namespace=_Args())
    settings = metadata["settings"]
    config = _validate_align_args(parser, args, settings)
    watch = watch_directory(args.watch_dir, working=True)
    work = partial(_run, parser, args, metadata, config, watch)
    operation = partial(checkpointed, None if args.dry_run else args.checkpoint, work)
    return run_command("align", args.progress, operation, watch=watch)


def _run(
    parser: ProgressParser,
    args: _Args,
    metadata: Mapping[str, object],
    config: AlignConfig | None,
    watch: WatchDirectory | None,
) -> Result | None:
    import tomojax as tj

    if not Path(args.data).exists():
        raise FileNotFoundError(f"input not found: {args.data}")
    outputs = [] if args.dry_run else [args.out, args.manifest]
    check_paths(parser, inputs=[args.data], outputs=outputs, force=args.force)
    progress.check_canceled()
    scan = tj.load(args.data, apply_poses=args.apply_poses)
    progress.check_canceled()
    from tomojax._workflow.operations import align_resolved, alignment_task
    from tomojax._workflow.smooth_alignment import JointSelection, resolve_alignment
    from tomojax.cli._plans import announce, dry_run, plan_product

    fit = method_fit(args.method, None if args.fit is None else tuple(args.fit))
    shape = None if args.grid is None else (args.grid[2], args.grid[1], args.grid[0])
    with alignment_task(args.method, fit):
        work = resolve_alignment(
            scan, fit=fit, method=args.method, region=args.region, shape=shape, config=config
        )
        method = "joint" if isinstance(work, JointSelection) else work.method
        if args.dry_run:
            plan = plan_product(scan, work)
            document = _dry_run_plan(args, fit, scan, work, plan)
            return dry_run(args.progress, args.data, plan, scan, work, document=document)
        used = tuple(f for f, v in (("--region", args.region), ("--grid", args.grid)) if v)
        _ = announce("align", scan, work, used=used)
        with progress.previews() if watch is not None else nullcontext():
            result = align_resolved(scan, work, checkpoint=args.checkpoint)
    progress.check_canceled()
    assert args.out is not None
    tj.save(args.out, result)
    if args.manifest is not None:
        progress.check_canceled()
        masked = isinstance(work, JointSelection) and work.cylinder is not None
        _write_manifest(args, metadata, scan, result, fit=fit, masked=masked, method=method)
    if args.progress == "json":
        from tomojax.cli._results import saved_result

        outputs = tuple(p for p in (args.out, args.manifest) if p is not None)
        return saved_result(args.out, result, method=method, outputs=outputs)
    print(f"wrote {args.out} with fitted {', '.join(fit)} for {len(scan.angles)} views")
    for line in _fit_summary(result):
        print(line)
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
