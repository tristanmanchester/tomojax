"""``tomojax align``: one geometry fit on a saved scan."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import logging
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax._field_of_view import (
    REGION_CHOICES,
    REGION_HELP,
    region_grid,
    zero_outside_cylinder,
)
from tomojax.cli._options import EXIT_STATUS, add_config, add_force, check_paths
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax import Alignment, Grid, Scan
    from tomojax._workflow import Region
    from tomojax.alignment import AlignConfig


class _Args(argparse.Namespace):
    data: str
    out: str | None
    force: bool
    fit: list[str]
    method: str
    motion: str | None
    iterations: int | None
    image_iterations: int | None
    tolerance: float | None
    interpolation: str | None
    resolution: int | None
    region: str
    grid: list[int] | None
    apply_poses: bool
    manifest: str | None
    dry_run: bool


# The everyday settings: tj.align's keywords, each a field of AlignConfig.
_SETTINGS = ("motion", "iterations", "image_iterations", "tolerance", "interpolation", "resolution")


def build_parser() -> argparse.ArgumentParser:
    """The ``tomojax align`` parser, with the same fit names as Python."""
    parser = argparse.ArgumentParser(
        prog="tomojax align",
        description=(
            "Fit selected geometry parameters jointly with a reconstruction, or estimate "
            "the rotation centre from near-opposite views (--method opposite-views). "
            "OUTPUT holds the corrected scan and poses, plus the working image for joint fits."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax align scan.nxs -o aligned.nxs\n"
            "  tomojax align scan.nxs -o aligned.nxs --fit center\n"
            "  tomojax align scan.nxs -o aligned.nxs --fit translations tilt roll\n"
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
    _ = parser.add_argument(
        "--fit",
        nargs="+",
        choices=("center", "tilt", "roll", "translations", "rotations"),
        default=["translations"],
        metavar="NAME",
        help=(
            "Parameters to fit: center, tilt, roll, translations, rotations (the axis's wobble; "
            "view angles as recorded) (default translations). All other parameters stay fixed."
        ),
    )
    _ = parser.add_argument(
        "--method",
        choices=("joint", "opposite-views"),
        default="joint",
        help="joint (default): fit an image with the parameters; opposite-views: --fit "
        "center alone from near-opposite views of an axial parallel scan, without an image",
    )
    _ = parser.add_argument(
        "--motion",
        choices=("independent", "smooth"),
        help="independent (default): each view may differ from its neighbours; smooth: drift "
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
    _ = parser.add_argument("--region", choices=REGION_CHOICES, default="auto", help=REGION_HELP)
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
    add_config(parser, settings=_config_settings)
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fit selection and grid as JSON without aligning",
    )
    return parser


def _config_settings() -> dict[str, object]:
    """A config file's settings: every AlignConfig field the flags do not set."""
    from tomojax.alignment import AlignConfig

    return {k: v for k, v in asdict(AlignConfig()).items() if k not in _SETTINGS}


def _build_align_config(
    args: _Args,
    settings: Mapping[str, Any],
) -> AlignConfig:
    """Validate expert TOML settings and construct ``AlignConfig`` before loading data."""
    from tomojax.alignment import AlignConfig

    defaults = asdict(AlignConfig())
    coerced: dict[str, Any] = {}
    for key, value in settings.items():
        if key == "levels":
            if not isinstance(value, list | tuple) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            ):
                raise ValueError("levels must be a list of positive integers ending at 1")
            coerced[key] = tuple(int(item) for item in value)
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
    given = {name: getattr(args, name) for name in _SETTINGS if getattr(args, name) is not None}
    return replace(AlignConfig(**coerced), **given)


def _validate_opposite_views(
    args: _Args,
    settings: Mapping[str, object],
    explicit: set[str],
    config_file_values: Mapping[str, object],
) -> None:
    """Reject unsupported --fit and joint-only options for ``--method opposite-views``."""
    fit = tuple(args.fit)
    if fit != ("center",):
        raise ValueError(f"--method opposite-views fits --fit center alone, not {list(fit)!r}")
    joint_given = [name for name in _SETTINGS if getattr(args, name) is not None]
    if args.grid is not None:
        joint_given.append("grid")
    if "region" in explicit or "region" in config_file_values:
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
    masked: bool,
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
        "fit": list(args.fit),
        "method": args.method,
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
    logging.info("Saved alignment manifest to %s", args.manifest)


def _validate_align_args(
    parser: argparse.ArgumentParser,
    args: _Args,
    settings: Mapping[str, Any],
    explicit: set[str],
    config_values: Mapping[str, object],
) -> AlignConfig | None:
    """Validate CLI and TOML arguments before loading scan data."""
    if len(args.fit) != len(set(args.fit)):
        parser.error("--fit names must not repeat")
    if not args.dry_run and not args.out:
        parser.error("the following arguments are required: -o/--output")
    if args.grid is not None and min(args.grid) < 1:
        parser.error("--grid dimensions must be positive integers")
    try:
        if args.method == "opposite-views":
            _validate_opposite_views(args, settings, explicit, config_values)
            return None
        return _build_align_config(args, settings)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def _dry_run_plan(
    args: _Args,
    fit: tuple[str, ...],
    scan: Scan,
    grid: Grid | None,
    config: AlignConfig | None,
) -> dict[str, object]:
    """Return the JSON summary for ``tomojax align --dry-run``."""
    plan: dict[str, object] = {
        "input_path": args.data,
        "output_path": args.out,
        "fit": list(fit),
        "method": args.method,
    }
    if args.method == "joint":
        assert config is not None
        plan["config"] = asdict(config)
        plan["grid"] = (scan.grid if grid is None else grid).to_dict()
        plan["bounded_working_grid"] = grid is None
    return plan


def main() -> None:
    """Run ``tomojax align``."""
    import tomojax as tj
    from tomojax.core import log_jax_env, setup_logging

    parser = build_parser()
    namespace, metadata = parse_args_with_config(parser)
    args = cast("_Args", namespace)
    settings = cast("dict[str, Any]", metadata["settings"])
    explicit = set(cast("list[str]", metadata["explicit_cli_keys"]))
    config_values = cast("dict[str, object]", metadata["config_file_values"])
    config = _validate_align_args(parser, args, settings, explicit, config_values)
    outputs = [] if args.dry_run else [args.out, args.manifest]
    check_paths(parser, inputs=[args.data], outputs=outputs, force=args.force)
    scan = tj.load(args.data, apply_poses=args.apply_poses)
    fit = tuple(args.fit)
    grid, cylinder = (None, None)
    if args.method == "joint" and (args.grid is not None or args.region != "auto"):
        shape = None if args.grid is None else (args.grid[2], args.grid[1], args.grid[0])
        region = None if args.region == "off" else args.region
        grid, cylinder = region_grid(scan, region, shape=shape)
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args, fit, scan, grid, config), indent=2, sort_keys=True))
        return
    setup_logging()
    log_jax_env()
    if args.method == "opposite-views":
        result = tj.align(scan, fit=fit, method="opposite-views")
    elif args.grid is None and args.region != "auto":
        region_arg = cast("Region | None", None if args.region == "off" else args.region)
        result = tj.align(scan, fit=fit, region=region_arg, config=config)
    else:
        result = tj.align(scan, fit=fit, grid=grid, config=config)
        if cylinder is not None and result.image is not None and result.grid is not None:
            masked_image = zero_outside_cylinder(result.image, result.grid, cylinder)
            result = replace(result, image=masked_image)
    assert args.out is not None
    tj.save(args.out, result)
    logging.info("Saved alignment results to %s", args.out)
    if args.manifest is not None:
        _write_manifest(args, metadata, scan, result, masked=cylinder is not None)


if __name__ == "__main__":  # pragma: no cover
    main()
