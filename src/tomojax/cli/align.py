"""``tomojax align``: one joint geometry fit on a saved scan."""

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
from tomojax.cli._options import EXIT_STATUS, add_config, add_output, check_paths
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from tomojax import Alignment, Scan


class _Args(argparse.Namespace):
    data: str
    out: str
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
            "Fit selected geometry parameters jointly with a reconstruction. "
            "OUTPUT holds the corrected scan, its poses and the volume."
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
    add_output(parser, "Output .nxs: the corrected scan, its poses and volume")
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
    from tomojax import AlignConfig

    return {k: v for k, v in asdict(AlignConfig()).items() if k not in _SETTINGS}


def _write_manifest(args: _Args, scan: Scan, result: Alignment, *, masked: bool) -> None:
    """Save the requested inputs, fitted geometry and solver record."""
    from tomojax.cli.manifest import build_manifest, save_manifest

    payload: dict[str, object] = {
        "input_path": args.data,
        "output_path": args.out,
        "input_projection_shape": list(np.shape(scan.projections)),
        "input_poses": scan.poses is not None,
        "fit": result.info["fit"],
        "region": {
            "requested": args.region,
            "grid_changed": result.grid != scan.grid,
            "cylindrical_output_mask": masked,
        },
        "reconstruction_grid": None if result.grid is None else result.grid.to_dict(),
        "detector": result.scan.detector.to_dict(),
        "poses_shape": list(result.poses.shape),
        "image_shape": None if result.image is None else list(np.shape(result.image)),
        "alignment": result.info,
    }
    save_manifest(
        cast("str", args.manifest),
        build_manifest("tomojax align", list(sys.argv), args, payload),
    )
    logging.info("Saved reproducibility manifest to %s", args.manifest)


def main() -> None:
    """Run ``tomojax align`` with the same joint solver as :func:`tomojax.align`."""
    import tomojax as tj
    from tomojax.core import log_jax_env, setup_logging

    parser = build_parser()
    namespace, metadata = parse_args_with_config(parser)
    args = cast("_Args", namespace)
    settings = cast("dict[str, Any]", metadata["settings"])
    if len(args.fit) != len(set(args.fit)):
        parser.error("--fit names must not repeat")
    check_paths(
        parser,
        inputs=[args.data],
        outputs=[args.out, args.manifest],
        force=args.force,
    )
    scan = tj.load(args.data, apply_poses=args.apply_poses)
    grid, cylinder = (None, None)
    if args.grid is not None or args.region != "auto":
        shape = None if args.grid is None else (args.grid[0], args.grid[1], args.grid[2])
        region = None if args.region == "off" else args.region
        grid, cylinder = region_grid(scan, region, shape=shape)
    fit = tuple(args.fit)
    given = {name: getattr(args, name) for name in _SETTINGS if getattr(args, name) is not None}
    if "levels" in settings:  # a TOML array
        settings["levels"] = tuple(settings["levels"])
    config = replace(tj.AlignConfig(**settings), **given)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_path": args.data,
                    "output_path": args.out,
                    "fit": list(fit),
                    "method": args.method,
                    "config": asdict(config),
                    "grid": (scan.grid if grid is None else grid).to_dict(),
                    "bounded_working_grid": grid is None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    setup_logging()
    log_jax_env()
    if args.method == "opposite-views":
        result = tj.align(scan, fit=fit, method="opposite-views")
    else:
        result = tj.align(scan, fit=fit, grid=grid, config=config)
    if cylinder is not None and result.image is not None and result.grid is not None:
        result = replace(result, image=zero_outside_cylinder(result.image, result.grid, cylinder))
    tj.save(args.out, result)
    logging.info("Saved alignment results to %s", args.out)
    if args.manifest is not None:
        _write_manifest(args, scan, result, masked=cylinder is not None)


__all__ = ["build_parser", "main"]
