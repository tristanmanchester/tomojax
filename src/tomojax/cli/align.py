"""``tomojax align``: one joint geometry fit on a saved scan."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import sys
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.cli._options import EXIT_STATUS, add_output, check_paths
from tomojax.cli._reconstruction_region import (
    ROI_CHOICES,
    ROI_HELP,
    region_grid,
    zero_outside_cylinder,
)

if TYPE_CHECKING:
    from tomojax import Alignment, Scan


class _Args(argparse.Namespace):
    data: str
    out: str
    force: bool
    fit: list[str]
    roi: str
    grid: list[int] | None
    poses: bool
    manifest: str | None
    dry_run: bool


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
            "  tomojax align scan.nxs -o aligned.nxs --fit centre\n"
            "  tomojax align scan.nxs -o aligned.nxs --fit translations tilt roll\n"
            "  tomojax recon aligned.nxs -o recon.nxs\n\n" + EXIT_STATUS
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument("data", metavar="INPUT", help="Scan to align (.nxs, .h5, .npz)")
    add_output(parser, "Output .nxs: the corrected scan, its poses and volume")
    _ = parser.add_argument(
        "--fit",
        nargs="+",
        choices=("centre", "tilt", "roll", "translations", "rotations"),
        default=["translations"],
        metavar="NAME",
        help=(
            "Parameters to fit: centre, tilt, roll, translations, rotations "
            "(default translations). All other parameters stay fixed."
        ),
    )
    _ = parser.add_argument("--roi", choices=ROI_CHOICES, default="auto", help=ROI_HELP)
    _ = parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Resample the selected field of view to NX x NY x NZ voxels",
    )
    _ = parser.add_argument(
        "--poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Correct on top of saved poses (--no-poses uses the nominal geometry)",
    )
    _ = parser.add_argument("--manifest", metavar="JSON", help="Also save a JSON record of the run")
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fit selection and grid as JSON without aligning",
    )
    return parser


def _write_manifest(args: _Args, scan: Scan, result: Alignment, *, masked: bool) -> None:
    """Save the requested inputs, fitted geometry and solver record."""
    from tomojax.cli.manifest import build_manifest, save_manifest

    payload: dict[str, object] = {
        "input_path": args.data,
        "output_path": args.out,
        "input_projection_shape": list(np.shape(scan.projections)),
        "input_poses": scan.poses is not None,
        "fit": result.info["fit"],
        "roi": {
            "requested": args.roi,
            "grid_changed": result.scan.grid != scan.grid,
            "cylindrical_output_mask": masked,
        },
        "reconstruction_grid": result.scan.grid.to_dict(),
        "detector": result.scan.detector.to_dict(),
        "poses_shape": list(result.poses.shape),
        "volume_shape": list(np.shape(result.volume)),
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
    args = cast("_Args", parser.parse_args())
    if len(args.fit) != len(set(args.fit)):
        parser.error("--fit names must not repeat")
    check_paths(
        parser,
        inputs=[args.data],
        outputs=[args.out, args.manifest],
        force=args.force,
    )
    scan = tj.load(args.data, poses=args.poses)
    grid, masked = (None, False)
    if args.grid is not None or args.roi != "auto":
        shape = None if args.grid is None else (args.grid[0], args.grid[1], args.grid[2])
        grid, masked = region_grid(scan, roi=args.roi, grid=shape)
    fit = tuple(args.fit)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_path": args.data,
                    "output_path": args.out,
                    "fit": list(fit),
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
    result = tj.align(scan, fit=fit, grid=grid)
    if masked:
        result = replace(
            result, volume=zero_outside_cylinder(result.volume, result.scan.grid, scan.detector)
        )
    tj.save(args.out, result)
    logging.info("Saved alignment results to %s", args.out)
    if args.manifest is not None:
        _write_manifest(args, scan, result, masked=masked)


__all__ = ["build_parser", "main"]
