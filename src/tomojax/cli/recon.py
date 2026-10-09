"""``tomojax recon``: :func:`tomojax.reconstruct` on the command line.

The command loads INPUT with its saved poses (unless ``--no-poses``); chooses
the grid (``--roi``, ``--grid``); builds the method's configuration from a
``--config`` file's expert settings, which are fields of that configuration
class; reconstructs with :func:`tomojax.reconstruct`, whose keywords the other
options are; and saves the result with :func:`tomojax.save`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import logging
import os
import sys
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from tomojax.cli._options import add_config, add_output, check_paths, hide_expert
from tomojax.cli._reconstruction_region import (
    ROI_CHOICES,
    ROI_HELP,
    cylinder_support,
    region_grid,
    zero_outside_cylinder,
)
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax import Reconstruction, Scan
    from tomojax.recon.api import MethodConfig

type _Method = Literal["fbp", "cgls", "fista", "spdhg"]
_METHODS: tuple[_Method, ...] = ("fbp", "cgls", "fista", "spdhg")
# The options that are tj.reconstruct keywords besides the method and grid.
_KEYWORDS = ("filter", "iterations", "tv_weight", "nonnegative", "warm_start", "seed")
# Configuration fields a TOML file cannot hold.
_NOT_SETTINGS = frozenset({"support", "devices"})

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
    "--poses",
    "--preview",
    "--manifest",
    "--progress",
)

_SETTINGS = (
    "Expert settings are --config keys, fields of the method's configuration:\n"
    "tomojax.recon.FBPConfig (fbp, and FDK for cone beams), CGLSConfig, FistaConfig\n"
    "or SPDHGConfig. Each replaces that field of the class's defaults; a setting the\n"
    "method's class has no field for fails with the ones it has.\n"
)
_EPILOG = (
    "Examples:\n"
    "  tomojax recon scan.nxs -o recon.nxs\n"
    "  tomojax recon scan.nxs -o recon.nxs --method cgls --iterations 100\n"
    "  tomojax recon scan.nxs -o recon.nxs --method fista --tv-weight 0.002 --nonnegative"
)


def build_parser() -> argparse.ArgumentParser:
    """The ``tomojax recon`` parser; its expert settings are the methods' configuration fields."""
    p = argparse.ArgumentParser(
        prog="tomojax recon",
        description=(
            "Reconstruct a volume from a scan, with the poses saved with it (by tomojax "
            "align) unless --no-poses."
        ),
        epilog=_EPILOG,
    )
    _ = p.add_argument("data", metavar="INPUT", help="Scan to reconstruct (.nxs, .h5, .npz)")
    add_output(p, "Output .nxs: the volume, with the scan's projections and geometry")
    add_config(p, settings=_settings())
    _ = p.add_argument(
        "--method",
        choices=_METHODS,
        default="fbp",
        help=(
            "fbp: filtered backprojection, FDK for cone beams (default); cgls: least "
            "squares; fista, spdhg: least squares with total variation"
        ),
    )
    _ = p.add_argument(
        "--filter",
        choices=["ramp", "shepp-logan", "hann"],
        default=None,
        help="FBP filter (default ramp)",
    )
    _ = p.add_argument(
        "--iterations",
        type=int,
        default=None,
        metavar="N",
        help="Iterations of cgls and fista (default 50) and spdhg (400, a block of views each)",
    )
    _ = p.add_argument(
        "--tv-weight",
        type=float,
        default=None,
        metavar="W",
        help="Total-variation weight of fista and spdhg (default 0.005)",
    )
    _ = p.add_argument(
        "--nonnegative",
        action="store_true",
        default=None,
        help="Keep fista and spdhg voxels nonnegative",
    )
    _ = p.add_argument(
        "--warm-start",
        action="store_true",
        default=None,
        help="Start cgls, fista and spdhg from the FBP reconstruction",
    )
    _ = p.add_argument(
        "--seed", type=int, default=None, metavar="N", help="Seed of spdhg's view order (default 0)"
    )
    _ = p.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Reconstruct NX x NY x NZ voxels of the scan's voxel size",
    )
    _ = p.add_argument("--roi", choices=ROI_CHOICES, default="auto", help=ROI_HELP)
    _ = p.add_argument(
        "--poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the per-view corrections saved in INPUT (--no-poses: the nominal geometry)",
    )
    _ = p.add_argument(
        "--preview", metavar="PNG", default=None, help="Also write a PNG of the central x-y slice"
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
    hide_expert(p, _PUBLIC, settings=_SETTINGS)
    return p


def _settings() -> dict[str, object]:
    """A config file's expert settings: every method's configuration fields.

    Each has the default the methods share, or None where they differ.
    """
    from tomojax.recon.api import method_config

    defaults: dict[str, list[object]] = {}
    for method in _METHODS:
        config = method_config(method)
        for item in fields(config):
            if item.init and item.name not in _NOT_SETTINGS:
                value = cast("object", getattr(config, item.name))
                defaults.setdefault(item.name, []).append(value)
    return {
        name: values[0] if all(v == values[0] for v in values) else None
        for name, values in defaults.items()
    }


@dataclass(frozen=True)
class _Run:
    """The command's arguments."""

    data: str
    out: str
    method: _Method
    filter: str | None
    iterations: int | None
    tv_weight: float | None
    nonnegative: bool | None
    warm_start: bool | None
    seed: int | None
    grid: tuple[int, int, int] | None
    roi: str
    poses: bool
    preview: str | None
    manifest: str | None

    @classmethod
    def of(cls, args: argparse.Namespace) -> _Run:
        """The typed values of ``args``."""
        grid = cast("list[int] | None", args.grid)
        return cls(
            data=cast("str", args.data),
            out=cast("str", args.out),
            method=cast("_Method", args.method),
            filter=cast("str | None", args.filter),
            iterations=cast("int | None", args.iterations),
            tv_weight=cast("float | None", args.tv_weight),
            nonnegative=cast("bool | None", args.nonnegative),
            warm_start=cast("bool | None", args.warm_start),
            seed=cast("int | None", args.seed),
            grid=None if grid is None else (grid[0], grid[1], grid[2]),
            roi=cast("str", args.roi),
            poses=cast("bool", args.poses),
            preview=cast("str | None", args.preview),
            manifest=cast("str | None", args.manifest),
        )

    def keywords(self) -> dict[str, object]:
        """The options given that are :func:`tomojax.reconstruct` keywords."""
        given = {name: cast("object", getattr(self, name)) for name in _KEYWORDS}
        return {name: value for name, value in given.items() if value is not None}

    def config(self, settings: Mapping[str, object]) -> MethodConfig:
        """The method's configuration with the config file's ``settings``.

        The options are checked against it too, so one the method does not
        take is a usage error before the scan loads.
        """
        from tomojax.recon.api import method_config

        config = method_config(self.method, config=None, **settings)
        options = {k: v for k, v in self.keywords().items() if k != "warm_start"}
        _ = method_config(self.method, config=config, **options)
        if self.method == "fbp" and self.warm_start is not None:
            raise ValueError("method 'fbp' does not take warm_start")
        return config


def _write_manifest(
    args: argparse.Namespace,
    metadata: Mapping[str, object],
    scan: Scan,
    result: Reconstruction,
    *,
    masked: bool,
) -> None:
    """Write ``--manifest``: the run's inputs and settings and the reconstruction's record."""
    from tomojax.cli.manifest import build_manifest, save_manifest

    run = _Run.of(args)
    payload: dict[str, object] = {
        "input_path": run.data,
        "output_path": run.out,
        "preview_path": run.preview,
        "manifest_path": run.manifest,
        "config_path": metadata["config_path"],
        "config_file_values": metadata["config_file_values"],
        "explicit_cli_keys": metadata["explicit_cli_keys"],
        "effective_options": metadata["effective_options"],
        "settings": metadata["settings"],
        "method": result.method,
        "input_projection_shape": list(np.shape(scan.projections)),
        "input_poses": scan.poses is not None,
        "roi": {
            "requested": run.roi,
            "grid_changed": result.grid != scan.grid,
            "cylindrical_output_mask": masked,
        },
        "reconstruction_grid": result.grid.to_dict(),
        "detector": result.scan.detector.to_dict(),
        "volume_shape": list(np.shape(result.volume)),
        # The resolved configuration and the solver's record.
        "reconstruction": result.info,
    }
    manifest = build_manifest("tomojax recon", list(sys.argv), args, payload)
    save_manifest(str(run.manifest), manifest)
    logging.info("Saved reproducibility manifest to %s", run.manifest)


def main() -> None:
    """Run ``tomojax recon``."""
    import tomojax as tj
    from tomojax.core import log_jax_env, setup_logging

    parser = build_parser()
    args, metadata = parse_args_with_config(parser)
    run = _Run.of(args)
    check_paths(parser, inputs=[run.data], outputs=[run.out], force=cast("bool", args.force))
    try:
        config = run.config(cast("dict[str, object]", metadata["settings"]))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    setup_logging()
    log_jax_env()
    if cast("bool", args.progress):
        os.environ["TOMOJAX_PROGRESS"] = "1"

    scan = tj.load(run.data, poses=run.poses)
    grid, masked = region_grid(scan, roi=run.roi, grid=run.grid)
    if masked and hasattr(config, "support"):
        # The solvers that take a support constrain the solve to the cylinder.
        config = replace(config, support=cylinder_support(grid, scan.detector))
    result = tj.reconstruct(
        scan,
        run.method,
        grid=grid,
        config=config,
        filter=run.filter,
        iterations=run.iterations,
        tv_weight=run.tv_weight,
        nonnegative=run.nonnegative,
        warm_start=run.warm_start,
        seed=run.seed,
    )
    if masked:
        result = replace(result, volume=zero_outside_cylinder(result.volume, grid, scan.detector))
    tj.save(run.out, result)
    logging.info("Saved reconstruction to %s", run.out)
    if run.preview is not None:
        from tomojax.recon.quicklook import save_quicklook_png

        _ = save_quicklook_png(run.preview, np.asarray(result.volume))
        logging.info("Saved reconstruction preview to %s", run.preview)
    if run.manifest is not None:
        _write_manifest(args, metadata, scan, result, masked=masked)


__all__ = ["build_parser", "main"]
