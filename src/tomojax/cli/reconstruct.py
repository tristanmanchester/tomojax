"""``tomojax reconstruct``: :func:`tomojax.reconstruct` on the command line.

The command loads INPUT with its saved poses (unless ``--no-apply-poses``); chooses
the grid (``--region``, ``--grid``); builds the method's configuration from a
``--config`` file's expert settings, which are fields of that configuration
class; reconstructs with :func:`tomojax.reconstruct`, whose keywords the other
options are; and saves the result with :func:`tomojax.save`.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import fields
from functools import partial
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, TypedDict, cast, get_args

import numpy as np

from tomojax._parameters import Filter, Method, ReconMethod
from tomojax._workflow.field_of_view import (
    REGION_CHOICES,
    REGION_HELP,
)
from tomojax.cli._options import (
    add_checkpoint,
    add_config,
    add_force,
    check_paths,
    checkpointed,
    hide_expert,
)
from tomojax.cli._progress import ProgressParser, add_progress, run_command
from tomojax.cli._watch import watch_directory
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax import Reconstruction, Scan
    from tomojax._workflow.operations import Region
    from tomojax.cli._watch import WatchDirectory
    from tomojax.cli.config import ConfigMetadata
    from tomojax.recon.api import MethodConfig
    from tomojax.wire.progress import Result

LOG = logging.getLogger(__name__)

_METHODS = get_args(ReconMethod.__value__)
# Configuration fields a TOML file cannot hold.
_NOT_SETTINGS = frozenset({"support", "devices"})


_PUBLIC = (
    "--method",
    "--filter",
    "--iterations",
    "--tv-weight",
    "--nonnegative",
    "--warm-start",
    "--grid",
    "--region",
    "--apply-poses",
    "--include-scan",
    "--preview",
    "--manifest",
    "--watch-dir",
    "--checkpoint",
    "--progress",
    "--dry-run",
)

_SETTINGS = (
    "Expert settings are --config keys, fields of the method's configuration:\n"
    "tomojax.recon.FBPConfig (fbp, and FDK for cone beams), FourierConfig, CGLSConfig,\n"
    "FISTAConfig, SPDHGConfig, SIRTConfig or MLEMConfig. Each replaces\n"
    "a default field; a setting the method's class has no field for fails with the ones it has.\n"
    "Expert settings require an explicit --method rather than automatic selection.\n"
    "Pass --method NAME --config-keys to list one method's keys and defaults.\n"
)
_EPILOG = (
    "Examples:\n"
    "  tomojax reconstruct scan.nxs -o recon.nxs\n"
    "  tomojax reconstruct scan.nxs -o recon.nxs --method cgls --iterations 100\n"
    "  tomojax reconstruct scan.nxs -o recon.nxs --method fista --tv-weight 0.002 --nonnegative"
)


def build_parser() -> ProgressParser:
    """The ``tomojax reconstruct`` parser; expert settings are the methods' configuration fields."""
    p = ProgressParser(
        prog="tomojax reconstruct",
        description=(
            "Reconstruct a volume from a scan, with the poses saved with it (by tomojax "
            "align) unless --no-apply-poses."
        ),
        epilog=_EPILOG,
    )
    _ = p.add_argument("data", metavar="INPUT", help="Scan to reconstruct (.nxs, .h5, .npz)")
    _ = p.add_argument(
        "-o",
        "--output",
        dest="out",
        metavar="OUTPUT",
        help="Output .nxs: reconstructed volume, geometry and source reference "
        "(pass --include-scan to embed projections)",
    )
    add_force(p)
    _ = p.add_argument(
        "--dry-run", action="store_true", help="Print the plan (method, grid, memory) and stop"
    )
    add_config(p, settings=_settings, method_settings=_settings)
    _ = p.add_argument(
        "--method",
        choices=("auto", *_METHODS),
        default="auto",
        help=(
            "auto (default): choose from geometry; fbp: filtered backprojection/FDK; "
            "fourier: parallel Fourier inverse; "
            "cgls: least squares; sirt: normalized algebraic updates; fista, spdhg: TV; "
            "mlem: linear Poisson MLEM/OSEM"
        ),
    )
    _ = p.add_argument(
        "--filter",
        choices=get_args(Filter.__value__),
        default=None,
        help="FBP detector response (optional for Fourier, default ramp for FBP)",
    )
    _ = p.add_argument(
        "--iterations",
        type=int,
        default=None,
        metavar="N",
        help="Updates: cgls/sirt up to 50, mlem 50, fista 200, spdhg 600 (one view block each)",
    )
    _ = p.add_argument(
        "--tv-weight",
        type=float,
        default=None,
        metavar="W",
        help="Total-variation weight for fista/spdhg (default: auto from noise; 0 disables)",
    )
    _ = p.add_argument(
        "--nonnegative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep fista, spdhg or sirt voxels nonnegative (sets lower_bound=0)",
    )
    _ = p.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start iterative methods (cgls, fista, spdhg, sirt, mlem) from FBP/FDK",
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
        help="Resample the selected field of view to NX x NY x NZ voxels",
    )
    _ = p.add_argument("--region", choices=REGION_CHOICES, default=None, help=REGION_HELP)
    _ = p.add_argument(
        "--apply-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the per-view corrections saved in INPUT (--no-apply-poses: nominal geometry)",
    )
    _ = p.add_argument(
        "--include-scan",
        action="store_true",
        help="Also save the scan in OUTPUT, so it reloads as a scan too",
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
    add_checkpoint(p)
    add_progress(p, watch_dir=True)
    hide_expert(p, _PUBLIC, settings=_SETTINGS)
    return p


def _settings(*, method: str | None = None) -> dict[str, object]:
    """A config file's expert settings: every method's configuration fields, or one method's."""
    from tomojax.recon.api import method_config

    if method is not None and method != "auto" and method in _METHODS:
        config = method_config(method)
        selected: dict[str, object] = {}
        for item in fields(config):
            if item.init and item.name not in _NOT_SETTINGS:
                value: object = getattr(config, item.name)
                selected[item.name] = value
        return selected
    defaults: dict[str, list[tuple[str, object]]] = {}
    for name in _METHODS:
        config = method_config(name)
        for item in fields(config):
            if item.init and item.name not in _NOT_SETTINGS:
                default: object = getattr(config, item.name)
                defaults.setdefault(item.name, []).append((name, default))
    out: dict[str, object] = {}
    for key, pairs in defaults.items():
        values = [v for _, v in pairs]
        methods = ", ".join(m for m, _ in pairs)
        shared = values[0] if all(v == values[0] for v in values) else None
        out[key] = (shared, f"methods: {methods}")
    return out


class _MethodOptions(TypedDict, total=False):
    filter: Filter | None
    iterations: int | None
    tv_weight: float | None
    nonnegative: bool | None
    seed: int | None


class _Args(argparse.Namespace):
    """The command's arguments."""

    data: str
    out: str | None
    method: Method
    filter: Filter | None
    iterations: int | None
    tv_weight: float | None
    nonnegative: bool | None
    warm_start: bool | None
    seed: int | None
    grid: list[int] | None
    region: Region | None
    apply_poses: bool
    include_scan: bool
    preview: str | None
    manifest: str | None
    checkpoint: str | None

    force: bool
    dry_run: bool
    progress: str
    watch_dir: str | None

    def settings(self) -> _MethodOptions:
        """The CLI's method-specific arguments, with None for unset options."""
        return {
            "filter": self.filter,
            "iterations": self.iterations,
            "tv_weight": self.tv_weight,
            "nonnegative": self.nonnegative,
            "seed": self.seed,
        }

    def validate(
        self,
        parser: ProgressParser,
        settings: Mapping[str, object],
    ) -> MethodConfig | None:
        if self.grid is not None and min(self.grid) < 1:
            parser.error("--grid dimensions must be positive integers")
        if not Path(self.data).exists():
            raise FileNotFoundError(f"input not found: {self.data}")
        outputs = [] if self.dry_run else [self.out, self.preview, self.manifest]
        check_paths(parser, inputs=[self.data], outputs=outputs, force=self.force)
        try:
            return self.method_config(settings)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))

    def method_config(self, settings: Mapping[str, object]) -> MethodConfig | None:
        """The method's configuration with the config file's ``settings``.

        The options are checked against it too, so one the method does not
        take is a usage error before the scan loads.
        """
        from tomojax.recon.api import method_config

        options = {name: value for name, value in self.settings().items() if value is not None}
        if self.method == "auto":
            warm_start_key = [] if self.warm_start is None else ["warm_start"]
            method_specific = sorted([*settings, *options, *warm_start_key])
            if method_specific:
                raise ValueError(
                    f"method 'auto' does not take method-specific options "
                    f"({', '.join(method_specific)}); pass an explicit --method"
                )
            return None
        # The parser checks these dynamic keys against the solver fields before this call.
        expert = cast("Mapping[str, Any]", settings)
        config = method_config(self.method, config=None, warm_start=self.warm_start, **expert)
        _ = method_config(self.method, config=config, **cast("_MethodOptions", options))
        return config


def _write_manifest(
    args: _Args,
    metadata: ConfigMetadata,
    scan: Scan,
    result: Reconstruction,
    *,
    masked: bool,
) -> None:
    """Write ``--manifest``: the run's inputs and settings and the reconstruction's record."""
    from tomojax.cli.manifest import build_manifest, save_manifest

    payload: dict[str, object] = {
        "input_path": args.data,
        "output_path": args.out,
        "preview_path": args.preview,
        "manifest_path": args.manifest,
        "config_path": metadata["config_path"],
        "config_file_values": metadata["config_file_values"],
        "explicit_cli_keys": metadata["explicit_cli_keys"],
        "effective_options": metadata["effective_options"],
        "settings": metadata["settings"],
        "method": result.method,
        "input_projection_shape": list(np.shape(scan.projections)),
        "input_poses": scan.poses is not None,
        "region": {
            "requested": args.region,
            "grid_changed": result.grid != scan.grid,
            "cylindrical_output_mask": masked,
        },
        "reconstruction_grid": result.grid.to_dict(),
        "detector": result.geometry.detector.to_dict(),
        "volume_shape": list(np.shape(result.volume)),
        # The resolved configuration and the solver's record.
        "reconstruction": result.info,
    }
    manifest = build_manifest("tomojax reconstruct", list(sys.argv), args, payload)
    save_manifest(str(args.manifest), manifest)
    LOG.info("Saved reproducibility manifest to %s", args.manifest)


def main() -> int:
    """Run ``tomojax reconstruct``."""
    parser = build_parser()
    args, metadata = parse_args_with_config(parser, namespace=_Args())
    if not args.dry_run and not args.out:
        parser.error("the following arguments are required: -o/--output")
    watch = watch_directory(args.watch_dir, working=False)
    work = partial(_run, parser, args, metadata, watch)
    operation = partial(checkpointed, None if args.dry_run else args.checkpoint, work)
    return run_command("reconstruct", args.progress, operation, watch=watch)


def _run(
    parser: ProgressParser,
    args: _Args,
    metadata: ConfigMetadata,
    watch: WatchDirectory | None,
) -> Result | None:
    import tomojax as tj
    from tomojax import progress
    from tomojax.cli._runtime import report_automatic, solve_summary

    config = args.validate(parser, metadata["settings"])
    progress.check_canceled()

    scan = tj.load(args.data, apply_poses=args.apply_poses)
    progress.check_canceled()
    from tomojax._workflow.operations import reconstruct_resolved
    from tomojax._workflow.planning import resolve_reconstruction
    from tomojax.cli._plans import announce, dry_run, plan_product

    shape = None if args.grid is None else (args.grid[2], args.grid[1], args.grid[0])
    work = resolve_reconstruction(
        scan,
        args.method,
        region=args.region,
        shape=shape,
        config=config,
        warm_start=args.warm_start,
        **args.settings(),
    )
    if args.dry_run:
        return dry_run(args.progress, args.data, plan_product(scan, work), scan, work)
    used = tuple(f for f, v in (("--region", args.region), ("--grid", args.grid)) if v is not None)
    _ = announce("reconstruct", scan, work, used=used)
    assert args.out is not None
    with progress.previews() if watch is not None else nullcontext():
        result = reconstruct_resolved(work, checkpoint=args.checkpoint)
    report_automatic(result.info)
    source = scan.source
    if (
        source is not None
        and source.geometry_metadata.get("detector_center_status") == "not_recorded"
    ):
        LOG.warning(
            "%s records no rotation center; the detector middle was used. "
            "Fit one with tomojax align --fit center before reconstructing.",
            args.data,
        )
    progress.check_canceled()
    tj.save(args.out, result, include_scan=args.include_scan)
    if args.preview is not None:
        from tomojax.cli._images import save_volume_plane

        progress.check_canceled()
        label = f"z plane · {result.method} · {Path(args.out).name}"
        _ = save_volume_plane(Path(args.preview), result.volume, result.grid, "z", label)
    if args.manifest is not None:
        progress.check_canceled()
        _write_manifest(args, metadata, scan, result, masked=args.region == "cyl")
    if args.progress == "json":
        from tomojax.cli._results import saved_result

        outputs = tuple(str(p) for p in (args.out, args.preview, args.manifest) if p is not None)
        return saved_result(args.out, result, outputs=outputs, include_scan=args.include_scan)
    stopped = solve_summary(result.info)
    ended = "" if stopped is None else f"; {stopped}"
    print(f"wrote {args.out}: {result.method} volume {tuple(np.shape(result.volume))}{ended}")
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
