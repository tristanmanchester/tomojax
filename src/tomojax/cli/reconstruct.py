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
from dataclasses import dataclass, fields
from functools import partial
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np

from tomojax._field_of_view import (
    REGION_CHOICES,
    REGION_HELP,
)
from tomojax._parameters import Filter, Method
from tomojax.cli._options import add_config, add_force, check_paths, hide_expert
from tomojax.cli._plans import announce, dry_run
from tomojax.cli._progress import ProgressParser, add_progress, run_command
from tomojax.cli._watch import admit_or_exit, watch_directory
from tomojax.cli.config import parse_args_with_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax import Reconstruction, Scan
    from tomojax._workflow import Region
    from tomojax.cli._watch import WatchDirectory
    from tomojax.recon.api import MethodConfig
    from tomojax.wire.progress import Result

LOG = logging.getLogger(__name__)

_METHODS = get_args(Method.__value__)[1:]
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
    "--region",
    "--apply-poses",
    "--include-scan",
    "--preview",
    "--manifest",
    "--watch-dir",
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


def build_parser() -> argparse.ArgumentParser:
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
    add_config(p, settings=_settings)
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
        help="FBP filter (default ramp)",
    )
    _ = p.add_argument(
        "--iterations",
        type=int,
        default=None,
        metavar="N",
        help="Iterations: cgls/fista/sirt/mlem 50, spdhg 600 (one view block each)",
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
    add_progress(p, watch_dir=True)
    hide_expert(p, _PUBLIC, settings=_SETTINGS)
    return p


def _settings(*, method: str | None = None) -> dict[str, object]:
    """A config file's expert settings: every method's configuration fields, or one method's."""
    from tomojax.recon.api import method_config

    if method is not None and method != "auto" and method in _METHODS:
        config = method_config(method)
        return {
            item.name: cast("object", getattr(config, item.name))
            for item in fields(config)
            if item.init and item.name not in _NOT_SETTINGS
        }
    defaults: dict[str, list[tuple[str, object]]] = {}
    for name in _METHODS:
        config = method_config(name)
        for item in fields(config):
            if item.init and item.name not in _NOT_SETTINGS:
                defaults.setdefault(item.name, []).append(
                    (name, cast("object", getattr(config, item.name)))
                )
    out: dict[str, object] = {}
    for key, pairs in defaults.items():
        values = [v for _, v in pairs]
        methods = ", ".join(m for m, _ in pairs)
        shared = values[0] if all(v == values[0] for v in values) else None
        out[key] = (shared, f"methods: {methods}")
    return out


@dataclass(frozen=True)
class _Run:
    """The command's arguments."""

    data: str
    out: str | None
    method: Method
    filter: str | None
    iterations: int | None
    tv_weight: float | None
    nonnegative: bool | None
    warm_start: bool | None
    seed: int | None
    grid: tuple[int, int, int] | None
    region: Region | None
    apply_poses: bool
    include_scan: bool
    preview: str | None
    manifest: str | None

    @classmethod
    def of(cls, args: argparse.Namespace) -> _Run:
        """The typed values of ``args``."""
        grid = cast("list[int] | None", args.grid)
        return cls(
            data=cast("str", args.data),
            out=cast("str | None", args.out),
            method=cast("Method", args.method),
            filter=cast("str | None", args.filter),
            iterations=cast("int | None", args.iterations),
            tv_weight=cast("float | None", args.tv_weight),
            nonnegative=cast("bool | None", args.nonnegative),
            warm_start=cast("bool | None", args.warm_start),
            seed=cast("int | None", args.seed),
            grid=None if grid is None else (grid[0], grid[1], grid[2]),
            region=cast("Region | None", args.region),
            apply_poses=cast("bool", args.apply_poses),
            include_scan=cast("bool", args.include_scan),
            preview=cast("str | None", args.preview),
            manifest=cast("str | None", args.manifest),
        )

    def keywords(self) -> dict[str, object]:
        """The options given that are :func:`tomojax.reconstruct` keywords."""
        given = {name: cast("object", getattr(self, name)) for name in _KEYWORDS}
        return {name: value for name, value in given.items() if value is not None}

    def validate(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        metadata: Mapping[str, object],
    ) -> MethodConfig | None:
        if self.grid is not None and min(self.grid) < 1:
            parser.error("--grid dimensions must be positive integers")
        if not Path(self.data).exists():
            raise FileNotFoundError(f"input not found: {self.data}")
        outputs = [] if args.dry_run else [self.out, self.preview, self.manifest]
        check_paths(parser, inputs=[self.data], outputs=outputs, force=cast("bool", args.force))
        try:
            return self.config(cast("dict[str, object]", metadata["settings"]))
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))

    def config(self, settings: Mapping[str, object]) -> MethodConfig | None:
        """The method's configuration with the config file's ``settings``.

        The options are checked against it too, so one the method does not
        take is a usage error before the scan loads.
        """
        from tomojax.recon.api import method_config

        if self.method == "auto":
            method_specific = sorted([*settings, *self.keywords()])
            if method_specific:
                raise ValueError(
                    f"method 'auto' does not take method-specific options "
                    f"({', '.join(method_specific)}); pass an explicit --method"
                )
            return None
        expert = cast("dict[str, Any]", dict(settings))
        config = method_config(self.method, config=None, warm_start=self.warm_start, **expert)
        options = cast("dict[str, Any]", self.keywords())
        options.pop("warm_start", None)
        _ = method_config(self.method, config=config, **options)
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
        "region": {
            "requested": run.region,
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
    save_manifest(str(run.manifest), manifest)
    LOG.info("Saved reproducibility manifest to %s", run.manifest)


def main() -> int:
    """Run ``tomojax reconstruct``."""
    parser = build_parser()
    args, metadata = parse_args_with_config(parser)
    if not args.dry_run and not args.out:
        parser.error("the following arguments are required: -o/--output")
    watch = watch_directory(args.watch_dir, "reconstruct", working=False)
    operation = partial(_run, parser, args, metadata, watch)
    return run_command("reconstruct", args.progress, operation, watch=watch)


def _run(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    metadata: Mapping[str, object],
    watch: WatchDirectory | None,
) -> Result | None:
    import tomojax as tj
    from tomojax import progress
    from tomojax.cli._runtime import report_automatic

    run = _Run.of(args)
    config = run.validate(parser, args, metadata)
    related = (
        run.data,
        run.out,
        run.preview,
        run.manifest,
        cast("str | None", metadata["config_path"]),
    )
    if not args.dry_run:
        admit_or_exit(parser, watch, force=cast("bool", args.force), related=related)
    progress.check_cancelled()

    scan = tj.load(run.data, apply_poses=run.apply_poses)
    progress.check_cancelled()
    from tomojax._planning import resolve_reconstruction
    from tomojax._workflow import reconstruct_resolved

    keywords = {name: cast("Any", getattr(run, name)) for name in _KEYWORDS}
    shape = None if run.grid is None else (run.grid[2], run.grid[1], run.grid[0])
    work = resolve_reconstruction(
        scan, run.method, region=run.region, shape=shape, config=config, **keywords
    )
    if args.dry_run:
        from tomojax.cli._plans import plan_product

        return dry_run(args.progress, run.data, plan_product(scan, work), scan, work)
    used = tuple(f for f, v in (("--region", run.region), ("--grid", run.grid)) if v is not None)
    _ = announce("reconstruct", scan, work, used=used)
    assert run.out is not None
    with progress.previews() if watch is not None else nullcontext():
        result = reconstruct_resolved(work)
    report_automatic(result.info)
    source = scan.source
    if (
        source is not None
        and source.geometry_metadata.get("detector_center_status") == "not_recorded"
    ):
        LOG.warning(
            "%s records no rotation centre; the detector middle was used. "
            "Fit one with tomojax align --fit center before reconstructing.",
            run.data,
        )
    progress.check_cancelled()
    tj.save(run.out, result, include_scan=run.include_scan)
    if run.preview is not None:
        from tomojax.cli._images import save_volume_plane

        progress.check_cancelled()
        label = f"z plane · {result.method} · {Path(run.out).name}"
        _ = save_volume_plane(Path(run.preview), result.volume, result.grid, "z", label)
    if run.manifest is not None:
        progress.check_cancelled()
        _write_manifest(args, metadata, scan, result, masked=run.region == "cyl")
    if args.progress == "json":
        from tomojax.cli._results import saved_result

        outputs = tuple(str(p) for p in (run.out, run.preview, run.manifest) if p is not None)
        return saved_result(run.out, result, outputs=outputs, include_scan=run.include_scan)
    print(f"wrote {run.out}: {result.method} volume {tuple(np.shape(result.volume))}")
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
