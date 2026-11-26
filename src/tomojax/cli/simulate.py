"""``tomojax simulate``: write a synthetic scan of a phantom."""

from __future__ import annotations

from dataclasses import fields, replace
import os
from typing import TYPE_CHECKING, Literal, cast, get_args

from tomojax._parameters import GeometryName, Phantom
from tomojax.cli._options import add_config, add_output, check_paths, hide_expert
from tomojax.cli._progress import ProgressParser, add_progress, run_command
from tomojax.cli.config import parse_args_with_config

from ._runtime import transfer_guard_context

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

    from tomojax.datasets import SimulationArtifacts
    from tomojax.wire.progress import Result

TransferGuardName = Literal["off", "log", "disallow"]


_PUBLIC = (
    "--progress",
    "--size",
    "--n-views",
    "--geometry",
    "--phantom",
    "--rotation",
    "--tilt",
    "--source-to-axis",
    "--source-to-detector",
    "--poisson-scale",
    "--translation-sigma",
    "--seed",
)


def _build_parser() -> ProgressParser:
    """Build the simulate command parser."""
    parser = ProgressParser(
        prog="tomojax simulate",
        description=(
            "Write a synthetic scan of a phantom (projections, geometry and the true "
            "volume) to try reconstruction and alignment on. Deterministic for a --seed."
        ),
        epilog=(
            "Examples:\n"
            "  tomojax simulate -o phantom.nxs\n"
            "  tomojax simulate -o cone.nxs --geometry cone --size 128 --n-views 360\n"
            "  tomojax simulate -o lamino.nxs --geometry lamino --tilt 30 --poisson-scale 1000"
        ),
    )
    add_progress(parser)
    add_output(parser, "Dataset to write (.nxs)")
    add_config(parser, settings=_settings)
    _ = parser.add_argument(
        "--size",
        type=int,
        default=64,
        metavar="N",
        help="Volume edge in voxels, and the detector size",
    )
    _ = parser.add_argument(
        "--n-views",
        type=int,
        default=None,
        metavar="N",
        help="Number of views (default 2 x size)",
    )
    _ = parser.add_argument(
        "--geometry", choices=get_args(GeometryName.__value__), default="parallel"
    )
    _ = parser.add_argument(
        "--phantom",
        choices=get_args(Phantom.__value__),
        default="shepp",
    )
    _ = parser.add_argument(
        "--rotation",
        type=float,
        default=None,
        metavar="DEG",
        help="Rotation range (default 180 for parallel, 360 otherwise)",
    )
    _ = parser.add_argument(
        "--tilt",
        type=float,
        default=None,
        metavar="DEG",
        help="Rotation axis tilt (default 30 for lamino, 0 otherwise)",
    )
    _ = parser.add_argument(
        "--source-to-axis",
        type=float,
        default=None,
        metavar="DIST",
        help="Cone beam, in voxels (default 3 x size)",
    )
    _ = parser.add_argument(
        "--source-to-detector",
        type=float,
        default=None,
        metavar="DIST",
        help="Cone beam, in voxels (default 1.5 x source-to-axis)",
    )
    _ = parser.add_argument(
        "--poisson-scale",
        type=float,
        default=0.0,
        metavar="SCALE",
        help="Add Poisson noise, drawing counts of value x SCALE (higher is quieter)",
    )
    _ = parser.add_argument(
        "--translation-sigma",
        type=float,
        default=0.0,
        metavar="DIST",
        help="Standard deviation per axis of random object shifts (voxel pitches)",
    )
    _ = parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Unit-spaced volume shape (voxel pitch 1), overriding --size",
    )
    _ = parser.add_argument(
        "--detector",
        type=int,
        nargs=2,
        metavar=("NU", "NV"),
        default=None,
        help="Unit-spaced centered detector shape (pixel pitch 1; default from --size)",
    )
    _ = parser.add_argument(
        "--seed", type=int, default=0, metavar="N", help="Random seed (phantom, motion and noise)"
    )
    _ = parser.add_argument(
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help=(
            "JAX transfer guard mode during compute "
            "(default: off; use log/disallow for strict transfer checks)"
        ),
    )
    hide_expert(parser, _PUBLIC)
    return parser


# Phantom details a config file may set; the detector artifacts besides
# --poisson-scale are config settings too.
_PHANTOM_SETTINGS = (
    "tilt_about",
    "single_size",
    "single_value",
    "single_rotate",
    "n_cubes",
    "n_spheres",
    "min_size",
    "max_size",
    "min_value",
    "max_value",
    "max_rotation",
    "lamino_thickness_ratio",
)


def _settings() -> dict[str, object]:
    """A config file's expert settings: phantom details and detector artifacts."""
    from tomojax.datasets import SimConfig, SimulationArtifacts

    phantom = SimConfig(nx=1, ny=1, nz=1, nu=1, nv=1, n_views=1)
    artifacts = SimulationArtifacts()
    return {name: getattr(phantom, name) for name in _PHANTOM_SETTINGS} | {
        f.name: getattr(artifacts, f.name) for f in fields(artifacts) if f.name != "poisson_scale"
    }


def _validate_args(
    parser: ProgressParser,
    args: argparse.Namespace,
    artifact_settings: dict[str, object],
) -> SimulationArtifacts | None:
    from tomojax.datasets import SimulationArtifacts

    if cast("int", args.size) < 1:
        parser.error("--size must be a positive integer")
    n_views = cast("int | None", args.n_views)
    if n_views is not None and n_views < 1:
        parser.error("--n-views must be a positive integer")
    shape = cast("list[int] | None", args.grid)
    if shape is not None and min(shape) < 1:
        parser.error("--grid dimensions must be positive integers")
    pixels = cast("list[int] | None", args.detector)
    if pixels is not None and min(pixels) < 1:
        parser.error("--detector dimensions must be positive integers")
    if cast("float", args.poisson_scale) < 0:
        parser.error("--poisson-scale must be nonnegative")
    geometry = cast("str", args.geometry)
    if geometry != "cone" and (
        args.source_to_axis is not None or args.source_to_detector is not None
    ):
        parser.error("--source-to-axis and --source-to-detector require --geometry cone")
    if geometry == "parallel" and args.tilt is not None:
        parser.error("--tilt applies to --geometry lamino or cone, not parallel")
    try:
        return replace(SimulationArtifacts(), **artifact_settings) if artifact_settings else None
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax simulate``: :func:`tomojax.simulate`, saved with its scan."""
    parser = _build_parser()
    args, metadata = parse_args_with_config(parser, argv)

    import tomojax as tj
    from tomojax._simulate import build_sim_config, simulate_config

    def run() -> Result | None:
        settings = cast("dict[str, object]", metadata["settings"])
        out = cast("str", args.out)
        check_paths(parser, outputs=[out], force=cast("bool", args.force))
        artifacts = _validate_args(
            parser, args, {k: v for k, v in settings.items() if k not in _PHANTOM_SETTINGS}
        )
        shape = cast("list[int] | None", args.grid)
        pixels = cast("list[int] | None", args.detector)
        grid = None if shape is None else tj.Grid(shape[0], shape[1], shape[2], 1, 1, 1)
        detector = None if pixels is None else tj.Detector(pixels[0], pixels[1], 1, 1)
        phantom_keys = {name: settings[name] for name in _PHANTOM_SETTINGS if name in settings}
        with transfer_guard_context(cast("TransferGuardName", args.transfer_guard)):
            resolved_grid, resolved_detector, cfg = build_sim_config(
                cast("Phantom", args.phantom),
                size=cast("int", args.size),
                n_views=cast("int | None", args.n_views),
                geometry=cast("GeometryName", args.geometry),
                rotation=cast("float | None", args.rotation),
                tilt=cast("float | None", args.tilt),
                source_to_axis=cast("float | None", args.source_to_axis),
                source_to_detector=cast("float | None", args.source_to_detector),
                poisson_scale=cast("float", args.poisson_scale),
                translation_sigma=args.translation_sigma,
                artifacts=artifacts,
                grid=grid,
                detector=detector,
                seed=cast("int", args.seed),
                phantom_settings=phantom_keys or None,
            )
            result = simulate_config(cfg, grid=resolved_grid, detector=resolved_detector)
        tj.save(out, result)
        if args.progress == "json":
            from tomojax.cli._results import saved_result

            return saved_result(out, result, outputs=(out,))
        scan, phantom = result
        print(
            f"wrote {out}: {args.geometry} scan of a {'x'.join(map(str, phantom.shape[::-1]))} "
            f"{args.phantom} phantom, {len(scan.angles)} views on a "
            f"{scan.detector.nu}x{scan.detector.nv} detector"
        )
        return None

    return run_command("simulate", args.progress, run)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
