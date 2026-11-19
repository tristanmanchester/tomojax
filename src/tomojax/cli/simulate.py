"""``tomojax simulate``: write a synthetic scan of a phantom."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import os
from typing import TYPE_CHECKING, Literal, cast

from tomojax.cli._options import add_config, add_output, check_paths, hide_expert
from tomojax.cli.config import parse_args_with_config
from tomojax.core import log_jax_env, setup_logging
from tomojax.datasets import SimConfig, SimulationArtefacts

from ._runtime import transfer_guard_context

if TYPE_CHECKING:
    from collections.abc import Sequence

GeometryName = Literal["parallel", "lamino", "cone"]
PhantomName = Literal["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"]
TransferGuardName = Literal["off", "log", "disallow"]


_PUBLIC = (
    "--size",
    "--n-views",
    "--geometry",
    "--phantom",
    "--rotation",
    "--tilt",
    "--source-to-axis",
    "--source-to-detector",
    "--poisson-scale",
    "--seed",
    "--progress",
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the simulate command parser."""
    parser = argparse.ArgumentParser(
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
        "--geometry", choices=["parallel", "lamino", "cone"], default="parallel"
    )
    _ = parser.add_argument(
        "--phantom",
        choices=["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"],
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
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Volume shape, overriding --size",
    )
    _ = parser.add_argument(
        "--detector",
        type=int,
        nargs=2,
        metavar=("NU", "NV"),
        default=None,
        help="Detector shape (default: --size, or enough to see the whole cone-beam volume)",
    )
    _ = parser.add_argument(
        "--seed", type=int, default=0, metavar="N", help="Random seed (phantom and noise)"
    )
    _ = parser.add_argument(
        "--progress", action="store_true", help="Show progress bars if tqdm is available"
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


# Phantom details a config file may set; the detector artefacts besides
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
    "max_rot_deg",
    "lamino_thickness_ratio",
)
_ARTEFACT_SETTINGS = tuple(f.name for f in fields(SimulationArtefacts) if f.name != "poisson_scale")


def _settings() -> dict[str, object]:
    """A config file's expert settings: phantom details and detector artefacts."""
    phantom = SimConfig(nx=1, ny=1, nz=1, nu=1, nv=1, n_views=1)
    artefacts = SimulationArtefacts()
    return {name: getattr(phantom, name) for name in _PHANTOM_SETTINGS} | {
        name: getattr(artefacts, name) for name in _ARTEFACT_SETTINGS
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``tomojax simulate``: :func:`tomojax.simulate`, saved with its scan."""
    import tomojax as tj

    parser = _build_parser()
    args, metadata = parse_args_with_config(parser, argv)
    settings = cast("dict[str, object]", metadata["settings"])
    out = cast("str", args.out)
    check_paths(parser, outputs=[out], force=cast("bool", args.force))
    artefacts = replace(
        SimulationArtefacts(),
        **{name: settings[name] for name in _ARTEFACT_SETTINGS if name in settings},
    )
    config = replace(
        SimConfig(nx=1, ny=1, nz=1, nu=1, nv=1, n_views=1, artefacts=artefacts),
        **{name: settings[name] for name in _PHANTOM_SETTINGS if name in settings},
    )
    shape = cast("list[int] | None", args.grid)
    pixels = cast("list[int] | None", args.detector)
    setup_logging()
    log_jax_env()
    if cast("bool", args.progress):
        os.environ["TOMOJAX_PROGRESS"] = "1"
    try:
        with transfer_guard_context(cast("TransferGuardName", args.transfer_guard)):
            result = tj.simulate(
                cast("PhantomName", args.phantom),
                size=cast("int", args.size),
                n_views=cast("int | None", args.n_views),
                geometry=cast("GeometryName", args.geometry),
                rotation=cast("float | None", args.rotation),
                tilt=cast("float | None", args.tilt),
                source_to_axis=cast("float | None", args.source_to_axis),
                source_to_detector=cast("float | None", args.source_to_detector),
                poisson_scale=cast("float", args.poisson_scale),
                grid=None if shape is None else tj.Grid(shape[0], shape[1], shape[2], 1, 1, 1),
                detector=None if pixels is None else tj.Detector(pixels[0], pixels[1], 1, 1),
                seed=cast("int", args.seed),
                config=config,
            )
    except ValueError as exc:
        parser.error(str(exc))
    tj.save(out, result, include_scan=True)
    assert result.scan is not None  # a simulation's scan
    print(
        f"wrote {out}: {args.geometry} scan of a {'x'.join(map(str, result.volume.shape[::-1]))} "
        f"{args.phantom} phantom, {len(result.scan.angles)} views on a "
        f"{result.scan.detector.nu}x{result.scan.detector.nv} detector"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
