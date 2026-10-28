"""``tomojax simulate``: write a synthetic scan of a phantom."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import math
import os
from typing import TYPE_CHECKING, Literal, cast

from tomojax.cli._options import add_config, add_output, check_paths, hide_expert
from tomojax.cli.config import parse_args_with_config
from tomojax.core import log_jax_env, setup_logging
from tomojax.datasets import SimConfig, SimulationArtefacts, simulate_to_file

from ._runtime import transfer_guard_context

if TYPE_CHECKING:
    from collections.abc import Sequence

GeometryName = Literal["parallel", "lamino", "cone"]
PhantomName = Literal["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"]
TransferGuardName = Literal["off", "log", "disallow"]


@dataclass(frozen=True)
class SimulateCommand:
    """Typed command plan for synthetic dataset simulation."""

    out: str
    config: SimConfig
    transfer_guard: TransferGuardName
    progress: bool


_PUBLIC = (
    "--size",
    "--n-views",
    "--geometry",
    "--phantom",
    "--rotation-deg",
    "--tilt-deg",
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
            "  tomojax simulate -o lamino.nxs --geometry lamino --tilt-deg 30 --poisson-scale 1000"
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
        "--rotation-deg",
        type=float,
        default=None,
        metavar="DEG",
        help="Rotation range (default 180 for parallel, 360 otherwise)",
    )
    _ = parser.add_argument(
        "--tilt-deg",
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


def _cone_detector(size: int, source_to_axis: float, source_to_detector: float) -> int:
    """Detector pixels that see a ``size``-voxel cube from ``source_to_axis``."""
    radius = size / math.sqrt(2.0)
    if source_to_axis <= radius:
        raise ValueError("the source must be outside the volume: increase --source-to-axis")
    across = radius * source_to_detector / math.sqrt(source_to_axis**2 - radius**2)
    up = 0.5 * size * source_to_detector / (source_to_axis - radius)
    return 2 * math.ceil(max(across, up))


def _parse_command(argv: Sequence[str] | None) -> SimulateCommand:
    """Parse CLI arguments into a typed simulation command plan."""
    parser = _build_parser()
    args, metadata = parse_args_with_config(parser, argv)
    settings = cast("dict[str, object]", metadata["settings"])
    check_paths(parser, outputs=[cast("str", args.out)], force=cast("bool", args.force))
    artefacts = replace(
        SimulationArtefacts(poisson_scale=cast("float", args.poisson_scale)),
        **{name: settings[name] for name in _ARTEFACT_SETTINGS if name in settings},
    )
    rotation_deg = cast("float | None", args.rotation_deg)
    geometry = cast("GeometryName", args.geometry)
    tilt_deg = cast("float | None", args.tilt_deg)
    size = cast("int", args.size)
    nx, ny, nz = cast("list[int] | None", args.grid) or (size, size, size)
    sod = cast("float | None", args.source_to_axis)
    sdd = cast("float | None", args.source_to_detector)
    detector = cast("list[int] | None", args.detector)
    if detector is not None:
        nu, nv = detector
    elif geometry == "cone":
        sod_default = 3.0 * max(nx, ny, nz) if sod is None else sod
        try:
            n = _cone_detector(
                max(nx, ny, nz), sod_default, 1.5 * sod_default if sdd is None else sdd
            )
        except ValueError as exc:
            parser.error(str(exc))
        nu = nv = n
    else:
        nu, nv = max(nx, ny), nz
    n_views = cast("int | None", args.n_views) or 2 * size
    config = SimConfig(
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nu=int(nu),
        nv=int(nv),
        n_views=int(n_views),
        geometry=geometry,
        tilt_deg=tilt_deg if tilt_deg is not None else (30.0 if geometry == "lamino" else 0.0),
        source_to_axis=sod,
        source_to_detector=sdd,
        rotation_deg=rotation_deg,
        phantom=cast("PhantomName", args.phantom),
        seed=cast("int", args.seed),
        artefacts=artefacts if artefacts.has_enabled() else None,
    )
    config = replace(
        config, **{name: settings[name] for name in _PHANTOM_SETTINGS if name in settings}
    )
    return SimulateCommand(
        out=cast("str", args.out),
        config=config,
        transfer_guard=cast("TransferGuardName", args.transfer_guard),
        progress=cast("bool", args.progress),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the synthetic dataset simulation command."""
    command = _parse_command(argv)

    setup_logging()
    log_jax_env()
    if command.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    with transfer_guard_context(command.transfer_guard):
        out = simulate_to_file(command.config, command.out)
    cfg = command.config
    print(
        f"wrote {out}: {cfg.geometry} scan of a {cfg.nx}x{cfg.ny}x{cfg.nz} {cfg.phantom} "
        f"phantom, {cfg.n_views} views on a {cfg.nu}x{cfg.nv} detector"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
