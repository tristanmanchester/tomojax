"""``tomojax simulate``: write a synthetic scan of a phantom."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from typing import TYPE_CHECKING, Literal, cast

from tomojax.cli._options import add_config, add_output, check_paths, hide_expert
from tomojax.cli.config import parse_args_with_config
from tomojax.core import log_jax_env, setup_logging
from tomojax.datasets import (
    SimConfig,
    SimulationArtefacts,
    simulate_to_file,
)
from tomojax.datasets.api import validate_simulation_artefacts

from ._runtime import transfer_guard_context

if TYPE_CHECKING:
    from collections.abc import Sequence

GeometryName = Literal["parallel", "lamino", "cone"]
TiltAxis = Literal["x", "z"]
PhantomName = Literal["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"]
TransferGuardName = Literal["off", "log", "disallow"]
IntensityDriftModeName = Literal["none", "linear", "sinusoidal"]


@dataclass(frozen=True)
class SimulateCommand:
    """Typed command plan for synthetic dataset simulation."""

    out: str
    config: SimConfig
    transfer_guard: TransferGuardName
    progress: bool


_PUBLIC = (
    "--size",
    "--views",
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
            "  tomojax simulate -o cone.nxs --geometry cone --size 128 --views 360\n"
            "  tomojax simulate -o lamino.nxs --geometry lamino --tilt 30 --poisson-scale 1000"
        ),
    )
    add_output(parser, "Dataset to write (.nxs)")
    add_config(parser)
    _ = parser.add_argument(
        "--size",
        type=int,
        default=64,
        metavar="N",
        help="Volume edge in voxels, and the detector size",
    )
    _ = parser.add_argument(
        "--views",
        dest="n_views",
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
        dest="rotation_deg",
        type=float,
        default=None,
        metavar="DEG",
        help="Rotation range (default 180 for parallel, 360 otherwise)",
    )
    _ = parser.add_argument(
        "--tilt",
        dest="tilt_deg",
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
    _ = parser.add_argument("--tilt-about", choices=["x", "z"], default="x")
    _ = parser.add_argument(
        "--single-rotate",
        dest="single_rotate",
        action="store_true",
        default=True,
        help="Rotate the single cube randomly in 3D (default: on)",
    )
    _ = parser.add_argument("--no-single-rotate", dest="single_rotate", action="store_false")
    _ = parser.add_argument(
        "--single-size",
        type=float,
        default=0.5,
        help="Relative size of cube side or sphere diameter (0-1). Default 0.5",
    )
    _ = parser.add_argument(
        "--single-value", type=float, default=1.0, help="Intensity value for the single object"
    )
    _ = parser.add_argument("--n-cubes", type=int, default=8)
    _ = parser.add_argument("--n-spheres", type=int, default=7)
    _ = parser.add_argument("--min-size", type=int, default=4)
    _ = parser.add_argument("--max-size", type=int, default=32)
    _ = parser.add_argument("--min-value", type=float, default=0.1)
    _ = parser.add_argument("--max-value", type=float, default=1.0)
    _ = parser.add_argument("--max-rot-deg", type=float, default=180.0)
    _ = parser.add_argument(
        "--lamino-thickness-ratio",
        type=float,
        default=0.2,
        help="Relative slab thickness (0-1) used by the lamino disk phantom",
    )
    _ = parser.add_argument("--gaussian-sigma", type=float, default=0.0)
    _ = parser.add_argument("--dead-pixel-fraction", type=float, default=0.0)
    _ = parser.add_argument("--dead-pixel-value", type=float, default=0.0)
    _ = parser.add_argument("--hot-pixel-fraction", type=float, default=0.0)
    _ = parser.add_argument("--hot-pixel-value", type=float, default=1.0)
    _ = parser.add_argument("--zinger-fraction", type=float, default=0.0)
    _ = parser.add_argument("--zinger-value", type=float, default=1.0)
    _ = parser.add_argument("--stripe-fraction", type=float, default=0.0)
    _ = parser.add_argument("--stripe-gain-sigma", type=float, default=0.0)
    _ = parser.add_argument("--dropped-view-fraction", type=float, default=0.0)
    _ = parser.add_argument("--dropped-view-fill", type=float, default=0.0)
    _ = parser.add_argument("--detector-blur-sigma", type=float, default=0.0)
    _ = parser.add_argument("--intensity-drift-amplitude", type=float, default=0.0)
    _ = parser.add_argument(
        "--intensity-drift-mode",
        choices=["none", "linear", "sinusoidal"],
        default="none",
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
    args, _ = parse_args_with_config(parser, argv)
    check_paths(parser, outputs=[cast("str", args.out)], force=cast("bool", args.force))
    artefacts = _build_artefacts(args)
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
        tilt_about=cast("TiltAxis", args.tilt_about),
        source_to_axis=sod,
        source_to_detector=sdd,
        rotation_deg=rotation_deg,
        phantom=cast("PhantomName", args.phantom),
        seed=cast("int", args.seed),
        artefacts=artefacts,
        single_size=cast("float", args.single_size),
        single_value=cast("float", args.single_value),
        single_rotate=cast("bool", args.single_rotate),
        n_cubes=cast("int", args.n_cubes),
        n_spheres=cast("int", args.n_spheres),
        min_size=cast("int", args.min_size),
        max_size=cast("int", args.max_size),
        min_value=cast("float", args.min_value),
        max_value=cast("float", args.max_value),
        max_rot_deg=cast("float", args.max_rot_deg),
        lamino_thickness_ratio=cast("float", args.lamino_thickness_ratio),
    )
    return SimulateCommand(
        out=cast("str", args.out),
        config=config,
        transfer_guard=cast("TransferGuardName", args.transfer_guard),
        progress=cast("bool", args.progress),
    )


def _build_artefacts(args: argparse.Namespace) -> SimulationArtefacts | None:
    """Build validated optional artefact config from parsed arguments."""
    artefacts = SimulationArtefacts(
        poisson_scale=cast("float", args.poisson_scale),
        gaussian_sigma=cast("float", args.gaussian_sigma),
        dead_pixel_fraction=cast("float", args.dead_pixel_fraction),
        dead_pixel_value=cast("float", args.dead_pixel_value),
        hot_pixel_fraction=cast("float", args.hot_pixel_fraction),
        hot_pixel_value=cast("float", args.hot_pixel_value),
        zinger_fraction=cast("float", args.zinger_fraction),
        zinger_value=cast("float", args.zinger_value),
        stripe_fraction=cast("float", args.stripe_fraction),
        stripe_gain_sigma=cast("float", args.stripe_gain_sigma),
        dropped_view_fraction=cast("float", args.dropped_view_fraction),
        dropped_view_fill=cast("float", args.dropped_view_fill),
        detector_blur_sigma=cast("float", args.detector_blur_sigma),
        intensity_drift_amplitude=cast("float", args.intensity_drift_amplitude),
        intensity_drift_mode=cast("IntensityDriftModeName", args.intensity_drift_mode),
    )
    validate_simulation_artefacts(artefacts)
    if not artefacts.has_enabled():
        return None
    return artefacts


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
