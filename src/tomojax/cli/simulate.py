"""CLI: simulate synthetic TomoJAX datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
import sys
from typing import TYPE_CHECKING, Literal, cast

from tomojax.core import log_jax_env, setup_logging
from tomojax.datasets import (
    SimConfig,
    SimulationArtefacts,
    simulate_to_file,
    validate_simulation_artefacts,
)

from ._runtime import transfer_guard_context

if TYPE_CHECKING:
    from collections.abc import Sequence

GeometryName = Literal["parallel", "lamino"]
TiltAxis = Literal["x", "z"]
PhantomName = Literal["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"]
NoiseName = Literal["none", "gaussian", "poisson"]
TransferGuardName = Literal["off", "log", "disallow"]
IntensityDriftModeName = Literal["none", "linear", "sinusoidal"]

_ARTEFACT_OPTION_STRINGS = {
    "--poisson-scale",
    "--gaussian-sigma",
    "--dead-pixel-fraction",
    "--dead-pixel-value",
    "--hot-pixel-fraction",
    "--hot-pixel-value",
    "--zinger-fraction",
    "--zinger-value",
    "--stripe-fraction",
    "--stripe-gain-sigma",
    "--dropped-view-fraction",
    "--dropped-view-fill",
    "--detector-blur-sigma",
    "--intensity-drift-amplitude",
    "--intensity-drift-mode",
}


@dataclass(frozen=True)
class SimulateCommand:
    """Typed command plan for synthetic dataset simulation."""

    out: str
    config: SimConfig
    transfer_guard: TransferGuardName
    progress: bool


def _artefact_options_present(argv: Sequence[str]) -> bool:
    for token in argv:
        for option in _ARTEFACT_OPTION_STRINGS:
            if token == option or token.startswith(f"{option}="):
                return True
    return False


def _build_parser() -> argparse.ArgumentParser:
    """Build the simulate command parser."""
    parser = argparse.ArgumentParser(description="Simulate tomographic dataset and save to .nxs")
    _ = parser.add_argument("--out", required=True, help="Output .nxs path")
    _ = parser.add_argument("--nx", type=int, required=True)
    _ = parser.add_argument("--ny", type=int, required=True)
    _ = parser.add_argument("--nz", type=int, required=True)
    _ = parser.add_argument("--nu", type=int, required=True)
    _ = parser.add_argument("--nv", type=int, required=True)
    _ = parser.add_argument("--n-views", type=int, required=True)
    _ = parser.add_argument(
        "--rotation-deg",
        type=float,
        default=None,
        help="Total rotation range in degrees. Defaults: 180 for parallel, 360 for lamino.",
    )
    _ = parser.add_argument("--geometry", choices=["parallel", "lamino"], default="parallel")
    _ = parser.add_argument("--tilt-deg", type=float, default=30.0)
    _ = parser.add_argument("--tilt-about", choices=["x", "z"], default="x")
    _ = parser.add_argument(
        "--phantom",
        choices=["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"],
        default="shepp",
        help="Phantom type. Use 'cube' or 'sphere' for a single centered object.",
    )
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
    _ = parser.add_argument("--noise", choices=["none", "gaussian", "poisson"], default="none")
    _ = parser.add_argument("--noise-level", type=float, default=0.0)
    _ = parser.add_argument("--poisson-scale", type=float, default=0.0)
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
    _ = parser.add_argument("--seed", type=int, default=0)
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
    return parser


def _parse_command(argv: Sequence[str] | None) -> SimulateCommand:
    """Parse CLI arguments into a typed simulation command plan."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv_list)
    artefacts = _build_artefacts(args, _artefact_options_present(argv_list))
    rotation_deg = cast("float | None", args.rotation_deg)
    config = SimConfig(
        nx=cast("int", args.nx),
        ny=cast("int", args.ny),
        nz=cast("int", args.nz),
        nu=cast("int", args.nu),
        nv=cast("int", args.nv),
        n_views=cast("int", args.n_views),
        geometry=cast("GeometryName", args.geometry),
        tilt_deg=cast("float", args.tilt_deg),
        tilt_about=cast("TiltAxis", args.tilt_about),
        rotation_deg=rotation_deg,
        phantom=cast("PhantomName", args.phantom),
        noise=cast("NoiseName", args.noise),
        noise_level=cast("float", args.noise_level),
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


def _build_artefacts(
    args: argparse.Namespace,
    explicit_artefacts: bool,
) -> SimulationArtefacts | None:
    """Build validated optional artefact config from parsed arguments."""
    if not explicit_artefacts:
        return None

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
    if cast("NoiseName", args.noise) != "none" and cast("float", args.noise_level) > 0.0:
        logging.warning(
            "Ignoring shorthand --noise/--noise-level because explicit artefact "
            "options were supplied"
        )
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
    logging.info("Wrote dataset: %s", out)


if __name__ == "__main__":  # pragma: no cover
    main()
