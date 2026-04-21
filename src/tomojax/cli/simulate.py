from __future__ import annotations

import argparse
import logging
import os
import sys

from ..data.artefacts import SimulationArtefacts, validate_simulation_artefacts
from ..data.simulate import SimConfig, simulate_to_file
from ..utils.logging import setup_logging, log_jax_env
from ._runtime import transfer_guard_context


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


def _artefact_options_present(argv: list[str]) -> bool:
    for token in argv:
        for option in _ARTEFACT_OPTION_STRINGS:
            if token == option or token.startswith(f"{option}="):
                return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Simulate tomographic dataset and save to .nxs")
    p.add_argument("--out", required=True, help="Output .nxs path")
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--ny", type=int, required=True)
    p.add_argument("--nz", type=int, required=True)
    p.add_argument("--nu", type=int, required=True)
    p.add_argument("--nv", type=int, required=True)
    p.add_argument("--n-views", type=int, required=True)
    p.add_argument(
        "--rotation-deg",
        type=float,
        default=None,
        help="Total rotation range in degrees. Defaults: 180 for parallel, 360 for lamino.",
    )
    p.add_argument("--geometry", choices=["parallel", "lamino"], default="parallel")
    p.add_argument("--tilt-deg", type=float, default=30.0)
    p.add_argument("--tilt-about", choices=["x", "z"], default="x")
    p.add_argument(
        "--phantom",
        choices=["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"],
        default="shepp",
        help="Phantom type. Use 'cube' or 'sphere' for a single centered object.",
    )
    # rotate the single cube randomly by default; sphere is unaffected
    p.add_argument("--single-rotate", dest="single_rotate", action="store_true", default=True,
                   help="Rotate the single cube randomly in 3D (default: on)")
    p.add_argument("--no-single-rotate", dest="single_rotate", action="store_false")
    # single-object phantom args (used for phantom=cube|sphere)
    p.add_argument("--single-size", type=float, default=0.5, help="Relative size of cube side or sphere diameter (0-1). Default 0.5")
    p.add_argument("--single-value", type=float, default=1.0, help="Intensity value for the single object")
    # random_shapes args
    p.add_argument("--n-cubes", type=int, default=8)
    p.add_argument("--n-spheres", type=int, default=7)
    p.add_argument("--min-size", type=int, default=4)
    p.add_argument("--max-size", type=int, default=32)
    p.add_argument("--min-value", type=float, default=0.1)
    p.add_argument("--max-value", type=float, default=1.0)
    p.add_argument("--max-rot-deg", type=float, default=180.0)
    p.add_argument(
        "--lamino-thickness-ratio",
        type=float,
        default=0.2,
        help="Relative slab thickness (0-1) used by the lamino disk phantom",
    )
    p.add_argument("--noise", choices=["none", "gaussian", "poisson"], default="none")
    p.add_argument("--noise-level", type=float, default=0.0)
    p.add_argument("--poisson-scale", type=float, default=0.0)
    p.add_argument("--gaussian-sigma", type=float, default=0.0)
    p.add_argument("--dead-pixel-fraction", type=float, default=0.0)
    p.add_argument("--dead-pixel-value", type=float, default=0.0)
    p.add_argument("--hot-pixel-fraction", type=float, default=0.0)
    p.add_argument("--hot-pixel-value", type=float, default=1.0)
    p.add_argument("--zinger-fraction", type=float, default=0.0)
    p.add_argument("--zinger-value", type=float, default=1.0)
    p.add_argument("--stripe-fraction", type=float, default=0.0)
    p.add_argument("--stripe-gain-sigma", type=float, default=0.0)
    p.add_argument("--dropped-view-fraction", type=float, default=0.0)
    p.add_argument("--dropped-view-fill", type=float, default=0.0)
    p.add_argument("--detector-blur-sigma", type=float, default=0.0)
    p.add_argument("--intensity-drift-amplitude", type=float, default=0.0)
    p.add_argument(
        "--intensity-drift-mode",
        choices=["none", "linear", "sinusoidal"],
        default="none",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress", action="store_true", help="Show progress bars if tqdm is available")
    p.add_argument(
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help="JAX transfer guard mode during compute (default: off; use log/disallow when debugging)",
    )
    args = p.parse_args()

    setup_logging()
    log_jax_env()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    explicit_artefacts = _artefact_options_present(sys.argv[1:])
    artefacts = None
    if explicit_artefacts:
        artefacts = SimulationArtefacts(
            poisson_scale=args.poisson_scale,
            gaussian_sigma=args.gaussian_sigma,
            dead_pixel_fraction=args.dead_pixel_fraction,
            dead_pixel_value=args.dead_pixel_value,
            hot_pixel_fraction=args.hot_pixel_fraction,
            hot_pixel_value=args.hot_pixel_value,
            zinger_fraction=args.zinger_fraction,
            zinger_value=args.zinger_value,
            stripe_fraction=args.stripe_fraction,
            stripe_gain_sigma=args.stripe_gain_sigma,
            dropped_view_fraction=args.dropped_view_fraction,
            dropped_view_fill=args.dropped_view_fill,
            detector_blur_sigma=args.detector_blur_sigma,
            intensity_drift_amplitude=args.intensity_drift_amplitude,
            intensity_drift_mode=args.intensity_drift_mode,
        )
        validate_simulation_artefacts(artefacts)
        if artefacts.has_enabled():
            if args.noise != "none" and float(args.noise_level) > 0.0:
                logging.warning(
                    "Ignoring legacy --noise/--noise-level because explicit artefact "
                    "options were supplied"
                )
        else:
            artefacts = None

    cfg = SimConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        nu=args.nu, nv=args.nv, n_views=args.n_views,
        geometry=args.geometry, tilt_deg=args.tilt_deg, tilt_about=args.tilt_about,
        rotation_deg=(float(args.rotation_deg) if args.rotation_deg is not None else None),
        phantom=args.phantom, noise=args.noise, noise_level=args.noise_level, seed=args.seed,
        artefacts=artefacts,
        single_size=args.single_size, single_value=args.single_value, single_rotate=bool(args.single_rotate),
        n_cubes=args.n_cubes, n_spheres=args.n_spheres,
        min_size=args.min_size, max_size=args.max_size,
        min_value=args.min_value, max_value=args.max_value,
        max_rot_deg=args.max_rot_deg,
        lamino_thickness_ratio=args.lamino_thickness_ratio,
    )
    with transfer_guard_context(args.transfer_guard):
        out = simulate_to_file(cfg, args.out)
    logging.info("Wrote dataset: %s", out)


if __name__ == "__main__":  # pragma: no cover
    main()
