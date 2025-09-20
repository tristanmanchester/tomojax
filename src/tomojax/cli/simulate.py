from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext as _nullcontext

from ..data.simulate import SimConfig, simulate_to_file
from ..utils.logging import setup_logging, log_jax_env


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
        choices=["shepp", "cube", "blobs", "random_shapes", "lamino_disk"],
        default="shepp",
    )
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

    cfg = SimConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        nu=args.nu, nv=args.nv, n_views=args.n_views,
        geometry=args.geometry, tilt_deg=args.tilt_deg, tilt_about=args.tilt_about,
        rotation_deg=(float(args.rotation_deg) if args.rotation_deg is not None else None),
        phantom=args.phantom, noise=args.noise, noise_level=args.noise_level, seed=args.seed,
        n_cubes=args.n_cubes, n_spheres=args.n_spheres,
        min_size=args.min_size, max_size=args.max_size,
        min_value=args.min_value, max_value=args.max_value,
        max_rot_deg=args.max_rot_deg,
        lamino_thickness_ratio=args.lamino_thickness_ratio,
    )
    def _transfer_guard_ctx(mode: str | None = None):
        # Allow overriding via env var: off|log|disallow
        if mode is None:
            mode = os.environ.get("TOMOJAX_TRANSFER_GUARD", "log").lower()
        if mode in ("off", "none", "disable", "disabled"):
            return _nullcontext()
        try:
            import jax as _jax
            tg = getattr(_jax, "transfer_guard", None)
            if tg is not None:
                return tg(mode)
            try:
                from jax.experimental import transfer_guard as _tg  # type: ignore
                return _tg(mode)
            except Exception:
                return _nullcontext()
        except Exception:
            return _nullcontext()

    with _transfer_guard_ctx(args.transfer_guard):
        out = simulate_to_file(cfg, args.out)
    logging.info("Wrote dataset: %s", out)


if __name__ == "__main__":  # pragma: no cover
    main()
