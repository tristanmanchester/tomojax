from __future__ import annotations

import argparse
import logging
import numpy as np
import jax.numpy as jnp
import os
from contextlib import nullcontext as _nullcontext

from ..data.io_hdf5 import load_nxtomo, save_nxtomo
from ..core.geometry import Grid, Detector, ParallelGeometry, LaminographyGeometry
from ..recon.fbp import fbp
from ..recon.fista_tv import fista_tv
from ..utils.logging import setup_logging, log_jax_env


def build_geometry(meta: dict):
    grid_d = meta["grid"]; det_d = meta["detector"]; thetas = meta["thetas_deg"]; gtype = meta.get("geometry_type", "parallel")
    grid = Grid(**{k: grid_d[k] for k in ("nx","ny","nz","vx","vy","vz")})
    detector = Detector(**{k: det_d[k] for k in ("nu","nv","du","dv")}, det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    if gtype == "parallel":
        geom = ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)
    else:
        tilt_deg = float(meta.get("tilt_deg", 30.0))
        tilt_about = str(meta.get("tilt_about", "x"))
        geom = LaminographyGeometry(grid=grid, detector=detector, thetas_deg=thetas, tilt_deg=tilt_deg, tilt_about=tilt_about)
    return grid, detector, geom

def _transfer_guard_ctx(mode: str | None = None):
    # Allow overriding via env var: off|log|disallow
    if mode is None:
        mode = os.environ.get("TOMOJAX_TRANSFER_GUARD", "log").lower()
    if mode in ("off", "none", "disable", "disabled"):
        return _nullcontext()
    try:
        import jax as _jax  # local import for flexibility
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


def main() -> None:
    p = argparse.ArgumentParser(description="Reconstruct volume from dataset (.nxs)")
    p.add_argument("--data", required=True, help="Input .nxs")
    p.add_argument("--algo", choices=["fbp", "fista"], default="fbp")
    p.add_argument("--filter", default="ramp", help="FBP filter: ramp|shepp|hann")
    p.add_argument("--iters", type=int, default=50, help="FISTA iterations")
    p.add_argument("--lambda-tv", type=float, default=0.005, help="TV regularization weight")
    p.add_argument("--tv-prox-iters", type=int, default=10, help="Inner iterations for TV proximal operator")
    p.add_argument("--L", type=float, default=None, help="Fixed Lipschitz constant for FISTA (skip power-method)")
    p.add_argument("--gather-dtype", choices=["fp32", "bf16", "fp16"], default="fp32", help="Projector gather dtype (accumulation stays fp32)")
    ck = p.add_mutually_exclusive_group()
    ck.add_argument("--checkpoint-projector", dest="checkpoint_projector", action="store_true", help="Enable projector checkpointing")
    ck.add_argument("--no-checkpoint-projector", dest="checkpoint_projector", action="store_false", help="Disable projector checkpointing")
    p.set_defaults(checkpoint_projector=True)
    p.add_argument("--out", required=True, help="Output .nxs containing recon (and copying projections)")
    p.add_argument(
        "--frame",
        choices=["sample", "lab"],
        default="sample",
        help="Frame to record for saved volume (default: sample).",
    )
    p.add_argument("--progress", action="store_true", help="Show progress bars if tqdm is available")
    args = p.parse_args()

    setup_logging(); log_jax_env()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    meta = load_nxtomo(args.data)
    grid, detector, geom = build_geometry(meta)
    proj = jnp.asarray(meta["projections"], dtype=jnp.float32)

    # Hidden defaults: stream one view at a time; unroll=1
    vpb_val: int | None = 1

    if args.algo == "fbp":
        with _transfer_guard_ctx("log"):
            vol = fbp(
                geom,
                grid,
                detector,
                proj,
                filter_name=args.filter,
                views_per_batch=int(vpb_val),
                projector_unroll=1,
                checkpoint_projector=bool(args.checkpoint_projector),
                gather_dtype=str(args.gather_dtype),
            )
    else:
        vpb = int(vpb_val) if int(vpb_val) > 0 else None
        with _transfer_guard_ctx("log"):
            vol, info = fista_tv(
                geom,
                grid,
                detector,
                proj,
                iters=args.iters,
                lambda_tv=args.lambda_tv,
                L=(float(args.L) if args.L is not None else None),
                views_per_batch=vpb,
                projector_unroll=1,
                checkpoint_projector=bool(args.checkpoint_projector),
                gather_dtype=str(args.gather_dtype),
                tv_prox_iters=int(args.tv_prox_iters),
            )

    # Note: Reconstructions are computed on the object (sample) frame. We persist that by default.
    # If a lab-frame export is desired, we currently only record metadata; a resampling export
    # may be added later if needed for interop.
    # Avoid copying projections back from device: reuse host array from metadata
    save_nxtomo(
        args.out,
        projections=meta["projections"],
        thetas_deg=np.asarray(meta["thetas_deg"]),
        grid=meta.get("grid"),
        detector=meta.get("detector"),
        geometry_type=meta.get("geometry_type", "parallel"),
        geometry_meta=meta.get("geometry_meta"),
        volume=np.asarray(vol),
        frame=str(args.frame),
    )
    logging.info("Saved reconstruction to %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
