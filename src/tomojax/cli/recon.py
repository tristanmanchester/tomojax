from __future__ import annotations

import argparse
import logging
import numpy as np
import jax.numpy as jnp
import os

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


def main() -> None:
    p = argparse.ArgumentParser(description="Reconstruct volume from dataset (.nxs)")
    p.add_argument("--data", required=True, help="Input .nxs")
    p.add_argument("--algo", choices=["fbp", "fista"], default="fbp")
    p.add_argument("--filter", default="ramp", help="FBP filter: ramp|shepp|hann")
    p.add_argument("--iters", type=int, default=50, help="FISTA iterations")
    p.add_argument("--lambda-tv", type=float, default=0.005, help="TV regularization weight")
    p.add_argument("--L", type=float, default=None, help="Fixed Lipschitz constant for FISTA (skip power-method)")
    p.add_argument("--views-per-batch", default="0", help="Batch views to control memory (applies to FISTA and FBP). Use 'auto' to estimate.")
    p.add_argument("--projector-unroll", type=int, default=1, help="Unroll factor inside projector scan")
    p.add_argument("--gather-dtype", choices=["fp32", "bf16", "fp16"], default="fp32", help="Projector gather dtype (accumulation stays fp32)")
    ck = p.add_mutually_exclusive_group()
    ck.add_argument("--checkpoint-projector", dest="checkpoint_projector", action="store_true", help="Enable projector checkpointing")
    ck.add_argument("--no-checkpoint-projector", dest="checkpoint_projector", action="store_false", help="Disable projector checkpointing")
    p.set_defaults(checkpoint_projector=True)
    p.add_argument("--out", required=True, help="Output .nxs containing recon (and copying projections)")
    p.add_argument("--progress", action="store_true", help="Show progress bars if tqdm is available")
    args = p.parse_args()

    setup_logging(); log_jax_env()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    meta = load_nxtomo(args.data)
    grid, detector, geom = build_geometry(meta)
    proj = jnp.asarray(meta["projections"], dtype=jnp.float32)

    # Resolve views_per_batch (int or 'auto')
    vpb_str = str(args.views_per_batch)
    vpb_val: int | None
    if vpb_str.strip().lower() == "auto":
        from ..utils.memory import estimate_views_per_batch
        est = estimate_views_per_batch(
            n_views=int(proj.shape[0]),
            grid_nxyz=(int(grid.nx), int(grid.ny), int(grid.nz)),
            det_nuv=(int(detector.nv), int(detector.nu)),
            gather_dtype=str(args.gather_dtype),
            checkpoint_projector=bool(args.checkpoint_projector),
            algo=str(args.algo),
        )
        vpb_val = est
    else:
        vpb_val = int(vpb_str)

    if args.algo == "fbp":
        vol = fbp(
            geom,
            grid,
            detector,
            proj,
            filter_name=args.filter,
            views_per_batch=int(vpb_val),
            projector_unroll=int(args.projector_unroll),
            checkpoint_projector=bool(args.checkpoint_projector),
            gather_dtype=str(args.gather_dtype),
        )
    else:
        vpb = int(vpb_val) if int(vpb_val) > 0 else None
        vol, info = fista_tv(
            geom,
            grid,
            detector,
            proj,
            iters=args.iters,
            lambda_tv=args.lambda_tv,
            L=(float(args.L) if args.L is not None else None),
            views_per_batch=vpb,
            projector_unroll=int(args.projector_unroll),
            checkpoint_projector=bool(args.checkpoint_projector),
            gather_dtype=str(args.gather_dtype),
        )

    save_nxtomo(
        args.out,
        projections=np.asarray(proj),
        thetas_deg=np.asarray(meta["thetas_deg"]),
        grid=meta.get("grid"),
        detector=meta.get("detector"),
        geometry_type=meta.get("geometry_type", "parallel"),
        volume=np.asarray(vol),
    )
    logging.info("Saved reconstruction to %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
