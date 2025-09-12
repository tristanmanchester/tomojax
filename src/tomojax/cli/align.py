from __future__ import annotations

import argparse
import logging
import numpy as np
import jax.numpy as jnp
import os

from ..data.io_hdf5 import load_nxtomo, save_nxtomo
from ..core.geometry import Grid, Detector, ParallelGeometry, LaminographyGeometry
from ..align.pipeline import align, AlignConfig
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
    p = argparse.ArgumentParser(description="Joint reconstruction + alignment on dataset (.nxs)")
    p.add_argument("--data", required=True, help="Input .nxs")
    p.add_argument("--outer-iters", type=int, default=5)
    p.add_argument("--recon-iters", type=int, default=10)
    p.add_argument("--lambda-tv", type=float, default=0.005)
    p.add_argument("--lr-rot", type=float, default=1e-3)
    p.add_argument("--lr-trans", type=float, default=1e-1)
    p.add_argument("--levels", type=int, nargs="+", default=None, help="Optional multires factors, e.g., 4 2 1")
    p.add_argument("--views-per-batch", default="0", help="Batch views to control memory (0=all). Use 'auto' to estimate.")
    p.add_argument("--projector-unroll", type=int, default=1, help="Unroll factor inside projector scan")
    p.add_argument("--gather-dtype", choices=["fp32", "bf16", "fp16"], default="fp32", help="Projector gather dtype")
    ck = p.add_mutually_exclusive_group()
    ck.add_argument("--checkpoint-projector", dest="checkpoint_projector", action="store_true")
    ck.add_argument("--no-checkpoint-projector", dest="checkpoint_projector", action="store_false")
    p.set_defaults(checkpoint_projector=True)
    p.add_argument("--opt-method", choices=["gd", "gn"], default="gd", help="Alignment optimizer: gd or gn")
    p.add_argument("--gn-damping", type=float, default=1e-3, help="Levenberg-Marquardt damping for GN")
    p.add_argument("--w-rot", type=float, default=1e-3, help="Smoothness weight for rotations")
    p.add_argument("--w-trans", type=float, default=1e-3, help="Smoothness weight for translations")
    p.add_argument("--seed-translations", action="store_true", help="Phase-correlation init for dx,dz at coarsest level")
    p.add_argument("--log-summary", action="store_true", help="Print per-outer summaries (FISTA loss, alignment loss before/after)")
    p.add_argument("--recon-L", type=float, default=None, help="Fixed Lipschitz constant for FISTA inside alignment (skip power-method)")
    p.add_argument("--out", required=True, help="Output .nxs with recon and alignment params")
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
    if vpb_str.strip().lower() == "auto":
        from ..utils.memory import estimate_views_per_batch
        vpb_est = estimate_views_per_batch(
            n_views=int(proj.shape[0]),
            grid_nxyz=(int(grid.nx), int(grid.ny), int(grid.nz)),
            det_nuv=(int(detector.nv), int(detector.nu)),
            gather_dtype=str(args.gather_dtype),
            checkpoint_projector=bool(args.checkpoint_projector),
            algo="fista",
        )
    else:
        vpb_est = int(vpb_str)

    cfg = AlignConfig(
        outer_iters=args.outer_iters,
        recon_iters=args.recon_iters,
        lambda_tv=args.lambda_tv,
        lr_rot=args.lr_rot,
        lr_trans=args.lr_trans,
        views_per_batch=int(vpb_est),
        projector_unroll=int(args.projector_unroll),
        checkpoint_projector=bool(args.checkpoint_projector),
        gather_dtype=str(args.gather_dtype),
        opt_method=str(args.opt_method),
        gn_damping=float(args.gn_damping),
        w_rot=float(args.w_rot),
        w_trans=float(args.w_trans),
        seed_translations=bool(args.seed_translations),
        log_summary=bool(args.log_summary),
        recon_L=(float(args.recon_L) if args.recon_L is not None else None),
    )
    if args.levels is not None and len(args.levels) > 0:
        from ..align.pipeline import align_multires
        x, params5, info = align_multires(geom, grid, detector, proj, factors=args.levels, cfg=cfg)
    else:
        x, params5, info = align(geom, grid, detector, proj, cfg=cfg)

    save_nxtomo(
        args.out,
        projections=np.asarray(proj),
        thetas_deg=np.asarray(meta["thetas_deg"]),
        grid=meta.get("grid"),
        detector=meta.get("detector"),
        geometry_type=meta.get("geometry_type", "parallel"),
        volume=np.asarray(x),
        align_params=np.asarray(params5),
    )
    logging.info("Saved alignment results to %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
