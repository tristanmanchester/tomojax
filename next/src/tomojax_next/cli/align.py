from __future__ import annotations

import argparse
import logging
import numpy as np
import jax.numpy as jnp

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
    p.add_argument("--out", required=True, help="Output .nxs with recon and alignment params")
    args = p.parse_args()

    setup_logging(); log_jax_env()
    meta = load_nxtomo(args.data)
    grid, detector, geom = build_geometry(meta)
    proj = jnp.asarray(meta["projections"], dtype=jnp.float32)

    cfg = AlignConfig(
        outer_iters=args.outer_iters,
        recon_iters=args.recon_iters,
        lambda_tv=args.lambda_tv,
        lr_rot=args.lr_rot,
        lr_trans=args.lr_trans,
    )
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

