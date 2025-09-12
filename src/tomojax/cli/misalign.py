from __future__ import annotations

import argparse
import logging
import os
import numpy as np
import jax
import jax.numpy as jnp

from ..data.io_hdf5 import load_nxtomo, save_nxtomo
from ..core.geometry import Grid, Detector, ParallelGeometry, LaminographyGeometry
from ..align.parametrizations import se3_from_5d
from ..core.projector import forward_project_view_T
from ..utils.logging import setup_logging, log_jax_env


def main() -> None:
    p = argparse.ArgumentParser(description="Create a misaligned (and optionally noisy) dataset from a ground-truth NXtomo file.")
    p.add_argument("--data", required=True, help="Input .nxs containing ground-truth volume and geometry")
    p.add_argument("--out", required=True, help="Output .nxs path")
    p.add_argument("--rot-deg", type=float, default=1.0, help="Max abs rotation per-axis (alpha,beta,phi) in degrees")
    p.add_argument("--trans-px", type=float, default=10.0, help="Max abs translation in detector pixels (dx,dz)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for misalignment")
    p.add_argument("--poisson", type=float, default=0.0, help="Photons per pixel for Poisson noise (0 disables)")
    p.add_argument("--progress", action="store_true", help="Show progress bars if tqdm is available")
    args = p.parse_args()

    setup_logging(); log_jax_env()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    meta = load_nxtomo(args.data)
    grid_d = meta.get("grid"); det_d = meta.get("detector")
    # Grid: infer from volume if not provided
    if grid_d is None:
        if "volume" in meta:
            nx, ny, nz = map(int, meta["volume"].shape)
            grid_d = {"nx": nx, "ny": ny, "nz": nz, "vx": 1.0, "vy": 1.0, "vz": 1.0}
        else:
            raise ValueError("Missing grid metadata and no ground-truth volume to infer from.")
    if det_d is None:
        # Fallback: infer from projections shape
        n_views, nv, nu = meta["projections"].shape
        det_d = {"nu": int(nu), "nv": int(nv), "du": 1.0, "dv": 1.0, "det_center": (0.0, 0.0)}
    if "volume" not in meta:
        raise ValueError("Input file does not contain a ground-truth volume under /entry/processing/tomojax/volume.")

    grid = Grid(**{k: grid_d[k] for k in ("nx","ny","nz","vx","vy","vz")})
    det = Detector(**{k: det_d[k] for k in ("nu","nv","du","dv")}, det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    thetas = meta.get("thetas_deg")
    geom_type = meta.get("geometry_type", "parallel")
    if geom_type == "parallel":
        geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    else:
        tilt_deg = float(meta.get("tilt_deg", 30.0))
        tilt_about = str(meta.get("tilt_about", "x"))
        geom = LaminographyGeometry(grid=grid, detector=det, thetas_deg=thetas, tilt_deg=tilt_deg, tilt_about=tilt_about)

    vol = jnp.asarray(meta["volume"], jnp.float32)
    n_views = int(len(thetas))
    T_nom = jnp.stack([jnp.asarray(geom.pose_for_view(i), jnp.float32) for i in range(n_views)], axis=0)

    # Random per-view parameters
    rng = np.random.default_rng(args.seed)
    rot_scale = np.deg2rad(float(args.rot_deg)).astype(np.float32)
    params5 = np.zeros((n_views, 5), np.float32)
    params5[:, 0] = rng.uniform(-rot_scale, rot_scale, n_views)  # alpha
    params5[:, 1] = rng.uniform(-rot_scale, rot_scale, n_views)  # beta
    params5[:, 2] = rng.uniform(-rot_scale, rot_scale, n_views)  # phi
    params5[:, 3] = rng.uniform(-float(args.trans_px), float(args.trans_px), n_views).astype(np.float32) * float(det.du)
    params5[:, 4] = rng.uniform(-float(args.trans_px), float(args.trans_px), n_views).astype(np.float32) * float(det.dv)
    params5 = jnp.asarray(params5, jnp.float32)

    T_aug = T_nom @ jax.vmap(se3_from_5d)(params5)
    vm_project = jax.vmap(lambda T, v: forward_project_view_T(T, grid, det, v, use_checkpoint=True), in_axes=(0, None))
    proj = vm_project(T_aug, vol).astype(jnp.float32)

    # Optional noise
    if args.poisson and float(args.poisson) > 0:
        s = float(args.poisson)
        lam = np.clip(np.asarray(proj), 0.0, None) * s
        noisy = np.random.default_rng(args.seed + 1).poisson(lam=lam).astype(np.float32) / max(1e-6, s)
        proj = jnp.asarray(noisy, jnp.float32)

    save_nxtomo(
        args.out,
        projections=np.asarray(proj),
        thetas_deg=np.asarray(thetas),
        grid=grid.to_dict(),
        detector=det.to_dict(),
        geometry_type=geom_type,
        volume=np.asarray(vol),
        align_params=np.asarray(params5),
    )
    logging.info("Wrote dataset: %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
