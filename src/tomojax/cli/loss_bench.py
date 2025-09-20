from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from ..data.simulate import SimConfig, simulate_to_file
from ..data.io_hdf5 import load_nxtomo, save_nxtomo
from ..align.pipeline import align, AlignConfig
from ..core.geometry import Grid, Detector, ParallelGeometry, LaminographyGeometry
from ..core.projector import forward_project_view_T, get_detector_grid_device
from ..align.parametrizations import se3_from_5d
from ..utils.logging import setup_logging, log_jax_env


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _make_gt_dataset(expdir: str, *, nx: int, ny: int, nz: int, nu: int, nv: int, n_views: int, geometry: str, seed: int) -> str:
    gt_path = os.path.join(expdir, "gt.nxs")
    if os.path.exists(gt_path):
        return gt_path
    cfg = SimConfig(
        nx=nx, ny=ny, nz=nz, nu=nu, nv=nv, n_views=n_views,
        geometry=geometry, phantom="shepp", rotation_deg=None, seed=seed,
    )
    simulate_to_file(cfg, gt_path)
    return gt_path


def _make_misaligned_dataset(expdir: str, gt_path: str, *, rot_deg: float, trans_px: float, seed: int) -> str:
    mis_path = os.path.join(expdir, "misaligned.nxs")
    if os.path.exists(mis_path):
        return mis_path
    meta = load_nxtomo(gt_path)
    grid_d = meta["grid"]; det_d = meta["detector"]
    grid = Grid(**{k: grid_d[k] for k in ("nx","ny","nz","vx","vy","vz")})
    det = Detector(**{k: det_d[k] for k in ("nu","nv","du","dv")}, det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    thetas = np.asarray(meta["thetas_deg"], dtype=np.float32)
    geom_type = meta.get("geometry_type", "parallel")
    if geom_type == "parallel":
        geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    else:
        tilt_deg = float(meta.get("tilt_deg", 30.0)); tilt_about = str(meta.get("tilt_about", "x"))
        geom = LaminographyGeometry(grid=grid, detector=det, thetas_deg=thetas, tilt_deg=tilt_deg, tilt_about=tilt_about)
    vol = jnp.asarray(meta["volume"], jnp.float32)
    n = len(thetas)
    # Random offsets within ranges
    rng = np.random.default_rng(seed)
    rot_scale = np.deg2rad(float(rot_deg))
    params5 = np.zeros((n, 5), np.float32)
    params5[:, 0] = rng.uniform(-rot_scale, rot_scale, n).astype(np.float32)  # alpha
    params5[:, 1] = rng.uniform(-rot_scale, rot_scale, n).astype(np.float32)  # beta
    params5[:, 2] = rng.uniform(-rot_scale, rot_scale, n).astype(np.float32)  # phi
    params5[:, 3] = rng.uniform(-float(trans_px), float(trans_px), n).astype(np.float32) * float(det.du)
    params5[:, 4] = rng.uniform(-float(trans_px), float(trans_px), n).astype(np.float32) * float(det.dv)
    params5_j = jnp.asarray(params5, jnp.float32)
    T_nom = jnp.stack([jnp.asarray(geom.pose_for_view(i), jnp.float32) for i in range(n)], axis=0)
    T_aug = T_nom @ jax.vmap(se3_from_5d)(params5_j)
    det_grid = get_detector_grid_device(det)
    vm_project = jax.vmap(lambda T: forward_project_view_T(T, grid, det, vol, use_checkpoint=True, det_grid=det_grid), in_axes=0)
    projections = vm_project(T_aug).astype(jnp.float32)
    save_nxtomo(
        mis_path,
        projections=np.asarray(projections),
        thetas_deg=np.asarray(thetas),
        grid=grid.to_dict(),
        detector=det.to_dict(),
        geometry_type=geom_type,
        geometry_meta=meta.get("geometry_meta"),
        volume=np.asarray(vol),
        align_params=np.asarray(params5),
        frame=str(meta.get("frame", "sample")),
    )
    return mis_path


def _metrics(params_true: np.ndarray, params_est: np.ndarray, du: float, dv: float) -> Dict[str, float]:
    # angles in deg, translations in pixels
    assert params_true.shape == params_est.shape
    d = params_est - params_true
    d_deg = np.rad2deg(d[:, :3])
    dx_px = d[:, 3] / max(1e-12, float(du))
    dz_px = d[:, 4] / max(1e-12, float(dv))
    rot_rmse = float(np.sqrt(np.mean(d_deg ** 2)))
    trans_rmse = float(np.sqrt(np.mean(np.stack([dx_px, dz_px], axis=1) ** 2)))
    rot_mae = float(np.mean(np.abs(d_deg)))
    trans_mae = float(np.mean(np.abs(np.stack([dx_px, dz_px], axis=1))))
    return {
        "rot_rmse_deg": rot_rmse,
        "trans_rmse_px": trans_rmse,
        "rot_mae_deg": rot_mae,
        "trans_mae_px": trans_mae,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark multiple alignment losses on a small misaligned phantom.")
    p.add_argument("--expdir", default="runs/loss_experiment", help="Experiment output directory (datasets, logs, results)")
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--nu", type=int, default=128)
    p.add_argument("--nv", type=int, default=128)
    p.add_argument("--n-views", type=int, default=60)
    p.add_argument("--geometry", choices=["parallel", "lamino"], default="parallel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rot-deg", type=float, default=1.0, help="Max |rotation| per-axis used to generate misalignment (degrees)")
    p.add_argument("--trans-px", type=float, default=5.0, help="Max |translation| used to generate misalignment (pixels)")
    p.add_argument("--outer-iters", type=int, default=4)
    p.add_argument("--recon-iters", type=int, default=10)
    p.add_argument("--progress", action="store_true")
    p.add_argument(
        "--losses",
        nargs="*",
        default=[
            "l2","charbonnier","huber","cauchy","barron","student_t","correntropy",
            "zncc","ssim","ms-ssim","mi","nmi","renyi_mi",
            "grad_l1","edge_l2","ngf","grad_orient","phasecorr","fft_mag","chamfer_edge",
            "l2_otsu","ssim_otsu","tversky","swd","mind","pwls","poisson"
        ],
        help="Subset of losses to run; default is a comprehensive set",
    )
    args = p.parse_args()

    _ensure_dir(args.expdir); _ensure_dir(os.path.join(args.expdir, "logs"))
    setup_logging(); log_jax_env()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    # 1) Create datasets if needed
    gt = _make_gt_dataset(args.expdir, nx=args.nx, ny=args.ny, nz=args.nz, nu=args.nu, nv=args.nv, n_views=args.n_views, geometry=args.geometry, seed=args.seed)
    mis = _make_misaligned_dataset(args.expdir, gt, rot_deg=args.rot_deg, trans_px=args.trans_px, seed=args.seed + 1)
    meta_mis = load_nxtomo(mis)
    true_params = np.asarray(meta_mis.get("align_params"), dtype=np.float32)
    det_d = meta_mis.get("detector"); du, dv = float(det_d.get("du", 1.0)), float(det_d.get("dv", 1.0))

    # 2) Prepare geometry and arrays
    grid_d = meta_mis["grid"]; grid = Grid(**{k: grid_d[k] for k in ("nx","ny","nz","vx","vy","vz")})
    det = Detector(**{k: det_d[k] for k in ("nu","nv","du","dv")}, det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    thetas = np.asarray(meta_mis["thetas_deg"], dtype=np.float32)
    geom_type = meta_mis.get("geometry_type", "parallel")
    if geom_type == "parallel":
        geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    else:
        tilt_deg = float(meta_mis.get("tilt_deg", 30.0)); tilt_about = str(meta_mis.get("tilt_about", "x"))
        geom = LaminographyGeometry(grid=grid, detector=det, thetas_deg=thetas, tilt_deg=tilt_deg, tilt_about=tilt_about)
    projections = jnp.asarray(meta_mis["projections"], jnp.float32)

    # 3) Run alignment for each loss
    results: List[Dict[str, object]] = []
    for loss in args.losses:
        run_name = str(loss).lower()
        out_path = os.path.join(args.expdir, f"align_{run_name}.nxs")
        log_path = os.path.join(args.expdir, "logs", f"{run_name}.log")
        # Per-run logging to file
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logging.getLogger().addHandler(fh)
        start = time.perf_counter()
        status = "ok"; err_msg = None
        metrics = {}
        try:
            # Reasonable defaults for small problems; use GD for all losses
            cfg = AlignConfig(
                outer_iters=args.outer_iters,
                recon_iters=args.recon_iters,
                lambda_tv=0.005,
                tv_prox_iters=10,
                lr_rot=1e-3,
                lr_trans=1e-1,
                views_per_batch=1,
                projector_unroll=1,
                checkpoint_projector=True,
                gather_dtype="auto",
                opt_method="gd",
                gn_damping=1e-3,
                w_rot=1e-3,
                w_trans=1e-3,
                seed_translations=False,
                log_summary=True,
                log_compact=True,
                recon_L=None,
                early_stop=True,
                early_stop_rel_impr=1e-3,
                early_stop_patience=2,
                loss_kind=run_name,
                loss_params=None,
            )
            x_est, p_est, info = align(geom, grid, det, projections, cfg=cfg)
            # Save output
            save_nxtomo(
                out_path,
                projections=np.asarray(meta_mis["projections"]),
                thetas_deg=np.asarray(thetas),
                grid=meta_mis.get("grid"),
                detector=meta_mis.get("detector"),
                geometry_type=meta_mis.get("geometry_type", "parallel"),
                geometry_meta=meta_mis.get("geometry_meta"),
                volume=np.asarray(x_est),
                align_params=np.asarray(p_est),
                frame=str(meta_mis.get("frame", "sample")),
            )
            # Metrics
            metrics = _metrics(true_params, np.asarray(p_est), du=du, dv=dv)
        except Exception as e:
            status = "error"
            err_msg = str(e)
            logging.exception("Loss %s failed", run_name)
        elapsed = time.perf_counter() - start
        # Remove file handler to avoid duplicating logs next run
        logging.getLogger().removeHandler(fh)
        try:
            fh.close()
        except Exception:
            pass
        rec = {
            "loss": run_name,
            "status": status,
            "seconds": elapsed,
            **metrics,
            "log": os.path.relpath(log_path, args.expdir),
            "output": os.path.relpath(out_path, args.expdir),
            "error": err_msg,
        }
        results.append(rec)
        logging.info("%s: status=%s time=%.2fs metrics=%s", run_name, status, elapsed, {k:v for k,v in metrics.items()})

    # 4) Save summary
    with open(os.path.join(args.expdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "config": {k: getattr(args, k) for k in vars(args)}}, f, indent=2)
    # Simple CSV too
    csv_path = os.path.join(args.expdir, "results.csv")
    keys = ["loss", "status", "seconds", "rot_rmse_deg", "trans_rmse_px", "rot_mae_deg", "trans_mae_px", "log", "output", "error"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            row = [str(r.get(k, "")) for k in keys]
            f.write(",".join(row) + "\n")
    best = None
    try:
        ok = [r for r in results if r.get("status") == "ok"]
        best = min(ok, key=lambda r: (r.get("rot_rmse_deg", 1e9) + r.get("trans_rmse_px", 1e9))) if ok else None
    except Exception:
        best = None
    if best:
        logging.info("Best by (rot_rmse_deg + trans_rmse_px): %s", best)
    print(f"Results written to {args.expdir} (results.json, results.csv)")


if __name__ == "__main__":  # pragma: no cover
    main()
