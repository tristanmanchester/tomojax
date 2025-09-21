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


def _metrics_abs(params_true: np.ndarray, params_est: np.ndarray, du: float, dv: float) -> Dict[str, float]:
    """Absolute parameter RMSE/MAE in degrees/pixels (gauge-sensitive)."""
    assert params_true.shape == params_est.shape
    d = params_est - params_true
    d_deg = np.rad2deg(d[:, :3])
    dx_px = d[:, 3] / max(1e-12, float(du))
    dz_px = d[:, 4] / max(1e-12, float(dv))
    rot_rmse = float(np.sqrt(np.mean(d_deg ** 2)))
    trans_rmse = float(np.sqrt(np.mean(np.stack([dx_px, dz_px], axis=1) ** 2)))
    rot_mse = float(np.mean(d_deg ** 2))
    trans_mse = float(np.mean(np.stack([dx_px, dz_px], axis=1) ** 2))
    rot_mae = float(np.mean(np.abs(d_deg)))
    trans_mae = float(np.mean(np.abs(np.stack([dx_px, dz_px], axis=1))))
    return {
        "rot_rmse_deg": rot_rmse,
        "trans_rmse_px": trans_rmse,
        "rot_mse_deg": rot_mse,
        "trans_mse_px": trans_mse,
        "rot_mae_deg": rot_mae,
        "trans_mae_px": trans_mae,
    }


def _so3_geodesic_deg(R: np.ndarray) -> float:
    """Angle in degrees of rotation matrix R (assumes proper rotation)."""
    tr = np.clip((np.trace(R[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def _inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]; t = T[:3, 3]
    Rt = R.T
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = Rt
    Ti[:3, 3] = -Rt @ t
    return Ti


def _params_to_T(params: np.ndarray) -> np.ndarray:
    """Convert (N,5) params to (N,4,4) transforms using se3_from_5d."""
    p_j = jnp.asarray(params, jnp.float32)
    T = jax.vmap(se3_from_5d)(p_j)
    return np.asarray(T)


def _metrics_relative(params_true: np.ndarray, params_est: np.ndarray, du: float, dv: float, *, k_step: int = 1) -> Dict[str, float]:
    """Gauge-invariant relative-motion error over k-step differences.

    Computes D_true = T_{i+k} T_i^{-1} and D_est analogously, then errors from
    E_i = D_true^{-1} D_est.
    """
    assert params_true.shape == params_est.shape
    n = params_true.shape[0]
    if n <= k_step:
        return {"rot_rel_rmse_deg": float("nan"), "trans_rel_rmse_px": float("nan")}
    Tt = _params_to_T(params_true)
    Te = _params_to_T(params_est)
    rot_errs: List[float] = []
    trans_errs_sq: List[float] = []
    for i in range(n - k_step):
        Dt = Tt[i + k_step] @ _inv_T(Tt[i])
        De = Te[i + k_step] @ _inv_T(Te[i])
        E = _inv_T(Dt) @ De
        rot_errs.append(_so3_geodesic_deg(E))
        tx, tz = float(E[0, 3]), float(E[2, 3])
        ex = tx / max(1e-12, float(du))
        ez = tz / max(1e-12, float(dv))
        trans_errs_sq.append(ex * ex + ez * ez)
    rot_rel_rmse = float(np.sqrt(np.mean(np.square(rot_errs)))) if rot_errs else float("nan")
    trans_rel_rmse = float(np.sqrt(np.mean(trans_errs_sq))) if trans_errs_sq else float("nan")
    return {
        "rot_rel_rmse_deg": rot_rel_rmse,
        "trans_rel_rmse_px": trans_rel_rmse,
    }


def _metrics_gauge_fixed(params_true: np.ndarray, params_est: np.ndarray, du: float, dv: float) -> Dict[str, float]:
    """Gauge-fixed error by solving a single global G that aligns est to true.

    Rotation via orthogonal Procrustes on SO(3); translation by mean residual.
    """
    Tt = _params_to_T(params_true)
    Te = _params_to_T(params_est)
    Rsum = np.zeros((3, 3), dtype=np.float64)
    for i in range(Tt.shape[0]):
        Rsum += Tt[i, :3, :3] @ Te[i, :3, :3].T
    U, _, Vt = np.linalg.svd(Rsum)
    Rg = U @ Vt
    if np.linalg.det(Rg) < 0:
        U[:, -1] *= -1
        Rg = U @ Vt
    # Translation offset
    t_res = []
    for i in range(Tt.shape[0]):
        t_true = Tt[i, :3, 3]
        t_est = Te[i, :3, 3]
        t_res.append(t_true - Rg @ t_est)
    t_res = np.asarray(t_res)
    tg = np.mean(t_res, axis=0)
    G = np.eye(4, dtype=np.float32)
    G[:3, :3] = Rg.astype(np.float32)
    G[:3, 3] = tg.astype(np.float32)
    # Residuals after gauge-fix
    rot_errs: List[float] = []
    trans_errs_sq: List[float] = []
    for i in range(Tt.shape[0]):
        E = _inv_T(Tt[i]) @ (G @ Te[i])
        rot_errs.append(_so3_geodesic_deg(E))
        tx, tz = float(E[0, 3]), float(E[2, 3])
        ex = tx / max(1e-12, float(du))
        ez = tz / max(1e-12, float(dv))
        trans_errs_sq.append(ex * ex + ez * ez)
    rot_rmse = float(np.sqrt(np.mean(np.square(rot_errs)))) if rot_errs else float("nan")
    trans_rmse = float(np.sqrt(np.mean(trans_errs_sq))) if trans_errs_sq else float("nan")
    return {
        "rot_gf_rmse_deg": rot_rmse,
        "trans_gf_rmse_px": trans_rmse,
    }


def _gt_projection_mse(
    x_gt: jnp.ndarray,
    grid: Grid,
    det: Detector,
    geom: ParallelGeometry | LaminographyGeometry,
    params_est: np.ndarray,
) -> float:
    """Forward-project the GT volume with estimated poses; return MSE vs measured proj.

    Uses current process device; modest sizes recommended.
    """
    thetas = np.asarray(geom.thetas_deg, dtype=np.float32)
    T_nom = jnp.stack([jnp.asarray(geom.pose_for_view(i), jnp.float32) for i in range(len(thetas))], axis=0)
    S_est = jax.vmap(se3_from_5d)(jnp.asarray(params_est, jnp.float32))
    T_est_full = T_nom @ S_est
    det_grid = get_detector_grid_device(det)
    vm_project = jax.vmap(lambda T: forward_project_view_T(T, grid, det, x_gt, use_checkpoint=True, det_grid=det_grid), in_axes=0)
    y_hat = vm_project(T_est_full).astype(jnp.float32)
    # Measured projections must be fetched from geom context; caller will provide via closure
    return y_hat


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
    p.add_argument("--metrics-only", action="store_true", help="Do not run alignment; only (re)compute metrics for existing outputs")
    p.add_argument("--k-step", type=int, default=1, help="k-step for relative-motion metric (default: 1)")
    p.add_argument("--gt-metric", choices=["none","mse"], default="mse", help="Physics-aware metric against GT projections")
    p.add_argument("--levels", type=int, nargs="*", default=None, help="Optional multiresolution pyramid factors (e.g., 4 2 1). Applied to all losses.")
    p.add_argument(
        "--losses",
        nargs="*",
        default=["l2", "l2_otsu", "pwls"],
        help="Subset of losses to run; default tests LS-like losses (GN-only)",
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

    # 3) Loss-specific configuration (tiered settings)
    gn_losses = {"l2", "l2_otsu", "pwls", "edge_l2"}
    high_iter_losses = {"l2", "l2_otsu", "pwls", "edge_l2"}
    default_levels = {
        "l2": (4, 2, 1),
        "l2_otsu": (4, 2, 1),
        "edge_l2": (4, 2, 1),
        "pwls": (4, 2, 1),
    }
    rot_rates = {
        "l2": 5e-4,
        "l2_otsu": 5e-4,
        "edge_l2": 5e-4,
        "pwls": 5e-4,
    }
    trans_rates = {
        "l2": 5e-2,
        "l2_otsu": 5e-2,
        "edge_l2": 5e-2,
        "pwls": 5e-2,
    }
    # LBFGS removed

    # 4) Run alignment for each loss or load existing outputs
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
        is_gn = (run_name in gn_losses)
        levels = default_levels.get(run_name)
        if args.levels is not None and len(args.levels) > 0:
            levels = tuple(int(v) for v in args.levels)
        logging.info("=== [%s] starting (GN) ===", run_name.upper())
        if levels is not None:
            logging.info("[%s] multires levels: %s", run_name, levels)
        start = time.perf_counter()
        status = "ok"; err_msg = None
        metrics: Dict[str, float] = {}
        try:
            p_est_np: np.ndarray
            x_est_np: np.ndarray | None = None
            if os.path.exists(out_path):
                logging.info("[%s] output exists; skipping alignment and recomputing metrics", run_name)
                meta_out = load_nxtomo(out_path)
                p_est_np = np.asarray(meta_out.get("align_params"), dtype=np.float32)
                # prefer saved recon if present for possible future metrics
                x_est_np = np.asarray(meta_out.get("volume")) if meta_out.get("volume") is not None else None
            elif args.metrics_only:
                logging.warning("[%s] metrics-only set and output missing; skipping (status=missing)", run_name)
                status = "missing"
                p_est_np = None  # type: ignore
            else:
                # Run alignment (GN-only for LS-like losses)
                if not is_gn:
                    logging.warning("[%s] skipped: not an LS-like loss (GN-only mode)", run_name)
                    status = "skipped"
                    p_est_np = None  # type: ignore
                else:
                    outer_iters = args.outer_iters if run_name not in high_iter_losses else max(args.outer_iters, 8)
                    recon_iters = args.recon_iters if run_name not in high_iter_losses else max(args.recon_iters, 30)
                    opt_method = "gn"
                    cfg = AlignConfig(
                        outer_iters=outer_iters,
                        recon_iters=recon_iters,
                        lambda_tv=0.005,
                        tv_prox_iters=10,
                        lr_rot=rot_rates.get(run_name, 2e-4 if not is_gn else 5e-4),
                        lr_trans=trans_rates.get(run_name, 5e-2 if not is_gn else 5e-2),
                        views_per_batch=1,
                        projector_unroll=1,
                        checkpoint_projector=True,
                        gather_dtype="auto",
                        opt_method=opt_method,
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
                    if levels is not None:
                        from ..align.pipeline import align_multires
                        x_est, p_est, info = align_multires(geom, grid, det, projections, factors=levels, cfg=cfg)
                    else:
                        x_est, p_est, info = align(geom, grid, det, projections, cfg=cfg)
                    p_est_np = np.asarray(p_est)
                    x_est_np = np.asarray(x_est)
                    # Save output
                    save_nxtomo(
                        out_path,
                        projections=np.asarray(meta_mis["projections"]),
                        thetas_deg=np.asarray(thetas),
                        grid=meta_mis.get("grid"),
                        detector=meta_mis.get("detector"),
                        geometry_type=meta_mis.get("geometry_type", "parallel"),
                        geometry_meta=meta_mis.get("geometry_meta"),
                        volume=x_est_np,
                        align_params=p_est_np,
                        frame=str(meta_mis.get("frame", "sample")),
                    )

            # Metrics (only if we have estimates)
            if status != "missing":
                abs_m = _metrics_abs(true_params, p_est_np, du=du, dv=dv)
                rel_m = _metrics_relative(true_params, p_est_np, du=du, dv=dv, k_step=args.k_step)
                gf_m = _metrics_gauge_fixed(true_params, p_est_np, du=du, dv=dv)
                metrics.update(abs_m)
                metrics.update(rel_m)
                metrics.update(gf_m)
                if args.gt_metric != "none":
                    # Compute physics-aware GT projection MSE
                    y_hat = _gt_projection_mse(jnp.asarray(meta_mis["volume"], jnp.float32), grid, det, geom, p_est_np)
                    y = jnp.asarray(meta_mis["projections"], jnp.float32)
                    gt_mse = float(jnp.mean((y_hat - y) ** 2).item())
                    metrics["gt_mse"] = gt_mse
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
        logging.info(
            "[%s] status=%s time=%.2fs | abs: rot_rmse=%.3f° trans_rmse=%.3fpx | rel(k=%d): rot=%.3f° trans=%.3fpx | gf: rot=%.3f° trans=%.3fpx | gt_mse=%.3e",
            run_name,
            status,
            elapsed,
            metrics.get("rot_rmse_deg", float("nan")),
            metrics.get("trans_rmse_px", float("nan")),
            args.k_step,
            metrics.get("rot_rel_rmse_deg", float("nan")),
            metrics.get("trans_rel_rmse_px", float("nan")),
            metrics.get("rot_gf_rmse_deg", float("nan")),
            metrics.get("trans_gf_rmse_px", float("nan")),
            metrics.get("gt_mse", float("nan")),
        )
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
    keys = [
        "loss",
        "status",
        "seconds",
        "rot_rmse_deg",
        "rot_mse_deg",
        "trans_rmse_px",
        "trans_mse_px",
        "rot_mae_deg",
        "trans_mae_px",
        "rot_rel_rmse_deg",
        "trans_rel_rmse_px",
        "rot_gf_rmse_deg",
        "trans_gf_rmse_px",
        "gt_mse",
        "log",
        "output",
        "error",
    ]
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
