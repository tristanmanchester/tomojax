from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import jax.numpy as jnp

from ..bench.loss_experiment import (
    make_gt_dataset as _make_gt_dataset,
    make_misaligned_dataset as _make_misaligned_dataset,
    metrics_abs as _metrics_abs,
    metrics_gauge_fixed as _metrics_gauge_fixed,
    metrics_relative as _metrics_relative,
    project_gt_with_estimated_poses as _project_gt_with_estimated_poses,
)
from ..data.geometry_meta import build_geometry_from_meta
from ..data.io_hdf5 import NXTomoMetadata, load_nxtomo, save_nxtomo
from ..align.losses import parse_loss_spec
from ..align.pipeline import align, AlignConfig
from ..utils.logging import setup_logging, log_jax_env


type BenchmarkResultValue = str | float | None


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _run_benchmark_workflow(args: argparse.Namespace) -> list[dict[str, BenchmarkResultValue]]:
    gt = _make_gt_dataset(
        args.expdir,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        nu=args.nu,
        nv=args.nv,
        n_views=args.n_views,
        geometry=args.geometry,
        seed=args.seed,
    )
    mis = _make_misaligned_dataset(
        args.expdir,
        gt,
        rot_deg=args.rot_deg,
        trans_px=args.trans_px,
        seed=args.seed + 1,
    )
    meta_mis = load_nxtomo(mis)
    if meta_mis.align_params is None:
        raise ValueError("Misaligned benchmark dataset is missing align_params")
    true_params = np.asarray(meta_mis.align_params, dtype=np.float32)
    volume = meta_mis.volume

    grid, det, geom = build_geometry_from_meta(
        meta_mis.geometry_inputs(),
        apply_saved_alignment=False,
        volume_shape=(np.asarray(volume).shape if volume is not None else None),
    )
    du, dv = float(det.du), float(det.dv)
    thetas = np.asarray(meta_mis.thetas_deg, dtype=np.float32)
    projections = jnp.asarray(meta_mis.projections, jnp.float32)

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

    results: list[dict[str, BenchmarkResultValue]] = []
    for loss in args.losses:
        run_name = str(loss).lower()
        out_path = os.path.join(args.expdir, f"align_{run_name}.nxs")
        log_path = os.path.join(args.expdir, "logs", f"{run_name}.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logging.getLogger().addHandler(fh)
        is_gn = run_name in gn_losses
        levels = default_levels.get(run_name)
        if args.levels is not None and len(args.levels) > 0:
            levels = tuple(int(v) for v in args.levels)
        logging.info("=== [%s] starting (GN) ===", run_name.upper())
        if levels is not None:
            logging.info("[%s] multires levels: %s", run_name, levels)
        start = time.perf_counter()
        status = "ok"
        err_msg = None
        metrics: dict[str, float] = {}
        try:
            p_est_np: np.ndarray
            x_est_np: np.ndarray | None = None
            if os.path.exists(out_path):
                logging.info(
                    "[%s] output exists; skipping alignment and recomputing metrics",
                    run_name,
                )
                meta_out = load_nxtomo(out_path)
                if meta_out.align_params is None:
                    raise ValueError(f"{out_path} is missing align_params")
                p_est_np = np.asarray(meta_out.align_params, dtype=np.float32)
                x_est_np = np.asarray(meta_out.volume) if meta_out.volume is not None else None
            elif args.metrics_only:
                logging.warning(
                    "[%s] metrics-only set and output missing; skipping (status=missing)",
                    run_name,
                )
                status = "missing"
                p_est_np = None  # type: ignore[assignment]
            else:
                if not is_gn:
                    logging.warning("[%s] skipped: not an LS-like loss (GN-only mode)", run_name)
                    status = "skipped"
                    p_est_np = None  # type: ignore[assignment]
                else:
                    outer_iters = (
                        args.outer_iters
                        if run_name not in high_iter_losses
                        else max(args.outer_iters, 8)
                    )
                    recon_iters = (
                        args.recon_iters
                        if run_name not in high_iter_losses
                        else max(args.recon_iters, 30)
                    )
                    cfg = AlignConfig(
                        outer_iters=outer_iters,
                        recon_iters=recon_iters,
                        lambda_tv=0.005,
                        tv_prox_iters=10,
                        lr_rot=rot_rates.get(run_name, 5e-4),
                        lr_trans=trans_rates.get(run_name, 5e-2),
                        views_per_batch=1,
                        projector_unroll=1,
                        checkpoint_projector=True,
                        gather_dtype="auto",
                        opt_method="gn",
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
                        loss=parse_loss_spec(run_name),
                    )
                    if levels is not None:
                        from ..align.pipeline import align_multires

                        x_est, p_est, info = align_multires(
                            geom, grid, det, projections, factors=levels, cfg=cfg
                        )
                    else:
                        x_est, p_est, info = align(geom, grid, det, projections, cfg=cfg)
                    p_est_np = np.asarray(p_est)
                    x_est_np = np.asarray(x_est)
                    save_meta = meta_mis.copy_metadata()
                    save_meta.thetas_deg = np.asarray(thetas)
                    save_meta.volume = x_est_np
                    save_meta.align_params = p_est_np
                    save_meta.frame = str(meta_mis.frame or "sample")
                    save_nxtomo(
                        out_path,
                        projections=np.asarray(meta_mis.projections),
                        metadata=save_meta,
                    )

            if status == "ok" and p_est_np is not None:
                abs_m = _metrics_abs(true_params, p_est_np, du=du, dv=dv)
                rel_m = _metrics_relative(true_params, p_est_np, du=du, dv=dv, k_step=args.k_step)
                gf_m = _metrics_gauge_fixed(true_params, p_est_np, du=du, dv=dv)
                metrics.update(abs_m)
                metrics.update(rel_m)
                metrics.update(gf_m)
                if args.gt_metric != "none":
                    y_hat = _project_gt_with_estimated_poses(
                        jnp.asarray(meta_mis.volume, jnp.float32),
                        grid,
                        det,
                        geom,
                        p_est_np,
                    )
                    y = jnp.asarray(meta_mis.projections, jnp.float32)
                    gt_mse = float(jnp.mean((y_hat - y) ** 2).item())
                    metrics["gt_mse"] = gt_mse
        except Exception as e:
            status = "error"
            err_msg = str(e)
            logging.exception("Loss %s failed", run_name)
        elapsed = time.perf_counter() - start
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
        results.append(
            {
                "loss": run_name,
                "status": status,
                "seconds": elapsed,
                **metrics,
                "log": os.path.relpath(log_path, args.expdir),
                "output": os.path.relpath(out_path, args.expdir) if status == "ok" else None,
                "error": err_msg,
            }
        )
        logging.info(
            "%s: status=%s time=%.2fs metrics=%s",
            run_name,
            status,
            elapsed,
            {k: v for k, v in metrics.items()},
        )
    return results


def _write_results_summary(
    expdir: str,
    args: argparse.Namespace,
    results: list[dict[str, BenchmarkResultValue]],
) -> None:
    with open(os.path.join(expdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "config": {k: getattr(args, k) for k in vars(args)}},
            f,
            indent=2,
        )
    csv_path = os.path.join(expdir, "results.csv")
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
        for result in results:
            row = [str(result.get(k, "")) for k in keys]
            f.write(",".join(row) + "\n")


def _best_result(
    results: list[dict[str, BenchmarkResultValue]],
) -> dict[str, BenchmarkResultValue] | None:
    ok = [result for result in results if result.get("status") == "ok"]
    if not ok:
        return None
    return min(
        ok,
        key=lambda result: (
            result.get("rot_rmse_deg", 1e9) + result.get("trans_rmse_px", 1e9)
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark multiple alignment losses on a small misaligned phantom."
    )
    p.add_argument(
        "--expdir",
        default="runs/loss_experiment",
        help="Experiment output directory (datasets, logs, results)",
    )
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--nu", type=int, default=128)
    p.add_argument("--nv", type=int, default=128)
    p.add_argument("--n-views", type=int, default=60)
    p.add_argument("--geometry", choices=["parallel", "lamino"], default="parallel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--rot-deg",
        type=float,
        default=1.0,
        help="Max |rotation| per-axis used to generate misalignment (degrees)",
    )
    p.add_argument(
        "--trans-px",
        type=float,
        default=5.0,
        help="Max |translation| used to generate misalignment (pixels)",
    )
    p.add_argument("--outer-iters", type=int, default=4)
    p.add_argument("--recon-iters", type=int, default=10)
    p.add_argument("--progress", action="store_true")
    p.add_argument(
        "--metrics-only",
        action="store_true",
        help="Do not run alignment; only (re)compute metrics for existing outputs",
    )
    p.add_argument(
        "--k-step",
        type=int,
        default=1,
        help="k-step for relative-motion metric (default: 1)",
    )
    p.add_argument(
        "--gt-metric",
        choices=["none", "mse"],
        default="mse",
        help="Physics-aware metric against GT projections",
    )
    p.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=None,
        help="Optional multiresolution pyramid factors (e.g., 4 2 1). Applied to all losses.",
    )
    p.add_argument(
        "--losses",
        nargs="*",
        default=["l2", "l2_otsu", "pwls"],
        help="Subset of losses to run; default tests LS-like losses (GN-only)",
    )
    args = p.parse_args()

    _ensure_dir(args.expdir)
    _ensure_dir(os.path.join(args.expdir, "logs"))
    setup_logging()
    log_jax_env()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"
    results = _run_benchmark_workflow(args)
    _write_results_summary(args.expdir, args, results)
    best = _best_result(results)
    if best:
        logging.info("Best by (rot_rmse_deg + trans_rmse_px): %s", best)
    print(f"Results written to {args.expdir} (results.json, results.csv)")


if __name__ == "__main__":  # pragma: no cover
    main()
