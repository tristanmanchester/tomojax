"""User-facing loss-benchmark CLI.

This entry point owns experiment-directory orchestration and user-visible outputs for
the loss benchmark. Shared benchmark building blocks belong under ``tomojax.bench``,
while fixed controller-profile policy stays in ``bench/`` and ad hoc exploration stays
under ``scripts/``.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import csv
import json
import logging
from pathlib import Path
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from tomojax.align.objectives.loss_specs import parse_loss_spec
from tomojax.align.pipeline import AlignConfig, align
from tomojax.bench.loss_experiment import (
    make_gt_dataset as _make_gt_dataset,
    make_misaligned_dataset as _make_misaligned_dataset,
    metrics_abs as _metrics_abs,
    metrics_gauge_fixed as _metrics_gauge_fixed,
    metrics_relative as _metrics_relative,
    project_gt_with_estimated_poses as _project_gt_with_estimated_poses,
)
from tomojax.core import log_jax_env, setup_logging
from tomojax.io import (
    build_geometry_from_dataset_metadata,
    load_projection_payload,
    save_projection_payload,
)

type BenchmarkResultValue = str | float | None


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _jnp_float32_array(value: object) -> Any:
    return jnp.asarray(value, dtype=jnp.float32)  # pyright: ignore[reportUnknownMemberType]


def _result_metric(result: dict[str, BenchmarkResultValue], key: str) -> float:
    value = result.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 1e9


def _run_benchmark_workflow(  # noqa: PLR0912, PLR0915
    args: argparse.Namespace,
) -> list[dict[str, BenchmarkResultValue]]:
    expdir = Path(args.expdir)
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
    dataset_mis = load_projection_payload(mis)
    metadata_mis = dataset_mis.copy_metadata()
    if metadata_mis.align_params is None:
        raise ValueError("Misaligned benchmark dataset is missing align_params")
    true_params = np.asarray(metadata_mis.align_params, dtype=np.float32)
    volume = metadata_mis.volume

    grid, det, geom = build_geometry_from_dataset_metadata(
        dataset_mis.geometry_inputs(),
        apply_saved_alignment=False,
        volume_shape=(np.asarray(volume).shape if volume is not None else None),
    )
    du, dv = float(det.du), float(det.dv)
    thetas = np.asarray(dataset_mis.angles_deg, dtype=np.float32)
    projections = _jnp_float32_array(dataset_mis.projections)

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
        out_path = expdir / f"align_{run_name}.nxs"
        log_path = expdir / "logs" / f"{run_name}.log"
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
            p_est_np: np.ndarray | None = None
            x_est_np: np.ndarray | None = None
            if out_path.exists():
                logging.info(
                    "[%s] output exists; skipping alignment and recomputing metrics",
                    run_name,
                )
                dataset_out = load_projection_payload(str(out_path))
                metadata_out = dataset_out.copy_metadata()
                if metadata_out.align_params is None:
                    raise ValueError(f"{out_path} is missing align_params")
                p_est_np = np.asarray(metadata_out.align_params, dtype=np.float32)
                x_est_np = (
                    np.asarray(metadata_out.volume) if metadata_out.volume is not None else None
                )
            elif args.metrics_only:
                logging.warning(
                    "[%s] metrics-only set and output missing; skipping (status=missing)",
                    run_name,
                )
                status = "missing"
                p_est_np = None
            elif not is_gn:
                logging.warning("[%s] skipped: not an LS-like loss (GN-only mode)", run_name)
                status = "skipped"
                p_est_np = None
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
                    from tomojax.align.pipeline import align_multires

                    x_est, p_est, _info = align_multires(
                        geom, grid, det, projections, factors=levels, cfg=cfg
                    )
                else:
                    x_est, p_est, _info = align(geom, grid, det, projections, cfg=cfg)
                p_est_np = np.asarray(p_est)
                x_est_np = np.asarray(x_est)
                save_meta = dataset_mis.copy_metadata()
                save_meta.thetas_deg = np.asarray(thetas)
                save_meta.volume = x_est_np
                save_meta.align_params = p_est_np
                save_meta.frame = str(metadata_mis.frame or "sample")
                save_projection_payload(
                    str(out_path),
                    projections=np.asarray(dataset_mis.projections),
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
                        _jnp_float32_array(metadata_mis.volume),
                        grid,
                        det,
                        geom,
                        p_est_np,
                    )
                    y = _jnp_float32_array(dataset_mis.projections)
                    gt_mse = float(jnp.mean((y_hat - y) ** 2).item())
                    metrics["gt_mse"] = gt_mse
        except Exception as e:
            status = "error"
            err_msg = str(e)
            logging.exception("Loss %s failed", run_name)
        elapsed = time.perf_counter() - start
        logging.getLogger().removeHandler(fh)
        with suppress(Exception):
            fh.close()
        logging.info(
            "[%s] status=%s time=%.2fs | abs: rot_rmse=%.3f° trans_rmse=%.3fpx | "
            "rel(k=%d): rot=%.3f° trans=%.3fpx | gf: rot=%.3f° trans=%.3fpx | "
            "gt_mse=%.3e",
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
                "log": str(log_path.relative_to(expdir)),
                "output": str(out_path.relative_to(expdir)) if status == "ok" else None,
                "error": err_msg,
            }
        )
        logging.info(
            "%s: status=%s time=%.2fs metrics=%s",
            run_name,
            status,
            elapsed,
            dict(metrics),
        )
    return results


def _write_results_summary(
    expdir: str,
    args: argparse.Namespace,
    results: list[dict[str, BenchmarkResultValue]],
) -> None:
    expdir_path = Path(expdir)
    with (expdir_path / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "config": {k: getattr(args, k) for k in vars(args)}},
            f,
            indent=2,
        )
    csv_path = expdir_path / "results.csv"
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
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in keys})


def _best_result(
    results: list[dict[str, BenchmarkResultValue]],
) -> dict[str, BenchmarkResultValue] | None:
    ok = [result for result in results if result.get("status") == "ok"]
    if not ok:
        return None
    return min(
        ok,
        key=lambda result: (
            _result_metric(result, "rot_rmse_deg") + _result_metric(result, "trans_rmse_px")
        ),
    )


def _has_failed_result(results: list[dict[str, BenchmarkResultValue]]) -> bool:
    return any(result.get("status") == "error" for result in results)


def main() -> None:
    """Run the developer loss benchmark."""
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
    _ensure_dir(str(Path(args.expdir) / "logs"))
    setup_logging()
    log_jax_env()
    if args.progress:
        from os import environ

        environ["TOMOJAX_PROGRESS"] = "1"
    results = _run_benchmark_workflow(args)
    _write_results_summary(args.expdir, args, results)
    best = _best_result(results)
    if best:
        logging.info("Best by (rot_rmse_deg + trans_rmse_px): %s", best)
    print(f"Results written to {args.expdir} (results.json, results.csv)")
    if _has_failed_result(results):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
