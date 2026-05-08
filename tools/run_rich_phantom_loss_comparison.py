"""Run PHANTOM94 loss-mode comparison through align-auto sidecars."""
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from tomojax.datasets import generate_synthetic_dataset

LOSS_MODES = ("otsu_l2", "pseudo_huber", "otsu_pseudo_huber")
CASES = (
    ("setup_global_fixed_truth", "fixed_synthetic_truth"),
    ("setup_global_stopped", "stopped_reconstruction"),
)


def main() -> int:
    """Run all configured loss/case combinations and write aggregate summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128, choices=(32, 64, 128))
    parser.add_argument("--views", type=int, default=128)
    parser.add_argument("--profile", default="balanced")
    args = parser.parse_args()

    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    dataset_paths = generate_synthetic_dataset(
        "rich_phantom94_setup_global_tomo",
        root / "datasets",
        size=args.size,
        clean=True,
        views=args.views,
        supported_only=True,
    )
    rows: list[dict[str, Any]] = []
    for case_name, volume_source in CASES:
        for loss_mode in LOSS_MODES:
            run_dir = root / f"{case_name}_{loss_mode}"
            start = time.perf_counter()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tomojax.cli.align_auto",
                    "--out-dir",
                    str(run_dir),
                    "--profile",
                    str(args.profile),
                    "--size",
                    str(args.size),
                    "--views",
                    str(args.views),
                    "--synthetic-dataset",
                    "rich_phantom94_setup_global_tomo",
                    "--synthetic-dataset-dir",
                    str(dataset_paths.dataset_dir),
                    "--projection-loss-mode",
                    loss_mode,
                    "--geometry-update-volume-source",
                    volume_source,
                    "--geometry-update-pose-frozen",
                    "--geometry-update-active-setup-parameters",
                    "det_u_px",
                    "--preview-volume-support",
                    "cylindrical",
                    "--preview-initialization",
                    "backprojection",
                    "--preview-tv-scale",
                    "1.0",
                    "--preview-residual-filter-mode",
                    "continuation",
                    "--preview-center-l2-weight",
                    "0.02",
                ],
                check=False,
                env=_jax_subprocess_env(),
            )
            elapsed = time.perf_counter() - start
            if completed.returncode != 0:
                raise RuntimeError(f"align-auto failed for {case_name}/{loss_mode}")
            rows.append(_summary_row(case_name, loss_mode, run_dir, elapsed))
    _write_summary(root, rows)
    _write_plots(root, rows)
    return 0


def _summary_row(case_name: str, loss_mode: str, run_dir: Path, elapsed: float) -> dict[str, Any]:
    result = json.loads((run_dir / "benchmark_result.json").read_text(encoding="utf-8"))
    geometry = result["geometry_recovery"]
    reconstruction = result["reconstruction"]
    runtime = result["runtime"]
    sidecar = result.get("dataset", {})
    schur = json.loads((run_dir / "schur_diagnostics.json").read_text(encoding="utf-8"))
    diagnostics = schur.get("diagnostics", {})
    return {
        "case": case_name,
        "loss_mode": loss_mode,
        "status": result.get("status"),
        "phantom_id": "PHANTOM94",
        "phantom_seed": 20260893,
        "phantom_kind": "phantom94_random_cubes_spheres",
        "views": sidecar.get("projection_views"),
        "artifact_dir": str(run_dir),
        "det_u_realized_rmse_px": geometry.get("det_u_realized_rmse_px"),
        "det_v_realized_rmse_px": geometry.get("det_v_realized_rmse_px"),
        "theta_realized_rmse_rad": geometry.get("theta_realized_rmse_rad"),
        "alpha_beta_rmse_rad": geometry.get("alpha_beta_rmse_rad"),
        "final_volume_final_geometry_loss": reconstruction.get(
            "final_volume_final_geometry_loss_all_views"
        ),
        "final_volume_true_geometry_loss": reconstruction.get(
            "final_volume_true_geometry_loss_all_views"
        ),
        "true_volume_final_geometry_loss": reconstruction.get(
            "true_volume_final_geometry_loss_all_views"
        ),
        "true_volume_true_geometry_loss": reconstruction.get(
            "true_volume_true_geometry_loss_all_views"
        ),
        "volume_nmse": reconstruction.get("volume_nmse"),
        "schur_accepted": diagnostics.get("accepted"),
        "classification": reconstruction.get("projection_loss_classification"),
        "runtime_seconds": runtime.get("total_wall_seconds", elapsed),
        "peak_memory": None,
    }


def _write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Rich Phantom Loss Comparison",
        "",
        "| Case | Loss | det_u RMSE px | Volume NMSE | Final residual | "
        "Schur accepted | Classification |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    lines.extend(
        (
            "| {case} | {loss_mode} | {det_u_realized_rmse_px} | {volume_nmse} | "
            "{final_volume_final_geometry_loss} | {schur_accepted} | {classification} |".format(
                **row
            )
        )
        for row in rows
    )
    lines.extend(["", "Fixed-truth rows are oracle diagnostics, not production passes."])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [f"{row['case']}\n{row['loss_mode']}" for row in rows]
    det_u = [float(row["det_u_realized_rmse_px"]) for row in rows]
    nmse = [float(row["volume_nmse"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(rows) * 1.4), 7), constrained_layout=True)
    axes[0].bar(labels, det_u)
    axes[0].set_ylabel("det_u RMSE px")
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[1].bar(labels, nmse)
    axes[1].set_ylabel("Volume NMSE")
    axes[1].tick_params(axis="x", labelrotation=35)
    fig.savefig(root / "loss_comparison_metrics.png", dpi=160)
    plt.close(fig)


def _jax_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return env


if __name__ == "__main__":
    raise SystemExit(main())
