from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_LOSS_RE = re.compile(
    r"Loss\s+([0-9.+\-eE]+)\s+->\s+([0-9.+\-eE]+)\s+\(.*?([-+][0-9.]+)%\)"
)
_ALIGN_RE = re.compile(
    r"Align\s+\|\s+time\s+([0-9.+\-eE]+)s.*?"
    r"loss\s+([0-9.+\-eE]+)->([0-9.+\-eE]+).*?([-+][0-9.]+)%"
)
_RECON_RE = re.compile(r"Recon \(FISTA\)\s+\|\s+time\s+([0-9.+\-eE]+)s")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _python_module_cmd(env: dict[str, str], module: str) -> list[str]:
    python = env.get("TOMOJAX_BENCH_PYTHON")
    if python:
        return [python, "-m", module]
    entrypoints = {
        "tomojax.cli.simulate": "tomojax-simulate",
        "tomojax.cli.misalign": "tomojax-misalign",
        "tomojax.cli.align": "tomojax-align",
        "tomojax.cli.recon": "tomojax-recon",
    }
    return ["uv", "run", entrypoints[module]]


def _read_volume(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["entry/processing/tomojax/volume"], dtype=np.float32)


def _mse(recon: np.ndarray, truth: np.ndarray) -> float:
    diff = np.asarray(recon, dtype=np.float32) - np.asarray(truth, dtype=np.float32)
    return float(np.mean(diff * diff))


def _metrics(recon: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    mse = _mse(recon, truth)
    rmse = float(math.sqrt(mse))
    peak = float(max(np.max(truth) - np.min(truth), 1e-12))
    psnr = float(20.0 * math.log10(peak / max(rmse, 1e-12)))
    return {
        "mse": mse,
        "rmse": rmse,
        "psnr_db": psnr,
        "min": float(np.min(recon)),
        "max": float(np.max(recon)),
        "mean": float(np.mean(recon)),
    }


def _theta_stats(path: Path) -> dict[str, dict[str, float]]:
    with h5py.File(path, "r") as handle:
        theta = np.asarray(handle["entry/processing/tomojax/align/thetas"], dtype=np.float32)
    stats: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(("alpha", "beta", "phi", "dx", "dz")):
        values = theta[:, idx]
        entry = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "abs_mean": float(np.mean(np.abs(values))),
            "abs_max": float(np.max(np.abs(values))),
        }
        if name in {"alpha", "beta", "phi"}:
            for key in ("min", "max", "mean", "std", "abs_mean", "abs_max"):
                entry[f"{key}_deg"] = float(entry[key] * 180.0 / math.pi)
        stats[name] = entry
    return stats


def _parse_alignment_log(log: str) -> dict[str, Any]:
    summaries = [
        {
            "initial_loss": float(match.group(1)),
            "final_loss": float(match.group(2)),
            "delta_percent": float(match.group(3)),
        }
        for match in _LOSS_RE.finditer(log)
    ]
    align_steps = [
        {
            "align_time_sec": float(match.group(1)),
            "initial_loss": float(match.group(2)),
            "final_loss": float(match.group(3)),
            "delta_percent": float(match.group(4)),
        }
        for match in _ALIGN_RE.finditer(log)
    ]
    recon_times = [float(match.group(1)) for match in _RECON_RE.finditer(log)]
    initial = align_steps[0]["initial_loss"] if align_steps else None
    final = align_steps[-1]["final_loss"] if align_steps else None
    delta_percent = None
    if initial not in (None, 0.0) and final is not None:
        delta_percent = float((final - initial) / initial * 100.0)
    return {
        "level_summaries": summaries,
        "align_steps": align_steps,
        "initial_loss": initial,
        "final_loss": final,
        "delta_percent": delta_percent,
        "recon_time_sec": float(sum(recon_times)),
        "align_time_sec": float(sum(step["align_time_sec"] for step in align_steps)),
    }


def _write_png(
    out: Path,
    *,
    truth_path: Path,
    misaligned_recon_path: Path | None,
    aligned_path: Path,
) -> dict[str, Any]:
    truth = _read_volume(truth_path)
    aligned = _read_volume(aligned_path)
    volumes: dict[str, np.ndarray] = {"Ground truth": truth}
    if misaligned_recon_path is not None and misaligned_recon_path.exists():
        volumes["Misaligned FISTA"] = _read_volume(misaligned_recon_path)
    volumes["Aligned FISTA"] = aligned

    z_index = int(np.argmax(truth.sum(axis=(1, 2))))
    shared_vmax = float(np.percentile(truth, 99.7))
    fig, axes = plt.subplots(2, len(volumes), figsize=(3.2 * len(volumes), 6.4), constrained_layout=True)
    if len(volumes) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    metrics: dict[str, Any] = {"z_index": z_index}
    for col, (label, volume) in enumerate(volumes.items()):
        slice_ = volume[z_index]
        mse = 0.0 if label == "Ground truth" else _mse(volume, truth)
        metrics[label] = {
            "mse_vs_ground_truth": mse,
            "min": float(np.min(volume)),
            "max": float(np.max(volume)),
            "mean": float(np.mean(volume)),
        }
        axes[0, col].imshow(
            slice_,
            cmap="magma",
            vmin=0.0,
            vmax=shared_vmax,
            origin="lower",
            interpolation="nearest",
        )
        axes[0, col].set_title(label, fontsize=10)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        axes[0, col].set_xlabel(f"shared scale, MSE={mse:.4f}", fontsize=8)

        lo = float(np.percentile(slice_, 1))
        hi = float(np.percentile(slice_, 99.5))
        if hi <= lo:
            lo = float(np.min(slice_))
            hi = float(np.max(slice_)) if np.max(slice_) > np.min(slice_) else lo + 1e-6
        axes[1, col].imshow(
            slice_,
            cmap="magma",
            vmin=lo,
            vmax=hi,
            origin="lower",
            interpolation="nearest",
        )
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        axes[1, col].set_xlabel(f"panel scale [{lo:.3g}, {hi:.3g}]", fontsize=8)

    axes[0, 0].set_ylabel(f"z={z_index}\nshared", fontsize=9)
    axes[1, 0].set_ylabel("per-panel", fontsize=9)
    fig.suptitle("TomoJAX 24^3 full-pose alignment smoke", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return metrics


def _ensure_fixture(args: argparse.Namespace, *, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    fixture = args.fixture_dir
    fixture.mkdir(parents=True, exist_ok=True)
    truth = fixture / "align24_gt.nxs"
    misaligned = fixture / "align24_strong_misaligned.nxs"
    created: list[str] = []

    if not truth.exists():
        _run(
            [
                *_python_module_cmd(env, "tomojax.cli.simulate"),
                "--out",
                str(truth),
                "--nx",
                str(args.size),
                "--ny",
                str(args.size),
                "--nz",
                str(args.size),
                "--nu",
                str(args.size),
                "--nv",
                str(args.size),
                "--n-views",
                str(args.views),
                "--rotation-deg",
                "180",
                "--phantom",
                "random_shapes",
                "--n-cubes",
                str(args.n_cubes),
                "--n-spheres",
                str(args.n_spheres),
                "--min-size",
                str(args.min_size),
                "--max-size",
                str(args.max_size),
                "--seed",
                str(args.seed),
            ],
            cwd=cwd,
            env=env,
        )
        created.append(str(truth))

    if not misaligned.exists():
        _run(
            [
                *_python_module_cmd(env, "tomojax.cli.misalign"),
                "--data",
                str(truth),
                "--out",
                str(misaligned),
                "--with-random",
                "--rot-deg",
                str(args.misalignment_rot_deg),
                "--trans-px",
                str(args.misalignment_trans_px),
                "--seed",
                str(args.misalignment_seed),
            ],
            cwd=cwd,
            env=env,
        )
        created.append(str(misaligned))

    return {
        "truth": truth,
        "misaligned": misaligned,
        "created": created,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny TomoJAX alignment smoke benchmark.")
    parser.add_argument("--tomojax-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--slice-png", type=Path, required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--git-branch", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--n-cubes", type=int, default=3)
    parser.add_argument("--n-spheres", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--misalignment-rot-deg", type=float, default=5.0)
    parser.add_argument("--misalignment-trans-px", type=float, default=3.0)
    parser.add_argument("--misalignment-seed", type=int, default=8)
    parser.add_argument("--levels", nargs="+", type=int, default=[1])
    parser.add_argument("--outer-iters", type=int, default=3)
    parser.add_argument("--recon-iters", type=int, default=4)
    parser.add_argument("--loss", default="l2")
    parser.add_argument("--schedule", default="pose_only")
    parser.add_argument("--lambda-tv", type=float, default=0.001)
    parser.add_argument("--recon-L", type=float, default=5000.0)
    parser.add_argument("--views-per-batch", type=int, default=0)
    args = parser.parse_args()

    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
    fixture = _ensure_fixture(args, cwd=args.tomojax_dir, env=env)

    stem = args.out.with_suffix("")
    aligned = stem.with_name(stem.name + "_aligned.nxs")
    params_json = stem.with_name(stem.name + "_params.json")
    manifest_json = stem.with_name(stem.name + "_manifest.json")
    misaligned_recon = stem.with_name(stem.name + "_misaligned_recon.nxs")
    misaligned_recon_manifest = stem.with_name(stem.name + "_misaligned_recon_manifest.json")

    start = time.perf_counter()
    align_result = _run(
        [
            *_python_module_cmd(env, "tomojax.cli.align"),
            "--data",
            str(fixture["misaligned"]),
            "--levels",
            *[str(level) for level in args.levels],
            "--outer-iters",
            str(args.outer_iters),
            "--recon-iters",
            str(args.recon_iters),
            "--recon-L",
            str(args.recon_L),
            "--lambda-tv",
            str(args.lambda_tv),
            "--views-per-batch",
            str(args.views_per_batch),
            "--schedule",
            args.schedule,
            "--loss",
            args.loss,
            "--no-early-stop",
            "--no-log-compact",
            "--log-summary",
            "--out",
            str(aligned),
            "--save-params-json",
            str(params_json),
            "--save-manifest",
            str(manifest_json),
        ],
        cwd=args.tomojax_dir,
        env=env,
    )
    wall_sec = time.perf_counter() - start

    _run(
        [
            *_python_module_cmd(env, "tomojax.cli.recon"),
            "--data",
            str(fixture["misaligned"]),
            "--algo",
            "fista",
            "--iters",
            str(args.recon_iters),
            "--L",
            str(args.recon_L),
            "--lambda-tv",
            str(args.lambda_tv),
            "--views-per-batch",
            str(args.views_per_batch),
            "--out",
            str(misaligned_recon),
            "--save-manifest",
            str(misaligned_recon_manifest),
        ],
        cwd=args.tomojax_dir,
        env=env,
    )

    truth = _read_volume(fixture["truth"])
    aligned_volume = _read_volume(aligned)
    misaligned_recon_volume = _read_volume(misaligned_recon)
    log_metrics = _parse_alignment_log(align_result.stdout)
    slice_metrics = _write_png(
        args.slice_png,
        truth_path=fixture["truth"],
        misaligned_recon_path=misaligned_recon,
        aligned_path=aligned,
    )

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    resolved = manifest.get("resolved_config", {})
    report = {
        "benchmark": "tomojax_alignment_smoke",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "experiment": {
            "note": args.note,
            "git_branch": args.git_branch,
            "git_commit": args.git_commit,
        },
        "config": {
            "size": args.size,
            "views": args.views,
            "phantom": "random_shapes",
            "seed": args.seed,
            "n_cubes": args.n_cubes,
            "n_spheres": args.n_spheres,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "misalignment_rot_deg": args.misalignment_rot_deg,
            "misalignment_trans_px": args.misalignment_trans_px,
            "misalignment_seed": args.misalignment_seed,
            "levels": args.levels,
            "outer_iters": args.outer_iters,
            "recon_iters": args.recon_iters,
            "schedule": args.schedule,
            "loss": args.loss,
            "lambda_tv": args.lambda_tv,
            "recon_L": args.recon_L,
            "views_per_batch": args.views_per_batch,
        },
        "fixture": {
            "truth": str(fixture["truth"]),
            "misaligned": str(fixture["misaligned"]),
            "created": fixture["created"],
            "misalignment_stats": _theta_stats(fixture["misaligned"]),
        },
        "artifacts": {
            "aligned_nxs": str(aligned),
            "params_json": str(params_json),
            "manifest_json": str(manifest_json),
            "misaligned_recon_nxs": str(misaligned_recon),
            "misaligned_recon_manifest_json": str(misaligned_recon_manifest),
            "slice_png": str(args.slice_png),
        },
        "timing": {
            "wall_sec": float(wall_sec),
            "recon_time_sec": log_metrics["recon_time_sec"],
            "align_time_sec": log_metrics["align_time_sec"],
        },
        "loss": {
            "initial": log_metrics["initial_loss"],
            "final": log_metrics["final_loss"],
            "delta_percent": log_metrics["delta_percent"],
            "level_summaries": log_metrics["level_summaries"],
            "align_steps": log_metrics["align_steps"],
            "manifest_final_loss": (resolved.get("run_info") or {}).get("final_loss"),
            "manifest_loss_count": (resolved.get("run_info") or {}).get("loss_count"),
            "manifest_loss_kind": (resolved.get("run_info") or {}).get("loss_kind"),
        },
        "quality": {
            "misaligned_recon_vs_truth": _metrics(misaligned_recon_volume, truth),
            "aligned_recon_vs_truth": _metrics(aligned_volume, truth),
            "slice_metrics": slice_metrics,
        },
        "resolved": {
            "active_dofs": resolved.get("active_dofs"),
            "levels": resolved.get("levels"),
            "objective_kind": resolved.get("objective_kind"),
            "gather_dtype": resolved.get("gather_dtype"),
        },
        "stdout": align_result.stdout,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            if value == 0:
                return "0"
            if abs(value) < 0.001 or abs(value) >= 10000:
                return f"{value:.4e}"
            return f"{value:.4f}"
        return str(value)

    args.summary_md.write_text(
        "\n".join(
            [
                "# TomoJAX Alignment Smoke Benchmark",
                "",
                f"- Size: `{args.size}^3`",
                f"- Views: `{args.views}`",
                f"- Misalignment: `±{args.misalignment_rot_deg:g} deg`, `±{args.misalignment_trans_px:g} px`",
                f"- Levels: `{' '.join(str(level) for level in args.levels)}`",
                f"- Outer iterations: `{args.outer_iters}`",
                f"- Recon iterations per outer: `{args.recon_iters}`",
                f"- Schedule: `{args.schedule}`",
                f"- Loss: `{args.loss}`",
                f"- Active DOFs: `{', '.join(report['resolved']['active_dofs'] or [])}`",
                "",
                "## Results",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Wall time | {fmt(report['timing']['wall_sec'])} sec |",
                f"| Recon time | {fmt(report['timing']['recon_time_sec'])} sec |",
                f"| Align time | {fmt(report['timing']['align_time_sec'])} sec |",
                f"| Initial loss | {fmt(report['loss']['initial'])} |",
                f"| Final loss | {fmt(report['loss']['final'])} |",
                f"| Loss delta | {fmt(report['loss']['delta_percent'])}% |",
                f"| Misaligned FISTA MSE vs GT | {fmt(report['quality']['misaligned_recon_vs_truth']['mse'])} |",
                f"| Aligned FISTA MSE vs GT | {fmt(report['quality']['aligned_recon_vs_truth']['mse'])} |",
                f"| Aligned FISTA PSNR vs GT | {fmt(report['quality']['aligned_recon_vs_truth']['psnr_db'])} dB |",
                "",
                f"Slice PNG: `{args.slice_png}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
