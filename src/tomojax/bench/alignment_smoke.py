from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
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
_PARAMETER_ORDER = ("alpha", "beta", "phi", "dx", "dz")
_ROTATION_DOFS = frozenset(("alpha", "beta", "phi"))


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


def _run_align_in_process(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run tomojax-align in this process while preserving CLI outputs."""
    from tomojax.cli import align as align_cli

    old_argv = sys.argv
    sys.argv = ["tomojax-align", *argv]
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        parser = align_cli._build_parser()
        args, config_metadata = align_cli.parse_args_with_config(
            parser,
            argv,
            required=("data", "out"),
        )
        align_cli.log_jax_env()
        align_cli._init_jax_compilation_cache()
        if args.progress:
            os.environ["TOMOJAX_PROGRESS"] = "1"
        plan = align_cli._build_align_cli_run_plan(parser, args, config_metadata)
        checkpoint_callbacks = align_cli._make_align_cli_checkpoint_callbacks(plan)
        execution = align_cli._execute_alignment_plan(
            plan,
            single_checkpoint_callback=checkpoint_callbacks.single,
            multires_checkpoint_callback=checkpoint_callbacks.multires,
        )
        align_cli._write_alignment_outputs(plan, execution)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        sys.argv = old_argv
    return subprocess.CompletedProcess(
        args=["tomojax-align", *argv],
        returncode=0,
        stdout=log_stream.getvalue(),
        stderr="",
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


def _read_theta_array(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        theta = np.asarray(handle["entry/processing/tomojax/align/thetas"], dtype=np.float32)
    if theta.ndim != 2 or theta.shape[1] != 5:
        raise ValueError(f"alignment theta dataset must have shape (n_views, 5), got {theta.shape}")
    return theta


def _read_projection_shape(path: Path) -> tuple[int, ...]:
    with h5py.File(path, "r") as handle:
        return tuple(int(v) for v in handle["entry/data/projections"].shape)


def _params_json_array(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    views = payload.get("views")
    if not isinstance(views, list):
        raise ValueError("alignment parameter JSON must contain a views list")
    rows = []
    for record in views:
        rows.append(
            [
                float(record["alpha_rad"]),
                float(record["beta_rad"]),
                float(record["phi_rad"]),
                float(record["dx_px"]),
                float(record["dz_px"]),
            ]
        )
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"alignment parameter JSON must resolve to shape (n_views, 5), got {arr.shape}")
    return arr, payload


def _rmse(values: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(np.square(values, dtype=np.float64)))))


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values), dtype=np.float64))


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _pose_recovery_metrics(*, truth_path: Path, params_json: Path) -> dict[str, Any]:
    """Compare recovered alignment parameters with fixture truth.

    The fixture stores the synthetic per-view perturbation in
    ``entry/processing/tomojax/align/thetas``. Exported alignment parameters use
    the same five-column convention. Translation comparison applies the exported
    mean-translation gauge to the fixture truth when that gauge was active.
    """
    truth = _read_theta_array(truth_path).astype(np.float64)
    recovered, payload = _params_json_array(params_json)
    recovered = recovered.astype(np.float64)
    if recovered.shape != truth.shape:
        raise ValueError(
            "recovered alignment parameters and fixture truth have different shapes: "
            f"{recovered.shape} vs {truth.shape}"
        )

    truth_for_compare = truth.copy()
    gauge = payload.get("gauge_fix") or {}
    gauge_mode = str(gauge.get("mode") or "none")
    if gauge_mode == "mean_translation":
        for idx in (3, 4):
            truth_for_compare[:, idx] -= float(np.mean(truth_for_compare[:, idx]))

    err = recovered - truth_for_compare
    per_dof: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(_PARAMETER_ORDER):
        values = err[:, idx]
        entry = {
            "rmse": _rmse(values),
            "mae": _mae(values),
            "max_abs": _max_abs(values),
        }
        if name in _ROTATION_DOFS:
            deg = values * 180.0 / math.pi
            entry.update(
                {
                    "rmse_deg": _rmse(deg),
                    "mae_deg": _mae(deg),
                    "max_abs_deg": _max_abs(deg),
                }
            )
        per_dof[name] = entry

    rot_err_deg = err[:, :3] * 180.0 / math.pi
    trans_err_px = err[:, 3:5]
    initial_truth = truth_for_compare
    return {
        "parameter_order": list(_PARAMETER_ORDER),
        "comparison": "recovered_minus_fixture_truth",
        "translation_gauge": gauge_mode,
        "per_dof": per_dof,
        "rot_rmse_deg": _rmse(rot_err_deg),
        "trans_rmse_px": _rmse(trans_err_px),
        "initial_rot_rmse_deg": _rmse(initial_truth[:, :3] * 180.0 / math.pi),
        "initial_trans_rmse_px": _rmse(initial_truth[:, 3:5]),
    }


def _projection_residual_metrics(
    *,
    final_loss: float | None,
    loss_kind: str | None,
    projection_shape: tuple[int, ...],
) -> dict[str, Any]:
    n_rays = int(np.prod(projection_shape, dtype=np.int64))
    out: dict[str, Any] = {
        "loss_kind": loss_kind,
        "final_loss": final_loss,
        "projection_shape": list(projection_shape),
        "n_rays": n_rays,
        "rmse_per_ray": None,
    }
    if loss_kind == "l2" and final_loss is not None and n_rays > 0:
        out["rmse_per_ray"] = float(math.sqrt(max(0.0, 2.0 * float(final_loss)) / n_rays))
    return out


def _success_flags(report: dict[str, Any]) -> dict[str, bool]:
    loss = report["loss"]
    quality = report["quality"]
    pose = report["pose_recovery"]
    initial_loss = loss.get("initial")
    final_loss = loss.get("final")
    misaligned_mse = quality["misaligned_recon_vs_truth"]["mse"]
    aligned_mse = quality["aligned_recon_vs_truth"]["mse"]
    return {
        "finite_final_loss": final_loss is not None and math.isfinite(float(final_loss)),
        "loss_decreased": (
            initial_loss is not None
            and final_loss is not None
            and math.isfinite(float(initial_loss))
            and math.isfinite(float(final_loss))
            and float(final_loss) < float(initial_loss)
        ),
        "loss_drop_at_least_50_percent": (
            loss.get("delta_percent") is not None and float(loss["delta_percent"]) <= -50.0
        ),
        "aligned_mse_improved_vs_misaligned": float(aligned_mse) < float(misaligned_mse),
        "pose_metrics_finite": all(
            math.isfinite(float(pose[key])) for key in ("rot_rmse_deg", "trans_rmse_px")
        ),
    }


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
    parser.add_argument("--regulariser", choices=["tv", "huber_tv"], default="tv")
    parser.add_argument("--huber-delta", type=float, default=1e-2)
    parser.add_argument("--recon-L", type=float, default=5000.0)
    parser.add_argument("--views-per-batch", type=int, default=1)
    parser.add_argument("--projector-unroll", type=int, default=4)
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
    align_argv = [
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
        "--regulariser",
        args.regulariser,
        "--huber-delta",
        str(args.huber_delta),
        "--views-per-batch",
        str(args.views_per_batch),
        "--projector-unroll",
        str(args.projector_unroll),
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
    ]
    align_result = _run_align_in_process(align_argv)
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
            "--regulariser",
            args.regulariser,
            "--huber-delta",
            str(args.huber_delta),
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
    pose_recovery = _pose_recovery_metrics(
        truth_path=fixture["misaligned"],
        params_json=params_json,
    )
    slice_metrics = _write_png(
        args.slice_png,
        truth_path=fixture["truth"],
        misaligned_recon_path=misaligned_recon,
        aligned_path=aligned,
    )

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    resolved = manifest.get("resolved_config", {})
    run_info = resolved.get("run_info") or {}
    final_loss = run_info.get("final_loss")
    loss_kind = run_info.get("loss_kind")
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
            "regulariser": args.regulariser,
            "huber_delta": args.huber_delta,
            "recon_L": args.recon_L,
            "views_per_batch": args.views_per_batch,
            "projector_unroll": args.projector_unroll,
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
            "manifest_final_loss": final_loss,
            "manifest_loss_count": run_info.get("loss_count"),
            "manifest_loss_kind": loss_kind,
        },
        "pose_recovery": pose_recovery,
        "projection_residual": _projection_residual_metrics(
            final_loss=final_loss,
            loss_kind=loss_kind,
            projection_shape=_read_projection_shape(fixture["misaligned"]),
        ),
        "quality": {
            "misaligned_recon_vs_truth": _metrics(misaligned_recon_volume, truth),
            "aligned_recon_vs_truth": _metrics(aligned_volume, truth),
            "reference_recon_vs_truth": None,
            "reference_recon_note": (
                "Not run in smoke benchmark; aligned_recon_vs_truth uses the configured "
                "working FISTA reconstruction."
            ),
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
    report["success"] = _success_flags(report)

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
                f"- Reconstruction regulariser: `{args.regulariser}`",
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
                f"| Projection residual RMSE/ray | {fmt(report['projection_residual']['rmse_per_ray'])} |",
                f"| Pose rotation RMSE | {fmt(report['pose_recovery']['rot_rmse_deg'])} deg |",
                f"| Pose translation RMSE | {fmt(report['pose_recovery']['trans_rmse_px'])} px |",
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
