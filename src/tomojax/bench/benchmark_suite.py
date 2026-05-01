from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_PRESETS: dict[str, list[dict[str, Any]]] = {
    "quick": [
        {"name": "quick_64", "size": 64, "detector": 64, "views": 90, "warmup": 1, "repeat": 1},
    ],
    "guard": [
        {"name": "headline_128", "size": 128, "detector": 128, "views": 180, "warmup": 1, "repeat": 7},
        {"name": "sanity_64", "size": 64, "detector": 64, "views": 90, "warmup": 1, "repeat": 7},
    ],
    "publication": [
        {"name": "scale_64", "size": 64, "detector": 64, "views": 90, "warmup": 3, "repeat": 10},
        {"name": "scale_128", "size": 128, "detector": 128, "views": 180, "warmup": 3, "repeat": 10},
        {"name": "scale_192", "size": 192, "detector": 192, "views": 270, "warmup": 2, "repeat": 7},
    ],
}

EVIDENCE_CLASS = {
    "quick": "quick_invalid_for_claims",
    "guard": "guard_invalid_for_claims",
    "publication": "publication_evidence_for_this_machine",
}


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> subprocess.CompletedProcess[str]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.perf_counter() - start
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "$ " + " ".join(cmd) + "\n"
        f"# exit={proc.returncode} wall_sec={duration:.6f}\n\n"
        + proc.stdout,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def _device_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        env["nvidia_smi_gpu"] = proc.stdout.strip()
    except Exception:
        env["nvidia_smi_gpu"] = None
    return env


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 10000:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def _case_summary(case: dict[str, Any], report: dict[str, Any], artifact_rel: dict[str, str]) -> dict[str, Any]:
    timing = report["timing_summary"]
    cold = report.get("cold_timing_summary", {})
    memory = report["gpu_memory_summary_mb"]
    speedups = report["speedups"]
    forward = report["forward_projection"]
    recon = report["reconstruction"]
    fbp_path = report.get("fbp_path", {})
    return {
        "case": case["name"],
        "size": case["size"],
        "detector": case["detector"],
        "views": case["views"],
        "warmup": case["warmup"],
        "repeat": case["repeat"],
        "evidence_class": EVIDENCE_CLASS.get(str(report.get("suite_mode", "")), ""),
        "timed_fbp_path": fbp_path.get("timed_fbp_path"),
        "public_fbp_timed": fbp_path.get("public_fbp_timed"),
        "specialized_pallas_fbp_timed": fbp_path.get("specialized_pallas_fbp_timed"),
        "tomojax_forward_cold_sec": (cold.get("tomojax_forward") or {}).get("seconds"),
        "tomojax_pallas_forward_cold_sec": (cold.get("tomojax_pallas_forward") or {}).get(
            "seconds"
        ),
        "astra_forward_cold_sec": (cold.get("astra_parallel3d_forward") or {}).get("seconds"),
        "tomojax_fbp_cold_sec": (cold.get("tomojax_fbp") or {}).get("seconds"),
        "astra_slice_fbp_cold_sec": (cold.get("astra_slice_fbp") or {}).get("seconds"),
        "tomojax_forward_median_sec": timing["tomojax_forward"]["median_sec"],
        "tomojax_pallas_forward_median_sec": timing["tomojax_pallas_forward"]["median_sec"],
        "astra_forward_median_sec": timing["astra_parallel3d_forward"]["median_sec"],
        "tomojax_fbp_median_sec": timing["tomojax_fbp"]["median_sec"],
        "astra_slice_fbp_median_sec": timing["astra_slice_fbp"]["median_sec"],
        "astra_forward_vs_tomojax_forward": speedups["astra_forward_vs_tomojax_forward_median"],
        "pallas_forward_vs_tomojax_forward": speedups["pallas_forward_vs_tomojax_forward_median"],
        "astra_forward_vs_pallas_forward": speedups["astra_forward_vs_pallas_forward_median"],
        "astra_slice_fbp_vs_tomojax_fbp": speedups["astra_slice_fbp_vs_tomojax_fbp_median"],
        "pallas_rel_l2_vs_jax": (
            forward.get("tomojax_pallas_vs_tomojax") or {}
        ).get("relative_l2_vs_tomojax"),
        "astra_rel_l2_vs_jax": forward["astra_parallel3d_vs_tomojax"]["relative_l2_vs_tomojax"],
        "tomojax_fbp_mse": recon["tomojax_fbp_vs_truth"]["mse"],
        "tomojax_fbp_psnr_db": recon["tomojax_fbp_vs_truth"]["psnr_db"],
        "tomojax_direct_vs_generic_fbp_rel_l2": recon[
            "tomojax_direct_fbp_vs_generic_fbp"
        ]["relative_l2_vs_tomojax"],
        "tomojax_direct_vs_generic_fbp_max_abs": recon[
            "tomojax_direct_fbp_vs_generic_fbp"
        ]["max_abs_vs_tomojax"],
        "tomojax_pallas_peak_delta_mb": memory["tomojax_pallas_forward"]["peak_delta_process_mb"],
        "tomojax_fbp_peak_delta_mb": memory["tomojax_fbp"]["peak_delta_process_mb"],
        "json": artifact_rel["json"],
        "markdown": artifact_rel["markdown"],
        "summary_csv": artifact_rel["summary_csv"],
        "quality_csv": artifact_rel["quality_csv"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, suite: dict[str, Any]) -> None:
    lines = [
        "# TomoJAX Benchmark Suite",
        "",
        f"- Mode: `{suite['mode']}`",
        f"- Evidence class: `{suite['evidence_class']}`",
        f"- Branch: `{suite['git_branch']}`",
        f"- Commit: `{suite['git_commit']}`",
        f"- Note: {suite['note'] or '-'}",
        f"- Created: `{suite['created_at']}`",
        "",
        "## Cases",
        "",
        "| Case | Size | Views | JAX Fwd warm | Pallas Fwd warm | ASTRA Fwd warm | Specialized FBP warm | FBP path | ASTRA FBP warm | Pallas vs JAX | ASTRA vs Pallas | FBP: ASTRA vs Specialized | Pallas rel L2 | Direct/Generic FBP L2 | FBP PSNR |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in suite["case_summaries"]:
        lines.append(
            "| {case} | {size} | {views} | {jax} | {pallas} | {astra} | {tfbp} | {fbp_path} | {afbp} | {pallas_speed}x | {astra_pallas}x | {fbp_speed}x | {pallas_l2} | {direct_generic_l2} | {psnr} |".format(
                case=row["case"],
                size=row["size"],
                views=row["views"],
                jax=_fmt(row["tomojax_forward_median_sec"]),
                pallas=_fmt(row["tomojax_pallas_forward_median_sec"]),
                astra=_fmt(row["astra_forward_median_sec"]),
                tfbp=_fmt(row["tomojax_fbp_median_sec"]),
                fbp_path=_fmt(row.get("timed_fbp_path")),
                afbp=_fmt(row["astra_slice_fbp_median_sec"]),
                pallas_speed=_fmt(row["pallas_forward_vs_tomojax_forward"]),
                astra_pallas=_fmt(row["astra_forward_vs_pallas_forward"]),
                fbp_speed=_fmt(row["astra_slice_fbp_vs_tomojax_fbp"]),
                pallas_l2=_fmt(row["pallas_rel_l2_vs_jax"]),
                direct_generic_l2=_fmt(row["tomojax_direct_vs_generic_fbp_rel_l2"]),
                psnr=_fmt(row["tomojax_fbp_psnr_db"]),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Scope",
            "",
            (
                "This suite is an optimization guard, not publication evidence."
                if suite["mode"] in {"quick", "guard"}
                else "This suite is publication evidence for the recorded machine and environment only."
            ),
        ]
    )

    sanity = suite.get("pallas_sanity")
    if sanity:
        lines.extend(
            [
                "",
                "## Pallas Changed-Input Sanity",
                "",
                "| Check | Value |",
                "|---|---:|",
                f"| Status | {sanity['status']} |",
                f"| Scaled sum ratio | {_fmt(sanity['checks']['scaled_sum_ratio'])} |",
                f"| Scaled output rel L2 vs base | {_fmt(sanity['checks']['scaled_rel_l2_vs_base'])} |",
                f"| Shifted pose rel L2 vs base | {_fmt(sanity['checks']['shifted_pose_rel_l2_vs_base'])} |",
                f"| Base Pallas vs JAX rel L2 | {_fmt(sanity['checks']['base_pallas_vs_jax_rel_l2'])} |",
            ]
        )

    alignment = suite.get("alignment_smoke")
    if alignment:
        lines.extend(
            [
                "",
                "## Alignment Smoke",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Wall time | {_fmt(alignment['timing']['wall_sec'])} sec |",
                f"| Initial loss | {_fmt(alignment['loss']['initial'])} |",
                f"| Final loss | {_fmt(alignment['loss']['final'])} |",
                f"| Loss delta | {_fmt(alignment['loss']['delta_percent'])}% |",
                f"| Aligned MSE vs GT | {_fmt(alignment['quality']['aligned_recon_vs_truth']['mse'])} |",
                f"| Slice PNG | `{alignment['artifacts']['slice_png']}` |",
            ]
        )

    alignment_objective = suite.get("alignment_objective")
    if alignment_objective:
        summary = alignment_objective["summary"]
        lines.extend(
            [
                "",
                "## Alignment Objective",
                "",
                "- Suite: `alignment_objective`",
                "- Value+grad no-checkpoint speedup vs checkpointed: "
                f"`{_fmt(summary['no_checkpoint_speedup_vs_checkpointed'])}x`",
                "- Checkpointed warm median: "
                f"`{_fmt(summary['checkpointed_warm_seconds_median'])}` sec",
                "- No-checkpoint warm median: "
                f"`{_fmt(summary['no_checkpoint_warm_seconds_median'])}` sec",
            ]
        )

    residual = suite.get("forward_residual")
    if residual:
        fused = residual["summary"]["pallas_modes"]["pallas_fused"]
        lines.extend(
            [
                "",
                "## Forward Residual",
                "",
                f"- Suite: `{residual['suite']}`",
                "- Pallas fused geomean speedup vs JAX materialized: "
                f"`{_fmt(fused['geomean_speedup_vs_jax_materialized_warm_median'])}x`",
            ]
        )

    fista = suite.get("fista_iteration")
    if fista:
        lines.extend(["", "## FISTA Iteration", ""])
        for case in fista["cases"]:
            lines.append(
                f"- `{case['case_name']}`: warm median "
                f"`{_fmt(case['warm_seconds_median'])}` sec"
            )

    lines.extend(["", "## Artifacts", ""])
    for row in suite["case_summaries"]:
        lines.append(f"- `{row['case']}`: `{row['markdown']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TomoJAX benchmark case suites.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tomojax-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(CASE_PRESETS), default="guard")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--git-branch", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--include-alignment", action="store_true")
    parser.add_argument("--include-alignment-objective", action="store_true")
    parser.add_argument("--include-forward-residual", action="store_true")
    parser.add_argument("--include-fista-iteration", action="store_true")
    parser.add_argument("--include-pallas-sanity", action="store_true", default=True)
    parser.add_argument("--no-pallas-sanity", dest="include_pallas_sanity", action="store_false")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / "logs"
    cases_dir = args.out_dir / "cases"
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    src_path = str(args.tomojax_dir / "src")
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"
    )
    python_cmd = env.get("TOMOJAX_BENCH_PYTHON")
    python_prefix = [python_cmd] if python_cmd else ["uv", "run", "python"]

    case_summaries: list[dict[str, Any]] = []
    for case in CASE_PRESETS[args.mode]:
        case_prefix = cases_dir / f"{case['name']}_{args.stamp}"
        cmd = [
            *python_prefix,
            "-m",
            "tomojax.bench.astra_parallel",
            "--size",
            str(case["size"]),
            "--detector",
            str(case["detector"]),
            "--views",
            str(case["views"]),
            "--warmup",
            str(case["warmup"]),
            "--repeat",
            str(case["repeat"]),
            "--note",
            f"{args.note} [{args.mode}:{case['name']}]",
            "--git-branch",
            args.git_branch,
            "--git-commit",
            args.git_commit,
            "--out",
            str(case_prefix.with_suffix(".json")),
            "--summary-csv",
            str(case_prefix.with_name(case_prefix.name + "_summary.csv")),
            "--quality-csv",
            str(case_prefix.with_name(case_prefix.name + "_quality.csv")),
            "--summary-md",
            str(case_prefix.with_suffix(".md")),
        ]
        _run(cmd, cwd=args.tomojax_dir, env=env, log=logs_dir / f"{case['name']}.log")
        report = json.loads(case_prefix.with_suffix(".json").read_text(encoding="utf-8"))
        report["suite_mode"] = args.mode
        artifact_rel = {
            "json": str(case_prefix.with_suffix(".json").relative_to(args.out_dir)),
            "markdown": str(case_prefix.with_suffix(".md").relative_to(args.out_dir)),
            "summary_csv": str(
                case_prefix.with_name(case_prefix.name + "_summary.csv").relative_to(args.out_dir)
            ),
            "quality_csv": str(
                case_prefix.with_name(case_prefix.name + "_quality.csv").relative_to(args.out_dir)
            ),
        }
        case_summaries.append(_case_summary(case, report, artifact_rel))

    pallas_sanity = None
    if args.include_pallas_sanity:
        sanity_path = args.out_dir / "pallas_changed_input_sanity.json"
        cmd = [
            *python_prefix,
            "-m",
            "tomojax.bench.pallas_sanity",
            "--out",
            str(sanity_path),
            "--git-branch",
            args.git_branch,
            "--git-commit",
            args.git_commit,
            "--note",
            args.note,
        ]
        _run(cmd, cwd=args.tomojax_dir, env=env, log=logs_dir / "pallas_changed_input_sanity.log")
        pallas_sanity = json.loads(sanity_path.read_text(encoding="utf-8"))

    alignment = None
    if args.include_alignment:
        alignment_path = args.out_dir / "alignment_smoke.json"
        alignment_md = args.out_dir / "alignment_smoke.md"
        alignment_png = args.out_dir / "alignment_smoke_slices.png"
        cmd = [
            *python_prefix,
            "-m",
            "tomojax.bench.alignment_smoke",
            "--tomojax-dir",
            str(args.tomojax_dir),
            "--fixture-dir",
            str(args.root / "alignment-fixtures" / "full_pose_24_strong"),
            "--out",
            str(alignment_path),
            "--summary-md",
            str(alignment_md),
            "--slice-png",
            str(alignment_png),
            "--note",
            args.note,
            "--git-branch",
            args.git_branch,
            "--git-commit",
            args.git_commit,
            "--levels",
            "1",
            "--outer-iters",
            "3",
            "--recon-iters",
            "4",
            "--loss",
            "l2",
            "--schedule",
            "pose_only",
        ]
        _run(cmd, cwd=args.tomojax_dir, env=env, log=logs_dir / "alignment_smoke.log")
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        alignment["artifacts"]["slice_png"] = str(alignment_png.relative_to(args.out_dir))

    alignment_objective = None
    if args.include_alignment_objective:
        from tomojax.bench.alignment_objective import (
            run_alignment_objective_suite,
            write_benchmark_json,
        )

        alignment_objective = run_alignment_objective_suite("alignment_objective")
        write_benchmark_json(alignment_objective, args.out_dir / "alignment_objective.json")

    forward_residual = None
    if args.include_forward_residual:
        from tomojax.bench.forward_residual import (
            run_forward_residual_suite,
            write_benchmark_json,
        )

        forward_residual = run_forward_residual_suite("general_pose")
        write_benchmark_json(forward_residual, args.out_dir / "forward_residual_general_pose.json")

    fista_iteration = None
    if args.include_fista_iteration:
        from tomojax.bench.fista_iteration import (
            run_fista_iteration_suite,
            write_benchmark_json,
        )

        fista_iteration = run_fista_iteration_suite("fista_iteration")
        write_benchmark_json(fista_iteration, args.out_dir / "fista_iteration.json")

    suite = {
        "benchmark": "tomojax_benchmark_suite",
        "mode": args.mode,
        "evidence_class": EVIDENCE_CLASS[args.mode],
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "note": args.note,
        "git_branch": args.git_branch,
        "git_commit": args.git_commit,
        "environment": _device_environment(),
        "case_summaries": case_summaries,
        "pallas_sanity": pallas_sanity,
        "alignment_smoke": alignment,
        "alignment_objective": alignment_objective,
        "forward_residual": forward_residual,
        "fista_iteration": fista_iteration,
    }
    (args.out_dir / "suite.json").write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.out_dir / "cases.csv", case_summaries)
    _write_summary_md(args.out_dir / "summary.md", suite)
    if pallas_sanity is not None:
        (args.out_dir / "pallas_changed_input_sanity.json").write_text(
            json.dumps(pallas_sanity, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(suite, indent=2))


if __name__ == "__main__":
    main()
