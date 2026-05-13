from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from tomojax.bench.alignment_objective import (
    run_alignment_objective_suite,
)
from tomojax.bench.fista_iteration import (
    FistaIterationBenchmarkConfig,
    fista_iteration_pallas_tile_shape,
    run_fista_iteration_case,
)
from tomojax.bench.forward_projector import (
    ForwardSinogramBenchmarkConfig,
    run_forward_sinogram_benchmark,
)
from tomojax.bench.forward_residual import (
    ForwardResidualBenchmarkConfig,
    run_forward_residual_benchmark,
)


@dataclass(frozen=True)
class SampledCaseConfig:
    case_name: str
    family: str
    seed: int
    config: dict[str, Any]


AWKWARD_SHAPES = (
    (31, 37, 43, 29, 41, 47),
    (40, 32, 48, 37, 53, 41),
    (48, 40, 32, 41, 37, 47),
    (56, 48, 40, 59, 43, 53),
)

GENERAL_SHAPES = (
    (24, 24, 24, 24, 24, 24),
    (32, 28, 36, 31, 37, 37),
    (40, 32, 48, 41, 37, 47),
    (64, 56, 48, 59, 53, 64),
)

ALIGNMENT_OBJECTIVE_SHAPES = (
    (20, 20, 20, 20, 20, 18),
    (24, 24, 24, 24, 24, 24),
    (28, 28, 28, 31, 29, 32),
    (32, 28, 36, 31, 37, 40),
)

ALIGNMENT_SMOKE_SHAPES = (20, 22, 24)

FISTA_SHAPES = (
    (24, 24, 24, 24, 24, 24),
    (32, 28, 36, 31, 37, 37),
    (48, 40, 32, 41, 37, 47),
    (64, 64, 64, 64, 64, 90),
)


def _choice(rng: np.random.Generator, values: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return values[int(rng.integers(0, len(values)))]


def _sample_forward_config(rng: np.random.Generator, seed: int) -> ForwardSinogramBenchmarkConfig:
    shape_pool = AWKWARD_SHAPES if bool(rng.integers(0, 2)) else GENERAL_SHAPES
    nx, ny, nz, nu, nv, n_views = _choice(rng, shape_pool)
    return ForwardSinogramBenchmarkConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        nu=nu,
        nv=nv,
        n_views=n_views,
        seed=seed,
        warm_runs=5,
        gather_dtype="fp32",
        pose_mode="general_5d",
        pallas_state_mode="cached",
        pallas_tile_shape=tuple((int(rng.choice([8, 12, 16, 24])), int(rng.choice([4, 8])))),
        pallas_num_warps=1,
    )


def _sample_residual_config(
    forward_config: ForwardSinogramBenchmarkConfig,
) -> ForwardResidualBenchmarkConfig:
    return ForwardResidualBenchmarkConfig(**asdict(forward_config), target_delta=0.01)


def _sample_fista_config(rng: np.random.Generator, seed: int) -> FistaIterationBenchmarkConfig:
    nx, ny, nz, nu, nv, n_views = _choice(rng, FISTA_SHAPES)
    diagnostics = bool(rng.integers(0, 4) == 0)
    return FistaIterationBenchmarkConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        nu=nu,
        nv=nv,
        n_views=n_views,
        seed=seed,
        warm_runs=5,
        pose_mode="general_5d",
        pallas_tile_shape=fista_iteration_pallas_tile_shape(
            nv=nv,
            nu=nu,
            n_views=n_views,
        ),
        pallas_num_warps=1,
        compute_iteration_loss=diagnostics,
        compute_final_data_loss=diagnostics,
        compute_final_regulariser_value=diagnostics,
    )


def _sample_alignment_objective_overrides(
    rng: np.random.Generator,
    seed: int,
) -> dict[str, Any]:
    nx, ny, nz, nu, nv, n_views = _choice(rng, ALIGNMENT_OBJECTIVE_SHAPES)
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "nu": nu,
        "nv": nv,
        "n_views": n_views,
        "seed": seed,
        "warm_runs": 5,
        "projector_unroll": int(rng.choice([1, 2, 4])),
        "gather_dtype": str(rng.choice(["bf16", "fp32"])),
    }


def _sample_alignment_smoke_config(rng: np.random.Generator, seed: int) -> dict[str, Any]:
    size = int(rng.choice(ALIGNMENT_SMOKE_SHAPES))
    return {
        "size": size,
        "views": size,
        "seed": seed,
        "misalignment_seed": int(rng.integers(0, 2**31 - 1)),
        "misalignment_rot_deg": float(rng.choice([3.0, 5.0, 7.0])),
        "misalignment_trans_px": float(rng.choice([2.0, 3.0, 4.0])),
        "outer_iters": 3,
        "recon_iters": 4,
        "levels": [1],
        "align_profile": "lightning",
        "loss": "l2",
        "schedule": "pose_only",
        "regulariser": "huber_tv",
        "views_per_batch": 0,
    }


def run_sampled_alignment_smoke_case(
    *,
    case_name: str,
    config: dict[str, Any],
    tomojax_dir: Path,
    fixture_root: Path,
    out_dir: Path,
    note: str = "",
    git_branch: str = "",
    git_commit: str = "",
) -> dict[str, Any]:
    """Run one sampled full alignment smoke case and return its JSON report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = fixture_root / case_name
    out_json = out_dir / f"{case_name}.json"
    summary_md = out_dir / f"{case_name}.md"
    slice_png = out_dir / f"{case_name}_slices.png"
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
    src_path = str(tomojax_dir / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"
    env.setdefault("TOMOJAX_BENCH_PYTHON", sys.executable)
    cmd = [
        sys.executable,
        "-m",
        "tomojax.bench.alignment_smoke",
        "--tomojax-dir",
        str(tomojax_dir),
        "--fixture-dir",
        str(fixture_dir),
        "--out",
        str(out_json),
        "--summary-md",
        str(summary_md),
        "--slice-png",
        str(slice_png),
        "--note",
        note,
        "--git-branch",
        git_branch,
        "--git-commit",
        git_commit,
        "--size",
        str(config["size"]),
        "--views",
        str(config["views"]),
        "--seed",
        str(config["seed"]),
        "--misalignment-seed",
        str(config["misalignment_seed"]),
        "--misalignment-rot-deg",
        str(config["misalignment_rot_deg"]),
        "--misalignment-trans-px",
        str(config["misalignment_trans_px"]),
        "--levels",
        *[str(level) for level in config["levels"]],
        "--align-profile",
        str(config.get("align_profile", "lightning")),
        "--outer-iters",
        str(config["outer_iters"]),
        "--recon-iters",
        str(config["recon_iters"]),
        "--loss",
        str(config["loss"]),
        "--schedule",
        str(config["schedule"]),
        "--regulariser",
        str(config["regulariser"]),
        "--views-per-batch",
        str(config["views_per_batch"]),
    ]
    subprocess.run(
        cmd,
        cwd=tomojax_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    report = json.loads(out_json.read_text(encoding="utf-8"))
    report["case_name"] = case_name
    report["artifacts"]["json"] = str(out_json)
    report["artifacts"]["summary_md"] = str(summary_md)
    return report


def _requested_mode(results: list[dict[str, Any]], requested: str) -> dict[str, Any] | None:
    for row in results:
        if row.get("requested_mode") == requested:
            return row
    return None


def run_sampled_representative_suite(
    *,
    suite_seed: int,
    cases_per_family: int = 1,
    tomojax_dir: Path | None = None,
    fixture_root: Path | None = None,
    out_dir: Path | None = None,
    note: str = "",
    git_branch: str = "",
    git_commit: str = "",
) -> dict[str, Any]:
    """Run a seeded anti-overfitting panel of representative benchmark families."""
    rng = np.random.default_rng(int(suite_seed))
    sampled_cases: list[SampledCaseConfig] = []
    forward_cases: list[dict[str, Any]] = []
    residual_cases: list[dict[str, Any]] = []
    fista_cases: list[dict[str, Any]] = []
    alignment_objective_cases: list[dict[str, Any]] = []
    alignment_smoke_cases: list[dict[str, Any]] = []

    for i in range(int(cases_per_family)):
        case_seed = int(rng.integers(0, 2**31 - 1))
        forward_config = _sample_forward_config(rng, case_seed)
        sampled_cases.append(
            SampledCaseConfig(
                case_name=f"sampled-general-forward-{i}",
                family="general_pose_forward",
                seed=case_seed,
                config=asdict(forward_config),
            )
        )
        forward = run_forward_sinogram_benchmark(forward_config)
        forward["case_name"] = f"sampled-general-forward-{i}"
        forward_cases.append(forward)

        residual_config = _sample_residual_config(forward_config)
        sampled_cases.append(
            SampledCaseConfig(
                case_name=f"sampled-forward-residual-{i}",
                family="forward_residual",
                seed=case_seed,
                config=asdict(residual_config),
            )
        )
        residual = run_forward_residual_benchmark(residual_config)
        residual["case_name"] = f"sampled-forward-residual-{i}"
        residual_cases.append(residual)

        fista_seed = int(rng.integers(0, 2**31 - 1))
        fista_config = _sample_fista_config(rng, fista_seed)
        sampled_cases.append(
            SampledCaseConfig(
                case_name=f"sampled-fista-{i}",
                family="fista_iteration",
                seed=fista_seed,
                config=asdict(fista_config),
            )
        )
        fista = run_fista_iteration_case(fista_config)
        fista["case_name"] = f"sampled-fista-{i}"
        fista_cases.append(fista)

        objective_seed = int(rng.integers(0, 2**31 - 1))
        objective_overrides = _sample_alignment_objective_overrides(rng, objective_seed)
        sampled_cases.append(
            SampledCaseConfig(
                case_name=f"sampled-alignment-objective-{i}",
                family="alignment_objective",
                seed=objective_seed,
                config=objective_overrides,
            )
        )
        objective = run_alignment_objective_suite(
            "alignment_objective",
            overrides=objective_overrides,
        )
        objective["case_name"] = f"sampled-alignment-objective-{i}"
        alignment_objective_cases.append(objective)

        if tomojax_dir is not None and fixture_root is not None and out_dir is not None:
            smoke_seed = int(rng.integers(0, 2**31 - 1))
            smoke_config = _sample_alignment_smoke_config(rng, smoke_seed)
            sampled_cases.append(
                SampledCaseConfig(
                    case_name=f"sampled-alignment-smoke-{i}",
                    family="alignment_smoke",
                    seed=smoke_seed,
                    config=smoke_config,
                )
            )
            smoke = run_sampled_alignment_smoke_case(
                case_name=f"sampled-alignment-smoke-{i}",
                config=smoke_config,
                tomojax_dir=tomojax_dir,
                fixture_root=fixture_root,
                out_dir=out_dir / "alignment_smoke_cases",
                note=note,
                git_branch=git_branch,
                git_commit=git_commit,
            )
            alignment_smoke_cases.append(smoke)

    forward_speedups = [
        float(row["speedup_vs_best_jax_warm_median"])
        for case in forward_cases
        for row in case["results"]
        if row.get("requested_mode") == "pallas_dispatch"
        and row.get("speedup_vs_best_jax_warm_median") is not None
    ]
    residual_speedups = [
        float(row["speedup_vs_jax_materialized_warm_median"])
        for case in residual_cases
        for row in case["results"]
        if row.get("requested_mode") == "pallas_dispatch"
        and row.get("speedup_vs_jax_materialized_warm_median") is not None
    ]
    fista_speedups = [
        float(case["speedup_vs_jax_warm_median"])
        for case in fista_cases
        if case.get("speedup_vs_jax_warm_median") is not None
    ]
    alignment_smoke_wall_times = [
        float(case["timing"]["wall_sec"])
        for case in alignment_smoke_cases
        if case.get("timing", {}).get("wall_sec") is not None
    ]
    alignment_smoke_success_counts = [
        sum(bool(value) for value in case.get("success", {}).values())
        for case in alignment_smoke_cases
    ]
    alignment_smoke_success_totals = [
        len(case.get("success", {})) for case in alignment_smoke_cases
    ]

    def geomean(values: list[float]) -> float | None:
        return float(np.exp(np.mean(np.log(values)))) if values and min(values) > 0.0 else None

    return {
        "benchmark": "sampled_representative_suite",
        "suite_seed": int(suite_seed),
        "sampler_version": 2,
        "cases_per_family": int(cases_per_family),
        "sampling_policy": (
            "Seeded random case generation. Artifacts record every sampled config so each "
            "case is exactly reproducible."
        ),
        "sampled_cases": [asdict(case) for case in sampled_cases],
        "summary": {
            "general_forward_dispatch_geomean_speedup_vs_best_jax": geomean(forward_speedups),
            "general_forward_dispatch_worst_speedup_vs_best_jax": min(forward_speedups)
            if forward_speedups
            else None,
            "forward_residual_dispatch_geomean_speedup_vs_jax": geomean(residual_speedups),
            "forward_residual_dispatch_worst_speedup_vs_jax": min(residual_speedups)
            if residual_speedups
            else None,
            "fista_geomean_speedup_vs_jax": geomean(fista_speedups),
            "fista_worst_speedup_vs_jax": min(fista_speedups) if fista_speedups else None,
            "alignment_smoke_median_wall_sec": float(np.median(alignment_smoke_wall_times))
            if alignment_smoke_wall_times
            else None,
            "alignment_smoke_successful_cases": int(
                sum(
                    count == total and total > 0
                    for count, total in zip(
                        alignment_smoke_success_counts,
                        alignment_smoke_success_totals,
                        strict=False,
                    )
                )
            ),
            "alignment_smoke_total_cases": len(alignment_smoke_cases),
        },
        "forward_cases": forward_cases,
        "residual_cases": residual_cases,
        "fista_cases": fista_cases,
        "alignment_objective_cases": alignment_objective_cases,
        "alignment_smoke_cases": alignment_smoke_cases,
    }


def write_benchmark_json(metrics: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
