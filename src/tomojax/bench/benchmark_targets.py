from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkTarget:
    metric: str
    baseline_value: float
    target_value: float
    direction: str
    unit: str
    source_commit: str
    source_artifact: str
    note: str


SOURCE_COMMIT = "994342ad4ba8964aa0bfc5a901b4a9156fdef6c3"
SOURCE_ARTIFACT = "results/guard_20260504_071737_994342a/summary.md"


TARGETS_2X: tuple[BenchmarkTarget, ...] = (
    BenchmarkTarget(
        metric="alignment_smoke_wall_seconds",
        baseline_value=2.2030,
        target_value=1.1015,
        direction="lower_is_better",
        unit="seconds",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="24^3 full 5-axis alignment smoke wall time",
    ),
    BenchmarkTarget(
        metric="alignment_objective_checkpointed_value_grad_seconds",
        baseline_value=0.0014588110207114369,
        target_value=0.0007294055103557184,
        direction="lower_is_better",
        unit="seconds",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="24^3 alignment objective checkpointed value+grad warm median",
    ),
    BenchmarkTarget(
        metric="general_pose_forward_dispatch_speedup_vs_best_jax",
        baseline_value=5.7033,
        target_value=11.4066,
        direction="higher_is_better",
        unit="x",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="General-pose forward Pallas dispatch geomean speedup",
    ),
    BenchmarkTarget(
        metric="general_pose_forward_dispatch_worst_case_speedup_vs_best_jax",
        baseline_value=4.6405,
        target_value=9.2810,
        direction="higher_is_better",
        unit="x",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="General-pose forward Pallas dispatch worst-case speedup",
    ),
    BenchmarkTarget(
        metric="forward_residual_dispatch_speedup_vs_jax_materialized",
        baseline_value=5.6801,
        target_value=11.3602,
        direction="higher_is_better",
        unit="x",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="General-pose residual Pallas dispatch geomean speedup",
    ),
    BenchmarkTarget(
        metric="fista_iter_24_pallas_seconds",
        baseline_value=0.00041540400707162917,
        target_value=0.00020770200353581459,
        direction="lower_is_better",
        unit="seconds",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="24^3 one-iteration Pallas FISTA warm median",
    ),
    BenchmarkTarget(
        metric="fista_iter_64_pallas_seconds",
        baseline_value=0.0048012729967013,
        target_value=0.00240063649835065,
        direction="lower_is_better",
        unit="seconds",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="64^3 one-iteration Pallas FISTA warm median",
    ),
    BenchmarkTarget(
        metric="fista_iter_24_speedup_vs_jax",
        baseline_value=5.1025,
        target_value=10.2050,
        direction="higher_is_better",
        unit="x",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="24^3 one-iteration FISTA speedup vs JAX",
    ),
    BenchmarkTarget(
        metric="fista_iter_64_speedup_vs_jax",
        baseline_value=11.2923,
        target_value=22.5846,
        direction="higher_is_better",
        unit="x",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="64^3 one-iteration FISTA speedup vs JAX",
    ),
    BenchmarkTarget(
        metric="regular_z_128_pallas_forward_seconds",
        baseline_value=0.0062,
        target_value=0.0031,
        direction="lower_is_better",
        unit="seconds",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="128^3 regular z-axis Pallas forward warm median",
    ),
    BenchmarkTarget(
        metric="regular_z_128_specialized_fbp_seconds",
        baseline_value=0.0029,
        target_value=0.00145,
        direction="lower_is_better",
        unit="seconds",
        source_commit=SOURCE_COMMIT,
        source_artifact=SOURCE_ARTIFACT,
        note="128^3 specialized Pallas parallel-z FBP helper warm median",
    ),
)


def benchmark_targets_report() -> dict[str, Any]:
    return {
        "source_commit": SOURCE_COMMIT,
        "source_artifact": SOURCE_ARTIFACT,
        "target_policy": "Improve each current best guard value by 2x while preserving quality/parity gates.",
        "targets": [asdict(target) for target in TARGETS_2X],
    }

