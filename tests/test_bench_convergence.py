from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

fitness = importlib.import_module("fitness")

def test_convergence_summary_marks_threshold_crossing() -> None:
    summary = fitness._convergence_summary_from_trace(
        metric="gt_mse",
        threshold=0.5,
        trace=[
            {"outer_idx": 1, "elapsed_seconds": 10.0, "quality_value": 1.2},
            {"outer_idx": 2, "elapsed_seconds": 18.0, "quality_value": 0.4},
        ],
        stopped_on_threshold=True,
    )

    assert summary.threshold_met is True
    assert summary.stopped_on_threshold is True
    assert summary.stopped_on_plateau is False
    assert summary.stopped_on_budget is False
    assert summary.seconds_to_threshold == 18.0
    assert summary.outer_iters_to_threshold == 2
    assert summary.best_quality_value == 0.4
    assert summary.total_outer_iters_executed == 2

def test_convergence_summary_handles_threshold_miss() -> None:
    summary = fitness._convergence_summary_from_trace(
        metric="gt_mse",
        threshold=0.2,
        trace=[
            {"outer_idx": 1, "elapsed_seconds": 5.0, "quality_value": 1.0},
            {"outer_idx": 2, "elapsed_seconds": 9.0, "quality_value": 0.7},
        ],
        stopped_on_plateau=True,
    )

    assert summary.threshold_met is False
    assert summary.stopped_on_threshold is False
    assert summary.stopped_on_plateau is True
    assert summary.stopped_on_budget is False
    assert summary.seconds_to_threshold is None
    assert summary.outer_iters_to_threshold is None
    assert summary.best_quality_value == 0.7
    assert summary.best_quality_elapsed_seconds == 9.0


def test_meaningful_relative_improvement_requires_margin() -> None:
    assert fitness._is_meaningful_relative_improvement(10.0, 9.0, 0.02) is True
    assert fitness._is_meaningful_relative_improvement(10.0, 9.9, 0.02) is False


def test_aggregate_warm_convergence_runs_requires_multiple_hits() -> None:
    run1 = fitness.ConvergenceRunSummary(
        metric="gt_mse",
        threshold=1.0,
        threshold_met=True,
        stopped_on_threshold=True,
        stopped_on_plateau=False,
        stopped_on_budget=False,
        seconds_to_threshold=12.0,
        outer_iters_to_threshold=2,
        best_quality_value=0.8,
        best_quality_elapsed_seconds=12.0,
        total_outer_iters_executed=2,
        trace=[{"outer_idx": 1, "elapsed_seconds": 5.0, "quality_value": 1.2}],
    )
    run2 = fitness.ConvergenceRunSummary(
        metric="gt_mse",
        threshold=1.0,
        threshold_met=False,
        stopped_on_threshold=False,
        stopped_on_plateau=True,
        stopped_on_budget=False,
        seconds_to_threshold=None,
        outer_iters_to_threshold=None,
        best_quality_value=1.1,
        best_quality_elapsed_seconds=18.0,
        total_outer_iters_executed=3,
        trace=[{"outer_idx": 1, "elapsed_seconds": 6.0, "quality_value": 1.5}],
    )
    run3 = fitness.ConvergenceRunSummary(
        metric="gt_mse",
        threshold=1.0,
        threshold_met=True,
        stopped_on_threshold=True,
        stopped_on_plateau=False,
        stopped_on_budget=False,
        seconds_to_threshold=10.0,
        outer_iters_to_threshold=2,
        best_quality_value=0.9,
        best_quality_elapsed_seconds=10.0,
        total_outer_iters_executed=2,
        trace=[{"outer_idx": 1, "elapsed_seconds": 4.0, "quality_value": 1.3}],
    )

    aggregate = fitness._aggregate_warm_convergence_runs(
        [run1, run2, run3],
        required_successes=2,
    )

    assert aggregate["quality_threshold_met"] is True
    assert aggregate["warm_threshold_hit_count"] == 2
    assert aggregate["warm_threshold_total_runs"] == 3
    assert aggregate["warm_seconds_to_quality_threshold"] == 11.0
    assert aggregate["warm_outer_iters_to_quality_threshold"] == 2
    assert aggregate["warm_best_quality_value"] == 0.9
    assert aggregate["warm_stopped_on_plateau_count"] == 1
