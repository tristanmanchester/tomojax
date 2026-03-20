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
    )

    assert summary.threshold_met is True
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
    )

    assert summary.threshold_met is False
    assert summary.seconds_to_threshold is None
    assert summary.outer_iters_to_threshold is None
    assert summary.best_quality_value == 0.7
    assert summary.best_quality_elapsed_seconds == 9.0
