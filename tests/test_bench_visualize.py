from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("psutil")

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

visualize = importlib.import_module("visualize")
fitness = importlib.import_module("fitness")


def test_save_alignment_summary_writes_png(tmp_path: Path) -> None:
    gt_volume = np.linspace(0.0, 1.0, 27, dtype=np.float32).reshape(3, 3, 3)
    baseline_volume = gt_volume * 0.4
    final_volume = gt_volume * 0.9
    out_path = tmp_path / "align.summary.png"

    written = visualize.save_alignment_summary(
        out_path=out_path,
        profile_name="smoke_align",
        gt_volume=gt_volume,
        baseline_volume=baseline_volume,
        final_volume=final_volume,
        loss_history=[5.0, 2.5, 1.25],
        convergence_trace=[
            {"outer_idx": 1, "quality_value": 0.8},
            {"outer_idx": 2, "quality_value": 0.4},
        ],
        convergence_metric_name="gt_mse",
        quality_threshold_value=0.5,
        metrics={
            "objective_name": "gt_mse",
            "objective_value": 0.123,
            "warm_run_seconds_mean": 1.5,
            "warmup_seconds": 0.8,
            "warmup_incomplete": True,
            "primer_ran": True,
            "primer_seconds": 2.0,
            "primer_reached_finest_level": False,
            "warmup_stop_reason": "budget",
            "peak_gpu_memory_mb": 256.0,
            "gpu_memory_scope": "process",
            "quality_threshold_metric": "gt_mse",
            "quality_threshold_value": 0.5,
            "quality_threshold_met": True,
            "warm_seconds_to_quality_threshold": 1.2,
        },
        quality={
            "gt_mse": 0.123,
            "rot_rms_deg": 0.4,
            "trans_rms_px": 1.1,
        },
        fixture={
            "volume_shape": [3, 3, 3],
            "n_views": 12,
        },
    )

    assert written == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_visualize_helpers_handle_nonfinite_and_invalid_trace_rows() -> None:
    volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    slices = visualize._central_slices(volume)

    assert set(slices) == {"xy", "xz", "yz"}
    assert slices["xy"].shape == (3, 3)

    limits = visualize._display_limits(np.asarray([[np.nan, 2.0], [1.0, np.inf]], dtype=np.float32))
    assert limits == (1.0, 2.0)

    error_limit = visualize._error_limit(np.asarray([[np.nan, np.inf], [0.0, 3.0]], dtype=np.float32))
    assert error_limit >= 1e-6

    xs, ys, levels = visualize._trace_points(
        [
            {"outer_idx": 1, "quality_value": 0.8, "level_factor": 4},
            {"outer_idx": "bad", "quality_value": 0.7},
            {"outer_idx": 3, "quality_value": None},
            {"outer_idx": 4, "quality_value": 0.5, "level_factor": 1},
        ],
        value_key="quality_value",
    )

    np.testing.assert_array_equal(xs, np.asarray([1, 4], dtype=np.int32))
    np.testing.assert_allclose(ys, np.asarray([0.8, 0.5], dtype=np.float32))
    assert levels == [4, 1]


def test_plot_trace_metric_marks_threshold_crossing() -> None:
    fig, ax = visualize.plt.subplots()
    try:
        visualize._plot_trace_metric(
            ax,
            trace=[
                {"outer_idx": 1, "quality_value": 0.8, "level_factor": 4},
                {"outer_idx": 2, "quality_value": 0.4, "level_factor": 1},
            ],
            value_key="quality_value",
            title="gt_mse vs Outer Iteration",
            y_label="gt_mse",
            threshold=0.5,
        )
    finally:
        visualize.plt.close(fig)

    assert ax.get_title() == "gt_mse vs Outer Iteration"
    assert ax.get_xlabel() == "Outer Iteration"
    assert ax.get_ylabel() == "gt_mse"
    assert len(ax.lines) >= 3


def test_text_lines_include_contract_and_level_split_details(tmp_path: Path) -> None:
    lines = visualize._text_lines(
        profile_name="screen_align",
        out_path=tmp_path / "summary.png",
        metrics={
            "objective_name": "objective_time_memguard",
            "objective_value": 15.0,
            "warm_gt_mse_median": 0.12,
            "benchmark_valid": True,
            "reached_finest_level": True,
            "warm_run_seconds_mean": 12.5,
            "warmup_seconds": 3.0,
            "warmup_incomplete": False,
            "primer_ran": False,
            "primer_seconds": None,
            "time_budget_seconds": 20.0,
            "peak_gpu_memory_mb": 256.0,
            "gpu_memory_scope": "process",
            "quality_threshold_metric": "gt_mse",
            "quality_threshold_value": 0.2,
            "quality_threshold_scope": "finest_only",
            "quality_threshold_met": True,
            "stopped_on_threshold": True,
            "stopped_on_plateau": False,
            "stopped_on_budget": False,
            "final_stop_reason": "threshold",
            "final_stop_level_factor": 1,
            "first_threshold_crossing_level_factor": 1,
            "warm_seconds_to_quality_threshold": 12.0,
            "quality_contract_met": True,
            "warm_seconds_to_quality_contract": 12.0,
            "memory_guard_value_mb": 150.0,
            "memory_guard_penalty": 3.0,
            "objective_time_memguard": 15.0,
            "warm_level_summaries": [
                {
                    "level_factor": 4,
                    "elapsed_seconds_total": 5.0,
                    "outer_iters_executed": 2,
                    "final_gt_mse": 0.4,
                }
            ],
        },
        quality={"gt_mse": 0.12, "trans_gf_rmse_px": 0.3},
        fixture={"volume_shape": [16, 16, 16], "n_views": 64},
    )

    assert any("quality_contract_met: True" in line for line in lines)
    assert any("threshold_scope: finest_only" in line for line in lines)
    assert any("memory_guard_penalty: 3.0" in line for line in lines)
    assert any("level_split:" in line for line in lines)
    assert any("x4: 5.0s, 2 iters, final_gt_mse=0.4" in line for line in lines)


def test_should_render_alignment_summary_respects_profile_toggle() -> None:
    assert fitness._should_render_alignment_summary({"task": "align"}) is True
    assert (
        fitness._should_render_alignment_summary(
            {"task": "align", "visualization": {"enabled": False}}
        )
        is False
    )
    assert fitness._should_render_alignment_summary({"task": "recon"}) is False
