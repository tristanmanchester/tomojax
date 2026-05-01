from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.bench import astra_parallel


def _sample_report() -> dict:
    timings = {
        "tomojax_forward": {"runs": 1, "median_sec": 1.0, "mean_sec": 1.0, "min_sec": 1.0, "max_sec": 1.0},
        "tomojax_pallas_forward": {"runs": 1, "median_sec": 0.5, "mean_sec": 0.5, "min_sec": 0.5, "max_sec": 0.5},
        "astra_parallel3d_forward": {"runs": 1, "median_sec": 0.25, "mean_sec": 0.25, "min_sec": 0.25, "max_sec": 0.25},
        "tomojax_fbp": {"runs": 1, "median_sec": 0.2, "mean_sec": 0.2, "min_sec": 0.2, "max_sec": 0.2},
        "astra_slice_fbp": {"runs": 1, "median_sec": 0.4, "mean_sec": 0.4, "min_sec": 0.4, "max_sec": 0.4},
    }
    memory = {
        key: {"peak_process_mb": 10.0, "peak_delta_process_mb": 1.0}
        for key in timings
    }
    return {
        "config": {"size": 8, "detector": 8, "views": 4, "warmup": 1, "repeat": 1},
        "timing_summary": timings,
        "cold_timing_summary": {key: {"seconds": index + 0.1} for index, key in enumerate(timings)},
        "gpu_memory_summary_mb": memory,
        "speedups": {
            "astra_forward_vs_tomojax_forward_median": 4.0,
            "pallas_forward_vs_tomojax_forward_median": 2.0,
            "astra_forward_vs_pallas_forward_median": 2.0,
            "astra_slice_fbp_vs_tomojax_fbp_median": 0.5,
        },
        "fbp_path": {
            "timed_fbp_path": "specialized_pallas_parallel_z_helper",
            "public_fbp_timed": False,
            "specialized_pallas_fbp_timed": True,
        },
        "forward_projection": {
            "tomojax_pallas_vs_tomojax": {
                "mse_vs_tomojax": 0.0,
                "rmse_vs_tomojax": 0.0,
                "relative_l2_vs_tomojax": 0.0,
                "max_abs_vs_tomojax": 0.0,
            },
            "astra_parallel3d_vs_tomojax": {
                "mse_vs_tomojax": 0.1,
                "rmse_vs_tomojax": 0.2,
                "relative_l2_vs_tomojax": 0.3,
                "max_abs_vs_tomojax": 0.4,
            },
        },
        "reconstruction": {
            "tomojax_fbp_vs_truth": {"mse": 0.1, "rmse": 0.2, "psnr_db": 30.0},
            "astra_slice_fbp_vs_truth": {"mse": 0.2, "rmse": 0.3, "psnr_db": 25.0},
            "astra_slice_fbp_vs_tomojax_fbp": {"mse": 0.3, "rmse": 0.4, "psnr_db": 20.0},
            "tomojax_direct_fbp_vs_generic_fbp": {
                "mse_vs_tomojax": 0.01,
                "rmse_vs_tomojax": 0.02,
                "relative_l2_vs_tomojax": 0.03,
                "max_abs_vs_tomojax": 0.04,
            },
        },
    }


def test_time_call_with_cold_records_first_call_separately() -> None:
    calls = []

    def fn():
        calls.append("called")
        return jnp.asarray([len(calls)], dtype=jnp.float32)

    value, cold_sec, warm_times, memory = astra_parallel._time_call_with_cold(
        fn,
        warmup=1,
        repeat=2,
    )

    assert np.asarray(value).shape == (1,)
    assert cold_sec >= 0.0
    assert len(warm_times) == 2
    assert len(memory) == 2
    assert calls == ["called", "called", "called", "called"]


def test_astra_parallel_rows_include_cold_timing_and_direct_generic_quality() -> None:
    report = _sample_report()

    op_rows = astra_parallel._operation_rows(report)
    quality_rows = astra_parallel._quality_rows(report)

    assert op_rows[0]["cold_sec"] == pytest.approx(0.1)
    assert any(
        row["comparison"] == "TomoJAX specialized Pallas FBP helper vs TomoJAX generic FBP"
        and row["relative_l2"] == pytest.approx(0.03)
        for row in quality_rows
    )


def test_astra_parallel_markdown_includes_cold_and_direct_generic_quality(tmp_path) -> None:
    path = tmp_path / "summary.md"

    astra_parallel._write_markdown(path, _sample_report())

    text = path.read_text(encoding="utf-8")
    assert "Cold sec" in text
    assert "Timed TomoJAX FBP path" in text
    assert "TomoJAX specialized Pallas FBP helper vs TomoJAX generic FBP" in text


def test_astra_parallel_main_reports_missing_astra(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(astra_parallel, "astra", None)
    monkeypatch.setattr(
        "sys.argv",
        ["tomojax-astra-parallel-bench", "--out", str(tmp_path / "out.json")],
    )

    with pytest.raises(RuntimeError, match="ASTRA is required"):
        astra_parallel.main()
