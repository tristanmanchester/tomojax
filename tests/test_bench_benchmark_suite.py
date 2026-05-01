from __future__ import annotations

import json

from tomojax.bench import benchmark_suite


def _case_report() -> dict:
    return {
        "suite_mode": "guard",
        "timing_summary": {
            "tomojax_forward": {"median_sec": 1.0},
            "tomojax_pallas_forward": {"median_sec": 0.5},
            "astra_parallel3d_forward": {"median_sec": 0.25},
            "tomojax_fbp": {"median_sec": 0.2},
            "astra_slice_fbp": {"median_sec": 0.4},
        },
        "cold_timing_summary": {
            "tomojax_forward": {"seconds": 10.0},
            "tomojax_pallas_forward": {"seconds": 20.0},
            "astra_parallel3d_forward": {"seconds": 30.0},
            "tomojax_fbp": {"seconds": 40.0},
            "astra_slice_fbp": {"seconds": 50.0},
        },
        "gpu_memory_summary_mb": {
            "tomojax_pallas_forward": {"peak_delta_process_mb": 1.0},
            "tomojax_fbp": {"peak_delta_process_mb": 2.0},
        },
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
            "tomojax_pallas_vs_tomojax": {"relative_l2_vs_tomojax": 0.0},
            "astra_parallel3d_vs_tomojax": {"relative_l2_vs_tomojax": 0.1},
        },
        "reconstruction": {
            "tomojax_fbp_vs_truth": {"mse": 0.01, "psnr_db": 30.0},
            "tomojax_direct_fbp_vs_generic_fbp": {
                "relative_l2_vs_tomojax": 0.02,
                "max_abs_vs_tomojax": 0.03,
            },
        },
    }


def test_case_presets_have_distinct_evidence_classes() -> None:
    assert benchmark_suite.EVIDENCE_CLASS == {
        "quick": "quick_invalid_for_claims",
        "guard": "guard_invalid_for_claims",
        "publication": "publication_evidence_for_this_machine",
    }
    assert [case["name"] for case in benchmark_suite.CASE_PRESETS["guard"]] == [
        "headline_128",
        "sanity_64",
    ]


def test_case_summary_includes_cold_and_direct_generic_metrics() -> None:
    summary = benchmark_suite._case_summary(
        {"name": "case", "size": 8, "detector": 8, "views": 4, "warmup": 1, "repeat": 1},
        _case_report(),
        {"json": "case.json", "markdown": "case.md", "summary_csv": "case.csv", "quality_csv": "quality.csv"},
    )

    assert summary["evidence_class"] == "guard_invalid_for_claims"
    assert summary["timed_fbp_path"] == "specialized_pallas_parallel_z_helper"
    assert summary["public_fbp_timed"] is False
    assert summary["tomojax_forward_cold_sec"] == 10.0
    assert summary["tomojax_direct_vs_generic_fbp_rel_l2"] == 0.02


def test_suite_summary_marks_guard_as_not_publication_evidence(tmp_path) -> None:
    suite = {
        "mode": "guard",
        "evidence_class": "guard_invalid_for_claims",
        "git_branch": "bench/astra-hardening",
        "git_commit": "abc123",
        "note": "test",
        "created_at": "2026-05-01T00:00:00+00:00",
        "case_summaries": [
            benchmark_suite._case_summary(
                {
                    "name": "case",
                    "size": 8,
                    "detector": 8,
                    "views": 4,
                    "warmup": 1,
                    "repeat": 1,
                },
                _case_report(),
                {
                    "json": "case.json",
                    "markdown": "case.md",
                    "summary_csv": "case.csv",
                    "quality_csv": "quality.csv",
                },
            )
        ],
        "pallas_sanity": None,
        "alignment_smoke": None,
    }
    path = tmp_path / "summary.md"

    benchmark_suite._write_summary_md(path, suite)

    text = path.read_text(encoding="utf-8")
    assert "guard_invalid_for_claims" in text
    assert "optimization guard, not publication evidence" in text
    assert "Specialized FBP warm" in text
    assert "Direct/Generic FBP L2" in text


def test_suite_json_can_round_trip_evidence_class(tmp_path) -> None:
    payload = {
        "benchmark": "tomojax_benchmark_suite",
        "mode": "publication",
        "evidence_class": benchmark_suite.EVIDENCE_CLASS["publication"],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8"))["evidence_class"] == (
        "publication_evidence_for_this_machine"
    )
