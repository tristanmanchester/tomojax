from __future__ import annotations

import json

import pytest

from tomojax.bench import sampled
from tomojax.bench.benchmark_targets import TARGETS_2X, benchmark_targets_report


def test_benchmark_targets_are_2x_from_recorded_baselines() -> None:
    report = benchmark_targets_report()

    assert report["source_commit"].startswith("994342a")
    assert report["targets"]
    for target in TARGETS_2X:
        if target.direction == "lower_is_better":
            assert target.target_value == pytest.approx(target.baseline_value / 2.0)
        else:
            assert target.target_value == pytest.approx(target.baseline_value * 2.0)


def test_sampled_suite_records_reproducible_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_forward(config):
        return {
            "benchmark": "forward_sinogram",
            "config": {"nx": config.nx},
            "results": [
                {
                    "requested_mode": "pallas_dispatch",
                    "speedup_vs_best_jax_warm_median": 3.0,
                }
            ],
        }

    def fake_residual(config):
        return {
            "benchmark": "forward_residual",
            "config": {"nx": config.nx},
            "results": [
                {
                    "requested_mode": "pallas_dispatch",
                    "speedup_vs_jax_materialized_warm_median": 4.0,
                }
            ],
        }

    def fake_fista(config):
        return {
            "benchmark": "fista_iteration_comparison",
            "config": {"nx": config.nx},
            "speedup_vs_jax_warm_median": 5.0,
        }

    def fake_alignment_objective(name, *, overrides=None):
        return {
            "benchmark": "alignment_objective_suite",
            "suite": name,
            "summary": {"overrides": overrides},
        }

    monkeypatch.setattr(sampled, "run_forward_sinogram_benchmark", fake_forward)
    monkeypatch.setattr(sampled, "run_forward_residual_benchmark", fake_residual)
    monkeypatch.setattr(sampled, "run_fista_iteration_case", fake_fista)
    monkeypatch.setattr(sampled, "run_alignment_objective_suite", fake_alignment_objective)

    metrics = sampled.run_sampled_representative_suite(suite_seed=1234, cases_per_family=2)

    assert metrics["suite_seed"] == 1234
    assert metrics["sampler_version"] == 2
    assert len(metrics["sampled_cases"]) == 8
    assert {case["family"] for case in metrics["sampled_cases"]} == {
        "alignment_objective",
        "fista_iteration",
        "forward_residual",
        "general_pose_forward",
    }
    assert metrics["summary"]["general_forward_dispatch_geomean_speedup_vs_best_jax"] == (
        pytest.approx(3.0)
    )
    assert metrics["summary"]["forward_residual_dispatch_geomean_speedup_vs_jax"] == (
        pytest.approx(4.0)
    )
    assert metrics["summary"]["fista_geomean_speedup_vs_jax"] == pytest.approx(5.0)
    assert metrics["summary"]["alignment_smoke_total_cases"] == 0


def test_sampled_suite_records_alignment_smoke_success_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sampled,
        "run_forward_sinogram_benchmark",
        lambda config: {
            "results": [
                {
                    "requested_mode": "pallas_dispatch",
                    "speedup_vs_best_jax_warm_median": 3.0,
                }
            ]
        },
    )
    monkeypatch.setattr(
        sampled,
        "run_forward_residual_benchmark",
        lambda config: {
            "results": [
                {
                    "requested_mode": "pallas_dispatch",
                    "speedup_vs_jax_materialized_warm_median": 4.0,
                }
            ]
        },
    )
    monkeypatch.setattr(
        sampled,
        "run_fista_iteration_case",
        lambda config: {"speedup_vs_jax_warm_median": 5.0},
    )
    monkeypatch.setattr(
        sampled,
        "run_alignment_objective_suite",
        lambda name, *, overrides=None: {"summary": {"overrides": overrides}},
    )

    def fake_alignment_smoke_case(**kwargs):
        assert kwargs["note"] == "sampled note"
        assert kwargs["git_branch"] == "bench/test"
        assert kwargs["git_commit"] == "abc123"
        return {
            "case_name": kwargs["case_name"],
            "timing": {"wall_sec": 1.5},
            "success": {"loss_decreased": True, "pose_metrics_finite": True},
            "pose_recovery": {"rot_rmse_deg": 1.0, "trans_rmse_px": 0.25},
            "projection_residual": {"rmse_per_ray": 0.1},
        }

    monkeypatch.setattr(sampled, "run_sampled_alignment_smoke_case", fake_alignment_smoke_case)

    metrics = sampled.run_sampled_representative_suite(
        suite_seed=1234,
        cases_per_family=1,
        tomojax_dir=tmp_path / "tomojax",
        fixture_root=tmp_path / "fixtures",
        out_dir=tmp_path / "out",
        note="sampled note",
        git_branch="bench/test",
        git_commit="abc123",
    )

    assert len(metrics["sampled_cases"]) == 5
    assert "alignment_smoke" in {case["family"] for case in metrics["sampled_cases"]}
    assert metrics["summary"]["alignment_smoke_median_wall_sec"] == pytest.approx(1.5)
    assert metrics["summary"]["alignment_smoke_successful_cases"] == 1
    assert metrics["summary"]["alignment_smoke_total_cases"] == 1
    assert metrics["alignment_smoke_cases"][0]["success"]["pose_metrics_finite"] is True


def test_write_sampled_json(tmp_path) -> None:
    out = sampled.write_benchmark_json({"suite_seed": 1}, tmp_path / "sampled.json")

    assert json.loads(out.read_text()) == {"suite_seed": 1}
