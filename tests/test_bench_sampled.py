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
    assert metrics["sampler_version"] == 1
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


def test_write_sampled_json(tmp_path) -> None:
    out = sampled.write_benchmark_json({"suite_seed": 1}, tmp_path / "sampled.json")

    assert json.loads(out.read_text()) == {"suite_seed": 1}

