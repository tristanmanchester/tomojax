from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from tomojax.bench.forward_projector import (
    ForwardProjectorBenchmarkConfig,
    PRESET_NAMES,
    SUITE_NAMES,
    _geomean,
    _block_tree_ready,
    _time_blocked_call,
    benchmark_backend,
    make_forward_projector_fixture,
    preset_config,
    run_forward_projector_benchmark,
    run_forward_projector_suite,
    suite_cases,
    write_benchmark_json,
)


def test_preset_config_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="preset must be one of"):
        preset_config("unknown")


def test_suite_cases_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="suite must be one of"):
        suite_cases("unknown")


@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_preset_config_returns_named_workloads(preset_name: str) -> None:
    config = preset_config(preset_name)

    assert config.nx > 0
    assert config.ny > 0
    assert config.nz > 0
    assert config.nu > 0
    assert config.nv > 0
    if preset_name == "high-ray-count-128":
        assert config.nu * config.nv > config.nx * config.nz
    if preset_name == "high-ray-count-192":
        assert config.nu * config.nv > config.nx * config.nz
        assert config.step_size == pytest.approx(0.5)
    if preset_name == "noncubic-align-128":
        assert config.nz != config.nx
    if preset_name == "thin-noncubic-192":
        assert config.nz != config.nx
    if preset_name == "fine-step-128":
        assert config.step_size == pytest.approx(0.25)


@pytest.mark.parametrize("suite_name", SUITE_NAMES)
def test_suite_cases_returns_named_workloads(suite_name: str) -> None:
    cases = suite_cases(suite_name)

    assert cases
    assert all(case.name for case in cases)
    if suite_name == "quick":
        assert [case.name for case in cases] == ["high-ray-count-128"]
    if suite_name == "confirm":
        assert [case.name for case in cases] == [
            "profile-128",
            "noncubic-align-128",
            "high-ray-count-128",
        ]
        assert all(case.config.warm_runs == 25 for case in cases)
    if suite_name == "stress":
        assert [case.name for case in cases] == [
            "large-cubic-192",
            "thin-noncubic-192",
            "fine-step-128",
            "high-ray-count-192",
        ]
        assert all(case.config.warm_runs == 15 for case in cases)
        assert any(case.config.step_size == 0.25 for case in cases)
        assert any(case.config.nu * case.config.nv > 100_000 for case in cases)


def test_forward_projector_benchmark_reports_jax_and_pallas_fallback() -> None:
    config = ForwardProjectorBenchmarkConfig(
        nx=4,
        ny=4,
        nz=4,
        nu=4,
        nv=4,
        warm_runs=1,
        include_pallas=True,
    )

    metrics = run_forward_projector_benchmark(config)

    assert metrics["benchmark"] == "forward_projector"
    assert metrics["fixture_backend"] == "jax"
    assert metrics["config"]["nx"] == 4
    assert metrics["fixture"]["volume_shape"] == [4, 4, 4]
    assert metrics["fixture"]["detector_shape"] == [4, 4]
    assert metrics["fixture"]["n_rays"] == 16
    assert metrics["fixture"]["resolved_n_steps"] > 0
    assert metrics["fixture"]["total_ray_steps"] > 16
    assert [row["requested_backend"] for row in metrics["results"]] == ["jax", "pallas"]
    jax_row, pallas_row = metrics["results"]
    assert jax_row["actual_backend"] == "jax"
    assert jax_row["eligible_for_speed_claim"] is True
    assert jax_row["finite"] is True
    assert jax_row["warm_runs"] == 1
    assert pallas_row["actual_backend"] in {"jax", "pallas"}
    if pallas_row["actual_backend"] == "jax":
        assert pallas_row["eligible_for_speed_claim"] is False
        assert pallas_row["fallback_reason"]
        assert pallas_row["speedup_vs_jax_warm_median"] is None
    assert pallas_row["max_abs_error"] == pytest.approx(0.0)


def test_forward_projector_suite_reports_cases_and_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []

    def fake_run(config: ForwardProjectorBenchmarkConfig) -> dict:
        calls.append((config.warm_runs, config.include_pallas))
        return {
            "benchmark": "forward_projector",
            "fixture_backend": "jax",
            "config": {"warm_runs": config.warm_runs},
            "fixture": {"total_ray_steps": 1},
            "device": {},
            "results": [
                {
                    "requested_backend": "jax",
                    "actual_backend": "jax",
                    "eligible_for_speed_claim": True,
                    "warm_seconds_median": 2.0,
                    "finite": True,
                    "max_abs_error": 0.0,
                    "max_relative_error": 0.0,
                },
                {
                    "requested_backend": "pallas",
                    "actual_backend": "pallas",
                    "eligible_for_speed_claim": True,
                    "warm_seconds_median": 1.0,
                    "speedup_vs_jax_warm_median": 2.0,
                    "finite": True,
                    "max_abs_error": 0.0,
                    "max_relative_error": 0.0,
                },
            ],
        }

    monkeypatch.setattr("tomojax.bench.forward_projector.run_forward_projector_benchmark", fake_run)
    monkeypatch.setattr("tomojax.bench.forward_projector._device_metadata", lambda: {"test": True})

    metrics = run_forward_projector_suite("confirm", overrides={"warm_runs": 3})

    assert metrics["benchmark"] == "forward_projector_suite"
    assert metrics["suite"] == "confirm"
    assert [case["case_name"] for case in metrics["cases"]] == [
        "profile-128",
        "noncubic-align-128",
        "high-ray-count-128",
    ]
    assert calls == [(3, True), (3, True), (3, True)]
    assert metrics["summary"] == {
        "cases_total": 3,
        "cases_with_requested_pallas": 3,
        "cases_pallas_eligible": 3,
        "cases_parity_passed": 3,
        "geomean_speedup_vs_jax_warm_median": pytest.approx(2.0),
        "worst_case_speedup_vs_jax_warm_median": 2.0,
        "best_case_speedup_vs_jax_warm_median": 2.0,
    }


def test_geomean_returns_none_for_empty_or_invalid_values() -> None:
    assert _geomean([]) is None
    assert _geomean([1.0, 0.0]) is None


def test_benchmark_backend_blocks_each_timed_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = make_forward_projector_fixture(
        ForwardProjectorBenchmarkConfig(nx=2, ny=2, nz=2, nu=2, nv=2, warm_runs=2)
    )
    calls: list[str] = []

    def fake_call_jax(_fixture, _config):
        calls.append("call")
        return jnp.ones((2, 2), dtype=jnp.float32)

    def fake_block(value):
        calls.append("block")
        return value

    monkeypatch.setattr("tomojax.bench.forward_projector._call_jax", fake_call_jax)
    monkeypatch.setattr("tomojax.bench.forward_projector._block_tree_ready", fake_block)

    result, output = benchmark_backend("jax", fixture, fixture_config(warm_runs=2), oracle=None)

    assert output.shape == (2, 2)
    assert result["warm_runs"] == 2
    assert calls == ["call", "block", "call", "block", "call", "block"]


def test_time_blocked_call_waits_for_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_block(value):
        calls.append(f"block:{value}")
        return value

    monkeypatch.setattr("tomojax.bench.forward_projector._block_tree_ready", fake_block)

    seconds, output = _time_blocked_call(lambda: "result")

    assert seconds >= 0.0
    assert output == "result"
    assert calls == ["block:result"]


def test_write_benchmark_json(tmp_path: Path) -> None:
    out = write_benchmark_json({"b": 1, "a": {"c": 2}}, tmp_path / "metrics.json")

    assert out.exists()
    assert json.loads(out.read_text()) == {"a": {"c": 2}, "b": 1}


def fixture_config(*, warm_runs: int) -> ForwardProjectorBenchmarkConfig:
    return ForwardProjectorBenchmarkConfig(
        nx=2,
        ny=2,
        nz=2,
        nu=2,
        nv=2,
        warm_runs=warm_runs,
        include_pallas=False,
    )
