from __future__ import annotations

import json

import pytest

from tomojax.bench.fista_iteration import (
    FISTA_ITERATION_SUITE_NAMES,
    FistaIterationBenchmarkConfig,
    fista_iteration_suite_cases,
    run_fista_iteration_benchmark,
    run_fista_iteration_case,
    run_fista_iteration_suite,
    write_benchmark_json,
)


def test_fista_iteration_suite_cases_are_general_pose() -> None:
    cases = fista_iteration_suite_cases()

    assert [case.name for case in cases] == ["fista-iter-24", "fista-iter-64"]
    assert all(case.config.pose_mode == "general_5d" for case in cases)
    assert all(case.config.views_per_batch == 0 for case in cases)
    assert all(case.config.unroll is None for case in cases)
    assert all(case.config.forward_projector == "pallas" for case in cases)
    assert all(case.config.backprojector == "pallas" for case in cases)
    assert [case.config.pallas_tile_shape for case in cases] == [(12, 4), (8, 4)]
    assert all(not case.config.compute_iteration_loss for case in cases)
    assert all(not case.config.compute_final_data_loss for case in cases)
    assert all(not case.config.compute_final_regulariser_value for case in cases)


def test_fista_iteration_suite_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="fista iteration suite must be one of"):
        fista_iteration_suite_cases("unknown")


def test_fista_iteration_suite_reports_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_run(config: FistaIterationBenchmarkConfig) -> dict:
        calls.append(config.n_views)
        return {
            "benchmark": "fista_iteration_comparison",
            "api_surface": "internal_fista_tv_core_arrays",
            "warm_seconds_median": float(config.n_views),
            "speedup_vs_jax_warm_median": 2.0,
            "quality": {"finite": True, "repeat_rel_l2_vs_first": 0.0},
        }

    monkeypatch.setattr("tomojax.bench.fista_iteration.run_fista_iteration_case", fake_run)

    metrics = run_fista_iteration_suite(overrides={"warm_runs": 2})

    assert metrics["benchmark"] == "fista_iteration_suite"
    assert metrics["suite"] == "fista_iteration"
    assert [case["case_name"] for case in metrics["cases"]] == ["fista-iter-24", "fista-iter-64"]
    assert calls == [24, 90]
    assert metrics["summary"]["cases_total"] == 2
    assert metrics["summary"]["worst_case_speedup_vs_jax_warm_median"] == pytest.approx(2.0)


def test_fista_iteration_overrides_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[bool, bool, bool]] = []

    def fake_run(config: FistaIterationBenchmarkConfig) -> dict:
        seen.append(
            (
                config.compute_iteration_loss,
                config.compute_final_data_loss,
                config.compute_final_regulariser_value,
            )
        )
        return {
            "benchmark": "fista_iteration_comparison",
            "api_surface": "internal_fista_tv_core_arrays",
            "warm_seconds_median": 0.0,
            "speedup_vs_jax_warm_median": None,
            "quality": {"finite": True, "repeat_rel_l2_vs_first": 0.0},
        }

    monkeypatch.setattr("tomojax.bench.fista_iteration.run_fista_iteration_case", fake_run)

    run_fista_iteration_suite(
        overrides={
            "compute_iteration_loss": True,
            "compute_final_data_loss": True,
            "compute_final_regulariser_value": True,
        }
    )

    assert seen == [(True, True, True), (True, True, True)]


def test_fista_iteration_case_compares_jax_and_pallas(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_run(config: FistaIterationBenchmarkConfig) -> dict:
        seen.append((config.forward_projector, config.backprojector))
        warm = 4.0 if config.forward_projector == "jax" else 1.0
        return {
            "benchmark": "fista_iteration",
            "api_surface": "internal_fista_tv_core_arrays",
            "warm_seconds_median": warm,
            "quality": {"finite": True, "repeat_rel_l2_vs_first": 0.0},
        }

    monkeypatch.setattr("tomojax.bench.fista_iteration.run_fista_iteration_benchmark", fake_run)
    monkeypatch.setattr(
        "tomojax.bench.fista_iteration._make_fista_call",
        lambda config: (lambda: (0, (), (), (), ()), {}),
    )
    monkeypatch.setattr("tomojax.bench.fista_iteration._time_blocked_call", lambda fn: (0.0, fn()))

    metrics = run_fista_iteration_case(FistaIterationBenchmarkConfig(warm_runs=1))

    assert seen == [("jax", "jax"), ("pallas", "pallas")]
    assert metrics["speedup_vs_jax_warm_median"] == pytest.approx(4.0)
    assert metrics["baseline_mode"] == "jax"
    assert metrics["candidate_mode"] == "pallas"


def test_fista_iteration_quality_marks_data_loss_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_make_call(config: FistaIterationBenchmarkConfig):
        del config
        return (
            lambda: (
                0,
                [0.0],
                0.0,
                0.0,
                1,
            ),
            {},
        )

    monkeypatch.setattr("tomojax.bench.fista_iteration._make_fista_call", fake_make_call)
    monkeypatch.setattr("tomojax.bench.fista_iteration._time_blocked_call", lambda fn: (0.0, fn()))

    metrics = run_fista_iteration_benchmark(
        FistaIterationBenchmarkConfig(
            warm_runs=1,
            compute_iteration_loss=False,
            compute_final_data_loss=False,
        )
    )

    assert metrics["quality"]["data_loss_computed"] is False
    assert metrics["quality"]["data_loss_is_final"] is False
    assert metrics["quality"]["data_loss_is_last_gradient_point"] is False


def test_fista_iteration_public_suite_names() -> None:
    assert FISTA_ITERATION_SUITE_NAMES == ("fista_iteration",)


def test_write_fista_iteration_json(tmp_path) -> None:
    out = write_benchmark_json({"z": 1}, tmp_path / "fista.json")

    assert json.loads(out.read_text()) == {"z": 1}
