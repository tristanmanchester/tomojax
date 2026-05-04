from __future__ import annotations

import json

import pytest

from tomojax.bench.fista_iteration import (
    FISTA_ITERATION_SUITE_NAMES,
    FistaIterationBenchmarkConfig,
    fista_iteration_suite_cases,
    run_fista_iteration_suite,
    write_benchmark_json,
)


def test_fista_iteration_suite_cases_are_general_pose() -> None:
    cases = fista_iteration_suite_cases()

    assert [case.name for case in cases] == ["fista-iter-24", "fista-iter-64"]
    assert all(case.config.pose_mode == "general_5d" for case in cases)
    assert all(case.config.views_per_batch == 0 for case in cases)
    assert all(case.config.unroll == 4 for case in cases)
    assert all(case.config.forward_projector == "pallas" for case in cases)
    assert all(case.config.backprojector == "pallas" for case in cases)
    assert all(case.config.pallas_tile_shape == (16, 4) for case in cases)
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
            "benchmark": "fista_iteration",
            "api_surface": "internal_fista_tv_core_arrays",
            "warm_seconds_median": float(config.n_views),
            "quality": {"finite": True, "repeat_rel_l2_vs_first": 0.0},
        }

    monkeypatch.setattr("tomojax.bench.fista_iteration.run_fista_iteration_benchmark", fake_run)

    metrics = run_fista_iteration_suite(overrides={"warm_runs": 2})

    assert metrics["benchmark"] == "fista_iteration_suite"
    assert metrics["suite"] == "fista_iteration"
    assert [case["case_name"] for case in metrics["cases"]] == ["fista-iter-24", "fista-iter-64"]
    assert calls == [24, 90]
    assert metrics["summary"]["cases_total"] == 2


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
            "benchmark": "fista_iteration",
            "api_surface": "internal_fista_tv_core_arrays",
            "warm_seconds_median": 0.0,
            "quality": {"finite": True, "repeat_rel_l2_vs_first": 0.0},
        }

    monkeypatch.setattr("tomojax.bench.fista_iteration.run_fista_iteration_benchmark", fake_run)

    run_fista_iteration_suite(
        overrides={
            "compute_iteration_loss": True,
            "compute_final_data_loss": True,
            "compute_final_regulariser_value": True,
        }
    )

    assert seen == [(True, True, True), (True, True, True)]


def test_fista_iteration_public_suite_names() -> None:
    assert FISTA_ITERATION_SUITE_NAMES == ("fista_iteration",)


def test_write_fista_iteration_json(tmp_path) -> None:
    out = write_benchmark_json({"z": 1}, tmp_path / "fista.json")

    assert json.loads(out.read_text()) == {"z": 1}
