from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from tomojax.bench.forward_residual import (
    ForwardResidualBenchmarkConfig,
    PALLAS_DISPATCH_RAY_STEP_THRESHOLD,
    RESIDUAL_SUITE_NAMES,
    _geomean,
    benchmark_residual_mode,
    make_forward_residual_fixture,
    residual_dispatch_estimated_ray_steps,
    residual_dispatch_pallas_tile_shape,
    residual_dispatch_selected_mode,
    residual_suite_cases,
    run_forward_residual_benchmark,
    run_forward_residual_suite,
    write_benchmark_json,
)


def test_residual_suite_cases_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="residual suite must be one of"):
        residual_suite_cases("unknown")


def test_residual_suite_cases_returns_fixed_workloads() -> None:
    cases = residual_suite_cases("residual")

    assert [case.name for case in cases] == [
        "residual-64",
        "residual-128",
        "high-ray-residual-128",
    ]
    assert all(case.config.n_views > 1 for case in cases)
    assert any(case.config.nu * case.config.nv > 60_000 for case in cases)


def test_general_pose_residual_suite_uses_general_5d_poses() -> None:
    cases = residual_suite_cases("general_pose")

    assert [case.name for case in cases] == [
        "general-pose-residual-24",
        "general-pose-residual-64",
    ]
    assert all(case.config.pose_mode == "general_5d" for case in cases)
    assert all(case.config.pallas_state_mode == "cached" for case in cases)


def test_forward_residual_benchmark_reports_jax_and_pallas_fallback() -> None:
    config = ForwardResidualBenchmarkConfig(
        nx=4,
        ny=4,
        nz=4,
        nu=4,
        nv=4,
        n_views=2,
        warm_runs=1,
        include_pallas=True,
    )

    metrics = run_forward_residual_benchmark(config)

    assert metrics["benchmark"] == "forward_residual"
    assert metrics["fixture_backend"] == "jax_materialized"
    assert metrics["config"]["n_views"] == 2
    assert [row["requested_mode"] for row in metrics["results"]] == [
        "jax_materialized",
        "pallas_materialized",
        "pallas_fused",
        "pallas_dispatch",
    ]
    jax_row, pallas_materialized_row, pallas_fused_row, dispatch_row = metrics["results"]
    assert jax_row["actual_mode"] == "jax_materialized"
    assert jax_row["eligible_for_speed_claim"] is True
    assert jax_row["finite"] is True
    assert pallas_materialized_row["actual_mode"] in {"jax_materialized", "pallas_materialized"}
    assert pallas_fused_row["actual_mode"] in {"jax_materialized", "pallas_fused"}
    assert dispatch_row["actual_mode"] == "pallas_dispatch"
    assert dispatch_row["dispatch_selected_mode"] == "jax_materialized"
    assert dispatch_row["dispatch_estimated_ray_steps"] < PALLAS_DISPATCH_RAY_STEP_THRESHOLD
    assert dispatch_row["dispatch_timing_source"] == "jax_materialized_baseline"
    assert dispatch_row["speedup_vs_jax_materialized_warm_median"] == pytest.approx(1.0)
    if pallas_materialized_row["eligible_for_speed_claim"]:
        assert pallas_materialized_row["abs_error"] == pytest.approx(0.0, abs=1e-3)
    else:
        assert pallas_materialized_row["speedup_vs_jax_materialized_warm_median"] is None
    assert pallas_fused_row["abs_error"] == pytest.approx(0.0, abs=1e-3)
    assert dispatch_row["abs_error"] == pytest.approx(0.0)


def test_residual_mode_records_pallas_variant_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ForwardResidualBenchmarkConfig(
        nx=2,
        ny=2,
        nz=2,
        nu=2,
        nv=2,
        n_views=2,
        warm_runs=1,
        pallas_tile_shape=(4, 4),
        pallas_num_warps=2,
    )
    fixture = make_forward_residual_fixture(config)

    def fake_make_callable(requested_mode, _fixture, _config):
        return lambda: jnp.asarray(1.0, dtype=jnp.float32), requested_mode, None

    monkeypatch.setattr(
        "tomojax.bench.forward_residual._make_residual_callable",
        fake_make_callable,
    )

    result, _ = benchmark_residual_mode(
        "pallas_fused",
        fixture,
        config,
        oracle=jnp.asarray(1.0, dtype=jnp.float32),
        jax_median=2.0,
    )

    assert result["requested_pallas_variant"]["tile_shape"] == [4, 4]
    assert result["actual_pallas_variant"]["tile_shape"] == [2, 2]
    assert result["actual_pallas_variant"]["num_warps"] == 2
    assert result["actual_pallas_variant"]["effective_pallas_n_steps"] == 5
    assert result["speedup_vs_jax_materialized_warm_median"] is not None


def test_residual_dispatch_uses_pallas_for_material_general_workloads() -> None:
    tiny = ForwardResidualBenchmarkConfig(nx=4, ny=4, nz=4, nu=4, nv=4, n_views=2)
    general_24 = ForwardResidualBenchmarkConfig(
        nx=24,
        ny=24,
        nz=24,
        nu=24,
        nv=24,
        n_views=24,
        pose_mode="general_5d",
    )
    general_64 = ForwardResidualBenchmarkConfig(nx=64, ny=64, nz=64, nu=64, nv=64, n_views=90)
    high = ForwardResidualBenchmarkConfig(
        nx=128,
        ny=128,
        nz=128,
        nu=256,
        nv=256,
        n_views=90,
        step_size=0.5,
    )

    assert residual_dispatch_selected_mode(tiny) == "jax_materialized"
    assert residual_dispatch_selected_mode(general_24) == "pallas_materialized"
    assert residual_dispatch_selected_mode(general_64) == "pallas_materialized"
    assert residual_dispatch_selected_mode(high) == "pallas_materialized"
    assert residual_dispatch_estimated_ray_steps(tiny) < PALLAS_DISPATCH_RAY_STEP_THRESHOLD
    assert residual_dispatch_estimated_ray_steps(general_24) >= PALLAS_DISPATCH_RAY_STEP_THRESHOLD
    assert residual_dispatch_estimated_ray_steps(general_64) >= PALLAS_DISPATCH_RAY_STEP_THRESHOLD
    assert residual_dispatch_estimated_ray_steps(high) >= PALLAS_DISPATCH_RAY_STEP_THRESHOLD


def test_residual_dispatch_uses_general_pose_tile_policy() -> None:
    general = ForwardResidualBenchmarkConfig(
        nx=24,
        ny=24,
        nz=24,
        nu=24,
        nv=24,
        n_views=24,
        pose_mode="general_5d",
        pallas_tile_shape=(16, 8),
    )
    awkward = ForwardResidualBenchmarkConfig(
        nx=40,
        ny=32,
        nz=48,
        nu=37,
        nv=53,
        n_views=41,
        pose_mode="general_5d",
        pallas_tile_shape=(16, 8),
    )
    tiny = ForwardResidualBenchmarkConfig(
        nx=4,
        ny=4,
        nz=4,
        nu=4,
        nv=4,
        n_views=2,
        pallas_tile_shape=(16, 8),
    )

    assert residual_dispatch_selected_mode(general) == "pallas_materialized"
    assert residual_dispatch_selected_mode(awkward) == "pallas_materialized"
    assert residual_dispatch_pallas_tile_shape(general) == (16, 4)
    assert residual_dispatch_pallas_tile_shape(awkward) == (16, 4)
    assert residual_dispatch_pallas_tile_shape(tiny) == (16, 8)


def test_residual_dispatch_reports_effective_pallas_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ForwardResidualBenchmarkConfig(
        nx=24,
        ny=24,
        nz=24,
        nu=24,
        nv=24,
        n_views=24,
        pose_mode="general_5d",
        warm_runs=1,
        pallas_tile_shape=(16, 8),
    )
    fixture = make_forward_residual_fixture(config)

    def fake_make_callable(requested_mode, _fixture, _config):
        return lambda: jnp.asarray(1.0, dtype=jnp.float32), requested_mode, None

    monkeypatch.setattr(
        "tomojax.bench.forward_residual._make_residual_callable",
        fake_make_callable,
    )

    result, _ = benchmark_residual_mode(
        "pallas_dispatch",
        fixture,
        config,
        oracle=jnp.asarray(1.0, dtype=jnp.float32),
        jax_median=2.0,
    )

    assert result["requested_pallas_variant"]["tile_shape"] == [16, 8]
    assert result["dispatch_pallas_variant"]["tile_shape"] == [16, 4]
    assert result["actual_pallas_variant"]["tile_shape"] == [16, 4]


def test_forward_residual_suite_reports_cases_and_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, bool]] = []

    def fake_run(config: ForwardResidualBenchmarkConfig) -> dict:
        calls.append((config.n_views, config.warm_runs, config.include_pallas))
        return {
            "benchmark": "forward_residual",
            "fixture_backend": "jax_materialized",
            "config": {"warm_runs": config.warm_runs, "n_views": config.n_views},
            "fixture": {"total_ray_steps": 1},
            "device": {},
            "results": [
                {
                    "requested_mode": "jax_materialized",
                    "actual_mode": "jax_materialized",
                    "eligible_for_speed_claim": True,
                    "warm_seconds_median": 2.0,
                    "finite": True,
                    "abs_error": 0.0,
                    "relative_error": 0.0,
                },
                {
                    "requested_mode": "pallas_materialized",
                    "actual_mode": "pallas_materialized",
                    "eligible_for_speed_claim": True,
                    "warm_seconds_median": 1.0,
                    "speedup_vs_jax_materialized_warm_median": 2.0,
                    "finite": True,
                    "abs_error": 0.0,
                    "relative_error": 0.0,
                },
                {
                    "requested_mode": "pallas_fused",
                    "actual_mode": "pallas_fused",
                    "eligible_for_speed_claim": True,
                    "warm_seconds_median": 1.25,
                    "speedup_vs_jax_materialized_warm_median": 1.6,
                    "finite": True,
                    "abs_error": 0.0,
                    "relative_error": 0.0,
                },
                {
                    "requested_mode": "pallas_dispatch",
                    "actual_mode": "pallas_dispatch",
                    "eligible_for_speed_claim": True,
                    "warm_seconds_median": 1.0,
                    "speedup_vs_jax_materialized_warm_median": 2.0,
                    "finite": True,
                    "abs_error": 0.0,
                    "relative_error": 0.0,
                },
            ],
        }

    monkeypatch.setattr("tomojax.bench.forward_residual.run_forward_residual_benchmark", fake_run)
    monkeypatch.setattr("tomojax.bench.forward_residual._device_metadata", lambda: {"test": True})

    metrics = run_forward_residual_suite("residual", overrides={"warm_runs": 2})

    assert metrics["benchmark"] == "forward_residual_suite"
    assert metrics["suite"] == "residual"
    assert [case["case_name"] for case in metrics["cases"]] == [
        "residual-64",
        "residual-128",
        "high-ray-residual-128",
    ]
    assert calls == [(90, 2, True), (180, 2, True), (90, 2, True)]
    assert metrics["summary"] == {
        "cases_total": 3,
        "pallas_modes": {
            "pallas_materialized": {
                "cases_with_requested_pallas": 3,
                "cases_pallas_eligible": 3,
                "cases_parity_passed": 3,
                "geomean_speedup_vs_jax_materialized_warm_median": pytest.approx(2.0),
                "worst_case_speedup_vs_jax_materialized_warm_median": 2.0,
                "best_case_speedup_vs_jax_materialized_warm_median": 2.0,
            },
            "pallas_fused": {
                "cases_with_requested_pallas": 3,
                "cases_pallas_eligible": 3,
                "cases_parity_passed": 3,
                "geomean_speedup_vs_jax_materialized_warm_median": pytest.approx(1.6),
                "worst_case_speedup_vs_jax_materialized_warm_median": 1.6,
                "best_case_speedup_vs_jax_materialized_warm_median": 1.6,
            },
            "pallas_dispatch": {
                "cases_with_requested_pallas": 3,
                "cases_pallas_eligible": 3,
                "cases_parity_passed": 3,
                "geomean_speedup_vs_jax_materialized_warm_median": pytest.approx(2.0),
                "worst_case_speedup_vs_jax_materialized_warm_median": 2.0,
                "best_case_speedup_vs_jax_materialized_warm_median": 2.0,
            },
        },
        "primary_pallas_mode": "pallas_dispatch",
        "geomean_speedup_vs_jax_materialized_warm_median": pytest.approx(2.0),
        "worst_case_speedup_vs_jax_materialized_warm_median": 2.0,
        "best_case_speedup_vs_jax_materialized_warm_median": 2.0,
    }


def test_public_residual_suite_names() -> None:
    assert RESIDUAL_SUITE_NAMES == ("residual", "general_pose")


def test_geomean_alias_available() -> None:
    assert _geomean([2.0, 2.0]) == pytest.approx(2.0)


def test_write_residual_benchmark_json(tmp_path: Path) -> None:
    out = write_benchmark_json({"b": 1, "a": {"c": 2}}, tmp_path / "metrics.json")

    assert out.exists()
    assert json.loads(out.read_text()) == {"a": {"c": 2}, "b": 1}
