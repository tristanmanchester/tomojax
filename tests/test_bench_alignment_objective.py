from __future__ import annotations

import pytest

from tomojax.bench.alignment_objective import (
    AlignmentObjectiveBenchmarkConfig,
    benchmark_alignment_objective_variant,
    run_alignment_objective_suite,
)


def test_alignment_objective_variant_reports_value_and_grad_metrics() -> None:
    metrics = benchmark_alignment_objective_variant(
        "tiny",
        AlignmentObjectiveBenchmarkConfig(
            nx=4,
            ny=4,
            nz=4,
            nu=4,
            nv=4,
            n_views=2,
            warm_runs=1,
            gather_dtype="fp32",
        ),
    )

    assert metrics["benchmark"] == "alignment_objective_value_and_grad"
    assert metrics["case_name"] == "tiny"
    assert metrics["api_surface"] == "internal_fixed_volume_alignment_objective"
    assert metrics["implementation"] == "stacked"
    assert metrics["warm_runs"] == 1
    assert metrics["warm_seconds_median"] is not None
    assert metrics["value_finite"] is True
    assert metrics["grad_finite"] is True
    assert metrics["grad_shape"] == [2, 5]
    assert metrics["grad_norm"] > 0.0


def test_alignment_objective_manual_per_view_matches_stacked_value() -> None:
    config = AlignmentObjectiveBenchmarkConfig(
        nx=4,
        ny=4,
        nz=4,
        nu=4,
        nv=4,
        n_views=2,
        warm_runs=1,
        gather_dtype="fp32",
    )

    stacked = benchmark_alignment_objective_variant("stacked", config)
    manual = benchmark_alignment_objective_variant(
        "manual",
        config,
        implementation="manual_per_view",
    )

    assert manual["implementation"] == "manual_per_view"
    assert manual["value"] == pytest.approx(stacked["value"], rel=1e-5, abs=1e-5)
    assert manual["grad_shape"] == stacked["grad_shape"]
    assert manual["grad_finite"] is True


def test_alignment_objective_suite_compares_checkpoint_modes_and_batches_all_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, int, str]] = []

    def fake_benchmark(name, config, *, implementation="stacked"):
        calls.append(
            (bool(config.checkpoint_projector), int(config.views_per_batch), implementation)
        )
        return {
            "benchmark": "alignment_objective_value_and_grad",
            "case_name": name,
            "warm_seconds_median": (
                0.5
                if implementation == "manual_per_view"
                else (2.0 if config.checkpoint_projector else 1.0)
            ),
        }

    monkeypatch.setattr(
        "tomojax.bench.alignment_objective.benchmark_alignment_objective_variant",
        fake_benchmark,
    )
    monkeypatch.setattr("tomojax.bench.alignment_objective._device_metadata", dict)

    metrics = run_alignment_objective_suite(overrides={"warm_runs": 3})

    assert metrics["benchmark"] == "alignment_objective_suite"
    assert metrics["suite"] == "alignment_objective"
    assert calls == [
        (True, 0, "stacked"),
        (False, 0, "stacked"),
        (True, 0, "manual_per_view"),
    ]
    assert [case["case_name"] for case in metrics["cases"]] == [
        "checkpointed",
        "no_checkpoint",
        "manual_per_view",
    ]
    assert metrics["summary"]["no_checkpoint_speedup_vs_checkpointed"] == pytest.approx(2.0)
    assert metrics["summary"]["manual_per_view_speedup_vs_checkpointed"] == pytest.approx(4.0)


def test_alignment_objective_suite_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="alignment objective suite"):
        run_alignment_objective_suite("unknown")
