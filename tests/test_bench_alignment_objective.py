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
    assert metrics["warm_runs"] == 1
    assert metrics["warm_seconds_median"] is not None
    assert metrics["value_finite"] is True
    assert metrics["grad_finite"] is True
    assert metrics["grad_shape"] == [2, 5]
    assert metrics["grad_norm"] > 0.0


def test_alignment_objective_suite_compares_checkpoint_modes_and_batches_all_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, int]] = []

    def fake_benchmark(name, config):
        calls.append((bool(config.checkpoint_projector), int(config.views_per_batch)))
        return {
            "benchmark": "alignment_objective_value_and_grad",
            "case_name": name,
            "warm_seconds_median": 2.0 if config.checkpoint_projector else 1.0,
        }

    monkeypatch.setattr(
        "tomojax.bench.alignment_objective.benchmark_alignment_objective_variant",
        fake_benchmark,
    )
    monkeypatch.setattr("tomojax.bench.alignment_objective._device_metadata", lambda: {})

    metrics = run_alignment_objective_suite(overrides={"warm_runs": 3})

    assert metrics["benchmark"] == "alignment_objective_suite"
    assert metrics["suite"] == "alignment_objective"
    assert calls == [(True, 0), (False, 0)]
    assert [case["case_name"] for case in metrics["cases"]] == [
        "checkpointed",
        "no_checkpoint",
    ]
    assert metrics["summary"]["no_checkpoint_speedup_vs_checkpointed"] == pytest.approx(2.0)


def test_alignment_objective_suite_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="alignment objective suite"):
        run_alignment_objective_suite("unknown")
