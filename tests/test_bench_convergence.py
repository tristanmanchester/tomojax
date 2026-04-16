from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("psutil")

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

fitness = importlib.import_module("fitness")


def _summary(
    *,
    metric: str = "gt_mse",
    threshold: float | None = 1.0,
    threshold_met: bool = True,
    stopped_on_threshold: bool = False,
    stopped_on_plateau: bool = False,
    stopped_on_budget: bool = False,
    seconds_to_threshold: float | None = None,
    outer_iters_to_threshold: int | None = None,
    best_quality_value: float | None = None,
    best_quality_elapsed_seconds: float | None = None,
    total_outer_iters_executed: int = 1,
    final_stop_reason: str | None = None,
    final_stop_level_factor: int | None = 1,
    first_threshold_crossing_level_factor: int | None = None,
    reached_finest_level: bool = True,
    finest_level_first_elapsed_seconds: float | None = None,
    finest_level_first_outer_idx: int | None = None,
    level_summaries: list[dict] | None = None,
    trace: list[dict] | None = None,
) -> object:
    return fitness.ConvergenceRunSummary(
        metric=metric,
        threshold=threshold,
        threshold_met=threshold_met,
        stopped_on_threshold=stopped_on_threshold,
        stopped_on_plateau=stopped_on_plateau,
        stopped_on_budget=stopped_on_budget,
        seconds_to_threshold=seconds_to_threshold,
        outer_iters_to_threshold=outer_iters_to_threshold,
        best_quality_value=best_quality_value,
        best_quality_elapsed_seconds=best_quality_elapsed_seconds,
        total_outer_iters_executed=total_outer_iters_executed,
        final_stop_reason=final_stop_reason,
        final_stop_level_factor=final_stop_level_factor,
        first_threshold_crossing_level_factor=first_threshold_crossing_level_factor,
        reached_finest_level=reached_finest_level,
        finest_level_first_elapsed_seconds=finest_level_first_elapsed_seconds,
        finest_level_first_outer_idx=finest_level_first_outer_idx,
        level_summaries=level_summaries or [],
        trace=trace or [],
    )


def test_convergence_summary_marks_threshold_crossing() -> None:
    summary = fitness._convergence_summary_from_trace(
        metric="gt_mse",
        threshold=0.5,
        trace=[
            {"outer_idx": 1, "elapsed_seconds": 10.0, "quality_value": 1.2},
            {"outer_idx": 2, "elapsed_seconds": 18.0, "quality_value": 0.4},
        ],
        stopped_on_threshold=True,
    )

    assert summary.threshold_met is True
    assert summary.stopped_on_threshold is True
    assert summary.stopped_on_plateau is False
    assert summary.stopped_on_budget is False
    assert summary.seconds_to_threshold == 18.0
    assert summary.outer_iters_to_threshold == 2
    assert summary.best_quality_value == 0.4
    assert summary.total_outer_iters_executed == 2
    assert summary.final_stop_reason == "threshold"
    assert summary.first_threshold_crossing_level_factor is None


def test_convergence_summary_handles_threshold_miss() -> None:
    summary = fitness._convergence_summary_from_trace(
        metric="gt_mse",
        threshold=0.2,
        trace=[
            {"outer_idx": 1, "elapsed_seconds": 5.0, "quality_value": 1.0},
            {"outer_idx": 2, "elapsed_seconds": 9.0, "quality_value": 0.7},
        ],
        stopped_on_plateau=True,
    )

    assert summary.threshold_met is False
    assert summary.stopped_on_threshold is False
    assert summary.stopped_on_plateau is True
    assert summary.stopped_on_budget is False
    assert summary.seconds_to_threshold is None
    assert summary.outer_iters_to_threshold is None
    assert summary.best_quality_value == 0.7
    assert summary.best_quality_elapsed_seconds == 9.0
    assert summary.final_stop_reason == "plateau"


def test_meaningful_relative_improvement_requires_margin() -> None:
    assert fitness._is_meaningful_relative_improvement(10.0, 9.0, 0.02) is True
    assert fitness._is_meaningful_relative_improvement(10.0, 9.9, 0.02) is False


def test_convergence_action_advances_coarse_levels_and_stops_finest() -> None:
    assert (
        fitness._convergence_action_for_level(
            level_factor=4,
            finest_factor=1,
            budget_hit=False,
            threshold_hit=True,
            plateau_hit=False,
        )
        == "advance_level"
    )
    assert (
        fitness._convergence_action_for_level(
            level_factor=2,
            finest_factor=1,
            budget_hit=False,
            threshold_hit=False,
            plateau_hit=True,
        )
        == "advance_level"
    )
    assert (
        fitness._convergence_action_for_level(
            level_factor=1,
            finest_factor=1,
            budget_hit=False,
            threshold_hit=True,
            plateau_hit=False,
        )
        == "stop_run"
    )
    assert (
        fitness._convergence_action_for_level(
            level_factor=1,
            finest_factor=1,
            budget_hit=False,
            threshold_hit=False,
            plateau_hit=True,
        )
        == "stop_run"
    )


def test_aggregate_warm_convergence_runs_requires_multiple_hits() -> None:
    run1 = _summary(
        stopped_on_threshold=True,
        seconds_to_threshold=12.0,
        outer_iters_to_threshold=2,
        best_quality_value=0.8,
        best_quality_elapsed_seconds=12.0,
        total_outer_iters_executed=2,
        final_stop_reason="threshold",
        first_threshold_crossing_level_factor=2,
        finest_level_first_elapsed_seconds=12.0,
        finest_level_first_outer_idx=2,
        trace=[{"outer_idx": 1, "elapsed_seconds": 5.0, "quality_value": 1.2}],
    )
    run2 = _summary(
        threshold_met=False,
        stopped_on_plateau=True,
        seconds_to_threshold=None,
        outer_iters_to_threshold=None,
        best_quality_value=1.1,
        best_quality_elapsed_seconds=18.0,
        total_outer_iters_executed=3,
        final_stop_reason="plateau",
        first_threshold_crossing_level_factor=None,
        reached_finest_level=False,
        trace=[{"outer_idx": 1, "elapsed_seconds": 6.0, "quality_value": 1.5}],
    )
    run3 = _summary(
        stopped_on_threshold=True,
        seconds_to_threshold=10.0,
        outer_iters_to_threshold=2,
        best_quality_value=0.9,
        best_quality_elapsed_seconds=10.0,
        total_outer_iters_executed=2,
        final_stop_reason="threshold",
        first_threshold_crossing_level_factor=1,
        finest_level_first_elapsed_seconds=10.0,
        finest_level_first_outer_idx=2,
        trace=[{"outer_idx": 1, "elapsed_seconds": 4.0, "quality_value": 1.3}],
    )

    aggregate = fitness._aggregate_warm_convergence_runs(
        [run1, run2, run3],
        required_successes=2,
    )

    assert aggregate["quality_threshold_met"] is True
    assert aggregate["warm_threshold_hit_count"] == 2
    assert aggregate["warm_threshold_total_runs"] == 3
    assert aggregate["warm_seconds_to_quality_threshold"] == 11.0
    assert aggregate["warm_outer_iters_to_quality_threshold"] == 2
    assert aggregate["warm_best_quality_value"] == 0.9
    assert aggregate["warm_stopped_on_plateau_count"] == 1
    assert aggregate["final_stop_reason"] == "threshold"
    assert aggregate["first_threshold_crossing_level_factor"] == 1
    assert aggregate["benchmark_valid"] is False
    assert aggregate["invalid_reason"] == "did_not_reach_finest_level"


def test_normalize_recon_profile_config_coerces_supported_fields() -> None:
    profile = {
        "recon": {
            "algorithm": "FISTA_TV",
            "lambda_tv": "0.25",
            "L": "4.5",
            "views_per_batch": "3",
            "projector_unroll": "2",
            "checkpoint_projector": True,
            "gather_dtype": "bf16",
            "grad_mode": "manual",
            "tv_prox_iters": "7",
            "recon_rel_tol": "0.01",
            "recon_patience": "4",
        }
    }

    cfg = fitness._normalize_recon_profile_config(profile)

    assert cfg.algorithm == "fista_tv"
    assert cfg.lambda_tv == 0.25
    assert cfg.L == 4.5
    assert cfg.views_per_batch == 3
    assert cfg.projector_unroll == 2
    assert cfg.gather_dtype == "bf16"
    assert cfg.grad_mode == "manual"
    assert cfg.tv_prox_iters == 7
    assert cfg.recon_rel_tol == 0.01
    assert cfg.recon_patience == 4


def test_normalize_align_profile_config_builds_typed_align_config() -> None:
    profile = {
        "align": {
            "levels": [4, 2, 1],
            "outer_iters": "5",
            "recon_iters": "12",
            "lambda_tv": "0.01",
            "views_per_batch": "2",
            "projector_unroll": "3",
            "gather_dtype": "bf16",
            "loss_kind": "pwls",
            "loss_params": {"a": 2, "b": 0.5},
            "warmup_enabled": True,
            "warmup_outer_iters": "1",
            "warmup_recon_iters": "2",
            "warmup_time_budget_seconds": "90",
            "warmup_stop_on_first_finest_level": True,
            "k_step": "3",
        }
    }

    cfg = fitness._normalize_align_profile_config(profile)
    fake_mods = SimpleNamespace(
        parse_loss_spec=lambda kind, params: ("parsed", kind, params),
        AlignConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    steady = cfg.build_align_config(fake_mods)
    warmup = cfg.build_align_config(fake_mods, warmup=True)

    assert cfg.levels == (4, 2, 1)
    assert cfg.time_budget_seconds is None
    assert cfg.warmup_enabled is True
    assert cfg.warmup_time_budget_seconds == 90.0
    assert cfg.k_step == 3
    assert steady.outer_iters == 5
    assert steady.recon_iters == 12
    assert steady.views_per_batch == 2
    assert steady.projector_unroll == 3
    assert steady.gather_dtype == "bf16"
    assert steady.loss == ("parsed", "pwls", {"a": 2.0, "b": 0.5})
    assert steady.early_stop is True
    assert warmup.outer_iters == 1
    assert warmup.recon_iters == 2
    assert warmup.early_stop is False
    assert warmup.loss == ("parsed", "pwls", {"a": 2.0, "b": 0.5})


def test_align_profile_uses_unscored_primer_when_warmup_is_incomplete(tmp_path: Path) -> None:
    bundle = fitness.FixtureBundle(
        name="fixture",
        grid={},
        detector={},
        geometry_type="parallel",
        geometry_meta=None,
        thetas_deg=np.asarray([], dtype=np.float32),
        volume=np.zeros((1, 1), dtype=np.float32),
        projections=np.zeros((1, 1), dtype=np.float32),
        align_params=np.zeros((1, 5), dtype=np.float32),
    )
    profile = {
        "name": "screen_convergence_align_parallel_3d_64",
        "measurement": {},
        "visualization": {"enabled": False},
        "warm_runs": 2,
        "convergence": {
            "enabled": True,
            "metric": "gt_mse",
            "threshold": None,
            "stop_on_threshold": False,
            "stop_on_plateau": False,
            "required_warm_successes": 1,
        },
        "align": {
            "warmup_enabled": True,
            "warmup_time_budget_seconds": 30,
            "warmup_stop_on_first_finest_level": True,
            "warmup_outer_iters": 1,
            "warmup_recon_iters": 1,
            "time_budget_seconds": 60,
            "levels": [2, 1],
            "outer_iters": 2,
            "recon_iters": 2,
            "checkpoint_projector": True,
            "views_per_batch": 1,
            "projector_unroll": 1,
            "gather_dtype": "bf16",
            "opt_method": "gn",
            "loss_kind": "l2_otsu",
            "k_step": 1,
        },
    }

    fake_mods = SimpleNamespace(
        jnp=np,
        AlignConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        parse_loss_spec=lambda kind, params: ("parsed", kind, params),
        jax=SimpleNamespace(device_get=lambda value: value),
        loss_metrics_abs=lambda *args, **kwargs: {"trans_rmse_px": 1.0},
        loss_metrics_relative=lambda *args, **kwargs: {},
        loss_metrics_gf=lambda *args, **kwargs: {"trans_gf_rmse_px": 0.9},
        gt_projection_helper=lambda *args, **kwargs: np.zeros((1, 1), dtype=np.float32),
    )

    warmup_run = fitness.RunResult(
        output={"convergence": _summary(reached_finest_level=False, final_stop_reason="budget")},
        seconds=5.0,
        peak_host_rss_mb=100.0,
        peak_gpu_memory_mb=200.0,
        peak_gpu_memory_process_mb=None,
        peak_gpu_memory_device_mb=200.0,
        gpu_memory_backend="cuda",
        gpu_memory_scope="device",
        gpu_memory_process_source=None,
        gpu_memory_process_supported=False,
        gpu_memory_sample_interval_seconds=0.05,
        gpu_memory_sample_count=1,
        gpu_memory_observed_gpu_count=1,
        gpu_sampler_error=None,
    )
    primer_run = fitness.RunResult(
        output={
            "convergence": _summary(reached_finest_level=False, final_stop_reason="budget"),
            "params": np.zeros((1, 5), dtype=np.float32),
            "volume": np.zeros((1, 1), dtype=np.float32),
            "info": {},
        },
        seconds=30.0,
        peak_host_rss_mb=110.0,
        peak_gpu_memory_mb=210.0,
        peak_gpu_memory_process_mb=None,
        peak_gpu_memory_device_mb=210.0,
        gpu_memory_backend="cuda",
        gpu_memory_scope="device",
        gpu_memory_process_source=None,
        gpu_memory_process_supported=False,
        gpu_memory_sample_interval_seconds=0.05,
        gpu_memory_sample_count=2,
        gpu_memory_observed_gpu_count=1,
        gpu_sampler_error=None,
    )
    warm1 = fitness.RunResult(
        output={
            "convergence": _summary(
                reached_finest_level=True, finest_level_first_elapsed_seconds=40.0
            ),
            "params": np.zeros((1, 5), dtype=np.float32),
            "volume": np.zeros((1, 1), dtype=np.float32),
            "info": {},
        },
        seconds=10.0,
        peak_host_rss_mb=120.0,
        peak_gpu_memory_mb=220.0,
        peak_gpu_memory_process_mb=None,
        peak_gpu_memory_device_mb=220.0,
        gpu_memory_backend="cuda",
        gpu_memory_scope="device",
        gpu_memory_process_source=None,
        gpu_memory_process_supported=False,
        gpu_memory_sample_interval_seconds=0.05,
        gpu_memory_sample_count=3,
        gpu_memory_observed_gpu_count=1,
        gpu_sampler_error=None,
    )
    warm2 = fitness.RunResult(
        output={
            "convergence": _summary(
                reached_finest_level=True,
                finest_level_first_elapsed_seconds=38.0,
                level_summaries=[
                    {
                        "level_factor": 1,
                        "elapsed_seconds_total": 38.0,
                        "outer_iters_executed": 2,
                        "final_gt_mse": 0.25,
                    }
                ],
            ),
            "params": np.zeros((1, 5), dtype=np.float32),
            "volume": np.zeros((1, 1), dtype=np.float32),
            "info": {},
        },
        seconds=12.0,
        peak_host_rss_mb=130.0,
        peak_gpu_memory_mb=230.0,
        peak_gpu_memory_process_mb=None,
        peak_gpu_memory_device_mb=230.0,
        gpu_memory_backend="cuda",
        gpu_memory_scope="device",
        gpu_memory_process_source=None,
        gpu_memory_process_supported=False,
        gpu_memory_sample_interval_seconds=0.05,
        gpu_memory_sample_count=4,
        gpu_memory_observed_gpu_count=1,
        gpu_sampler_error=None,
    )

    scripted_runs = [warmup_run, primer_run, warm1, warm2]
    with (
        patch.object(
            fitness,
            "_bundle_geometry",
            return_value=(object(), SimpleNamespace(du=1.0, dv=1.0), object()),
        ),
        patch.object(fitness, "_timed_call", side_effect=scripted_runs),
        patch.object(fitness, "_maybe_save_jax_device_memory_profile", return_value=(None, None)),
    ):
        metrics = fitness._run_align_profile(
            bundle,
            profile,
            fake_mods,
            tmp_path / "metrics.json",
        )

    assert metrics["warmup_incomplete"] is True
    assert metrics["primer_ran"] is True
    assert metrics["primer_seconds"] == 30.0
    assert metrics["primer_reached_finest_level"] is False
    assert metrics["warm_run_seconds_mean"] == 11.0
    assert metrics["benchmark_valid"] is True
    assert metrics["success"] is True
