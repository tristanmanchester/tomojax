from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.bench.forward_projector import (
    ForwardSinogramBenchmarkConfig,
    _block_tree_ready,
    _device_metadata,
    make_forward_sinogram_fixture,
)
from tomojax.core.projector import forward_project_view_T
from tomojax.recon.fista_tv_core import FistaCoreConfig, fista_tv_core_arrays


FISTA_ITERATION_SUITE_NAMES = ("fista_iteration",)


@dataclass(frozen=True)
class FistaIterationBenchmarkConfig(ForwardSinogramBenchmarkConfig):
    lambda_tv: float = 0.001
    L: float = 5000.0
    views_per_batch: int = 1
    regulariser: str = "huber_tv"


@dataclass(frozen=True)
class FistaIterationSuiteCase:
    name: str
    config: FistaIterationBenchmarkConfig


def fista_iteration_suite_cases(name: str = "fista_iteration") -> tuple[FistaIterationSuiteCase, ...]:
    if name != "fista_iteration":
        raise ValueError(
            f"fista iteration suite must be one of: {', '.join(FISTA_ITERATION_SUITE_NAMES)}"
        )
    return (
        FistaIterationSuiteCase(
            "fista-iter-24",
            FistaIterationBenchmarkConfig(
                nx=24,
                ny=24,
                nz=24,
                nu=24,
                nv=24,
                n_views=24,
                warm_runs=7,
                pose_mode="general_5d",
            ),
        ),
        FistaIterationSuiteCase(
            "fista-iter-64",
            FistaIterationBenchmarkConfig(
                nx=64,
                ny=64,
                nz=64,
                nu=64,
                nv=64,
                n_views=90,
                warm_runs=5,
                pose_mode="general_5d",
            ),
        ),
    )


def _projection_fixture(config: FistaIterationBenchmarkConfig):
    fixture = make_forward_sinogram_fixture(config)

    @jax.jit
    def project_all(volume):
        return jax.vmap(
            lambda T: forward_project_view_T(
                T,
                fixture.grid,
                fixture.detector,
                volume,
                use_checkpoint=config.use_checkpoint,
                unroll=1 if config.unroll is None else int(config.unroll),
                gather_dtype=config.gather_dtype,
                det_grid=fixture.det_grid,
            )
        )(fixture.T_stack)

    projections = project_all(fixture.volume)
    _block_tree_ready(projections)
    return fixture, projections


def _make_fista_call(config: FistaIterationBenchmarkConfig) -> tuple[Callable[[], Any], dict[str, Any]]:
    fixture, projections = _projection_fixture(config)
    x0 = jnp.zeros_like(fixture.volume)
    cfg = FistaCoreConfig(
        iters=1,
        lambda_tv=float(config.lambda_tv),
        regulariser=config.regulariser,
        L=float(config.L),
        checkpoint_projector=bool(config.use_checkpoint),
        projector_unroll=1 if config.unroll is None else int(config.unroll),
        gather_dtype=str(config.gather_dtype),
        views_per_batch=int(config.views_per_batch),
    )

    @jax.jit
    def run():
        result = fista_tv_core_arrays(
            x0=x0,
            T_all=fixture.T_stack,
            det_grid=fixture.det_grid,
            projections=projections,
            grid=fixture.grid,
            detector=fixture.detector,
            cfg=cfg,
        )
        return (
            result.x,
            result.loss,
            result.data_loss,
            result.regulariser_value,
            result.effective_iters,
        )

    fixture_meta = {
        "volume_shape": [int(fixture.grid.nx), int(fixture.grid.ny), int(fixture.grid.nz)],
        "detector_shape": [int(fixture.detector.nv), int(fixture.detector.nu)],
        "n_views": int(config.n_views),
        "pose_mode": config.pose_mode,
    }
    return run, fixture_meta


def _time_blocked_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    out = fn()
    _block_tree_ready(out)
    return time.perf_counter() - start, out


def _timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "warm_runs": len(values),
        "warm_seconds": values,
        "warm_seconds_mean": float(statistics.mean(values)) if values else None,
        "warm_seconds_median": float(statistics.median(values)) if values else None,
        "warm_seconds_min": float(min(values)) if values else None,
        "warm_seconds_max": float(max(values)) if values else None,
    }


def run_fista_iteration_benchmark(config: FistaIterationBenchmarkConfig) -> dict[str, Any]:
    call, fixture = _make_fista_call(config)
    first_seconds, first = _time_blocked_call(call)
    warm_seconds: list[float] = []
    warm = first
    for _ in range(max(0, int(config.warm_runs))):
        seconds, warm = _time_blocked_call(call)
        warm_seconds.append(float(seconds))
    first_x = np.asarray(first[0])
    warm_x = np.asarray(warm[0])
    denom = float(np.linalg.norm(first_x.ravel())) or 1.0
    return {
        "benchmark": "fista_iteration",
        "api_surface": "internal_fista_tv_core_arrays",
        "config": asdict(config),
        "fixture": fixture,
        "device": _device_metadata(),
        "first_call_seconds": float(first_seconds),
        **_timing_summary(warm_seconds),
        "quality": {
            "finite": bool(np.isfinite(warm_x).all()),
            "repeat_rel_l2_vs_first": float(np.linalg.norm((warm_x - first_x).ravel()) / denom),
            "loss": [float(v) for v in np.asarray(warm[1]).ravel()],
            "data_loss": float(np.asarray(warm[2])),
            "regulariser_value": float(np.asarray(warm[3])),
            "effective_iters": int(np.asarray(warm[4])),
        },
    }


def run_fista_iteration_suite(
    name: str = "fista_iteration",
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cases = []
    for case in fista_iteration_suite_cases(name):
        config = replace(case.config, **(overrides or {}))
        metrics = run_fista_iteration_benchmark(config)
        metrics["case_name"] = case.name
        cases.append(metrics)
    return {
        "benchmark": "fista_iteration_suite",
        "suite": name,
        "device": _device_metadata(),
        "cases": cases,
        "summary": {
            "cases_total": len(cases),
            "warm_seconds_median_by_case": {
                case["case_name"]: case["warm_seconds_median"] for case in cases
            },
        },
    }


def write_benchmark_json(metrics: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
