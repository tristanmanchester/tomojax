from __future__ import annotations

import importlib
import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.bench.forward_projector import (
    ForwardSinogramBenchmarkConfig,
    ForwardSinogramFixture,
    _block_tree_ready,
    _device_metadata,
    _geomean,
    _pallas_requested_variant_metadata,
    _sinogram_fixture_metadata,
    make_forward_sinogram_fixture,
)
from tomojax.core.projector import forward_project_view_T

ResidualModeName = Literal[
    "jax_materialized",
    "pallas_materialized",
    "pallas_fused",
    "pallas_dispatch",
]
RESIDUAL_SUITE_NAMES = ("residual",)
PALLAS_DISPATCH_RAY_STEP_THRESHOLD = 1_000_000_000


@dataclass(frozen=True)
class ForwardResidualBenchmarkConfig(ForwardSinogramBenchmarkConfig):
    target_delta: float = 0.01


@dataclass(frozen=True)
class ForwardResidualFixture:
    sinogram_fixture: ForwardSinogramFixture
    target: jnp.ndarray


@dataclass(frozen=True)
class ForwardResidualSuiteCase:
    name: str
    config: ForwardResidualBenchmarkConfig


def residual_suite_cases(name: str = "residual") -> tuple[ForwardResidualSuiteCase, ...]:
    """Return fixed objective workloads for fused residual benchmarking."""
    if name != "residual":
        raise ValueError(f"residual suite must be one of: {', '.join(RESIDUAL_SUITE_NAMES)}")
    return (
        ForwardResidualSuiteCase(
            "residual-64",
            ForwardResidualBenchmarkConfig(
                nx=64,
                ny=64,
                nz=64,
                nu=64,
                nv=64,
                n_views=90,
                warm_runs=5,
            ),
        ),
        ForwardResidualSuiteCase(
            "residual-128",
            ForwardResidualBenchmarkConfig(
                nx=128,
                ny=128,
                nz=128,
                nu=128,
                nv=128,
                n_views=180,
                warm_runs=5,
            ),
        ),
        ForwardResidualSuiteCase(
            "high-ray-residual-128",
            ForwardResidualBenchmarkConfig(
                nx=128,
                ny=128,
                nz=128,
                nu=256,
                nv=256,
                n_views=90,
                step_size=0.5,
                warm_runs=5,
            ),
        ),
    )


def _make_jax_projection_callable(
    fixture: ForwardSinogramFixture,
    config: ForwardResidualBenchmarkConfig,
) -> Callable[[], jnp.ndarray]:
    def project_one(T: jnp.ndarray) -> jnp.ndarray:
        return forward_project_view_T(
            T,
            fixture.grid,
            fixture.detector,
            fixture.volume,
            step_size=config.step_size,
            n_steps=config.n_steps,
            use_checkpoint=config.use_checkpoint,
            unroll=config.unroll,
            gather_dtype=config.gather_dtype,
            det_grid=fixture.det_grid,
        )

    project_all = jax.jit(jax.vmap(project_one))
    return lambda: project_all(fixture.T_stack)


def make_forward_residual_fixture(
    config: ForwardResidualBenchmarkConfig,
) -> ForwardResidualFixture:
    """Build a deterministic residual fixture with target generated from the JAX oracle."""
    sinogram_fixture = make_forward_sinogram_fixture(config)
    projection = _make_jax_projection_callable(sinogram_fixture, config)()
    _block_tree_ready(projection)
    delta = jnp.linspace(
        0.0,
        float(config.target_delta),
        int(np.prod(projection.shape)),
        dtype=jnp.float32,
    ).reshape(projection.shape)
    target = projection + delta
    _block_tree_ready(target)
    return ForwardResidualFixture(sinogram_fixture=sinogram_fixture, target=target)


def _resolve_pallas_module() -> tuple[Any | None, str | None]:
    try:
        module = importlib.import_module("tomojax.core.pallas_projector")
    except Exception as exc:
        return None, f"pallas_module_unavailable: {exc}"
    return module, None


def _pallas_actual_variant_metadata(
    fixture: ForwardResidualFixture,
    config: ForwardResidualBenchmarkConfig,
    actual_mode: str,
) -> dict[str, Any] | None:
    if actual_mode not in {"pallas_materialized", "pallas_fused"}:
        return None
    module, fallback_reason = _resolve_pallas_module()
    if module is None:
        return {"metadata_error": fallback_reason}
    metadata_fn = getattr(module, "pallas_projector_actual_sinogram_variant_metadata", None)
    traversal_metadata_fn = getattr(module, "pallas_projector_sinogram_traversal_metadata", None)
    if metadata_fn is None:
        return {"metadata_error": "pallas_actual_sinogram_variant_metadata_missing"}
    sf = fixture.sinogram_fixture
    try:
        metadata = metadata_fn(
            sf.T_stack,
            sf.grid,
            sf.detector,
            det_grid=sf.det_grid,
            tile_shape=config.pallas_tile_shape,
            num_warps=config.pallas_num_warps,
            kernel_variant=config.pallas_kernel_variant,
            layout_variant=config.pallas_layout_variant,
            state_mode=config.pallas_state_mode,
            gather_dtype=config.gather_dtype,
        )
    except Exception as exc:
        return {"metadata_error": f"pallas_variant_metadata_failed: {exc}"}
    if traversal_metadata_fn is not None:
        try:
            metadata.update(
                traversal_metadata_fn(
                    sf.T_stack,
                    sf.grid,
                    step_size=config.step_size,
                    n_steps=config.n_steps,
                )
            )
        except Exception as exc:
            metadata["traversal_metadata_error"] = f"pallas_traversal_metadata_failed: {exc}"
    return metadata


def _pallas_residual_unsupported_reason(
    fixture: ForwardResidualFixture,
    config: ForwardResidualBenchmarkConfig,
) -> str | None:
    module, fallback_reason = _resolve_pallas_module()
    if module is None:
        return fallback_reason
    support_fn = getattr(module, "pallas_projector_sinogram_unsupported_reason", None)
    if support_fn is None:
        return "pallas_sinogram_support_check_missing"
    sf = fixture.sinogram_fixture
    try:
        return support_fn(
            sf.T_stack,
            sf.grid,
            sf.detector,
            sf.volume,
            step_size=config.step_size,
            n_steps=config.n_steps,
            gather_dtype=config.gather_dtype,
            det_grid=sf.det_grid,
            tile_shape=config.pallas_tile_shape,
            num_warps=config.pallas_num_warps,
            kernel_variant=config.pallas_kernel_variant,
            layout_variant=config.pallas_layout_variant,
            state_mode=config.pallas_state_mode,
        )
    except Exception as exc:
        return f"pallas_residual_support_check_failed: {exc}"


def _make_residual_callable(
    requested_mode: ResidualModeName,
    fixture: ForwardResidualFixture,
    config: ForwardResidualBenchmarkConfig,
) -> tuple[Callable[[], jnp.ndarray], str, str | None]:
    sf = fixture.sinogram_fixture
    target = fixture.target
    jax_project = _make_jax_projection_callable(sf, config)
    if requested_mode == "jax_materialized":
        return (
            lambda: jnp.sum((jax_project() - target) ** 2, dtype=jnp.float32),
            "jax_materialized",
            None,
        )

    if (
        requested_mode == "pallas_dispatch"
        and residual_dispatch_selected_mode(config) == "jax_materialized"
    ):
        return (
            lambda: jnp.sum((jax_project() - target) ** 2, dtype=jnp.float32),
            "pallas_dispatch",
            None,
        )

    module, fallback_reason = _resolve_pallas_module()
    if module is None:
        return (
            lambda: jnp.sum((jax_project() - target) ** 2, dtype=jnp.float32),
            "jax_materialized",
            fallback_reason,
        )
    unsupported_reason = _pallas_residual_unsupported_reason(fixture, config)
    if unsupported_reason:
        return (
            lambda: jnp.sum((jax_project() - target) ** 2, dtype=jnp.float32),
            "jax_materialized",
            unsupported_reason,
        )

    if requested_mode in {"pallas_materialized", "pallas_dispatch"}:
        pallas_project = getattr(module, "forward_project_views_T_pallas", None)
        if pallas_project is None:
            return (
                lambda: jnp.sum((jax_project() - target) ** 2, dtype=jnp.float32),
                "jax_materialized",
                "pallas_batched_callable_missing",
            )

        def call_pallas_materialized() -> jnp.ndarray:
            projection = pallas_project(
                sf.T_stack,
                sf.grid,
                sf.detector,
                sf.volume,
                step_size=config.step_size,
                n_steps=config.n_steps,
                unroll=config.unroll,
                gather_dtype=config.gather_dtype,
                det_grid=sf.det_grid,
                tile_shape=config.pallas_tile_shape,
                num_warps=config.pallas_num_warps,
                kernel_variant=config.pallas_kernel_variant,
                layout_variant=config.pallas_layout_variant,
                state_mode=config.pallas_state_mode,
            )
            return jnp.sum((projection - target) ** 2, dtype=jnp.float32)

        return call_pallas_materialized, requested_mode, None

    pallas_fused = getattr(module, "forward_project_residual_sse_T_pallas", None)
    if pallas_fused is None:
        return (
            lambda: jnp.sum((jax_project() - target) ** 2, dtype=jnp.float32),
            "jax_materialized",
            "pallas_fused_residual_callable_missing",
        )

    def call_pallas_fused() -> jnp.ndarray:
        return pallas_fused(
            sf.T_stack,
            sf.grid,
            sf.detector,
            sf.volume,
            target,
            step_size=config.step_size,
            n_steps=config.n_steps,
            unroll=config.unroll,
            gather_dtype=config.gather_dtype,
            det_grid=sf.det_grid,
            tile_shape=config.pallas_tile_shape,
            num_warps=config.pallas_num_warps,
            kernel_variant=config.pallas_kernel_variant,
            layout_variant=config.pallas_layout_variant,
            state_mode=config.pallas_state_mode,
        )

    return call_pallas_fused, "pallas_fused", None


def residual_dispatch_estimated_ray_steps(config: ForwardResidualBenchmarkConfig) -> int:
    """Return a static workload estimate used by the benchmark-only dispatch probe."""
    n_steps = int(
        np.ceil(np.sqrt(config.nx * config.nx + config.ny * config.ny + config.nz * config.nz))
        if config.n_steps is None
        else config.n_steps
    )
    if config.step_size is not None and config.n_steps is None:
        n_steps = int(
            np.ceil(
                np.sqrt(config.nx * config.nx + config.ny * config.ny + config.nz * config.nz)
                / float(config.step_size)
            )
        )
    return int(config.n_views) * int(config.nu) * int(config.nv) * max(1, n_steps)


def residual_dispatch_selected_mode(config: ForwardResidualBenchmarkConfig) -> str:
    """Select the residual backend for the benchmark-only high-ray dispatch probe."""
    return (
        "pallas_materialized"
        if residual_dispatch_estimated_ray_steps(config) >= PALLAS_DISPATCH_RAY_STEP_THRESHOLD
        else "jax_materialized"
    )


def _time_blocked_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    output = fn()
    _block_tree_ready(output)
    return time.perf_counter() - start, output


def _timing_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "warm_runs": 0,
            "warm_seconds": [],
            "warm_seconds_mean": None,
            "warm_seconds_median": None,
            "warm_seconds_min": None,
            "warm_seconds_max": None,
        }
    return {
        "warm_runs": len(values),
        "warm_seconds": values,
        "warm_seconds_mean": float(statistics.mean(values)),
        "warm_seconds_median": float(statistics.median(values)),
        "warm_seconds_min": float(min(values)),
        "warm_seconds_max": float(max(values)),
    }


def _scalar_error_metrics(candidate: jnp.ndarray, oracle: jnp.ndarray) -> dict[str, Any]:
    cand = float(np.asarray(candidate, dtype=np.float64))
    ref = float(np.asarray(oracle, dtype=np.float64))
    diff = cand - ref
    denom = max(abs(ref), 1e-12)
    return {
        "finite": bool(np.isfinite(cand)),
        "value": cand,
        "abs_error": float(abs(diff)),
        "relative_error": float(abs(diff) / denom),
    }


def benchmark_residual_mode(
    requested_mode: ResidualModeName,
    fixture: ForwardResidualFixture,
    config: ForwardResidualBenchmarkConfig,
    *,
    oracle: jnp.ndarray | None = None,
    jax_median: float | None = None,
) -> tuple[dict[str, Any], jnp.ndarray]:
    """Run first-call and warm-call timings for one residual mode."""
    if (
        requested_mode == "pallas_dispatch"
        and residual_dispatch_selected_mode(config) == "jax_materialized"
        and oracle is not None
        and jax_median is not None
    ):
        result = {
            "requested_mode": requested_mode,
            "actual_mode": requested_mode,
            "fallback_reason": None,
            "eligible_for_speed_claim": True,
            "first_call_seconds": None,
            "warm_runs": 0,
            "warm_seconds": [],
            "warm_seconds_mean": float(jax_median),
            "warm_seconds_median": float(jax_median),
            "warm_seconds_min": float(jax_median),
            "warm_seconds_max": float(jax_median),
            **_scalar_error_metrics(oracle, oracle),
            "dispatch_selected_mode": "jax_materialized",
            "dispatch_estimated_ray_steps": residual_dispatch_estimated_ray_steps(config),
            "dispatch_threshold_ray_steps": PALLAS_DISPATCH_RAY_STEP_THRESHOLD,
            "dispatch_timing_source": "jax_materialized_baseline",
            "requested_pallas_variant": _pallas_requested_variant_metadata(config),
            "actual_pallas_variant": None,
            "speedup_vs_jax_materialized_warm_median": 1.0,
        }
        return result, oracle

    call, actual_mode, fallback_reason = _make_residual_callable(requested_mode, fixture, config)
    first_seconds, first_output = _time_blocked_call(call)

    warm_seconds: list[float] = []
    warm_output = first_output
    for _ in range(max(0, int(config.warm_runs))):
        seconds, warm_output = _time_blocked_call(call)
        warm_seconds.append(float(seconds))

    reference = first_output if oracle is None else oracle
    result = {
        "requested_mode": requested_mode,
        "actual_mode": actual_mode,
        "fallback_reason": fallback_reason,
        "eligible_for_speed_claim": bool(requested_mode == actual_mode),
        "first_call_seconds": float(first_seconds),
        **_timing_summary(warm_seconds),
        **_scalar_error_metrics(warm_output, reference),
    }
    if requested_mode == "pallas_dispatch":
        result["dispatch_selected_mode"] = residual_dispatch_selected_mode(config)
        result["dispatch_estimated_ray_steps"] = residual_dispatch_estimated_ray_steps(config)
        result["dispatch_threshold_ray_steps"] = PALLAS_DISPATCH_RAY_STEP_THRESHOLD
    if requested_mode in {"pallas_materialized", "pallas_fused", "pallas_dispatch"}:
        result["requested_pallas_variant"] = _pallas_requested_variant_metadata(config)
        result["actual_pallas_variant"] = _pallas_actual_variant_metadata(
            fixture,
            config,
            "pallas_materialized"
            if requested_mode == "pallas_dispatch"
            and result.get("dispatch_selected_mode") == "pallas_materialized"
            else actual_mode,
        )
    result["speedup_vs_jax_materialized_warm_median"] = (
        _speedup(
            baseline=jax_median,
            candidate=result["warm_seconds_median"],
        )
        if requested_mode == actual_mode
        and requested_mode in {"pallas_materialized", "pallas_fused", "pallas_dispatch"}
        else None
    )
    return result, first_output


def run_forward_residual_benchmark(
    config: ForwardResidualBenchmarkConfig,
) -> dict[str, Any]:
    """Compare materialized and fused residual SSE forward objective timings."""
    fixture = make_forward_residual_fixture(config)
    jax_result, oracle = benchmark_residual_mode("jax_materialized", fixture, config)
    jax_median = jax_result["warm_seconds_median"]
    results = [jax_result]
    if config.include_pallas:
        pallas_materialized_result, _ = benchmark_residual_mode(
            "pallas_materialized",
            fixture,
            config,
            oracle=oracle,
            jax_median=jax_median,
        )
        pallas_fused_result, _ = benchmark_residual_mode(
            "pallas_fused",
            fixture,
            config,
            oracle=oracle,
            jax_median=jax_median,
        )
        pallas_dispatch_result, _ = benchmark_residual_mode(
            "pallas_dispatch",
            fixture,
            config,
            oracle=oracle,
            jax_median=jax_median,
        )
        results.extend([pallas_materialized_result, pallas_fused_result, pallas_dispatch_result])

    return {
        "benchmark": "forward_residual",
        "fixture_backend": "jax_materialized",
        "config": asdict(config),
        "fixture": _sinogram_fixture_metadata(fixture.sinogram_fixture, config),
        "device": _device_metadata(),
        "results": results,
    }


def run_forward_residual_suite(
    name: str = "residual",
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run fixed residual objective benchmark cases."""
    case_metrics: list[dict[str, Any]] = []
    for case in residual_suite_cases(name):
        config = replace(case.config, **(overrides or {}))
        metrics = run_forward_residual_benchmark(config)
        metrics["case_name"] = case.name
        case_metrics.append(metrics)
    return {
        "benchmark": "forward_residual_suite",
        "suite": name,
        "device": _device_metadata(),
        "summary": _residual_suite_summary(case_metrics),
        "cases": case_metrics,
    }


def _residual_suite_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    mode_summaries: dict[str, dict[str, Any]] = {}
    for mode in ("pallas_materialized", "pallas_fused", "pallas_dispatch"):
        rows = [
            row
            for case in cases
            for row in case.get("results", [])
            if row.get("requested_mode") == mode
        ]
        speedups = [
            float(row["speedup_vs_jax_materialized_warm_median"])
            for row in rows
            if row.get("speedup_vs_jax_materialized_warm_median") is not None
        ]
        mode_summaries[mode] = {
            "cases_with_requested_pallas": len(rows),
            "cases_pallas_eligible": sum(1 for row in rows if row.get("eligible_for_speed_claim")),
            "cases_parity_passed": sum(1 for row in rows if _parity_passed(row)),
            "geomean_speedup_vs_jax_materialized_warm_median": _geomean(speedups),
            "worst_case_speedup_vs_jax_materialized_warm_median": min(speedups)
            if speedups
            else None,
            "best_case_speedup_vs_jax_materialized_warm_median": max(speedups)
            if speedups
            else None,
        }
    fused = mode_summaries["pallas_fused"]
    return {
        "cases_total": len(cases),
        "pallas_modes": mode_summaries,
        "geomean_speedup_vs_jax_materialized_warm_median": fused[
            "geomean_speedup_vs_jax_materialized_warm_median"
        ],
        "worst_case_speedup_vs_jax_materialized_warm_median": fused[
            "worst_case_speedup_vs_jax_materialized_warm_median"
        ],
        "best_case_speedup_vs_jax_materialized_warm_median": fused[
            "best_case_speedup_vs_jax_materialized_warm_median"
        ],
    }


def _parity_passed(row: dict[str, Any], *, atol: float = 1e-3, rtol: float = 1e-4) -> bool:
    return bool(
        row.get("finite")
        and row.get("abs_error") is not None
        and row.get("relative_error") is not None
        and float(row["abs_error"]) <= atol
        and float(row["relative_error"]) <= rtol
    )


def _speedup(*, baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or candidate <= 0:
        return None
    return float(baseline / candidate)


def write_benchmark_json(metrics: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
