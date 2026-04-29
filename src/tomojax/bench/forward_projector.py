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

from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import _resolve_n_steps, forward_project_view_T, get_detector_grid_device

BackendName = Literal["jax", "pallas"]
SuiteName = Literal["quick", "confirm"]
PRESET_NAMES = ("tiny", "smoke", "profile-128", "noncubic-align-128", "high-ray-count-128")
SUITE_NAMES = ("quick", "confirm")


@dataclass(frozen=True)
class ForwardProjectorBenchmarkConfig:
    nx: int = 64
    ny: int = 64
    nz: int = 64
    nu: int = 64
    nv: int = 64
    n_steps: int | None = None
    step_size: float | None = None
    view_angle_deg: float = 17.0
    seed: int = 0
    warm_runs: int = 5
    gather_dtype: str = "fp32"
    unroll: int | None = None
    use_checkpoint: bool = True
    include_pallas: bool = True


@dataclass(frozen=True)
class ForwardProjectorFixture:
    grid: Grid
    detector: Detector
    T: jnp.ndarray
    volume: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]


@dataclass(frozen=True)
class ForwardProjectorSuiteCase:
    name: str
    config: ForwardProjectorBenchmarkConfig


def preset_config(name: str) -> ForwardProjectorBenchmarkConfig:
    """Return a named forward-projector benchmark size."""
    if name == "tiny":
        return ForwardProjectorBenchmarkConfig(nx=16, ny=16, nz=16, nu=16, nv=16, warm_runs=3)
    if name == "smoke":
        return ForwardProjectorBenchmarkConfig(nx=64, ny=64, nz=64, nu=64, nv=64, warm_runs=5)
    if name == "profile-128":
        return ForwardProjectorBenchmarkConfig(nx=128, ny=128, nz=128, nu=128, nv=128, warm_runs=7)
    if name == "noncubic-align-128":
        return ForwardProjectorBenchmarkConfig(
            nx=128,
            ny=128,
            nz=96,
            nu=128,
            nv=96,
            view_angle_deg=37.0,
            warm_runs=7,
        )
    if name == "high-ray-count-128":
        return ForwardProjectorBenchmarkConfig(
            nx=128,
            ny=128,
            nz=128,
            nu=256,
            nv=256,
            view_angle_deg=37.0,
            step_size=0.5,
            warm_runs=7,
        )
    raise ValueError(f"preset must be one of: {', '.join(PRESET_NAMES)}")


def suite_cases(name: str) -> tuple[ForwardProjectorSuiteCase, ...]:
    """Return the named benchmark suite as concrete per-case configs."""
    if name == "quick":
        return (ForwardProjectorSuiteCase("high-ray-count-128", preset_config("high-ray-count-128")),)
    if name == "confirm":
        return (
            ForwardProjectorSuiteCase(
                "profile-128",
                replace(preset_config("profile-128"), warm_runs=25),
            ),
            ForwardProjectorSuiteCase(
                "noncubic-align-128",
                replace(preset_config("noncubic-align-128"), warm_runs=25),
            ),
            ForwardProjectorSuiteCase(
                "high-ray-count-128",
                replace(preset_config("high-ray-count-128"), warm_runs=25),
            ),
        )
    raise ValueError(f"suite must be one of: {', '.join(SUITE_NAMES)}")


def make_forward_projector_fixture(
    config: ForwardProjectorBenchmarkConfig,
) -> ForwardProjectorFixture:
    """Build a deterministic single-view fixture shared by all compared backends."""
    grid = Grid(
        nx=int(config.nx),
        ny=int(config.ny),
        nz=int(config.nz),
        vx=1.0,
        vy=1.0,
        vz=1.0,
    )
    detector = Detector(
        nu=int(config.nu),
        nv=int(config.nv),
        du=1.0,
        dv=1.0,
        det_center=(0.0, 0.0),
    )
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=[config.view_angle_deg])
    rng = np.random.default_rng(int(config.seed))
    volume_np = rng.normal(size=(grid.nx, grid.ny, grid.nz)).astype(np.float32)
    # Smooth the distribution enough that interpolation deltas are not dominated by noise.
    volume_np = np.abs(volume_np)
    return ForwardProjectorFixture(
        grid=grid,
        detector=detector,
        T=jnp.asarray(geometry.pose_for_view(0), dtype=jnp.float32),
        volume=jnp.asarray(volume_np, dtype=jnp.float32),
        det_grid=get_detector_grid_device(detector),
    )


def _block_tree_ready(value: Any) -> Any:
    return jax.block_until_ready(value)


def _device_metadata() -> dict[str, Any]:
    devices = jax.devices()
    default_backend = jax.default_backend()
    return {
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", None),
        "default_backend": default_backend,
        "device_count": len(devices),
        "devices": [
            {
                "id": getattr(device, "id", None),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
            }
            for device in devices
        ],
    }


def _fixture_metadata(
    fixture: ForwardProjectorFixture,
    config: ForwardProjectorBenchmarkConfig,
) -> dict[str, Any]:
    step_size = float(config.step_size if config.step_size is not None else fixture.grid.vy)
    resolved_n_steps = _resolve_n_steps(fixture.grid, step_size, config.n_steps)
    n_rays = int(fixture.detector.nu) * int(fixture.detector.nv)
    return {
        "volume_shape": [int(fixture.grid.nx), int(fixture.grid.ny), int(fixture.grid.nz)],
        "detector_shape": [int(fixture.detector.nv), int(fixture.detector.nu)],
        "n_rays": n_rays,
        "step_size": step_size,
        "resolved_n_steps": int(resolved_n_steps),
        "total_ray_steps": int(n_rays * resolved_n_steps),
    }


def _resolve_pallas_callable() -> tuple[Callable[..., jnp.ndarray] | None, str | None]:
    try:
        module = importlib.import_module("tomojax.core.pallas_projector")
    except Exception as exc:
        return None, f"pallas_module_unavailable: {exc}"
    fn = getattr(module, "forward_project_view_T_pallas", None)
    if fn is None:
        return None, "pallas_callable_missing"
    return fn, None


def _call_jax(
    fixture: ForwardProjectorFixture,
    config: ForwardProjectorBenchmarkConfig,
) -> jnp.ndarray:
    return forward_project_view_T(
        fixture.T,
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


def _make_backend_callable(
    requested_backend: BackendName,
    fixture: ForwardProjectorFixture,
    config: ForwardProjectorBenchmarkConfig,
) -> tuple[Callable[[], jnp.ndarray], str, str | None]:
    if requested_backend == "jax":
        return lambda: _call_jax(fixture, config), "jax", None

    pallas_fn, fallback_reason = _resolve_pallas_callable()
    if pallas_fn is None:
        return lambda: _call_jax(fixture, config), "jax", fallback_reason

    def call_pallas() -> jnp.ndarray:
        return pallas_fn(
            fixture.T,
            fixture.grid,
            fixture.detector,
            fixture.volume,
            step_size=config.step_size,
            n_steps=config.n_steps,
            unroll=config.unroll,
            gather_dtype=config.gather_dtype,
            det_grid=fixture.det_grid,
        )

    return call_pallas, "pallas", None


def _time_blocked_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    output = fn()
    _block_tree_ready(output)
    return time.perf_counter() - start, output


def _error_metrics(candidate: jnp.ndarray, oracle: jnp.ndarray) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(oracle, dtype=np.float64)
    diff = cand - ref
    denom = np.maximum(np.abs(ref), 1e-12)
    return {
        "finite": bool(np.isfinite(cand).all()),
        "max_abs_error": float(np.max(np.abs(diff))),
        "max_relative_error": float(np.max(np.abs(diff) / denom)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


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


def benchmark_backend(
    requested_backend: BackendName,
    fixture: ForwardProjectorFixture,
    config: ForwardProjectorBenchmarkConfig,
    *,
    oracle: jnp.ndarray | None = None,
) -> tuple[dict[str, Any], jnp.ndarray]:
    """Run first-call and warm-call timings for one requested backend."""
    call, actual_backend, fallback_reason = _make_backend_callable(requested_backend, fixture, config)
    first_seconds, first_output = _time_blocked_call(call)

    warm_seconds: list[float] = []
    warm_output = first_output
    for _ in range(max(0, int(config.warm_runs))):
        seconds, warm_output = _time_blocked_call(call)
        warm_seconds.append(float(seconds))

    reference = first_output if oracle is None else oracle
    timings = _timing_summary(warm_seconds)
    result = {
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "fallback_reason": fallback_reason,
        "eligible_for_speed_claim": bool(requested_backend == actual_backend),
        "first_call_seconds": float(first_seconds),
        **timings,
        **_error_metrics(warm_output, reference),
    }
    return result, first_output


def run_forward_projector_benchmark(
    config: ForwardProjectorBenchmarkConfig,
) -> dict[str, Any]:
    """Compare the current JAX projector with the requested Pallas path."""
    fixture = make_forward_projector_fixture(config)
    jax_result, oracle = benchmark_backend("jax", fixture, config, oracle=None)
    results = [jax_result]
    if config.include_pallas:
        pallas_result, _ = benchmark_backend("pallas", fixture, config, oracle=oracle)
        pallas_result["speedup_vs_jax_warm_median"] = (
            _speedup(
                baseline=jax_result["warm_seconds_median"],
                candidate=pallas_result["warm_seconds_median"],
            )
            if pallas_result["eligible_for_speed_claim"]
            else None
        )
        results.append(pallas_result)

    return {
        "benchmark": "forward_projector",
        "fixture_backend": "jax",
        "config": asdict(config),
        "fixture": _fixture_metadata(fixture, config),
        "device": _device_metadata(),
        "results": results,
    }


def run_forward_projector_suite(
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a named set of forward-projector cases and return one JSON-ready report."""
    case_metrics: list[dict[str, Any]] = []
    for case in suite_cases(name):
        config = replace(case.config, **(overrides or {}))
        metrics = run_forward_projector_benchmark(config)
        metrics["case_name"] = case.name
        case_metrics.append(metrics)
    return {
        "benchmark": "forward_projector_suite",
        "suite": name,
        "device": _device_metadata(),
        "summary": _suite_summary(case_metrics),
        "cases": case_metrics,
    }


def _suite_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    pallas_rows = [
        row
        for case in cases
        for row in case.get("results", [])
        if row.get("requested_backend") == "pallas"
    ]
    speedups = [
        float(row["speedup_vs_jax_warm_median"])
        for row in pallas_rows
        if row.get("speedup_vs_jax_warm_median") is not None
    ]
    return {
        "cases_total": len(cases),
        "cases_with_requested_pallas": len(pallas_rows),
        "cases_pallas_eligible": sum(1 for row in pallas_rows if row.get("eligible_for_speed_claim")),
        "cases_parity_passed": sum(1 for row in pallas_rows if _parity_passed(row)),
        "geomean_speedup_vs_jax_warm_median": _geomean(speedups),
        "worst_case_speedup_vs_jax_warm_median": min(speedups) if speedups else None,
        "best_case_speedup_vs_jax_warm_median": max(speedups) if speedups else None,
    }


def _parity_passed(row: dict[str, Any], *, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    return bool(
        row.get("finite")
        and row.get("max_abs_error") is not None
        and row.get("max_relative_error") is not None
        and float(row["max_abs_error"]) <= atol
        and float(row["max_relative_error"]) <= rtol
    )


def _geomean(values: list[float]) -> float | None:
    if not values or any(value <= 0.0 for value in values):
        return None
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def _speedup(*, baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or candidate <= 0:
        return None
    return float(baseline / candidate)


def write_benchmark_json(metrics: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
