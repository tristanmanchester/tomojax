from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align.geometry.parametrizations import se3_from_5d
from tomojax.align.objectives.fixed_volume import project_and_score_stack
from tomojax.align.objectives.loss_adapters import build_loss_adapter
from tomojax.align.objectives.loss_specs import L2LossSpec
from tomojax.bench.forward_projector import _device_metadata
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device


@dataclass(frozen=True)
class AlignmentObjectiveBenchmarkConfig:
    nx: int = 24
    ny: int = 24
    nz: int = 24
    nu: int = 24
    nv: int = 24
    n_views: int = 24
    seed: int = 0
    warm_runs: int = 7
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "bf16"


@dataclass(frozen=True)
class AlignmentObjectiveFixture:
    grid: Grid
    detector: Detector
    nominal_pose_stack: jnp.ndarray
    volume: jnp.ndarray
    target: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]
    initial_params5: jnp.ndarray


def make_alignment_objective_fixture(
    config: AlignmentObjectiveBenchmarkConfig,
) -> AlignmentObjectiveFixture:
    """Build a deterministic full 5-DOF pose objective fixture."""
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
    thetas = np.linspace(0.0, 180.0, int(config.n_views), endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)
    nominal_pose_stack = jnp.asarray(
        np.stack(
            [
                np.asarray(geometry.pose_for_view(i), dtype=np.float32)
                for i in range(int(config.n_views))
            ]
        ),
        dtype=jnp.float32,
    )
    rng = np.random.default_rng(int(config.seed))
    volume = jnp.asarray(
        np.abs(rng.normal(size=(grid.nx, grid.ny, grid.nz)).astype(np.float32)),
        dtype=jnp.float32,
    )
    truth_params = np.zeros((int(config.n_views), 5), dtype=np.float32)
    truth_params[:, 0] = rng.uniform(-2.0, 2.0, size=int(config.n_views)) * np.float32(
        math.pi / 180.0
    )
    truth_params[:, 1] = rng.uniform(-2.0, 2.0, size=int(config.n_views)) * np.float32(
        math.pi / 180.0
    )
    truth_params[:, 2] = rng.uniform(-1.0, 1.0, size=int(config.n_views)) * np.float32(
        math.pi / 180.0
    )
    truth_params[:, 3] = rng.uniform(-1.0, 1.0, size=int(config.n_views))
    truth_params[:, 4] = rng.uniform(-1.0, 1.0, size=int(config.n_views))
    truth_params5 = jnp.asarray(truth_params, dtype=jnp.float32)
    det_grid = get_detector_grid_device(detector)

    def project_one(T: jnp.ndarray) -> jnp.ndarray:
        return forward_project_view_T(
            T,
            grid,
            detector,
            volume,
            use_checkpoint=bool(config.checkpoint_projector),
            unroll=int(config.projector_unroll),
            gather_dtype=str(config.gather_dtype),
            det_grid=det_grid,
        )

    target = jax.jit(jax.vmap(project_one))(
        nominal_pose_stack @ jax.vmap(se3_from_5d)(truth_params5)
    )
    jax.block_until_ready(target)
    return AlignmentObjectiveFixture(
        grid=grid,
        detector=detector,
        nominal_pose_stack=nominal_pose_stack,
        volume=volume,
        target=jax.lax.stop_gradient(target),
        det_grid=det_grid,
        initial_params5=jnp.zeros_like(truth_params5),
    )


def _make_value_and_grad_callable(
    fixture: AlignmentObjectiveFixture,
    config: AlignmentObjectiveBenchmarkConfig,
) -> Callable[[], tuple[jnp.ndarray, jnp.ndarray]]:
    loss_adapter = build_loss_adapter(L2LossSpec(), fixture.target)

    def loss(params5: jnp.ndarray) -> jnp.ndarray:
        pose_stack = fixture.nominal_pose_stack @ jax.vmap(se3_from_5d)(params5)
        return project_and_score_stack(
            pose_stack=pose_stack,
            grid=fixture.grid,
            detector=fixture.detector,
            volume=fixture.volume,
            det_grid=fixture.det_grid,
            targets=fixture.target,
            loss_adapter=loss_adapter,
            views_per_batch=int(config.views_per_batch),
            projector_unroll=int(config.projector_unroll),
            checkpoint_projector=bool(config.checkpoint_projector),
            gather_dtype=str(config.gather_dtype),
            view_indices=jnp.arange(int(config.n_views), dtype=jnp.int32),
        )

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    return lambda: value_and_grad(fixture.initial_params5)


def _time_blocked_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    output = fn()
    jax.block_until_ready(output)
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


def benchmark_alignment_objective_variant(
    name: str,
    config: AlignmentObjectiveBenchmarkConfig,
) -> dict[str, Any]:
    fixture = make_alignment_objective_fixture(config)
    call = _make_value_and_grad_callable(fixture, config)
    first_seconds, first_output = _time_blocked_call(call)
    warm_seconds: list[float] = []
    warm_output = first_output
    for _ in range(max(0, int(config.warm_runs))):
        seconds, warm_output = _time_blocked_call(call)
        warm_seconds.append(float(seconds))
    value, grad = warm_output
    grad_norm = jnp.linalg.norm(grad)
    return {
        "benchmark": "alignment_objective_value_and_grad",
        "case_name": name,
        "api_surface": "internal_fixed_volume_alignment_objective",
        "config": asdict(config),
        "device": _device_metadata(),
        "first_call_seconds": float(first_seconds),
        **_timing_summary(warm_seconds),
        "value": float(np.asarray(value, dtype=np.float64)),
        "value_finite": bool(np.isfinite(np.asarray(value))),
        "grad_norm": float(np.asarray(grad_norm, dtype=np.float64)),
        "grad_finite": bool(np.all(np.isfinite(np.asarray(grad)))),
        "grad_shape": [int(dim) for dim in grad.shape],
    }


def run_alignment_objective_suite(
    name: str = "alignment_objective",
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name != "alignment_objective":
        raise ValueError("alignment objective suite must be 'alignment_objective'")
    base = AlignmentObjectiveBenchmarkConfig()
    updates = dict(overrides or {})
    checkpointed = replace(base, **updates, checkpoint_projector=True)
    no_checkpoint = replace(base, **updates, checkpoint_projector=False)
    cases = [
        benchmark_alignment_objective_variant("checkpointed", checkpointed),
        benchmark_alignment_objective_variant("no_checkpoint", no_checkpoint),
    ]
    baseline = cases[0]["warm_seconds_median"]
    candidate = cases[1]["warm_seconds_median"]
    speedup = (
        float(baseline) / float(candidate)
        if baseline is not None and candidate is not None and float(candidate) > 0.0
        else None
    )
    return {
        "benchmark": "alignment_objective_suite",
        "suite": name,
        "summary": {
            "checkpointed_warm_seconds_median": baseline,
            "no_checkpoint_warm_seconds_median": candidate,
            "no_checkpoint_speedup_vs_checkpointed": speedup,
            "cases_total": len(cases),
        },
        "device": _device_metadata(),
        "cases": cases,
    }


def write_benchmark_json(metrics: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    metrics = run_alignment_objective_suite()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
