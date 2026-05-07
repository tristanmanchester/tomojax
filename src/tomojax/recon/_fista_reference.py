"""JAX reference FISTA preview reconstruction."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import csv
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.forward import (
    ResidualFilterConfig,
    apply_residual_filter_schedule,
    project_parallel_reference,
    residual_loss,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tomojax.geometry import GeometryState


@dataclass(frozen=True)
class ReferenceFISTAConfig:
    iterations: int = 8
    step_size: float = 1e-2
    tv_weight: float = 0.0
    tv_delta: float = 1e-3
    residual_sigma: float = 1.0
    residual_delta: float = 1.0
    residual_filters: tuple[ResidualFilterConfig, ...] = (ResidualFilterConfig(),)
    non_negative: bool = True


@dataclass(frozen=True)
class ReferenceFISTATraceRow:
    iteration: int
    loss: float
    data_loss: float
    regulariser: float
    step_size: float
    wall_time_s: float
    backend: str


@dataclass(frozen=True)
class ReferenceFISTAResult:
    volume: jax.Array
    trace: tuple[ReferenceFISTATraceRow, ...]
    config: ReferenceFISTAConfig


def fista_reconstruct_reference(
    projections: jax.Array,
    geometry: GeometryState,
    *,
    initial_volume: jax.Array | None = None,
    volume_support: jax.Array | None = None,
    mask: jax.Array | None = None,
    config: ReferenceFISTAConfig | None = None,
) -> ReferenceFISTAResult:
    """Run a tiny JAX reference FISTA preview reconstruction."""
    cfg = config or ReferenceFISTAConfig()
    observed = jnp.asarray(projections, dtype=jnp.float32)
    if observed.ndim != 3:
        raise ValueError("projections must have shape (views, rows, cols)")

    if initial_volume is None:
        volume = jnp.zeros(
            (observed.shape[1], observed.shape[1], observed.shape[2]),
            dtype=jnp.float32,
        )
    else:
        volume = jnp.asarray(initial_volume, dtype=jnp.float32)
        if volume.ndim != 3:
            raise ValueError("initial_volume must be 3D")
    volume_shape = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    support = _support_or_none(volume_support, shape=volume_shape)
    volume = _project_constraints(volume, support=support, config=cfg)

    y = volume
    t = jnp.asarray(1.0, dtype=jnp.float32)
    step_size = jnp.asarray(cfg.step_size, dtype=jnp.float32)
    trace: list[ReferenceFISTATraceRow] = []
    start = time.perf_counter()

    for iteration in range(max(0, int(cfg.iterations))):
        (loss_value, (data_value, regulariser_value)), gradient = jax.value_and_grad(
            _objective,
            has_aux=True,
        )(y, observed, geometry, mask, cfg)
        candidate = y - step_size * gradient
        candidate = _project_constraints(candidate, support=support, config=cfg)
        next_t = (1.0 + jnp.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = (t - 1.0) / next_t
        y = _project_constraints(
            candidate + momentum * (candidate - volume),
            support=support,
            config=cfg,
        )
        volume = candidate
        t = next_t
        trace.append(
            ReferenceFISTATraceRow(
                iteration=iteration,
                loss=float(loss_value),
                data_loss=float(data_value),
                regulariser=float(regulariser_value),
                step_size=float(step_size),
                wall_time_s=time.perf_counter() - start,
                backend="core_trilinear_ray",
            )
        )

    return ReferenceFISTAResult(volume=volume.astype(jnp.float32), trace=tuple(trace), config=cfg)


def write_fista_trace_csv(result: ReferenceFISTAResult, path: str | Path) -> Path:
    """Write the FISTA trace artifact as CSV."""
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "loss",
                "data_loss",
                "regulariser",
                "step_size",
                "wall_time_s",
                "backend",
            ],
        )
        writer.writeheader()
        for row in result.trace:
            writer.writerow(
                {
                    "iteration": row.iteration,
                    "loss": row.loss,
                    "data_loss": row.data_loss,
                    "regulariser": row.regulariser,
                    "step_size": row.step_size,
                    "wall_time_s": row.wall_time_s,
                    "backend": row.backend,
                }
            )
    return output_path


def _objective(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    mask: jax.Array | None,
    config: ReferenceFISTAConfig,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    predicted = project_parallel_reference(volume, geometry)
    residual = apply_residual_filter_schedule(
        (predicted - observed) / jnp.asarray(config.residual_sigma, dtype=jnp.float32),
        config.residual_filters,
        mask=mask,
    ).residual
    data = residual_loss(
        residual,
        jnp.zeros_like(residual),
        mask=None,
        sigma=1.0,
        delta=config.residual_delta,
    ).loss
    regulariser = jnp.asarray(config.tv_weight, dtype=jnp.float32) * _smoothed_tv(
        volume,
        delta=config.tv_delta,
    )
    total = data + regulariser
    return total, (data, regulariser)


def _support_or_none(
    volume_support: jax.Array | None, *, shape: tuple[int, int, int]
) -> jax.Array | None:
    if volume_support is None:
        return None
    support = jnp.asarray(volume_support, dtype=jnp.float32)
    if support.shape != shape:
        raise ValueError(f"volume_support must have shape {shape!r}")
    return support


def _project_constraints(
    volume: jax.Array,
    *,
    support: jax.Array | None,
    config: ReferenceFISTAConfig,
) -> jax.Array:
    projected = jnp.asarray(volume, dtype=jnp.float32)
    if config.non_negative:
        projected = jnp.maximum(projected, 0.0)
    if support is not None:
        projected = projected * support
    return projected


def _smoothed_tv(volume: jax.Array, *, delta: float) -> jax.Array:
    vol = jnp.asarray(volume, dtype=jnp.float32)
    d = jnp.asarray(delta, dtype=jnp.float32)
    dx = vol[1:, :, :] - vol[:-1, :, :]
    dy = vol[:, 1:, :] - vol[:, :-1, :]
    dz = vol[:, :, 1:] - vol[:, :, :-1]
    return (
        jnp.sum(jnp.sqrt(dx * dx + d * d) - d)
        + jnp.sum(jnp.sqrt(dy * dy + d * d) - d)
        + jnp.sum(jnp.sqrt(dz * dz + d * d) - d)
    ) / jnp.asarray(vol.size, dtype=jnp.float32)
