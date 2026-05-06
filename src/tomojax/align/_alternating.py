"""Small v2 alternating alignment smoke runner."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._continuation import (
    ContinuationLevel,
    ContinuationSchedule,
    reference_continuation_schedule,
)
from tomojax.datasets import make_benchmark_phantom
from tomojax.forward import project_parallel_reference, residual_loss
from tomojax.geometry import (
    GeometryState,
    canonicalize_geometry_gauges,
    write_geometry_json,
    write_pose_decomposition_csv,
    write_pose_params_csv,
)
from tomojax.recon import (
    ReferenceFISTAConfig,
    ReferenceFISTAResult,
    fista_reconstruct_reference,
    write_fista_trace_csv,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class AlternatingSmokeConfig:
    """Configuration for the deterministic 32^3 alternating smoke run."""

    seed: int = 17
    size: int = 32
    n_views: int = 4
    schedule: ContinuationSchedule | None = None
    verification_loss_tolerance: float = 1.0e-5


@dataclass(frozen=True)
class AlternatingLevelSummary:
    """Per-level alternating smoke summary."""

    level_factor: int
    role: str
    reconstruction_iterations: int
    geometry_updates: int
    loss_before: float
    loss_after: float
    verified: bool
    skipped_geometry: bool


@dataclass(frozen=True)
class AlternatingSmokeResult:
    """Result payload for the deterministic alternating smoke run."""

    final_volume: jax.Array
    initial_geometry: GeometryState
    final_geometry: GeometryState
    levels: tuple[AlternatingLevelSummary, ...]
    verification: Mapping[str, object]
    artifacts: Mapping[str, Path]


def run_alternating_solver_smoke(
    output_dir: str | Path,
    *,
    config: AlternatingSmokeConfig | None = None,
) -> AlternatingSmokeResult:
    """Run the smallest deterministic v2 alternating solver smoke slice."""
    cfg = config or AlternatingSmokeConfig()
    schedule = cfg.schedule or reference_continuation_schedule()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = jnp.asarray(make_benchmark_phantom(cfg.size, seed=cfg.seed), dtype=jnp.float32)
    initial_geometry = _synthetic_initial_geometry(cfg.n_views)
    observed = project_parallel_reference(truth, initial_geometry)
    mask = jnp.ones_like(observed, dtype=jnp.float32)

    geometry = initial_geometry
    volume: jax.Array | None = None
    summaries: list[AlternatingLevelSummary] = []
    fista_result: ReferenceFISTAResult | None = None
    coarse_verified = False

    initial_loss = _projection_loss(truth, observed, geometry, mask, schedule.levels[0])
    for level in schedule.levels:
        fista_result = fista_reconstruct_reference(
            observed,
            geometry,
            initial_volume=volume,
            mask=mask,
            config=ReferenceFISTAConfig(
                iterations=level.reconstruction_iterations,
                step_size=2.0e-3,
                tv_weight=0.0,
                residual_sigma=level.residual_sigma,
                residual_delta=level.residual_delta,
            ),
        )
        volume = jax.lax.stop_gradient(fista_result.volume)
        loss_before = _projection_loss(volume, observed, geometry, mask, level)
        skipped_geometry = level.geometry_updates == 0
        if level.geometry_updates > 0:
            geometry = _run_geometry_updates(geometry, level.geometry_updates)
        loss_after = _projection_loss(volume, observed, geometry, mask, level)
        verified = bool(loss_after <= loss_before + cfg.verification_loss_tolerance)
        coarse_verified = coarse_verified or (
            level.role == "preview" and level.skip_finer_if_verified and verified
        )
        summaries.append(
            AlternatingLevelSummary(
                level_factor=level.level_factor,
                role=level.role,
                reconstruction_iterations=level.reconstruction_iterations,
                geometry_updates=level.geometry_updates,
                loss_before=loss_before,
                loss_after=loss_after,
                verified=verified,
                skipped_geometry=skipped_geometry,
            )
        )

    if volume is None or fista_result is None:
        raise RuntimeError("continuation schedule produced no reconstruction")

    final_loss = summaries[-1].loss_after
    artifacts = _write_artifacts(
        out_dir,
        initial_geometry=initial_geometry,
        final_geometry=geometry,
        fista_result=fista_result,
        summaries=tuple(summaries),
        verification=_verification_payload(
            cfg=cfg,
            schedule=schedule,
            initial_loss=initial_loss,
            final_loss=final_loss,
            coarse_verified=coarse_verified,
            summaries=tuple(summaries),
        ),
    )
    verification = json.loads(artifacts["verification_json"].read_text(encoding="utf-8"))
    return AlternatingSmokeResult(
        final_volume=volume,
        initial_geometry=initial_geometry,
        final_geometry=geometry,
        levels=tuple(summaries),
        verification=verification,
        artifacts=artifacts,
    )


def _synthetic_initial_geometry(n_views: int) -> GeometryState:
    base = GeometryState.zeros(n_views)
    span = np.linspace(-1.0, 1.0, num=n_views, dtype=np.float64)
    return GeometryState(
        setup=base.setup,
        pose=base.pose.with_updates(
            phi_residual_rad=0.02 + 0.01 * span,
            dx_px=1.25 + 0.5 * span,
            dz_px=0.2 * span,
        ),
    )


def _run_geometry_updates(geometry: GeometryState, updates: int) -> GeometryState:
    updated = geometry
    for _ in range(updates):
        updated = canonicalize_geometry_gauges(updated).state
    return updated


def _projection_loss(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    mask: jax.Array,
    level: ContinuationLevel,
) -> float:
    predicted = project_parallel_reference(volume, geometry)
    result = residual_loss(
        predicted,
        observed,
        mask=mask,
        sigma=level.residual_sigma,
        delta=level.residual_delta,
    )
    return float(result.loss)


def _verification_payload(
    *,
    cfg: AlternatingSmokeConfig,
    schedule: ContinuationSchedule,
    initial_loss: float,
    final_loss: float,
    coarse_verified: bool,
    summaries: tuple[AlternatingLevelSummary, ...],
) -> dict[str, object]:
    return {
        "schema": "tomojax.alternating_smoke.verification.v1",
        "seed": cfg.seed,
        "size": cfg.size,
        "n_views": cfg.n_views,
        "schedule": schedule.name,
        "level_factors": list(schedule.level_factors),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "coarse_verified": coarse_verified,
        "level1_geometry_skipped": coarse_verified,
        "levels": [_summary_payload(summary) for summary in summaries],
    }


def _summary_payload(summary: AlternatingLevelSummary) -> dict[str, object]:
    return {
        "level_factor": summary.level_factor,
        "role": summary.role,
        "reconstruction_iterations": summary.reconstruction_iterations,
        "geometry_updates": summary.geometry_updates,
        "loss_before": summary.loss_before,
        "loss_after": summary.loss_after,
        "verified": summary.verified,
        "skipped_geometry": summary.skipped_geometry,
    }


def _write_artifacts(
    output_dir: Path,
    *,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    fista_result: ReferenceFISTAResult,
    summaries: tuple[AlternatingLevelSummary, ...],
    verification: Mapping[str, object],
) -> dict[str, Path]:
    artifacts = {
        "geometry_initial_json": output_dir / "geometry_initial.json",
        "geometry_final_json": output_dir / "geometry_final.json",
        "pose_params_csv": output_dir / "pose_params.csv",
        "pose_decomposition_csv": output_dir / "pose_decomposition.csv",
        "fista_trace_csv": output_dir / "fista_trace.csv",
        "alignment_summary_csv": output_dir / "alignment_summary.csv",
        "verification_json": output_dir / "verification.json",
        "artifact_index_json": output_dir / "artifact_index.json",
    }
    write_geometry_json(artifacts["geometry_initial_json"], initial_geometry)
    write_geometry_json(artifacts["geometry_final_json"], final_geometry)
    write_pose_params_csv(artifacts["pose_params_csv"], final_geometry.pose)
    write_pose_decomposition_csv(artifacts["pose_decomposition_csv"], final_geometry)
    _ = write_fista_trace_csv(fista_result, artifacts["fista_trace_csv"])
    _write_alignment_summary(artifacts["alignment_summary_csv"], summaries)
    _write_json(artifacts["verification_json"], verification)
    _write_json(artifacts["artifact_index_json"], _artifact_index_payload(artifacts))
    return artifacts


def _write_alignment_summary(
    path: Path,
    summaries: tuple[AlternatingLevelSummary, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "level_factor",
                "role",
                "reconstruction_iterations",
                "geometry_updates",
                "loss_before",
                "loss_after",
                "verified",
                "skipped_geometry",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_summary_payload(summary))


def _artifact_index_payload(artifacts: Mapping[str, Path]) -> dict[str, object]:
    return {
        "schema": "tomojax.artifact_index.v1",
        "artifacts": [
            {
                "name": name,
                "path": path.name,
                "media_type": _media_type(path),
            }
            for name, path in sorted(artifacts.items())
            if name != "artifact_index_json"
        ],
    }


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".csv":
        return "text/csv"
    return "application/octet-stream"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "AlternatingLevelSummary",
    "AlternatingSmokeConfig",
    "AlternatingSmokeResult",
    "run_alternating_solver_smoke",
]
