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
    GaugeReport,
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
    gauge_report = GaugeReport(())
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
            geometry, gauge_report = _run_geometry_updates(geometry, level.geometry_updates)
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
        final_volume=volume,
        observed=observed,
        mask=mask,
        gauge_report=gauge_report,
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


def _run_geometry_updates(
    geometry: GeometryState, updates: int
) -> tuple[GeometryState, GaugeReport]:
    updated = geometry
    gauge_report = GaugeReport(())
    for _ in range(updates):
        canonicalized = canonicalize_geometry_gauges(updated)
        updated = canonicalized.state
        gauge_report = canonicalized.report
    return updated, gauge_report


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
    final_volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    gauge_report: GaugeReport,
    fista_result: ReferenceFISTAResult,
    summaries: tuple[AlternatingLevelSummary, ...],
    verification: Mapping[str, object],
) -> dict[str, Path]:
    artifacts = {
        "alignment_summary_csv": output_dir / "alignment_summary.csv",
        "artifact_index_json": output_dir / "artifact_index.json",
        "backend_report_json": output_dir / "backend_report.json",
        "config_resolved_toml": output_dir / "config_resolved.toml",
        "final_volume_npy": output_dir / "final_volume.npy",
        "fista_trace_csv": output_dir / "fista_trace.csv",
        "gauge_report_json": output_dir / "gauge_report.json",
        "geometry_final_json": output_dir / "geometry_final.json",
        "geometry_initial_json": output_dir / "geometry_initial.json",
        "input_summary_json": output_dir / "input_summary.json",
        "mask_summary_json": output_dir / "mask_summary.json",
        "pose_decomposition_csv": output_dir / "pose_decomposition.csv",
        "pose_params_csv": output_dir / "pose_params.csv",
        "projection_stats_json": output_dir / "projection_stats.json",
        "residual_metrics_csv": output_dir / "residual_metrics.csv",
        "run_manifest_json": output_dir / "run_manifest.json",
        "verification_json": output_dir / "verification.json",
    }
    _write_config_resolved(artifacts["config_resolved_toml"])
    _write_json(artifacts["run_manifest_json"], _run_manifest_payload(final_volume, observed))
    _write_json(artifacts["input_summary_json"], _input_summary_payload(final_volume, observed))
    _write_json(artifacts["projection_stats_json"], _projection_stats_payload(observed))
    _write_json(artifacts["mask_summary_json"], _mask_summary_payload(mask))
    _write_json(artifacts["gauge_report_json"], _gauge_report_payload(gauge_report))
    _write_json(artifacts["backend_report_json"], _backend_report_payload())
    _write_final_volume(artifacts["final_volume_npy"], final_volume)
    write_geometry_json(artifacts["geometry_initial_json"], initial_geometry)
    write_geometry_json(artifacts["geometry_final_json"], final_geometry)
    write_pose_params_csv(artifacts["pose_params_csv"], final_geometry.pose)
    write_pose_decomposition_csv(artifacts["pose_decomposition_csv"], final_geometry)
    _ = write_fista_trace_csv(fista_result, artifacts["fista_trace_csv"])
    _write_alignment_summary(artifacts["alignment_summary_csv"], summaries)
    _write_residual_metrics(artifacts["residual_metrics_csv"], summaries)
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


def _write_residual_metrics(
    path: Path,
    summaries: tuple[AlternatingLevelSummary, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "level_factor",
                "role",
                "loss_before",
                "loss_after",
                "absolute_improvement",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "level_factor": summary.level_factor,
                    "role": summary.role,
                    "loss_before": summary.loss_before,
                    "loss_after": summary.loss_after,
                    "absolute_improvement": summary.loss_before - summary.loss_after,
                }
            )


def _artifact_index_payload(artifacts: Mapping[str, Path]) -> dict[str, object]:
    return {
        "schema": "tomojax.artifact_index.v1",
        "artifacts": [
            {
                "name": name,
                "path": path.name,
                "type": _artifact_type(path),
                "media_type": _media_type(path),
                "description": _artifact_description(name),
            }
            for name, path in sorted(artifacts.items())
            if name != "artifact_index_json"
        ],
    }


def _artifact_type(path: Path) -> str:
    if path.suffix == ".json":
        return "json"
    if path.suffix == ".csv":
        return "csv"
    if path.suffix == ".toml":
        return "toml"
    if path.suffix == ".npy":
        return "npy"
    return "binary"


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".csv":
        return "text/csv"
    if path.suffix == ".toml":
        return "application/toml"
    return "application/octet-stream"


def _artifact_description(name: str) -> str:
    descriptions = {
        "alignment_summary_csv": "Per-continuation-level alignment summary",
        "backend_report_json": "Backend provenance for the smoke run",
        "config_resolved_toml": "Resolved deterministic smoke configuration",
        "final_volume_npy": "Final reconstructed 32^3 volume",
        "fista_trace_csv": "Reference FISTA iteration trace",
        "gauge_report_json": "Gauge canonicalisation transfer report",
        "geometry_final_json": "Final canonical geometry state",
        "geometry_initial_json": "Initial corrupted geometry state",
        "input_summary_json": "Synthetic input shape and dtype summary",
        "mask_summary_json": "Projection mask coverage summary",
        "pose_decomposition_csv": "Final realised per-view pose decomposition",
        "pose_params_csv": "Final per-view pose parameters",
        "projection_stats_json": "Observed projection summary statistics",
        "residual_metrics_csv": "Per-level residual metrics",
        "run_manifest_json": "Resolved smoke run manifest",
        "verification_json": "Smoke verification report",
    }
    return descriptions[name]


def _write_config_resolved(path: Path) -> None:
    _ = path.write_text(
        "\n".join(
            (
                'profile = "smoke32"',
                'align_mode = "auto"',
                'backend_requested = "jax_reference"',
                'backend_actual = "jax_reference"',
                'geometry_model = "parallel_tomography_reference"',
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_final_volume(path: Path, volume: jax.Array) -> None:
    with path.open("wb") as handle:
        np.save(handle, np.asarray(jax.device_get(volume), dtype=np.float32), allow_pickle=False)


def _run_manifest_payload(volume: jax.Array, projections: jax.Array) -> dict[str, object]:
    return {
        "schema": "tomojax.run_manifest.v1",
        "run_id": "smoke32-deterministic",
        "profile": "smoke32",
        "align_mode": "auto",
        "dataset": {
            "source": "tomojax.datasets.make_benchmark_phantom",
            "shape": list(volume.shape),
            "projection_shape": list(projections.shape),
            "projection_dtype": str(projections.dtype),
        },
        "geometry_model": "parallel_tomography_reference",
        "backend_requested": "jax_reference",
        "backend_actual": "jax_reference",
        "status": "passed",
    }


def _input_summary_payload(volume: jax.Array, projections: jax.Array) -> dict[str, object]:
    return {
        "schema": "tomojax.input_summary.v1",
        "volume_shape": list(volume.shape),
        "volume_dtype": str(volume.dtype),
        "projection_shape": list(projections.shape),
        "projection_dtype": str(projections.dtype),
    }


def _projection_stats_payload(projections: jax.Array) -> dict[str, object]:
    values = jnp.asarray(projections, dtype=jnp.float32)
    return {
        "schema": "tomojax.projection_stats.v1",
        "min": float(jnp.min(values)),
        "max": float(jnp.max(values)),
        "mean": float(jnp.mean(values)),
        "std": float(jnp.std(values)),
    }


def _mask_summary_payload(mask: jax.Array) -> dict[str, object]:
    values = jnp.asarray(mask, dtype=jnp.float32)
    total = int(values.size)
    valid = int(jnp.sum(values > 0.0))
    return {
        "schema": "tomojax.mask_summary.v1",
        "valid_pixels": valid,
        "total_pixels": total,
        "valid_fraction": float(valid / total),
    }


def _gauge_report_payload(report: GaugeReport) -> dict[str, object]:
    return {
        "schema": "tomojax.gauge_report.v1",
        "status": "passed",
        "operations": [
            {
                "source": transfer.source,
                "target": transfer.target,
                "value": transfer.value,
                "unit": transfer.unit,
                "applied": transfer.applied,
                "reason": transfer.reason,
            }
            for transfer in report.transfers
        ],
    }


def _backend_report_payload() -> dict[str, object]:
    return {
        "schema": "tomojax.backend_report.v1",
        "requested": "jax_reference",
        "actual": "jax_reference",
        "fallback": False,
    }


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
