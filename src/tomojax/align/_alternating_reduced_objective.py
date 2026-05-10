"""Reduced-objective det_u refresh probes for alternating diagnostics."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from tomojax.align._alternating_heldout import _projection_loss
from tomojax.geometry import GeometryState
from tomojax.recon import (
    ReferenceFISTAConfig,
    centered_volume_support,
    fista_reconstruct_reference,
    reconstruct_average_reference,
    reconstruct_backprojection_reference,
    reference_fista_returned_quality,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tomojax.align._continuation import ContinuationLevel
    from tomojax.align._joint_schur_lm import JointSchurLMResult
    from tomojax.recon import ReferenceFISTATraceRow


@dataclass(frozen=True)
class ReducedObjectiveArtifacts:
    """Paths written by the reduced-objective probe."""

    csv_path: Path
    summary_path: Path
    curves_png_path: Path
    volume_sources_path: Path
    inner_solve_quality_path: Path


def write_reduced_objective_artifacts(
    artifacts: ReducedObjectiveArtifacts,
    *,
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    observed: jax.Array,
    alignment_mask: jax.Array,
    projection_valid_mask: jax.Array,
    level: ContinuationLevel,
    sigma: float,
    loss_mode: str,
    schur_result: JointSchurLMResult | None,
    preview_volume_support: str,
    preview_initialization: str,
    preview_tv_scale: float,
    preview_center_l2_weight: float,
    preview_views_per_batch: int,
) -> None:
    """Write reduced-objective probe CSV, summary, plot, and volume-source metadata."""
    candidates = _candidate_geometries(
        true_geometry=true_geometry,
        initial_geometry=initial_geometry,
        final_geometry=final_geometry,
        schur_result=schur_result,
    )
    rows = _probe_rows(
        candidates,
        observed=observed,
        alignment_mask=alignment_mask,
        projection_valid_mask=projection_valid_mask,
        level=level,
        sigma=sigma,
        loss_mode=loss_mode,
        preview_volume_support=preview_volume_support,
        preview_initialization=preview_initialization,
        preview_tv_scale=preview_tv_scale,
        preview_center_l2_weight=preview_center_l2_weight,
        preview_views_per_batch=preview_views_per_batch,
    )
    _write_probe_csv(artifacts.csv_path, rows)
    _write_json(artifacts.summary_path, _summary_payload(rows))
    _write_json(artifacts.volume_sources_path, _volume_sources_payload(rows))
    _write_json(artifacts.inner_solve_quality_path, _inner_solve_quality_payload(rows))
    _write_curves_png(artifacts.curves_png_path, rows)


def _candidate_geometries(
    *,
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    schur_result: JointSchurLMResult | None,
) -> tuple[tuple[str, GeometryState], ...]:
    candidates: list[tuple[str, GeometryState]] = [
        ("current_final", final_geometry),
        ("initial_corrupted", initial_geometry),
        ("true_synthetic_diagnostic", true_geometry),
    ]
    if schur_result is not None and schur_result.diagnostics.setup_update_by_parameter:
        step = float(schur_result.diagnostics.setup_update_by_parameter[0])
        current = float(final_geometry.setup.det_u_px.value)
        candidates.extend(
            (
                f"schur_backtrack_{scale:g}",
                _with_det_u(final_geometry, current + step * scale),
            )
            for scale in (1.0, 0.5, 0.25)
        )
    return tuple(_deduplicate_candidates(candidates))


def _deduplicate_candidates(
    candidates: list[tuple[str, GeometryState]],
) -> list[tuple[str, GeometryState]]:
    seen: set[float] = set()
    unique: list[tuple[str, GeometryState]] = []
    for name, geometry in candidates:
        det_u = round(float(geometry.setup.det_u_px.value), 6)
        if det_u in seen:
            continue
        seen.add(det_u)
        unique.append((name, geometry))
    return unique


def _probe_rows(
    candidates: tuple[tuple[str, GeometryState], ...],
    *,
    observed: jax.Array,
    alignment_mask: jax.Array,
    projection_valid_mask: jax.Array,
    level: ContinuationLevel,
    sigma: float,
    loss_mode: str,
    preview_volume_support: str,
    preview_initialization: str,
    preview_tv_scale: float,
    preview_center_l2_weight: float,
    preview_views_per_batch: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    support = _support(preview_volume_support, observed)
    config = ReferenceFISTAConfig(
        iterations=max(1, int(level.reconstruction_iterations)),
        step_size=_production_preview_step_size(observed),
        tv_weight=level.reconstruction_tv_weight * max(float(preview_tv_scale), 0.0),
        residual_sigma=sigma,
        residual_delta=level.residual_delta,
        residual_loss_mode=loss_mode,
        residual_filters=level.residual_filters,
        non_negative=True,
        center_l2_weight=max(float(preview_center_l2_weight), 0.0),
        views_per_batch=int(preview_views_per_batch),
    )
    for name, geometry in candidates:
        initial_volume, init_source = _initial_volume(
            preview_initialization,
            observed=observed,
            geometry=geometry,
            support=support,
        )
        result = fista_reconstruct_reference(
            observed,
            geometry,
            initial_volume=initial_volume,
            volume_support=support,
            mask=projection_valid_mask,
            config=config,
        )
        quality = reference_fista_returned_quality(
            result,
            observed,
            geometry,
            volume_support=support,
            mask=projection_valid_mask,
        )
        trace_last = result.trace[-1] if result.trace else None
        trace_first = result.trace[0] if result.trace else None
        alignment_loss = _projection_loss(
            result.volume,
            observed,
            geometry,
            alignment_mask,
            level,
            sigma=sigma,
            loss_mode=loss_mode,
        )
        valid_loss = _projection_loss(
            result.volume,
            observed,
            geometry,
            projection_valid_mask,
            level,
            sigma=sigma,
            loss_mode=loss_mode,
        )
        rows.append(
            {
                "candidate": name,
                "det_u_px": float(geometry.setup.det_u_px.value),
                "fista_iterations": int(config.iterations),
                "fista_step_size": float(config.step_size),
                "fista_initialization": init_source,
                "support_source": preview_volume_support,
                "fista_mask_role": "projection_valid_mask",
                "alignment_loss_mask_role": "alignment_loss_mask",
                "valid_loss_mask_role": "projection_valid_mask",
                "loss_normalisation": quality.loss_normalisation,
                "fista_loss_first": None if trace_first is None else trace_first.loss,
                "fista_loss_last": None if trace_last is None else trace_last.loss,
                "fista_data_loss_last": None if trace_last is None else trace_last.data_loss,
                "fista_regulariser_last": None if trace_last is None else trace_last.regulariser,
                "returned_candidate_loss": quality.returned_candidate_loss,
                "returned_data_loss": quality.returned_data_loss,
                "returned_regularizer": quality.returned_regularizer,
                "stationarity_proxy_trace_delta": _trace_delta(result.trace),
                "prox_gradient_norm": quality.projected_gradient_norm,
                "prox_gradient_note": "projected_gradient_mapping_at_returned_volume",
                "volume_rms": quality.volume_rms,
                "support_mass_fraction": (
                    "" if quality.support_mass_fraction is None else quality.support_mass_fraction
                ),
                "alignment_projection_loss": alignment_loss,
                "valid_projection_loss": valid_loss,
                "loss_mode": loss_mode,
                "residual_sigma": float(sigma),
            }
        )
    return rows


def _production_preview_step_size(observed: jax.Array) -> float:
    size = int(observed.shape[1])
    if size < 64:
        return 2.0e-3
    return 100.0 * max(float(size), 1.0) / 128.0


def _initial_volume(
    initialization: str,
    *,
    observed: jax.Array,
    geometry: GeometryState,
    support: jax.Array | None,
) -> tuple[jax.Array, str]:
    size = int(observed.shape[1])
    if initialization == "backprojection":
        volume = (
            reconstruct_average_reference(observed, depth=size)
            if size < 64
            else reconstruct_backprojection_reference(observed, geometry, depth=size)
        )
        return _apply_support(volume, support), "backprojection"
    if initialization == "average_projection":
        return _apply_support(reconstruct_average_reference(observed, depth=size), support), (
            "average_projection"
        )
    if initialization == "constant":
        fill = float(jnp.mean(jnp.asarray(observed, dtype=jnp.float32))) / max(size, 1)
        return _apply_support(jnp.full((size, size, size), fill, dtype=jnp.float32), support), (
            "constant"
        )
    return jnp.zeros((size, size, size), dtype=jnp.float32), "zero"


def _apply_support(volume: jax.Array, support: jax.Array | None) -> jax.Array:
    if support is None:
        return volume
    return jnp.asarray(volume, dtype=jnp.float32) * jnp.asarray(support, dtype=jnp.float32)


def _support(kind: str, observed: jax.Array) -> jax.Array | None:
    if kind == "none":
        return None
    shape = (int(observed.shape[1]), int(observed.shape[1]), int(observed.shape[2]))
    if kind == "cylindrical":
        return centered_volume_support(shape, kind="cylindrical")
    if kind == "spherical":
        return centered_volume_support(shape, kind="spherical")
    return None


def _trace_delta(trace: tuple[ReferenceFISTATraceRow, ...]) -> float | None:
    if len(trace) < 2:
        return None
    first = trace[0]
    last = trace[-1]
    return float(abs(float(last.loss) - float(first.loss)))


def _with_det_u(geometry: GeometryState, det_u_px: float) -> GeometryState:
    setup = geometry.setup.replace_parameter(
        "det_u_px",
        geometry.setup.det_u_px.with_value(float(det_u_px)),
    )
    return GeometryState(setup=setup, pose=geometry.pose, acquisition=geometry.acquisition)


def _summary_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "schema": "tomojax.reduced_objective_summary.v1",
            "status": "empty",
        }
    best_alignment = min(rows, key=lambda row: _float_value(row, "alignment_projection_loss"))
    best_valid = min(rows, key=lambda row: _float_value(row, "valid_projection_loss"))
    return {
        "schema": "tomojax.reduced_objective_summary.v1",
        "status": "inner_solve_underfit" if _is_underfit(rows) else "recorded",
        "inner_solve_classification": (
            "inner_solve_underfit" if _is_underfit(rows) else "inner_solve_informative"
        ),
        "candidate_count": len(rows),
        "best_alignment_candidate": str(best_alignment["candidate"]),
        "best_alignment_det_u_px": _float_value(best_alignment, "det_u_px"),
        "best_alignment_loss": _float_value(best_alignment, "alignment_projection_loss"),
        "best_valid_candidate": str(best_valid["candidate"]),
        "best_valid_det_u_px": _float_value(best_valid, "det_u_px"),
        "best_valid_loss": _float_value(best_valid, "valid_projection_loss"),
        "purpose": "diagnostic_reduced_objective_probe_not_production_center_search",
    }


def _volume_sources_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "tomojax.reduced_objective_volume_sources.v1",
        "status": "recorded",
        "fista_mask_role": "projection_valid_mask",
        "volume_saved": False,
        "candidates": [
            {
                "candidate": str(row["candidate"]),
                "det_u_px": _float_value(row, "det_u_px"),
                "fista_iterations": _int_value(row, "fista_iterations"),
                "fista_step_size": _float_value(row, "fista_step_size"),
                "initializer": str(row["fista_initialization"]),
                "support_source": str(row["support_source"]),
            }
            for row in rows
        ],
    }


def _inner_solve_quality_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "tomojax.reduced_objective_inner_solve_quality.v1",
        "status": "inner_solve_underfit" if _is_underfit(rows) else "recorded",
        "underfit_reasons": _underfit_reasons(rows),
        "candidate_count": len(rows),
        "max_volume_rms": max((_float_value(row, "volume_rms") for row in rows), default=0.0),
        "max_prox_gradient_norm": max(
            (_float_value(row, "prox_gradient_norm") for row in rows),
            default=0.0,
        ),
        "min_returned_candidate_loss": min(
            (_float_value(row, "returned_candidate_loss") for row in rows),
            default=0.0,
        ),
        "loss_normalisation": "full_projection_array_size",
        "rows": [
            {
                "candidate": str(row["candidate"]),
                "det_u_px": _float_value(row, "det_u_px"),
                "returned_candidate_loss": _float_value(row, "returned_candidate_loss"),
                "returned_data_loss": _float_value(row, "returned_data_loss"),
                "returned_regularizer": _float_value(row, "returned_regularizer"),
                "prox_gradient_norm": _float_value(row, "prox_gradient_norm"),
                "volume_rms": _float_value(row, "volume_rms"),
                "fista_iterations": _int_value(row, "fista_iterations"),
                "fista_step_size": _float_value(row, "fista_step_size"),
                "fista_initialization": str(row["fista_initialization"]),
            }
            for row in rows
        ],
    }


def _is_underfit(rows: list[dict[str, object]]) -> bool:
    return bool(_underfit_reasons(rows))


def _underfit_reasons(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return ["no_candidates"]
    reasons: list[str] = []
    max_volume_rms = max(_float_value(row, "volume_rms") for row in rows)
    max_trace_delta = max(
        (
            0.0
            if row["stationarity_proxy_trace_delta"] in (None, "")
            else _float_value(row, "stationarity_proxy_trace_delta")
        )
        for row in rows
    )
    min_prox_gradient = min(_float_value(row, "prox_gradient_norm") for row in rows)
    if max_volume_rms <= 1.0e-6:
        reasons.append("candidate_volumes_near_zero")
    if max_trace_delta <= 1.0e-8 and min_prox_gradient > 1.0e-6:
        reasons.append("no_loss_progress_with_nonzero_projected_gradient")
    return reasons


def _write_probe_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate",
        "det_u_px",
        "fista_iterations",
        "fista_step_size",
        "fista_initialization",
        "support_source",
        "fista_mask_role",
        "alignment_loss_mask_role",
        "valid_loss_mask_role",
        "loss_normalisation",
        "fista_loss_first",
        "fista_loss_last",
        "fista_data_loss_last",
        "fista_regulariser_last",
        "returned_candidate_loss",
        "returned_data_loss",
        "returned_regularizer",
        "stationarity_proxy_trace_delta",
        "prox_gradient_norm",
        "prox_gradient_note",
        "volume_rms",
        "support_mass_fraction",
        "alignment_projection_loss",
        "valid_projection_loss",
        "loss_mode",
        "residual_sigma",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _float_value(row: dict[str, object], key: str) -> float:
    return float(cast("float | int | str", row[key]))


def _int_value(row: dict[str, object], key: str) -> int:
    return int(cast("float | int | str", row[key]))


def _write_curves_png(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: _float_value(row, "det_u_px"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    _ = ax.plot(
        [_float_value(row, "det_u_px") for row in ordered],
        [_float_value(row, "alignment_projection_loss") for row in ordered],
        marker="o",
        label="alignment",
    )
    _ = ax.plot(
        [_float_value(row, "det_u_px") for row in ordered],
        [_float_value(row, "valid_projection_loss") for row in ordered],
        marker="s",
        label="valid",
    )
    _ = ax.set_xlabel("det_u_px")
    _ = ax.set_ylabel("Projection loss")
    _ = ax.set_title("Reduced-objective det_u probe")
    _ = ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
