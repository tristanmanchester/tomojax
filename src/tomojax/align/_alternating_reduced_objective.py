"""Reduced-objective det_u refresh probes for alternating diagnostics."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, cast

from tomojax.align._alternating_heldout import _projection_loss
from tomojax.geometry import GeometryState
from tomojax.recon import ReferenceFISTAConfig, centered_volume_support, fista_reconstruct_reference

if TYPE_CHECKING:
    from pathlib import Path

    import jax

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
        preview_views_per_batch=preview_views_per_batch,
    )
    _write_probe_csv(artifacts.csv_path, rows)
    _write_json(artifacts.summary_path, _summary_payload(rows))
    _write_json(artifacts.volume_sources_path, _volume_sources_payload(rows))
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
    preview_views_per_batch: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    support = _support(preview_volume_support, observed)
    config = ReferenceFISTAConfig(
        iterations=max(1, min(2, int(level.reconstruction_iterations))),
        step_size=2.0e-3,
        tv_weight=level.reconstruction_tv_weight,
        residual_sigma=sigma,
        residual_delta=level.residual_delta,
        residual_loss_mode=loss_mode,
        residual_filters=level.residual_filters,
        non_negative=True,
        views_per_batch=int(preview_views_per_batch),
    )
    for name, geometry in candidates:
        result = fista_reconstruct_reference(
            observed,
            geometry,
            initial_volume=None,
            volume_support=support,
            mask=projection_valid_mask,
            config=config,
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
                "fista_mask_role": "projection_valid_mask",
                "alignment_loss_mask_role": "alignment_loss_mask",
                "valid_loss_mask_role": "projection_valid_mask",
                "fista_loss_first": None if trace_first is None else trace_first.loss,
                "fista_loss_last": None if trace_last is None else trace_last.loss,
                "fista_data_loss_last": None if trace_last is None else trace_last.data_loss,
                "fista_regulariser_last": None if trace_last is None else trace_last.regulariser,
                "stationarity_proxy_trace_delta": _trace_delta(result.trace),
                "prox_gradient_norm": None,
                "prox_gradient_note": "not_computed_reference_fista_trace_only",
                "alignment_projection_loss": alignment_loss,
                "valid_projection_loss": valid_loss,
                "loss_mode": loss_mode,
                "residual_sigma": float(sigma),
            }
        )
    return rows


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
        "status": "recorded",
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
        "initializer": "neutral_zero_volume",
        "fista_mask_role": "projection_valid_mask",
        "volume_saved": False,
        "candidates": [
            {
                "candidate": str(row["candidate"]),
                "det_u_px": _float_value(row, "det_u_px"),
                "fista_iterations": _int_value(row, "fista_iterations"),
            }
            for row in rows
        ],
    }


def _write_probe_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate",
        "det_u_px",
        "fista_iterations",
        "fista_mask_role",
        "alignment_loss_mask_role",
        "valid_loss_mask_role",
        "fista_loss_first",
        "fista_loss_last",
        "fista_data_loss_last",
        "fista_regulariser_last",
        "stationarity_proxy_trace_delta",
        "prox_gradient_norm",
        "prox_gradient_note",
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
