"""Fixed-volume detector-u landscape artifacts for alternating diagnostics."""
# pyright: reportAny=false, reportArgumentType=false, reportPrivateUsage=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax.align._alternating_heldout import _projection_loss
from tomojax.geometry import GeometryState
from tomojax.recon import ReferenceFISTAConfig, fista_reconstruct_reference

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import jax

    from tomojax.align._continuation import ContinuationLevel


@dataclass(frozen=True)
class DetULandscapeArtifacts:
    """Paths written by the fixed-volume detector-u landscape diagnostic."""

    csv_path: Path
    loss_png_path: Path
    gradient_png_path: Path
    summary_path: Path
    inputs_path: Path


@dataclass(frozen=True)
class _VolumeSource:
    name: str
    volume: jax.Array
    mask_role: str


def write_detu_landscape_artifacts(
    artifacts: DetULandscapeArtifacts,
    *,
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    truth_volume: jax.Array,
    final_volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    projection_valid_mask: jax.Array,
    level: ContinuationLevel,
    sigma: float,
    loss_mode: str,
) -> None:
    """Write fixed-volume det_u loss/gradient landscape diagnostics."""
    det_u_values = _candidate_det_u_values(true_geometry, initial_geometry, final_geometry)
    base_config = _reconstruction_config(level, sigma=sigma, loss_mode=loss_mode)
    true_geometry_reconstructed = fista_reconstruct_reference(
        observed,
        true_geometry,
        initial_volume=None,
        mask=projection_valid_mask,
        config=base_config,
    ).volume
    preview_iteration_volume = fista_reconstruct_reference(
        observed,
        final_geometry,
        initial_volume=None,
        mask=projection_valid_mask,
        config=_reconstruction_config(level, sigma=sigma, loss_mode=loss_mode, iterations=1),
    ).volume
    preview_budget_volume = fista_reconstruct_reference(
        observed,
        final_geometry,
        initial_volume=None,
        mask=projection_valid_mask,
        config=base_config,
    ).volume
    bootstrap_refreshed_volume = fista_reconstruct_reference(
        observed,
        initial_geometry,
        initial_volume=None,
        mask=projection_valid_mask,
        config=base_config,
    ).volume
    sources = (
        _VolumeSource("true_volume", truth_volume, "alignment_loss_mask"),
        _VolumeSource("zero_initial_volume", truth_volume * 0.0, "alignment_loss_mask"),
        _VolumeSource(
            "true_geometry_reconstructed_volume",
            true_geometry_reconstructed,
            "alignment_loss_mask",
        ),
        _VolumeSource(
            "preview_iteration_1_volume",
            preview_iteration_volume,
            "alignment_loss_mask",
        ),
        _VolumeSource(
            "preview_budget_reconstructed_volume",
            preview_budget_volume,
            "alignment_loss_mask",
        ),
        _VolumeSource(
            "bootstrap_refreshed_volume",
            bootstrap_refreshed_volume,
            "alignment_loss_mask",
        ),
        _VolumeSource(
            "reduced_objective_refreshed_final_volume",
            preview_budget_volume,
            "alignment_loss_mask",
        ),
        _VolumeSource("final_stopped_volume", final_volume, "alignment_loss_mask"),
    )
    rows = _landscape_rows(
        det_u_values,
        sources=sources,
        base_geometry=final_geometry,
        observed=observed,
        mask=mask,
        level=level,
        sigma=sigma,
        loss_mode=loss_mode,
    )
    _write_landscape_csv(artifacts.csv_path, rows)
    summary = _landscape_summary(
        rows,
        true_geometry=true_geometry,
        initial_geometry=initial_geometry,
        final_geometry=final_geometry,
    )
    _write_json(artifacts.summary_path, summary)
    _write_json(
        artifacts.inputs_path,
        _landscape_inputs(
            det_u_values,
            true_geometry=true_geometry,
            initial_geometry=initial_geometry,
            final_geometry=final_geometry,
            level=level,
            sigma=sigma,
            loss_mode=loss_mode,
        ),
    )
    _write_landscape_pngs(artifacts.loss_png_path, artifacts.gradient_png_path, rows)


def _landscape_rows(
    det_u_values: np.ndarray,
    *,
    sources: Sequence[_VolumeSource],
    base_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
    level: ContinuationLevel,
    sigma: float,
    loss_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sources:
        losses = [
            _projection_loss(
                source.volume,
                observed,
                _with_det_u(base_geometry, float(det_u)),
                mask,
                level,
                sigma=sigma,
                loss_mode=loss_mode,
            )
            for det_u in det_u_values
        ]
        gradients = np.gradient(np.asarray(losses, dtype=np.float64), det_u_values)
        for det_u, loss, gradient in zip(det_u_values, losses, gradients, strict=True):
            rows.append(
                {
                    "volume_source": source.name,
                    "det_u_px": float(det_u),
                    "loss": float(loss),
                    "finite_difference_gradient": float(gradient),
                    "mask_role": source.mask_role,
                    "loss_mode": loss_mode,
                    "residual_sigma": float(sigma),
                    "residual_filters": "|".join(
                        filter_config.kind for filter_config in level.residual_filters
                    ),
                }
            )
    return rows


def _candidate_det_u_values(
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
) -> np.ndarray:
    anchors = np.asarray(
        [
            float(true_geometry.setup.det_u_px.value),
            float(initial_geometry.setup.det_u_px.value),
            float(final_geometry.setup.det_u_px.value),
        ],
        dtype=np.float64,
    )
    lo = float(np.min(anchors) - 2.0)
    hi = float(np.max(anchors) + 2.0)
    if hi <= lo:
        hi = lo + 4.0
    return np.linspace(lo, hi, 17, dtype=np.float64)


def _landscape_summary(
    rows: Sequence[dict[str, object]],
    *,
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
) -> dict[str, object]:
    by_source: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_source.setdefault(str(row["volume_source"]), []).append(row)
    curves = []
    for source, source_rows in sorted(by_source.items()):
        argmin = min(source_rows, key=lambda row: float(row["loss"]))
        curves.append(
            {
                "volume_source": source,
                "argmin_det_u_px": float(argmin["det_u_px"]),
                "argmin_loss": float(argmin["loss"]),
                "loss_at_true_det_u_px": _nearest_loss(
                    source_rows,
                    float(true_geometry.setup.det_u_px.value),
                ),
                "loss_at_initial_det_u_px": _nearest_loss(
                    source_rows,
                    float(initial_geometry.setup.det_u_px.value),
                ),
                "loss_at_final_det_u_px": _nearest_loss(
                    source_rows,
                    float(final_geometry.setup.det_u_px.value),
                ),
            }
        )
    return {
        "schema": "tomojax.detu_curve_summary.v1",
        "status": "recorded",
        "curves": curves,
        "unavailable_volume_sources": ["multires_carried_volumes"],
    }


def _landscape_inputs(
    det_u_values: np.ndarray,
    *,
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    level: ContinuationLevel,
    sigma: float,
    loss_mode: str,
) -> dict[str, object]:
    return {
        "schema": "tomojax.detu_curve_inputs.v1",
        "det_u_px_values": [float(value) for value in det_u_values],
        "true_det_u_px": float(true_geometry.setup.det_u_px.value),
        "initial_det_u_px": float(initial_geometry.setup.det_u_px.value),
        "final_det_u_px": float(final_geometry.setup.det_u_px.value),
        "level_factor": int(level.level_factor),
        "level_role": level.role,
        "loss_mode": loss_mode,
        "residual_sigma": float(sigma),
        "residual_filters": [filter_config.kind for filter_config in level.residual_filters],
        "purpose": "diagnostic_fixed_volume_landscape_not_production_center_search",
    }


def _with_det_u(geometry: GeometryState, det_u_px: float) -> GeometryState:
    setup = geometry.setup.replace_parameter(
        "det_u_px",
        geometry.setup.det_u_px.with_value(float(det_u_px)),
    )
    return GeometryState(setup=setup, pose=geometry.pose, acquisition=geometry.acquisition)


def _reconstruction_config(
    level: ContinuationLevel,
    *,
    sigma: float,
    loss_mode: str,
    iterations: int | None = None,
) -> ReferenceFISTAConfig:
    return ReferenceFISTAConfig(
        iterations=max(1, min(int(level.reconstruction_iterations), 4))
        if iterations is None
        else max(1, int(iterations)),
        step_size=2.0e-3,
        tv_weight=level.reconstruction_tv_weight,
        residual_sigma=sigma,
        residual_delta=level.residual_delta,
        residual_loss_mode=loss_mode,
        residual_filters=level.residual_filters,
        non_negative=True,
    )


def _nearest_loss(rows: Sequence[dict[str, object]], det_u_px: float) -> float:
    nearest = min(rows, key=lambda row: abs(float(row["det_u_px"]) - det_u_px))
    return float(nearest["loss"])


def _write_landscape_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "volume_source",
        "det_u_px",
        "loss",
        "finite_difference_gradient",
        "mask_role",
        "loss_mode",
        "residual_sigma",
        "residual_filters",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_landscape_pngs(
    loss_path: Path,
    gradient_path: Path,
    rows: Sequence[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_source: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_source.setdefault(str(row["volume_source"]), []).append(row)
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    gradient_path.parent.mkdir(parents=True, exist_ok=True)
    _write_plot(
        loss_path,
        by_source,
        y_key="loss",
        y_label="Projection loss",
        title="Fixed-volume det_u loss curves",
        plt=plt,
    )
    _write_plot(
        gradient_path,
        by_source,
        y_key="finite_difference_gradient",
        y_label="Finite-difference gradient",
        title="Fixed-volume det_u gradient curves",
        plt=plt,
    )


def _write_plot(
    path: Path,
    by_source: dict[str, list[dict[str, object]]],
    *,
    y_key: str,
    y_label: str,
    title: str,
    plt: Any,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for source, source_rows in sorted(by_source.items()):
        ordered = sorted(source_rows, key=lambda row: float(row["det_u_px"]))
        ax.plot(
            [float(row["det_u_px"]) for row in ordered],
            [float(row[y_key]) for row in ordered],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=source,
        )
    ax.set_xlabel("det_u_px")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
