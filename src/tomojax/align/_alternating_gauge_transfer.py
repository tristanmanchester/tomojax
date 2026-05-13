"""Gauge-transfer diagnostics for det_u volume absorbability."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from tomojax.forward import (
    core_projection_geometry_from_state,
    project_parallel_reference,
)
from tomojax.geometry import GeometryState
from tomojax.recon import sum_backproject_views_chunked

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tomojax.align._continuation import ContinuationLevel


@dataclass(frozen=True)
class GaugeTransferArtifacts:
    """Paths written by the gauge-transfer diagnostic."""

    json_path: Path
    csv_path: Path


def write_gauge_transfer_diagnostics(
    artifacts: GaugeTransferArtifacts,
    *,
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    volume: jax.Array,
    projection_valid_mask: jax.Array,
    level: ContinuationLevel,
) -> None:
    """Write diagnostic evidence for det_u absorbability by volume updates."""
    rows = [
        _diagnostic_row(
            "final",
            final_geometry,
            volume=volume,
            projection_valid_mask=projection_valid_mask,
            level=level,
        ),
        _diagnostic_row(
            "initial_corrupted",
            initial_geometry,
            volume=volume,
            projection_valid_mask=projection_valid_mask,
            level=level,
        ),
        _diagnostic_row(
            "true_synthetic_diagnostic",
            true_geometry,
            volume=volume,
            projection_valid_mask=projection_valid_mask,
            level=level,
        ),
    ]
    _write_json(artifacts.json_path, _payload(rows, level=level))
    _write_csv(artifacts.csv_path, rows)


def _diagnostic_row(
    source: str,
    geometry: GeometryState,
    *,
    volume: jax.Array,
    projection_valid_mask: jax.Array,
    level: ContinuationLevel,
) -> dict[str, object]:
    eps = 1.0e-2
    regularisation = 1.0e-3
    cg_max_iterations = 6
    mask = jnp.asarray(projection_valid_mask, dtype=jnp.float32)
    vol = jnp.asarray(volume, dtype=jnp.float32)
    tangent = _detu_projection_tangent(vol, geometry, eps=eps) * mask
    fixed_curvature = float(jnp.vdot(tangent, tangent).real)
    core = core_projection_geometry_from_state(
        (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])),
        geometry,
        detector_shape=(int(mask.shape[1]), int(mask.shape[2])),
    )
    rhs = sum_backproject_views_chunked(core, tangent)
    solution, cg_iterations, cg_residual = _cg_solve(
        _matvec_for_geometry(
            geometry=geometry,
            mask=mask,
            regularisation=regularisation,
        ),
        rhs,
        max_iterations=cg_max_iterations,
    )
    transferred = project_parallel_reference(solution, geometry) * mask
    transferred_curvature = float(jnp.vdot(tangent, transferred).real)
    reduced_curvature = fixed_curvature - transferred_curvature
    transfer_ratio = _safe_ratio(transferred_curvature, fixed_curvature)
    reduced_to_fixed_ratio = _safe_ratio(reduced_curvature, fixed_curvature)
    return {
        "geometry_source": source,
        "det_u_px": float(geometry.setup.det_u_px.value),
        "fixed_curvature": fixed_curvature,
        "transferred_curvature": transferred_curvature,
        "reduced_curvature_estimate": reduced_curvature,
        "transfer_ratio": transfer_ratio,
        "reduced_to_fixed_ratio": reduced_to_fixed_ratio,
        "cg_iterations": cg_iterations,
        "cg_residual_norm": cg_residual,
        "regularisation_lambda": regularisation,
        "finite_difference_step": eps,
        "mask_role": "projection_valid_mask",
        "residual_filter_kinds": "|".join(config.kind for config in level.residual_filters),
        "filter_application": "raw_projection_tangent_valid_mask",
        "interpretation": _interpretation(transfer_ratio, reduced_to_fixed_ratio),
    }


def _matvec_for_geometry(
    *,
    geometry: GeometryState,
    mask: jax.Array,
    regularisation: float,
) -> Callable[[jax.Array], jax.Array]:
    def matvec(candidate: jax.Array) -> jax.Array:
        return _normal_matvec(
            candidate,
            geometry=geometry,
            mask=mask,
            regularisation=regularisation,
        )

    return matvec


def _detu_projection_tangent(
    volume: jax.Array,
    geometry: GeometryState,
    *,
    eps: float,
) -> jax.Array:
    current = float(geometry.setup.det_u_px.value)
    plus = project_parallel_reference(volume, _with_det_u(geometry, current + eps))
    minus = project_parallel_reference(volume, _with_det_u(geometry, current - eps))
    return (plus - minus) / jnp.asarray(2.0 * eps, dtype=jnp.float32)


def _normal_matvec(
    volume: jax.Array,
    *,
    geometry: GeometryState,
    mask: jax.Array,
    regularisation: float,
) -> jax.Array:
    projected = project_parallel_reference(volume, geometry) * mask
    core = core_projection_geometry_from_state(
        (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])),
        geometry,
        detector_shape=(int(mask.shape[1]), int(mask.shape[2])),
    )
    return (
        sum_backproject_views_chunked(core, projected)
        + jnp.asarray(
            regularisation,
            dtype=jnp.float32,
        )
        * volume
    )


def _cg_solve(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    max_iterations: int,
) -> tuple[jax.Array, int, float]:
    x = jnp.zeros_like(rhs)
    residual = rhs - matvec(x)
    direction = residual
    rs_old = jnp.vdot(residual, residual).real
    iterations = 0
    for _ in range(max(1, int(max_iterations))):
        mat_direction = matvec(direction)
        denom = jnp.maximum(
            jnp.vdot(direction, mat_direction).real,
            jnp.asarray(1.0e-12, dtype=jnp.float32),
        )
        alpha = rs_old / denom
        x = x + alpha * direction
        residual = residual - alpha * mat_direction
        rs_new = jnp.vdot(residual, residual).real
        iterations += 1
        if float(jnp.sqrt(rs_new)) < 1.0e-4:
            rs_old = rs_new
            break
        beta = rs_new / jnp.maximum(rs_old, jnp.asarray(1.0e-12, dtype=jnp.float32))
        direction = residual + beta * direction
        rs_old = rs_new
    return x, iterations, float(jnp.sqrt(rs_old))


def _with_det_u(geometry: GeometryState, det_u_px: float) -> GeometryState:
    setup = geometry.setup.replace_parameter(
        "det_u_px",
        geometry.setup.det_u_px.with_value(float(det_u_px)),
    )
    return GeometryState(setup=setup, pose=geometry.pose, acquisition=geometry.acquisition)


def _payload(rows: list[dict[str, object]], *, level: ContinuationLevel) -> dict[str, object]:
    return {
        "schema": "tomojax.gauge_transfer_diagnostics.v1",
        "status": "recorded",
        "method": "regularized_cg_volume_tangent_transfer",
        "production_effect": "none_diagnostic_only",
        "mask_role": "projection_valid_mask",
        "residual_filter_kinds": [config.kind for config in level.residual_filters],
        "filter_application": "raw_projection_tangent_valid_mask",
        "rows": rows,
        "summary": _summary(rows),
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    final_row = next((row for row in rows if row["geometry_source"] == "final"), rows[0])
    return {
        "final_geometry_interpretation": str(final_row["interpretation"]),
        "final_transfer_ratio": float(cast("float", final_row["transfer_ratio"])),
        "final_reduced_to_fixed_ratio": float(cast("float", final_row["reduced_to_fixed_ratio"])),
    }


def _interpretation(transfer_ratio: float, reduced_ratio: float) -> str:
    if transfer_ratio >= 0.8 or reduced_ratio <= 0.2:
        return "absorbed_like"
    if transfer_ratio <= 0.3 and reduced_ratio >= 0.5:
        return "identifiable_like"
    return "mixed"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1.0e-12:
        return 0.0
    return float(numerator / denominator)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "geometry_source",
        "det_u_px",
        "fixed_curvature",
        "transferred_curvature",
        "reduced_curvature_estimate",
        "transfer_ratio",
        "reduced_to_fixed_ratio",
        "cg_iterations",
        "cg_residual_norm",
        "regularisation_lambda",
        "finite_difference_step",
        "mask_role",
        "residual_filter_kinds",
        "filter_application",
        "interpretation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
