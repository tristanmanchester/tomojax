"""Schur scalar normal-equation diagnostics for det_u landscapes."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import csv
import json
import math
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from tomojax.align._joint_schur_lm import JointSchurLMResult


def write_schur_scalar_diagnostics(
    path: Path,
    *,
    schur_result: JointSchurLMResult | None,
    detu_curve_csv: Path,
) -> None:
    """Write one-DOF Schur-vs-scalar-curve diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = schur_scalar_diagnostics_payload(
        schur_result=schur_result,
        detu_curve_csv=detu_curve_csv,
    )
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def schur_scalar_diagnostics_payload(
    *,
    schur_result: JointSchurLMResult | None,
    detu_curve_csv: Path,
) -> dict[str, object]:
    """Return JSON-ready scalar Schur diagnostics."""
    if schur_result is None:
        return _not_applicable("schur_result_missing")
    active_setup = tuple(str(value) for value in schur_result.active_setup_parameters)
    active_pose = tuple(str(value) for value in schur_result.active_pose_dofs)
    if active_setup != ("det_u_px",) or active_pose:
        return {
            "schema": "tomojax.schur_scalar_diagnostics.v1",
            "status": "not_applicable",
            "reason": "requires_det_u_only_setup_schur",
            "active_setup_parameters": list(active_setup),
            "active_pose_dofs": list(active_pose),
        }
    if not detu_curve_csv.exists():
        return _not_applicable("detu_curve_csv_missing")

    diagnostics = schur_result.diagnostics
    data_jtr = _sum_first_column(diagnostics.setup_gradient_by_view)
    data_jtj = _sum_first_column(diagnostics.setup_hessian_diag_by_view)
    damping = float(diagnostics.damping)
    damped_jtj = data_jtj + damping
    raw_newton_step = _safe_divide(-data_jtr, data_jtj)
    damped_lm_step = _safe_divide(-data_jtr, damped_jtj)
    selected_step = (
        float(diagnostics.setup_update_by_parameter[0])
        if diagnostics.setup_update_by_parameter
        else float("nan")
    )
    curve_rows = _read_curve_rows(detu_curve_csv)
    curve_sources = {
        source: _curve_comparison(rows, final_det_u=schur_result.geometry.setup.det_u_px.value)
        for source, rows in sorted(_rows_by_source(curve_rows).items())
    }
    final_curve = curve_sources.get("final_stopped_volume")
    true_curve = curve_sources.get("true_volume")
    return {
        "schema": "tomojax.schur_scalar_diagnostics.v1",
        "status": "recorded",
        "parameter": "det_u_px",
        "active_setup_parameters": list(active_setup),
        "active_pose_dofs": list(active_pose),
        "schur": {
            "data_JTr": data_jtr,
            "data_JTJ": data_jtj,
            "damping": damping,
            "damped_JTJ": damped_jtj,
            "raw_newton_step": raw_newton_step,
            "damped_lm_step": damped_lm_step,
            "selected_step": selected_step,
            "accepted": bool(diagnostics.accepted),
            "trust_scale": float(diagnostics.trust_scale),
            "setup_trust_scale": float(diagnostics.setup_trust_scale),
            "predicted_reduction": float(diagnostics.predicted_reduction),
            "actual_reduction": float(diagnostics.actual_reduction),
            "reduction_ratio": diagnostics.reduction_ratio,
        },
        "finite_difference_curves": curve_sources,
        "comparison": {
            "final_stopped_gradient_sign_agrees_with_JTr": _sign_agrees(
                data_jtr,
                _curve_gradient(final_curve),
            ),
            "true_volume_gradient_sign_agrees_with_JTr": _sign_agrees(
                data_jtr,
                _curve_gradient(true_curve),
            ),
            "final_stopped_curvature_positive": _curve_curvature_positive(final_curve),
            "true_volume_curvature_positive": _curve_curvature_positive(true_curve),
        },
        "interpretation_note": (
            "Diagnostic only: compares the scalar normal-equation evidence used "
            "by Schur with the sampled fixed-volume det_u landscape."
        ),
    }


def _not_applicable(reason: str) -> dict[str, object]:
    return {
        "schema": "tomojax.schur_scalar_diagnostics.v1",
        "status": "not_applicable",
        "reason": reason,
    }


def _sum_first_column(rows: tuple[tuple[float, ...], ...]) -> float:
    total = 0.0
    for row in rows:
        if row:
            total += float(row[0])
    return total


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-12:
        return None
    return float(numerator / denominator)


def _read_curve_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _rows_by_source(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_source: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_source.setdefault(str(row["volume_source"]), []).append(row)
    return by_source


def _curve_comparison(
    rows: list[dict[str, object]],
    *,
    final_det_u: float,
) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: float(cast("str", row["det_u_px"])))
    det_u = np.asarray([float(cast("str", row["det_u_px"])) for row in ordered], dtype=np.float64)
    losses = np.asarray([float(cast("str", row["loss"])) for row in ordered], dtype=np.float64)
    gradients = np.asarray(
        [float(cast("str", row["finite_difference_gradient"])) for row in ordered],
        dtype=np.float64,
    )
    curvatures = np.gradient(gradients, det_u)
    nearest_index = int(np.argmin(np.abs(det_u - float(final_det_u))))
    argmin_index = int(np.argmin(losses))
    return {
        "nearest_final_det_u_px": float(det_u[nearest_index]),
        "loss_at_nearest_final": float(losses[nearest_index]),
        "gradient_at_nearest_final": float(gradients[nearest_index]),
        "curvature_at_nearest_final": float(curvatures[nearest_index]),
        "argmin_det_u_px": float(det_u[argmin_index]),
        "argmin_loss": float(losses[argmin_index]),
        "sample_count": len(ordered),
    }


def _curve_gradient(curve: dict[str, object] | None) -> float | None:
    if curve is None:
        return None
    return float(cast("float", curve["gradient_at_nearest_final"]))


def _sign_agrees(left: float, right: float | None) -> bool | None:
    if right is None:
        return None
    if abs(left) <= 1.0e-12 or abs(right) <= 1.0e-12:
        return None
    return bool(math.copysign(1.0, left) == math.copysign(1.0, right))


def _curve_curvature_positive(curve: dict[str, object] | None) -> bool | None:
    if curve is None:
        return None
    return bool(float(cast("float", curve["curvature_at_nearest_final"])) > 0.0)
