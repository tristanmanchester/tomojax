"""Run focused det_u variable-projection diagnostics for an existing smoke run."""
# ruff: noqa: E402
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportArgumentType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, cast

_ = os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._alternating_heldout import _projection_loss
from tomojax.align.api import reference_continuation_schedule
from tomojax.geometry import GeometryState, read_geometry_json, read_pose_params_csv
from tomojax.recon import (
    ReferenceFISTAConfig,
    build_det_u_gauge_mode,
    build_scout_support,
    centered_volume_support,
    fista_reconstruct_reference,
    project_det_u_gauge_component,
    reconstruct_average_reference,
    reconstruct_backprojection_reference,
    reference_fista_returned_quality,
)

ObjectiveMode = Literal["fixed", "reduced"]


@dataclass(frozen=True)
class ObjectiveFamily:
    """Configuration for one fixed or reduced det_u objective family."""

    name: str
    mode: ObjectiveMode
    source_geometry: str
    nonnegative: bool
    support: str
    tv_weight: float
    center_l2_weight: float
    known_phantom_support: bool = False
    scout_soft_support: bool = False
    scout_low_frequency_anchor: bool = False
    tangent_gauge: bool = False


def main() -> int:
    """Run the diagnostic suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--run-dir", type=Path, required=True)
    _ = parser.add_argument("--out-dir", type=Path, required=True)
    _ = parser.add_argument("--profile", default="lightning")
    _ = parser.add_argument("--candidate-radius", type=float, default=2.0)
    _ = parser.add_argument("--candidate-step", type=float, default=1.0)
    _ = parser.add_argument("--fista-iterations", type=int, default=2)
    _ = parser.add_argument(
        "--reduced-init",
        choices=("zero", "backprojection", "average_projection"),
        default="zero",
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    context = _load_context(run_dir, profile=str(args.profile))
    det_u_values = _candidate_det_u_values(
        true_det_u=context["true_det_u"],
        initial_det_u=context["initial_det_u"],
        final_det_u=context["final_det_u"],
        radius=float(args.candidate_radius),
        step=float(args.candidate_step),
    )
    families = _objective_families(float(context["level"].reconstruction_tv_weight))
    family_summaries = []
    for family in families:
        family_dir = out_dir / family.name
        family_dir.mkdir(parents=True, exist_ok=True)
        rows = _evaluate_family(
            family,
            det_u_values=det_u_values,
            context=context,
            fista_iterations=max(1, int(args.fista_iterations)),
            reduced_init=str(args.reduced_init),
        )
        _write_family_artifacts(family_dir, family=family, rows=rows, context=context)
        family_summaries.append(_family_summary(family, rows, context=context))
    _write_root_summary(out_dir, family_summaries, context=context)
    return 0


def _load_context(run_dir: Path, *, profile: str) -> dict[str, Any]:
    true_geometry = read_geometry_json(
        run_dir / "geometry_true.json",
        read_pose_params_csv(run_dir / "pose_params.csv"),
    )
    final_geometry = read_geometry_json(
        run_dir / "geometry_final.json",
        read_pose_params_csv(run_dir / "pose_params.csv"),
    )
    initial_geometry = read_geometry_json(
        run_dir / "geometry_initial.json",
        read_pose_params_csv(run_dir / "pose_params.csv"),
    )
    observed = jnp.asarray(np.load(run_dir / "observed_projections.npy"), dtype=jnp.float32)
    alignment_mask = jnp.asarray(np.load(run_dir / "projection_mask.npy"), dtype=jnp.float32)
    valid_mask = jnp.ones_like(observed, dtype=jnp.float32)
    truth_volume = jnp.asarray(np.load(run_dir / "ground_truth_volume.npy"), dtype=jnp.float32)
    final_volume = jnp.asarray(np.load(run_dir / "final_volume.npy"), dtype=jnp.float32)
    level = reference_continuation_schedule(cast("Any", profile)).levels[-1]
    schur = json.loads((run_dir / "schur_scalar_diagnostics.json").read_text(encoding="utf-8"))
    scout = build_scout_support(
        observed,
        initial_geometry,
        projection_valid_mask=valid_mask,
        volume_shape=cast("tuple[int, int, int]", tuple(int(dim) for dim in truth_volume.shape)),
    )
    gauge_mode = build_det_u_gauge_mode(
        scout.low_frequency_anchor,
        initial_geometry,
        projection_valid_mask=valid_mask,
    )
    return {
        "run_dir": run_dir,
        "true_geometry": true_geometry,
        "initial_geometry": initial_geometry,
        "final_geometry": final_geometry,
        "observed": observed,
        "alignment_mask": alignment_mask,
        "valid_mask": valid_mask,
        "truth_volume": truth_volume,
        "final_volume": final_volume,
        "level": level,
        "sigma": float(level.residual_sigma),
        "loss_mode": "l2",
        "true_det_u": float(true_geometry.setup.det_u_px.value),
        "initial_det_u": float(initial_geometry.setup.det_u_px.value),
        "final_det_u": float(final_geometry.setup.det_u_px.value),
        "schur_data_jtr": float(schur.get("schur", {}).get("data_JTr", float("nan"))),
        "schur_data_jtj": float(schur.get("schur", {}).get("data_JTJ", float("nan"))),
        "scout_support": scout.support_probability,
        "scout_low_frequency_anchor": scout.low_frequency_anchor,
        "scout_provenance": scout.provenance,
        "det_u_gauge_mode": gauge_mode.mode,
        "det_u_gauge_mode_object": gauge_mode,
        "det_u_gauge_mode_provenance": gauge_mode.provenance,
        "det_u_gauge_transfer_ratio_before": gauge_mode.transfer_ratio_before,
    }


def _objective_families(level_tv_weight: float) -> tuple[ObjectiveFamily, ...]:
    return (
        ObjectiveFamily("true_volume_fixed_objective", "fixed", "true", True, "none", 0.0, 0.0),
        ObjectiveFamily(
            "wrong_geometry_recon_fixed_objective",
            "fixed",
            "initial_reconstruction",
            True,
            "none",
            0.0,
            0.0,
        ),
        ObjectiveFamily(
            "final_stopped_volume_fixed_objective",
            "fixed",
            "final_stopped",
            True,
            "none",
            0.0,
            0.0,
        ),
        ObjectiveFamily("honest_reduced_objective", "reduced", "candidate", True, "none", 0.0, 0.0),
        ObjectiveFamily(
            "reduced_nonnegative_only",
            "reduced",
            "candidate",
            True,
            "none",
            0.0,
            0.0,
        ),
        ObjectiveFamily(
            "reduced_support_only",
            "reduced",
            "candidate",
            False,
            "cylindrical",
            0.0,
            0.0,
        ),
        ObjectiveFamily(
            "reduced_support_nonnegative",
            "reduced",
            "candidate",
            True,
            "cylindrical",
            0.0,
            0.0,
        ),
        ObjectiveFamily(
            "reduced_support_tv",
            "reduced",
            "candidate",
            True,
            "cylindrical",
            float(level_tv_weight),
            0.0,
        ),
        ObjectiveFamily(
            "reduced_support_tv_center",
            "reduced",
            "candidate",
            True,
            "cylindrical",
            float(level_tv_weight),
            0.02,
        ),
        ObjectiveFamily(
            "reduced_known_phantom_support",
            "reduced",
            "candidate",
            True,
            "known_phantom",
            float(level_tv_weight),
            0.02,
            known_phantom_support=True,
        ),
        ObjectiveFamily(
            "reduced_scout_soft_support",
            "reduced",
            "candidate",
            True,
            "scout_soft",
            float(level_tv_weight),
            0.0,
            scout_soft_support=True,
        ),
        ObjectiveFamily(
            "reduced_scout_lowfreq_anchor",
            "reduced",
            "candidate",
            True,
            "none",
            float(level_tv_weight),
            0.0,
            scout_low_frequency_anchor=True,
        ),
        ObjectiveFamily(
            "reduced_scout_support_anchor",
            "reduced",
            "candidate",
            True,
            "scout_soft",
            float(level_tv_weight),
            0.0,
            scout_soft_support=True,
            scout_low_frequency_anchor=True,
        ),
        ObjectiveFamily(
            "reduced_scout_support_tangent_gauge",
            "reduced",
            "candidate",
            True,
            "scout_soft",
            float(level_tv_weight),
            0.0,
            scout_soft_support=True,
            scout_low_frequency_anchor=True,
            tangent_gauge=True,
        ),
    )


def _candidate_det_u_values(
    *,
    true_det_u: float,
    initial_det_u: float,
    final_det_u: float,
    radius: float,
    step: float,
) -> np.ndarray:
    values: set[float] = set()
    for anchor in (true_det_u, initial_det_u, final_det_u):
        for offset in np.arange(-radius, radius + step * 0.5, step):
            values.add(round(float(anchor + offset), 6))
        values.add(round(float(anchor), 6))
    return np.asarray(sorted(values), dtype=np.float64)


def _evaluate_family(
    family: ObjectiveFamily,
    *,
    det_u_values: np.ndarray,
    context: dict[str, Any],
    fista_iterations: int,
    reduced_init: str,
) -> list[dict[str, object]]:
    fixed_volume = (
        _fixed_volume_for_family(family, context, fista_iterations=fista_iterations)
        if family.mode == "fixed"
        else None
    )
    rows: list[dict[str, object]] = []
    for det_u in det_u_values:
        geometry = _with_det_u(context["final_geometry"], float(det_u))
        quality = None
        gauge_projection_report = None
        if family.mode == "fixed":
            if fixed_volume is None:
                raise AssertionError("fixed objective did not prepare a fixed volume")
            volume = fixed_volume
            reconstruction_nmse = None
        else:
            support = _support_for_family(family, context)
            config = _fista_config(family, context, iterations=fista_iterations)
            result = fista_reconstruct_reference(
                context["observed"],
                geometry,
                initial_volume=_initial_volume(
                    reduced_init,
                    context["observed"],
                    geometry,
                    support=support,
                ),
                volume_support=support,
                mask=context["valid_mask"],
                config=config,
            )
            quality = reference_fista_returned_quality(
                result,
                context["observed"],
                geometry,
                volume_support=support,
                mask=context["valid_mask"],
            )
            volume = result.volume
            if family.tangent_gauge:
                volume, gauge_projection_report = project_det_u_gauge_component(
                    volume,
                    cast("Any", context["det_u_gauge_mode_object"]),
                    reference=cast("jax.Array", context["scout_low_frequency_anchor"]),
                )
            reconstruction_nmse = _volume_nmse(volume, context["truth_volume"])
        loss = _projection_loss(
            volume,
            context["observed"],
            geometry,
            context["alignment_mask"],
            context["level"],
            sigma=float(context["sigma"]),
            loss_mode=str(context["loss_mode"]),
        )
        rows.append(
            {
                "objective_family": family.name,
                "det_u_px": float(det_u),
                "loss": float(loss),
                "fista_iterations": (
                    "not_applicable_fixed_volume"
                    if family.mode == "fixed"
                    else fista_iterations
                ),
                "fista_step_size": (
                    "not_applicable_fixed_volume"
                    if family.mode == "fixed"
                    else _fista_step_size(context["observed"])
                ),
                "volume_nmse": "" if reconstruction_nmse is None else reconstruction_nmse,
                "returned_candidate_loss": (
                    "" if quality is None else quality.returned_candidate_loss
                ),
                "returned_data_loss": "" if quality is None else quality.returned_data_loss,
                "returned_regularizer": "" if quality is None else quality.returned_regularizer,
                "prox_gradient_norm": "" if quality is None else quality.projected_gradient_norm,
                "volume_rms": "" if quality is None else quality.volume_rms,
                "gauge_projection_transfer_ratio_before": (
                    ""
                    if gauge_projection_report is None
                    else gauge_projection_report.transfer_ratio_before
                ),
                "gauge_projection_transfer_ratio_after": (
                    ""
                    if gauge_projection_report is None
                    else gauge_projection_report.transfer_ratio_after
                ),
                "gauge_projection_coefficient_before": (
                    ""
                    if gauge_projection_report is None
                    else gauge_projection_report.coefficient_before
                ),
                "gauge_projection_coefficient_after": (
                    ""
                    if gauge_projection_report is None
                    else gauge_projection_report.coefficient_after
                ),
                "support_mass_fraction": (
                    ""
                    if quality is None or quality.support_mass_fraction is None
                    else quality.support_mass_fraction
                ),
                "mask_role": "alignment_loss_mask",
                "reconstruction_mask_role": (
                    "not_applicable_fixed_volume"
                    if family.mode == "fixed"
                    else "projection_valid_mask"
                ),
                "loss_normalisation": "full_projection_array_size",
                "reduced_initialization": (
                    "not_applicable_fixed_volume" if family.mode == "fixed" else reduced_init
                ),
                "support": family.support,
                "nonnegative": family.nonnegative,
                "tv_weight": family.tv_weight,
                "center_l2_weight": family.center_l2_weight,
            }
        )
    _add_curve_derivatives(rows)
    return rows


def _initial_volume(
    init: str,
    observed: jax.Array,
    geometry: GeometryState,
    *,
    support: jax.Array | None,
) -> jax.Array | None:
    size = int(observed.shape[1])
    if init == "backprojection":
        volume = (
            reconstruct_average_reference(observed, depth=size)
            if size < 64
            else reconstruct_backprojection_reference(observed, geometry, depth=size)
        )
    elif init == "average_projection":
        volume = reconstruct_average_reference(observed, depth=size)
    else:
        return None
    if support is None:
        return volume
    return jnp.asarray(volume, dtype=jnp.float32) * jnp.asarray(support, dtype=jnp.float32)


def _fixed_volume_for_family(
    family: ObjectiveFamily,
    context: dict[str, Any],
    *,
    fista_iterations: int,
) -> jax.Array:
    if family.source_geometry == "true":
        return cast("jax.Array", context["truth_volume"])
    if family.source_geometry == "final_stopped":
        return cast("jax.Array", context["final_volume"])
    if family.source_geometry == "initial_reconstruction":
        result = fista_reconstruct_reference(
            context["observed"],
            context["initial_geometry"],
            initial_volume=None,
            volume_support=None,
            mask=context["valid_mask"],
            config=_fista_config(family, context, iterations=fista_iterations),
        )
        return result.volume
    raise ValueError(f"unsupported fixed volume source {family.source_geometry!r}")


def _support_for_family(family: ObjectiveFamily, context: dict[str, Any]) -> jax.Array | None:
    if family.known_phantom_support:
        return (jnp.asarray(context["truth_volume"]) > 1.0e-6).astype(jnp.float32)
    if family.scout_soft_support:
        return None
    if family.support == "cylindrical":
        shape = tuple(int(v) for v in context["truth_volume"].shape)
        return centered_volume_support(
            cast("tuple[int, int, int]", shape),
            kind="cylindrical",
        )
    return None


def _fista_config(
    family: ObjectiveFamily,
    context: dict[str, Any],
    *,
    iterations: int,
) -> ReferenceFISTAConfig:
    level = context["level"]
    return ReferenceFISTAConfig(
        iterations=int(iterations),
        step_size=_fista_step_size(context["observed"]),
        tv_weight=float(family.tv_weight),
        residual_sigma=float(context["sigma"]),
        residual_delta=level.residual_delta,
        residual_loss_mode=str(context["loss_mode"]),
        residual_filters=level.residual_filters,
        non_negative=bool(family.nonnegative),
        center_l2_weight=float(family.center_l2_weight),
        soft_support=(
            cast("jax.Array", context["scout_support"]) if family.scout_soft_support else None
        ),
        soft_support_outside_weight=0.1 if family.scout_soft_support else 0.0,
        low_frequency_anchor=(
            cast("jax.Array", context["scout_low_frequency_anchor"])
            if family.scout_low_frequency_anchor
            else None
        ),
        low_frequency_anchor_weight=0.05 if family.scout_low_frequency_anchor else 0.0,
        gauge_mode=cast("jax.Array", context["det_u_gauge_mode"]) if family.tangent_gauge else None,
        gauge_reference=(
            cast("jax.Array", context["scout_low_frequency_anchor"])
            if family.tangent_gauge
            else None
        ),
        gauge_mode_weight=0.2 if family.tangent_gauge else 0.0,
        views_per_batch=0,
    )


def _fista_step_size(observed: jax.Array) -> float:
    size = int(observed.shape[1])
    if size < 64:
        return 2.0e-3
    return 100.0 * max(float(size), 1.0) / 128.0


def _add_curve_derivatives(rows: list[dict[str, object]]) -> None:
    det_u = np.asarray([float(row["det_u_px"]) for row in rows], dtype=np.float64)
    losses = np.asarray([float(row["loss"]) for row in rows], dtype=np.float64)
    gradients = np.gradient(losses, det_u)
    curvatures = np.gradient(gradients, det_u)
    for row, gradient, curvature in zip(rows, gradients, curvatures, strict=True):
        row["finite_difference_gradient"] = float(gradient)
        row["finite_difference_curvature"] = float(curvature)


def _write_family_artifacts(
    family_dir: Path,
    *,
    family: ObjectiveFamily,
    rows: list[dict[str, object]],
    context: dict[str, Any],
) -> None:
    _write_csv(family_dir / "detu_loss_curves.csv", rows)
    _write_plot(family_dir / "detu_loss_curves.png", rows)
    summary = _family_summary(family, rows, context=context)
    _write_json(family_dir / "objective_summary.json", summary)
    _write_json(
        family_dir / "reconstruction_config.json",
        _reconstruction_config_payload(family, rows, context=context),
    )
    _write_json(family_dir / "mask_provenance.json", _mask_provenance_payload(family))
    _write_json(family_dir / "inner_solve_quality.json", _inner_solve_quality_payload(rows))
    _write_text(family_dir / "summary.md", _family_markdown(summary))


def _family_summary(
    family: ObjectiveFamily,
    rows: list[dict[str, object]],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    argmin = min(rows, key=lambda row: float(row["loss"]))
    truth = float(context["true_det_u"])
    final = float(context["final_det_u"])
    initial = float(context["initial_det_u"])
    return {
        "schema": "tomojax.variable_projection_detu_objective.v1",
        "objective_family": family.name,
        "mode": family.mode,
        "markers": {
            "initial_det_u_px": initial,
            "final_det_u_px": final,
            "true_det_u_px": truth,
        },
        "argmin_det_u_px": float(argmin["det_u_px"]),
        "argmin_error_from_truth_px": float(float(argmin["det_u_px"]) - truth),
        "loss_at_argmin": float(argmin["loss"]),
        "curvature_near_truth": _nearest(rows, truth, "finite_difference_curvature"),
        "curvature_near_final": _nearest(rows, final, "finite_difference_curvature"),
        "gradient_near_truth": _nearest(rows, truth, "finite_difference_gradient"),
        "gradient_near_final": _nearest(rows, final, "finite_difference_gradient"),
        "gradient_sign_agrees_with_schur": _sign_agrees(
            float(context["schur_data_jtr"]),
            _nearest(rows, final, "finite_difference_gradient"),
        ),
        "schur_data_JTr": float(context["schur_data_jtr"]),
        "schur_data_JTJ": float(context["schur_data_jtj"]),
        "candidate_count": len(rows),
        "volume_nmse_at_argmin": argmin["volume_nmse"],
        "inner_solve_status": _inner_solve_status(rows, mode=family.mode),
        "underfit_reasons": _underfit_reasons(rows, mode=family.mode),
        "interpretation": _interpretation(float(argmin["det_u_px"]), truth, final),
    }


def _nearest(rows: list[dict[str, object]], det_u: float, key: str) -> float:
    row = min(rows, key=lambda item: abs(float(item["det_u_px"]) - det_u))
    return float(row[key])


def _sign_agrees(left: float, right: float) -> bool | None:
    if not math.isfinite(left) or abs(left) <= 1.0e-12 or abs(right) <= 1.0e-12:
        return None
    return bool(math.copysign(1.0, left) == math.copysign(1.0, right))


def _interpretation(argmin: float, truth: float, final: float) -> str:
    truth_error = abs(argmin - truth)
    final_error = abs(argmin - final)
    if truth_error <= 0.75:
        return "geometry_information_present"
    if final_error <= truth_error:
        return "geometry_information_moved_or_absorbed"
    return "geometry_information_flat_or_ambiguous"


def _reconstruction_config_payload(
    family: ObjectiveFamily,
    rows: list[dict[str, object]],
    *,
    context: dict[str, Any],
) -> dict[str, object]:
    first = rows[0] if rows else {}
    return {
        "schema": "tomojax.variable_projection_reconstruction_config.v1",
        "objective_family": family.name,
        "mode": family.mode,
        "neutral_initializer": True,
        "state_carry_between_candidates": False,
        "fista_iterations": first.get("fista_iterations", ""),
        "fista_step_size": first.get("fista_step_size", ""),
        "initialization": first.get("reduced_initialization", ""),
        "mask_source": first.get("reconstruction_mask_role", ""),
        "loss_normalisation": "full_projection_array_size",
        "nonnegative": family.nonnegative,
        "support": family.support,
        "scout_support_provenance": context.get("scout_provenance", {}),
        "tv_weight": family.tv_weight,
        "center_l2_weight": family.center_l2_weight,
        "soft_support_outside_weight": 0.1 if family.scout_soft_support else 0.0,
        "low_frequency_anchor_weight": 0.05 if family.scout_low_frequency_anchor else 0.0,
        "tangent_gauge_weight": 0.2 if family.tangent_gauge else 0.0,
        "det_u_gauge_transfer_ratio_before": context.get(
            "det_u_gauge_transfer_ratio_before",
            "",
        ),
        "det_u_gauge_mode_provenance": context.get("det_u_gauge_mode_provenance", {}),
        "anchored_low_frequency_component": "not_available",
    }


def _inner_solve_quality_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    mode = str(rows[0]["objective_family"]) if rows else ""
    family_mode = (
        "fixed"
        if rows and rows[0]["reduced_initialization"] == "not_applicable_fixed_volume"
        else "reduced"
    )
    return {
        "schema": "tomojax.variable_projection_inner_solve_quality.v1",
        "objective_family": mode,
        "status": _inner_solve_status(rows, mode=family_mode),
        "underfit_reasons": _underfit_reasons(rows, mode=family_mode),
        "loss_normalisation": "full_projection_array_size",
        "rows": [
            {
                "det_u_px": row["det_u_px"],
                "returned_candidate_loss": row["returned_candidate_loss"],
                "returned_data_loss": row["returned_data_loss"],
                "returned_regularizer": row["returned_regularizer"],
                "prox_gradient_norm": row["prox_gradient_norm"],
                "volume_rms": row["volume_rms"],
                "gauge_projection_transfer_ratio_before": row[
                    "gauge_projection_transfer_ratio_before"
                ],
                "gauge_projection_transfer_ratio_after": row[
                    "gauge_projection_transfer_ratio_after"
                ],
                "gauge_projection_coefficient_before": row[
                    "gauge_projection_coefficient_before"
                ],
                "gauge_projection_coefficient_after": row[
                    "gauge_projection_coefficient_after"
                ],
                "support_mass_fraction": row["support_mass_fraction"],
                "reduced_initialization": row["reduced_initialization"],
            }
            for row in rows
        ],
    }


def _inner_solve_status(rows: list[dict[str, object]], *, mode: str) -> str:
    if mode == "fixed":
        return "not_applicable_fixed_volume"
    return "inner_solve_underfit" if _underfit_reasons(rows, mode=mode) else "recorded"


def _underfit_reasons(rows: list[dict[str, object]], *, mode: str) -> list[str]:
    if mode == "fixed":
        return []
    if not rows:
        return ["no_candidates"]
    reduced_rows = [row for row in rows if row["volume_rms"] != ""]
    if not reduced_rows:
        return ["no_reduced_candidates"]
    reasons: list[str] = []
    max_volume_rms = max(float(row["volume_rms"]) for row in reduced_rows)
    loss_values = np.asarray([float(row["loss"]) for row in reduced_rows], dtype=np.float64)
    if max_volume_rms <= 1.0e-6:
        reasons.append("candidate_volumes_near_zero")
    if float(np.ptp(loss_values)) <= max(float(np.min(loss_values)), 1.0) * 1.0e-8:
        reasons.append("flat_candidate_loss_curve")
    return reasons


def _mask_provenance_payload(family: ObjectiveFamily) -> dict[str, object]:
    return {
        "schema": "tomojax.variable_projection_mask_provenance.v1",
        "objective_family": family.name,
        "reconstruction_mask_role": (
            "not_applicable_fixed_volume"
            if family.mode == "fixed"
            else "projection_valid_mask"
        ),
        "objective_loss_mask_role": "alignment_loss_mask",
    }


def _write_root_summary(
    out_dir: Path,
    summaries: list[dict[str, Any]],
    *,
    context: dict[str, Any],
) -> None:
    payload = {
        "schema": "tomojax.variable_projection_detu_diagnostic.v1",
        "run_dir": str(context["run_dir"]),
        "summaries": summaries,
        "decision": _decision(summaries),
    }
    _write_json(out_dir / "objective_summary.json", payload)
    _write_text(out_dir / "summary.md", _root_markdown(payload))


def _decision(summaries: list[dict[str, Any]]) -> str:
    honest = next(
        item for item in summaries if item["objective_family"] == "honest_reduced_objective"
    )
    if honest["interpretation"] == "geometry_information_present":
        return "alternating_loop_or_state_handoff_wrong"
    restored = [
        item["objective_family"]
        for item in summaries
        if item["objective_family"].startswith("reduced_")
        and item["interpretation"] == "geometry_information_present"
    ]
    if restored:
        return f"constraint_restores_geometry_information:{restored[0]}"
    return "reduced_objective_flat_or_wrong_under_current_constraints"


def _family_markdown(summary: dict[str, Any]) -> str:
    return (
        f"# {summary['objective_family']}\n\n"
        f"argmin det_u: {summary['argmin_det_u_px']}\n\n"
        f"error from truth: {summary['argmin_error_from_truth_px']}\n\n"
        f"interpretation: `{summary['interpretation']}`\n"
    )


def _root_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# det_u Variable-Projection Diagnostic",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        "| Objective | Argmin det_u | Error from truth | Interpretation |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        (
            "| {objective_family} | {argmin_det_u_px} | {argmin_error_from_truth_px} | "
            "`{interpretation}` |".format(**item)
        )
        for item in cast("list[dict[str, Any]]", payload["summaries"])
    )
    return "\n".join(lines) + "\n"


def _volume_nmse(volume: jax.Array, truth: jax.Array) -> float:
    numerator = jnp.mean((volume - truth) ** 2)
    denominator = jnp.maximum(jnp.mean(truth**2), jnp.asarray(1.0e-12, dtype=jnp.float32))
    return float(numerator / denominator)


def _with_det_u(geometry: GeometryState, det_u_px: float) -> GeometryState:
    setup = geometry.setup.replace_parameter(
        "det_u_px",
        geometry.setup.det_u_px.with_value(float(det_u_px)),
    )
    return GeometryState(setup=setup, pose=geometry.pose, acquisition=geometry.acquisition)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([float(row["det_u_px"]) for row in rows], [float(row["loss"]) for row in rows])
    ax.set_xlabel("det_u_px")
    ax.set_ylabel("projection loss")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _ = path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
