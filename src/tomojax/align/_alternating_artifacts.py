"""Artifact writers for alternating smoke runs."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

from collections.abc import Mapping
import csv
from importlib.metadata import PackageNotFoundError, version
import json
import subprocess
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._alternating_verification import (
    _backend_report_payload,
    _failure_report_payload,
    _gauge_policy_payload,
    _gauge_report_payload,
    _input_summary_payload,
    _mask_summary_payload,
    _observability_report_payload,
    _projection_stats_payload,
    _recovery_tolerances_payload,
    _summary_payload,
)
from tomojax.align._joint_schur_lm import (
    joint_schur_normal_eq_summary,
)
from tomojax.forward import PROJECTION_OPERATOR, project_parallel_reference, residual_loss
from tomojax.geometry import (
    GaugeReport,
    GeometryState,
    canonicalize_geometry_gauges,
    write_geometry_json,
    write_pose_decomposition_csv,
    write_pose_params_csv,
)
from tomojax.recon import ReferenceFISTAResult, write_fista_trace_csv
from tomojax.verify import validate_run_artifacts

if TYPE_CHECKING:
    from pathlib import Path

    from tomojax.align._alternating_types import (
        AlternatingLevelSummary,
        GeometryUpdateVolumeSource,
    )
    from tomojax.align._continuation import ContinuationSchedule
    from tomojax.align._joint_schur_lm import JointSchurDiagnostics, JointSchurLMResult


def _write_artifacts(
    output_dir: Path,
    *,
    true_geometry: GeometryState,
    initial_geometry: GeometryState,
    final_geometry: GeometryState,
    truth_volume: jax.Array,
    final_volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    schedule: ContinuationSchedule,
    gauge_report: GaugeReport,
    fista_result: ReferenceFISTAResult,
    summaries: tuple[AlternatingLevelSummary, ...],
    schur_result: JointSchurLMResult | None,
    geometry_update_volume_source: GeometryUpdateVolumeSource,
    geometry_update_solver: str,
    geometry_update_setup_prior_strength: float | None,
    geometry_update_pose_prior_strength: float | None,
    geometry_update_pose_trust_radius: float | None,
    geometry_update_pose_frozen: bool,
    geometry_update_pose_activate_at_level_factor: int | None,
    geometry_update_theta_activate_at_level_factor: int | None,
    geometry_update_active_setup_parameters: tuple[str, ...],
    geometry_update_active_pose_dofs: tuple[str, ...],
    preview_volume_support: str,
    preview_initialization: str,
    preview_reconstruction_mask_source: str,
    preview_tv_scale: float,
    preview_residual_filter_mode: str,
    preview_center_l2_weight: float,
    stopped_preview_policy: str,
    fit_gain_offset_nuisance: bool,
    fit_background_nuisance: bool,
    verification: Mapping[str, object],
) -> dict[str, Path]:
    artifacts = {
        "alignment_summary_csv": output_dir / "alignment_summary.csv",
        "artifact_index_json": output_dir / "artifact_index.json",
        "benchmark_report_md": output_dir / "benchmark_report.md",
        "backend_report_json": output_dir / "backend_report.json",
        "benchmark_result_json": output_dir / "benchmark_result.json",
        "config_resolved_toml": output_dir / "config_resolved.toml",
        "final_volume_npy": output_dir / "final_volume.npy",
        "failure_report_json": output_dir / "failure_report.json",
        "fista_trace_csv": output_dir / "fista_trace.csv",
        "geometry_corrupted_json": output_dir / "geometry_corrupted.json",
        "gauge_policy_json": output_dir / "gauge_policy.json",
        "gauge_report_json": output_dir / "gauge_report.json",
        "geometry_final_json": output_dir / "geometry_final.json",
        "geometry_initial_json": output_dir / "geometry_initial.json",
        "geometry_trace_csv": output_dir / "geometry_trace.csv",
        "geometry_true_json": output_dir / "geometry_true.json",
        "ground_truth_volume_npy": output_dir / "ground_truth_volume.npy",
        "input_summary_json": output_dir / "input_summary.json",
        "mask_summary_json": output_dir / "mask_summary.json",
        "pose_decomposition_csv": output_dir / "pose_decomposition.csv",
        "pose_params_csv": output_dir / "pose_params.csv",
        "plots_summary_json": output_dir / "plots" / "summary.json",
        "observability_report_json": output_dir / "observability_report.json",
        "observed_projections_npy": output_dir / "observed_projections.npy",
        "preview_error_slice_npy": output_dir / "preview_slices" / "central_z_error.npy",
        "preview_final_slice_npy": output_dir / "preview_slices" / "central_z_final.npy",
        "preview_summary_json": output_dir / "preview_slices" / "summary.json",
        "preview_truth_slice_npy": output_dir / "preview_slices" / "central_z_truth.npy",
        "projection_mask_npy": output_dir / "projection_mask.npy",
        "projection_stats_json": output_dir / "projection_stats.json",
        "recovery_tolerances_json": output_dir / "recovery_tolerances.json",
        "residual_map_raw_npy": output_dir / "residual_maps" / "final_raw_residual.npy",
        "residual_map_summary_json": output_dir / "residual_maps" / "summary.json",
        "residual_metrics_csv": output_dir / "residual_metrics.csv",
        "run_manifest_json": output_dir / "run_manifest.json",
        "schur_diagnostics_json": output_dir / "schur_diagnostics.json",
        "verification_json": output_dir / "verification.json",
    }
    synthetic_dataset = verification.get("synthetic_dataset")
    if not isinstance(synthetic_dataset, dict):
        _ = artifacts.pop("benchmark_report_md")
        _ = artifacts.pop("benchmark_result_json")
    _write_config_resolved(
        artifacts["config_resolved_toml"],
        schedule,
        geometry_update_volume_source=geometry_update_volume_source,
        geometry_update_solver=geometry_update_solver,
        geometry_update_setup_prior_strength=geometry_update_setup_prior_strength,
        geometry_update_pose_prior_strength=geometry_update_pose_prior_strength,
        geometry_update_pose_trust_radius=geometry_update_pose_trust_radius,
        geometry_update_pose_frozen=geometry_update_pose_frozen,
        geometry_update_pose_activate_at_level_factor=(
            geometry_update_pose_activate_at_level_factor
        ),
        geometry_update_theta_activate_at_level_factor=(
            geometry_update_theta_activate_at_level_factor
        ),
        geometry_update_active_setup_parameters=geometry_update_active_setup_parameters,
        geometry_update_active_pose_dofs=geometry_update_active_pose_dofs,
        preview_volume_support=preview_volume_support,
        preview_initialization=preview_initialization,
        preview_reconstruction_mask_source=preview_reconstruction_mask_source,
        preview_tv_scale=preview_tv_scale,
        preview_residual_filter_mode=preview_residual_filter_mode,
        preview_center_l2_weight=preview_center_l2_weight,
        stopped_preview_policy=stopped_preview_policy,
        fit_gain_offset_nuisance=fit_gain_offset_nuisance,
        fit_background_nuisance=fit_background_nuisance,
        synthetic_dataset=verification.get("synthetic_dataset"),
    )
    _write_json(
        artifacts["run_manifest_json"],
        _run_manifest_payload(
            final_volume,
            observed,
            schedule,
            geometry_update_volume_source=geometry_update_volume_source,
            geometry_update_solver=geometry_update_solver,
            preview_volume_support=preview_volume_support,
            preview_initialization=preview_initialization,
            preview_reconstruction_mask_source=preview_reconstruction_mask_source,
            preview_tv_scale=preview_tv_scale,
            preview_residual_filter_mode=preview_residual_filter_mode,
            preview_center_l2_weight=preview_center_l2_weight,
            stopped_preview_policy=stopped_preview_policy,
            fit_gain_offset_nuisance=fit_gain_offset_nuisance,
            fit_background_nuisance=fit_background_nuisance,
            synthetic_dataset=verification.get("synthetic_dataset"),
        ),
    )
    _write_json(artifacts["input_summary_json"], _input_summary_payload(final_volume, observed))
    _write_json(artifacts["projection_stats_json"], _projection_stats_payload(observed))
    _write_json(artifacts["mask_summary_json"], _mask_summary_payload(mask))
    _write_array(artifacts["observed_projections_npy"], observed)
    _write_mask_array(artifacts["projection_mask_npy"], mask)
    _write_json(artifacts["recovery_tolerances_json"], _recovery_tolerances_payload())
    _write_array(artifacts["ground_truth_volume_npy"], truth_volume)
    _write_json(artifacts["gauge_policy_json"], _gauge_policy_payload())
    _write_json(artifacts["gauge_report_json"], _gauge_report_payload(gauge_report))
    _write_json(
        artifacts["observability_report_json"],
        _observability_report_payload(schur_result),
    )
    _write_json(artifacts["backend_report_json"], _backend_report_payload())
    failure_report = _failure_report_payload(
        final_volume=final_volume,
        final_geometry=final_geometry,
        observed=observed,
        mask=mask,
        summaries=summaries,
        schur_result=schur_result,
        verification=verification,
    )
    _write_json(
        artifacts["failure_report_json"],
        failure_report,
    )
    if isinstance(synthetic_dataset, dict):
        synthetic_dataset_payload = cast("dict[object, object]", synthetic_dataset)
        benchmark_result = _benchmark_result_payload(
            schedule=schedule,
            verification=verification,
            synthetic_dataset=synthetic_dataset_payload,
            summaries=summaries,
            failure_report=failure_report,
            schur_result=schur_result,
            true_geometry=true_geometry,
            final_geometry=final_geometry,
            final_volume=final_volume,
            observed=observed,
            mask=mask,
            geometry_update_volume_source=geometry_update_volume_source,
            geometry_update_solver=geometry_update_solver,
            preview_volume_support=preview_volume_support,
            preview_initialization=preview_initialization,
            preview_reconstruction_mask_source=preview_reconstruction_mask_source,
            preview_tv_scale=preview_tv_scale,
            preview_residual_filter_mode=preview_residual_filter_mode,
            preview_center_l2_weight=preview_center_l2_weight,
            stopped_preview_policy=stopped_preview_policy,
        )
        _write_json(artifacts["benchmark_result_json"], benchmark_result)
        _write_text(
            artifacts["benchmark_report_md"],
            _benchmark_report_markdown(benchmark_result),
        )
    _write_final_volume(artifacts["final_volume_npy"], final_volume)
    _write_preview_slice_artifacts(
        truth_path=artifacts["preview_truth_slice_npy"],
        final_path=artifacts["preview_final_slice_npy"],
        error_path=artifacts["preview_error_slice_npy"],
        summary_path=artifacts["preview_summary_json"],
        truth_volume=truth_volume,
        final_volume=final_volume,
    )
    write_geometry_json(artifacts["geometry_true_json"], true_geometry)
    write_geometry_json(artifacts["geometry_corrupted_json"], initial_geometry)
    write_geometry_json(artifacts["geometry_initial_json"], initial_geometry)
    write_geometry_json(artifacts["geometry_final_json"], final_geometry)
    write_pose_params_csv(artifacts["pose_params_csv"], final_geometry.pose)
    write_pose_decomposition_csv(artifacts["pose_decomposition_csv"], final_geometry)
    _ = write_fista_trace_csv(fista_result, artifacts["fista_trace_csv"])
    _write_alignment_summary(artifacts["alignment_summary_csv"], summaries)
    _write_geometry_trace(artifacts["geometry_trace_csv"], summaries)
    _write_json(
        artifacts["plots_summary_json"],
        _plots_summary_payload(fista_result=fista_result, summaries=summaries),
    )
    _write_json(
        artifacts["schur_diagnostics_json"],
        _schur_diagnostics_payload(
            schur_result,
            geometry_update_volume_source=geometry_update_volume_source,
        ),
    )
    _write_residual_map_artifacts(
        artifacts["residual_map_raw_npy"],
        artifacts["residual_map_summary_json"],
        final_volume=final_volume,
        final_geometry=final_geometry,
        observed=observed,
        mask=mask,
    )
    _write_residual_metrics(
        artifacts["residual_metrics_csv"],
        summaries,
        final_volume=final_volume,
        final_geometry=final_geometry,
        observed=observed,
        mask=mask,
    )
    _write_json(artifacts["verification_json"], verification)
    _write_json(artifacts["artifact_index_json"], _artifact_index_payload(output_dir, artifacts))
    _ = validate_run_artifacts(output_dir)
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
                "executed_geometry_updates",
                "residual_filter_kinds",
                "loss_before",
                "loss_after",
                "loss_nonincreasing",
                "finite_loss",
                "residual_sigma_estimated",
                "residual_sigma_effective",
                "prior_strength",
                "heldout_residual_before",
                "heldout_residual_after",
                "heldout_residual_passed",
                "gauge_stable",
                "parameter_update_norm",
                "parameter_update_small",
                "verified",
                "schur_accepted",
                "schur_condition",
                "schur_dense_step_difference_norm",
                "schur_predicted_reduction",
                "schur_actual_reduction",
                "schur_reduction_ratio",
                "skipped_geometry",
                "skipped_level",
                "early_exit_reason",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_summary_payload(summary))


def _write_geometry_trace(path: Path, summaries: tuple[AlternatingLevelSummary, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "level_factor",
                "role",
                "geometry_updates_requested",
                "geometry_updates_executed",
                "loss_before",
                "loss_after",
                "loss_delta",
                "loss_nonincreasing",
                "residual_sigma_estimated",
                "residual_sigma_effective",
                "prior_strength",
                "heldout_residual_before",
                "heldout_residual_after",
                "heldout_residual_passed",
                "gauge_stable",
                "parameter_update_norm",
                "parameter_update_small",
                "verified",
                "schur_accepted",
                "schur_condition",
                "schur_dense_step_difference_norm",
                "schur_predicted_reduction",
                "schur_actual_reduction",
                "schur_reduction_ratio",
                "skipped_geometry",
                "skipped_level",
                "early_exit_reason",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "level_factor": summary.level_factor,
                    "role": summary.role,
                    "geometry_updates_requested": summary.geometry_updates,
                    "geometry_updates_executed": summary.executed_geometry_updates,
                    "loss_before": summary.loss_before,
                    "loss_after": summary.loss_after,
                    "loss_delta": summary.loss_after - summary.loss_before,
                    "loss_nonincreasing": summary.loss_nonincreasing,
                    "residual_sigma_estimated": summary.residual_sigma_estimated,
                    "residual_sigma_effective": summary.residual_sigma_effective,
                    "prior_strength": summary.prior_strength,
                    "heldout_residual_before": summary.heldout_residual_before,
                    "heldout_residual_after": summary.heldout_residual_after,
                    "heldout_residual_passed": summary.heldout_residual_passed,
                    "gauge_stable": summary.gauge_stable,
                    "parameter_update_norm": summary.parameter_update_norm,
                    "parameter_update_small": summary.parameter_update_small,
                    "verified": summary.verified,
                    "schur_accepted": _optional_schur_value(summary.schur_diagnostics, "accepted"),
                    "schur_condition": _optional_schur_value(
                        summary.schur_diagnostics, "schur_condition"
                    ),
                    "schur_dense_step_difference_norm": _optional_schur_value(
                        summary.schur_diagnostics,
                        "dense_step_difference_norm",
                    ),
                    "schur_predicted_reduction": _optional_schur_value(
                        summary.schur_diagnostics,
                        "predicted_reduction",
                    ),
                    "schur_actual_reduction": _optional_schur_value(
                        summary.schur_diagnostics,
                        "actual_reduction",
                    ),
                    "schur_reduction_ratio": _optional_schur_value(
                        summary.schur_diagnostics,
                        "reduction_ratio",
                    ),
                    "skipped_geometry": summary.skipped_geometry,
                    "skipped_level": summary.skipped_level,
                    "early_exit_reason": summary.early_exit_reason,
                }
            )


def _optional_schur_value(diagnostics: JointSchurDiagnostics | None, field_name: str) -> object:
    if diagnostics is None:
        return ""
    value = getattr(diagnostics, field_name)
    return "" if value is None else value


def _plots_summary_payload(
    *,
    fista_result: ReferenceFISTAResult,
    summaries: tuple[AlternatingLevelSummary, ...],
) -> dict[str, object]:
    return {
        "schema": "tomojax.plots_summary.v1",
        "rendered": False,
        "reason": "smoke run stores plot-ready numeric traces without rendering dependencies",
        "fista_loss": [
            {
                "iteration": row.iteration,
                "loss": row.loss,
                "data_loss": row.data_loss,
                "regulariser": row.regulariser,
            }
            for row in fista_result.trace
        ],
        "geometry_loss": [
            {
                "level_factor": summary.level_factor,
                "role": summary.role,
                "loss_before": summary.loss_before,
                "loss_after": summary.loss_after,
                "loss_delta": summary.loss_after - summary.loss_before,
                "skipped_level": summary.skipped_level,
                "skipped_geometry": summary.skipped_geometry,
            }
            for summary in summaries
        ],
    }


def _schur_diagnostics_payload(
    result: JointSchurLMResult | None,
    *,
    geometry_update_volume_source: GeometryUpdateVolumeSource,
) -> dict[str, object]:
    if result is None:
        return {
            "schema": "tomojax.schur_diagnostics.v1",
            "status": "not_run",
            "solver": "joint_schur_lm_reference",
            "geometry_update_volume_source": geometry_update_volume_source,
        }
    payload = joint_schur_normal_eq_summary(result)
    payload["schema"] = "tomojax.schur_diagnostics.v1"
    payload["status"] = "passed" if result.final_loss <= result.initial_loss else "warning"
    payload["geometry_update_volume_source"] = geometry_update_volume_source
    return payload


def _write_residual_metrics(
    path: Path,
    summaries: tuple[AlternatingLevelSummary, ...],
    *,
    final_volume: jax.Array,
    final_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_type",
                "level_factor",
                "view_index",
                "role",
                "loss_before",
                "loss_after",
                "absolute_improvement",
                "rmse",
                "mae",
                "robust_loss",
                "valid_pixel_fraction",
                "outlier_fraction",
                "raw_rmse",
                "residual_filter_kinds",
                "loss_nonincreasing",
                "finite_loss",
                "residual_sigma_estimated",
                "residual_sigma_effective",
                "prior_strength",
                "heldout_residual_before",
                "heldout_residual_after",
                "heldout_residual_passed",
                "gauge_stable",
                "parameter_update_norm",
                "parameter_update_small",
                "skipped_level",
                "early_exit_reason",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "row_type": "level_summary",
                    "level_factor": summary.level_factor,
                    "view_index": "",
                    "role": summary.role,
                    "loss_before": summary.loss_before,
                    "loss_after": summary.loss_after,
                    "absolute_improvement": summary.loss_before - summary.loss_after,
                    "rmse": "",
                    "mae": "",
                    "robust_loss": "",
                    "valid_pixel_fraction": "",
                    "outlier_fraction": "",
                    "raw_rmse": "",
                    "residual_filter_kinds": "|".join(summary.residual_filter_kinds),
                    "loss_nonincreasing": summary.loss_nonincreasing,
                    "finite_loss": summary.finite_loss,
                    "residual_sigma_estimated": summary.residual_sigma_estimated,
                    "residual_sigma_effective": summary.residual_sigma_effective,
                    "prior_strength": summary.prior_strength,
                    "heldout_residual_before": summary.heldout_residual_before,
                    "heldout_residual_after": summary.heldout_residual_after,
                    "heldout_residual_passed": summary.heldout_residual_passed,
                    "gauge_stable": summary.gauge_stable,
                    "parameter_update_norm": summary.parameter_update_norm,
                    "parameter_update_small": summary.parameter_update_small,
                    "skipped_level": summary.skipped_level,
                    "early_exit_reason": summary.early_exit_reason,
                }
            )
        for row in _view_residual_metric_rows(final_volume, final_geometry, observed, mask):
            writer.writerow(row)


def _view_residual_metric_rows(
    final_volume: jax.Array,
    final_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
) -> tuple[dict[str, object], ...]:
    predicted = project_parallel_reference(final_volume, final_geometry)
    residual = jnp.asarray(predicted - observed, dtype=jnp.float32)
    mask_arr = jnp.asarray(mask, dtype=bool)
    rows: list[dict[str, object]] = []
    for view_index in range(int(residual.shape[0])):
        view_residual = residual[view_index]
        view_mask = mask_arr[view_index]
        valid = view_residual[view_mask]
        rmse = float(jnp.sqrt(jnp.mean(valid * valid)))
        mae = float(jnp.mean(jnp.abs(valid)))
        robust = residual_loss(
            view_residual,
            jnp.zeros_like(view_residual),
            mask=view_mask.astype(jnp.float32),
        ).loss
        rows.append(
            {
                "row_type": "view_residual",
                "level_factor": "final",
                "view_index": view_index,
                "role": "final",
                "loss_before": "",
                "loss_after": "",
                "absolute_improvement": "",
                "rmse": rmse,
                "mae": mae,
                "robust_loss": float(robust),
                "valid_pixel_fraction": float(jnp.mean(view_mask.astype(jnp.float32))),
                "outlier_fraction": 0.0,
                "raw_rmse": rmse,
                "residual_filter_kinds": "raw",
                "loss_nonincreasing": "",
                "finite_loss": "",
                "residual_sigma_estimated": "",
                "residual_sigma_effective": "",
                "prior_strength": "",
                "heldout_residual_before": "",
                "heldout_residual_after": "",
                "heldout_residual_passed": "",
                "gauge_stable": "",
                "parameter_update_norm": "",
                "parameter_update_small": "",
                "skipped_level": "",
                "early_exit_reason": "",
            }
        )
    return tuple(rows)


def _write_residual_map_artifacts(
    residual_path: Path,
    summary_path: Path,
    *,
    final_volume: jax.Array,
    final_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
) -> None:
    predicted = project_parallel_reference(final_volume, final_geometry)
    residual = jnp.asarray(predicted - observed, dtype=jnp.float32)
    mask_arr = jnp.asarray(mask, dtype=bool)
    _write_array(residual_path, residual)
    _write_json(
        summary_path,
        {
            "schema": "tomojax.residual_map_summary.v1",
            "residual_map": residual_path.name,
            "shape": list(residual.shape),
            "dtype": str(residual.dtype),
            "valid_pixel_fraction": float(jnp.mean(mask_arr.astype(jnp.float32))),
            "rmse": float(jnp.sqrt(jnp.mean(residual[mask_arr] * residual[mask_arr]))),
            "mae": float(jnp.mean(jnp.abs(residual[mask_arr]))),
            "min": float(jnp.min(residual)),
            "max": float(jnp.max(residual)),
        },
    )


def _write_preview_slice_artifacts(
    *,
    truth_path: Path,
    final_path: Path,
    error_path: Path,
    summary_path: Path,
    truth_volume: jax.Array,
    final_volume: jax.Array,
) -> None:
    center_index = int(truth_volume.shape[0] // 2)
    truth_slice = jnp.asarray(truth_volume[center_index, :, :], dtype=jnp.float32)
    final_slice = jnp.asarray(final_volume[center_index, :, :], dtype=jnp.float32)
    error_slice = jnp.asarray(final_slice - truth_slice, dtype=jnp.float32)
    _write_array(truth_path, truth_slice)
    _write_array(final_path, final_slice)
    _write_array(error_path, error_slice)
    _write_json(
        summary_path,
        {
            "schema": "tomojax.preview_slices.v1",
            "axis": "z",
            "index": center_index,
            "shape": list(truth_slice.shape),
            "dtype": str(truth_slice.dtype),
            "truth_slice": truth_path.name,
            "final_slice": final_path.name,
            "error_slice": error_path.name,
            "error_rmse": float(jnp.sqrt(jnp.mean(error_slice * error_slice))),
            "error_mae": float(jnp.mean(jnp.abs(error_slice))),
        },
    )


def _artifact_index_payload(output_dir: Path, artifacts: Mapping[str, Path]) -> dict[str, object]:
    return {
        "schema": "tomojax.artifact_index.v1",
        "artifacts": [
            {
                "name": name,
                "path": path.relative_to(output_dir).as_posix(),
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
        "benchmark_report_md": "Synthetic benchmark markdown report",
        "backend_report_json": "Backend provenance for the smoke run",
        "benchmark_result_json": "Synthetic benchmark case result",
        "config_resolved_toml": "Resolved deterministic smoke configuration",
        "final_volume_npy": "Final reconstructed 32^3 volume",
        "failure_report_json": "Failure status for the smoke run",
        "fista_trace_csv": "Reference FISTA iteration trace",
        "gauge_policy_json": "Gauge canonicalisation policy",
        "gauge_report_json": "Gauge canonicalisation transfer report",
        "geometry_corrupted_json": "Corrupted synthetic input geometry state",
        "geometry_final_json": "Final canonical geometry state",
        "geometry_initial_json": "Initial corrupted geometry state",
        "geometry_trace_csv": "Per-level geometry update trace",
        "geometry_true_json": "True uncorrupted synthetic geometry state",
        "ground_truth_volume_npy": "Ground-truth synthetic smoke volume",
        "input_summary_json": "Synthetic input shape and dtype summary",
        "mask_summary_json": "Projection mask coverage summary",
        "observability_report_json": "Schur observability and weak-DOF report",
        "observed_projections_npy": "Observed synthetic smoke projections",
        "plots_summary_json": "Plot-ready convergence summary",
        "pose_decomposition_csv": "Final realised per-view pose decomposition",
        "pose_params_csv": "Final per-view pose parameters",
        "preview_error_slice_npy": "Central final-minus-truth preview slice",
        "preview_final_slice_npy": "Central final-volume preview slice",
        "preview_summary_json": "Preview-slice summary",
        "preview_truth_slice_npy": "Central truth-volume preview slice",
        "projection_mask_npy": "Valid projection mask",
        "projection_stats_json": "Observed projection summary statistics",
        "recovery_tolerances_json": "Smoke recovery tolerance contract",
        "residual_map_raw_npy": "Final raw projection residual map",
        "residual_map_summary_json": "Final raw residual-map summary",
        "residual_metrics_csv": "Per-level residual metrics",
        "run_manifest_json": "Resolved smoke run manifest",
        "schur_diagnostics_json": "Joint Schur LM diagnostics summary",
        "verification_json": "Smoke verification report",
    }
    return descriptions[name]


def _benchmark_report_markdown(benchmark_result: Mapping[str, object]) -> str:
    dataset = cast("dict[object, object]", benchmark_result.get("dataset", {}))
    manifest_criteria = cast(
        "dict[object, object]",
        benchmark_result.get("benchmark_manifest_criteria", {}),
    )
    criteria_evaluation = cast(
        "dict[object, object]",
        benchmark_result.get("benchmark_manifest_evaluation", {}),
    )
    criteria_summary = cast(
        "dict[object, object]",
        benchmark_result.get("benchmark_manifest_evaluation_summary", {}),
    )
    reconstruction = cast("dict[object, object]", benchmark_result.get("reconstruction", {}))
    geometry = cast("dict[object, object]", benchmark_result.get("geometry_recovery", {}))
    backend = cast("dict[object, object]", benchmark_result.get("backend", {}))
    runtime = cast("dict[object, object]", benchmark_result.get("runtime", {}))
    failure_labels = benchmark_result.get("failure_labels")
    labels = (
        ", ".join(str(label) for label in cast("list[object]", failure_labels))
        if isinstance(failure_labels, list)
        else ""
    )
    if not labels:
        labels = "none"
    benchmark_name = _markdown_cell(benchmark_result.get("benchmark"))
    return "\n".join(
        [
            f"# Benchmark: {benchmark_name}",
            "",
            "## Summary",
            "",
            "| Impl | Profile | Status | Time to verified | Total time | "
            "Volume NMSE | Final residual |",
            "|---|---|---|---:|---:|---:|---:|",
            "| "
            + " | ".join(
                [
                    _markdown_cell(benchmark_result.get("implementation")),
                    _markdown_cell(benchmark_result.get("profile")),
                    _markdown_cell(benchmark_result.get("status")),
                    _markdown_cell(runtime.get("time_to_verified_geometry_seconds")),
                    _markdown_cell(runtime.get("total_wall_seconds")),
                    _markdown_cell(reconstruction.get("volume_nmse")),
                    _markdown_cell(reconstruction.get("final_residual")),
                ]
            )
            + " |",
            "",
            "## Dataset",
            "",
            f"- Name: {_markdown_cell(dataset.get('name'))}",
            f"- Artifact directory: {_markdown_cell(dataset.get('artifact_dir'))}",
            f"- Volume shape: {_markdown_cell(dataset.get('volume_shape'))}",
            f"- Projection views: {_markdown_cell(dataset.get('projection_views'))}",
            "",
            "## Benchmark Manifest Criteria",
            "",
            *[
                f"- {_markdown_cell(key)}: {_markdown_cell(value)}"
                for key, value in sorted(manifest_criteria.items(), key=lambda item: str(item[0]))
            ],
            "" if manifest_criteria else "none",
            "",
            "## Benchmark Manifest Evaluation",
            "",
            f"- Status: {_markdown_cell(criteria_summary.get('status'))}",
            f"- Passed: {_markdown_cell(criteria_summary.get('passed'))}",
            f"- Failed: {_markdown_cell(criteria_summary.get('failed'))}",
            f"- Not evaluated: {_markdown_cell(criteria_summary.get('not_evaluated'))}",
            "",
            "| Criterion | Status | Value | Threshold | Reason |",
            "|---|---|---:|---:|---|",
            *[
                _criteria_evaluation_row(key, value)
                for key, value in sorted(
                    criteria_evaluation.items(),
                    key=lambda item: str(item[0]),
                )
            ],
            "" if criteria_evaluation else "none",
            "",
            "## Geometry Recovery",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Passed | {_markdown_cell(geometry.get('passed'))} |",
            "| Supported DOFs improved | "
            f"{_markdown_cell(geometry.get('supported_dofs_improved'))} |",
            "| det_u realised RMSE px | "
            f"{_markdown_cell(geometry.get('det_u_realized_rmse_px'))} |",
            "| det_v realised RMSE px | "
            f"{_markdown_cell(geometry.get('det_v_realized_rmse_px'))} |",
            "| theta realised RMSE rad | "
            f"{_markdown_cell(geometry.get('theta_realized_rmse_rad'))} |",
            "",
            "## Backend Provenance",
            "",
            f"- Requested: {_markdown_cell(backend.get('requested'))}",
            f"- Actual: {_markdown_cell(backend.get('actual'))}",
            "",
            "## Projection Loss Provenance",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Schur train loss | {_markdown_cell(reconstruction.get('schur_train_loss'))} |",
            f"| Heldout loss | {_markdown_cell(reconstruction.get('heldout_loss'))} |",
            "| Final volume / final geometry | "
            f"{_markdown_cell(reconstruction.get('final_volume_final_geometry_loss_all_views'))} |",
            "| Final volume / true geometry | "
            f"{_markdown_cell(reconstruction.get('final_volume_true_geometry_loss_all_views'))} |",
            "| True volume / final geometry | "
            f"{_markdown_cell(reconstruction.get('true_volume_final_geometry_loss_all_views'))} |",
            "| True volume / true geometry | "
            f"{_markdown_cell(reconstruction.get('true_volume_true_geometry_loss_all_views'))} |",
            "| Classification | "
            f"{_markdown_cell(reconstruction.get('projection_loss_classification'))} |",
            "",
            "## Failure Labels",
            "",
            labels,
            "",
        ]
    )


def _markdown_cell(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(_markdown_cell(item) for item in cast("list[object]", value))
    return str(value)


def _criteria_evaluation_row(key: object, value: object) -> str:
    payload = cast("dict[object, object]", value) if isinstance(value, dict) else {}
    return (
        f"| {_markdown_cell(key)} | {_markdown_cell(payload.get('status'))} | "
        f"{_markdown_cell(payload.get('value'))} | {_markdown_cell(payload.get('threshold'))} | "
        f"{_markdown_cell(payload.get('reason'))} |"
    )


def _benchmark_result_payload(
    *,
    schedule: ContinuationSchedule,
    verification: Mapping[str, object],
    synthetic_dataset: dict[object, object],
    summaries: tuple[AlternatingLevelSummary, ...],
    failure_report: Mapping[str, object],
    schur_result: JointSchurLMResult | None,
    true_geometry: GeometryState,
    final_geometry: GeometryState,
    final_volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    geometry_update_volume_source: GeometryUpdateVolumeSource,
    geometry_update_solver: str,
    preview_volume_support: str,
    preview_initialization: str,
    preview_reconstruction_mask_source: str,
    preview_tv_scale: float,
    preview_residual_filter_mode: str,
    preview_center_l2_weight: float,
    stopped_preview_policy: str,
) -> dict[str, object]:
    metrics = cast("dict[object, object]", verification.get("metrics", {}))
    verification_runtime = cast("dict[object, object]", verification.get("runtime", {}))
    geometry_recovery = cast("dict[object, object]", verification.get("geometry_recovery", {}))
    sidecar_readback = cast(
        "dict[object, object]",
        synthetic_dataset.get("sidecar_readback", {}),
    )
    volume_size = verification.get("size")
    failed_gates = _failed_gate_names(failure_report)
    manifest_criteria = cast(
        "dict[object, object]",
        sidecar_readback.get("recovery_tolerances", {}),
    )
    backend_payload: dict[str, object] = {
        "requested": "core_trilinear_ray",
        "actual": "core_trilinear_ray",
        "projection_operator": PROJECTION_OPERATOR,
        "backprojector": "core_trilinear_ray_adjoint",
        "jax_default_backend": jax.default_backend(),
        "selected_jax_device": _selected_jax_device(),
        "fallbacks": _backend_fallbacks_from_sidecar(sidecar_readback),
    }
    bad_view_detection = _bad_view_detection_payload(
        final_volume=final_volume,
        final_geometry=final_geometry,
        observed=observed,
        mask=mask,
    )
    pose_jump_exclusion = _pose_jump_exclusion_payload(
        true_geometry=true_geometry,
        final_geometry=final_geometry,
    )
    object_motion_suspicion = _object_motion_suspicion_payload(
        final_geometry=final_geometry,
        sidecar_readback=sidecar_readback,
    )
    object_motion_recovery = _object_motion_recovery_payload(sidecar_readback=sidecar_readback)
    current_default_comparison = _current_default_comparison_payload(
        sidecar_readback=sidecar_readback,
        final_volume_nmse=metrics.get("volume_nmse"),
    )
    manifest_evaluation = _benchmark_manifest_evaluation(
        criteria=manifest_criteria,
        geometry_recovery=geometry_recovery,
        backend=backend_payload,
        weak_dof_policy=_weak_dof_policy_from_schur(schur_result),
        bad_view_detection=bad_view_detection,
        pose_jump_exclusion=pose_jump_exclusion,
        object_motion_suspicion=object_motion_suspicion,
        object_motion_recovery=object_motion_recovery,
        current_default_comparison=current_default_comparison,
    )
    return {
        "schema": "tomojax.synthetic_benchmark_result.v1",
        "benchmark": synthetic_dataset.get("name"),
        "implementation": "reimagined_align_auto_smoke",
        "profile": schedule.name,
        "status": verification.get("status"),
        "dataset": {
            "name": synthetic_dataset.get("name"),
            "source": synthetic_dataset.get("source"),
            "artifact_dir": synthetic_dataset.get("artifact_dir"),
            "nuisance_applied_to_projections": synthetic_dataset.get(
                "nuisance_applied_to_projections"
            ),
            "volume_shape": [volume_size, volume_size, volume_size],
            "projection_views": verification.get("n_views"),
        },
        "runtime": {
            "time_to_verified_geometry_seconds": verification_runtime.get(
                "time_to_verified_geometry_seconds"
            ),
            "total_wall_seconds": verification_runtime.get("total_wall_seconds"),
            "reconstruction_calls": sum(1 for summary in summaries if not summary.skipped_level),
            "geometry_updates_requested": sum(summary.geometry_updates for summary in summaries),
            "geometry_updates_executed": sum(
                summary.executed_geometry_updates for summary in summaries
            ),
            "projector_calls": None,
        },
        "reconstruction": {
            "volume_nmse": metrics.get("volume_nmse"),
            "final_residual": metrics.get("residual_after"),
            "relative_improvement": metrics.get("relative_improvement"),
            "schur_train_loss": metrics.get("schur_train_loss"),
            "heldout_loss": metrics.get("heldout_loss"),
            "final_volume_initial_geometry_loss_all_views": metrics.get(
                "final_volume_initial_geometry_loss_all_views"
            ),
            "final_volume_final_geometry_loss_all_views": metrics.get(
                "final_volume_final_geometry_loss_all_views"
            ),
            "final_volume_true_geometry_loss_all_views": metrics.get(
                "final_volume_true_geometry_loss_all_views"
            ),
            "true_volume_final_geometry_loss_all_views": metrics.get(
                "true_volume_final_geometry_loss_all_views"
            ),
            "true_volume_true_geometry_loss_all_views": metrics.get(
                "true_volume_true_geometry_loss_all_views"
            ),
            "projection_loss_classification": metrics.get("projection_loss_classification"),
        },
        "geometry_recovery": {
            "passed": geometry_recovery.get("passed"),
            "supported_dofs_improved": geometry_recovery.get("supported_dofs_improved"),
            "det_u_realized_rmse_px": geometry_recovery.get("det_u_realized_rmse_px"),
            "det_v_realized_rmse_px": geometry_recovery.get("det_v_realized_rmse_px"),
            "theta_realized_rmse_rad": geometry_recovery.get("theta_realized_rmse_rad"),
            "theta_scale_error": geometry_recovery.get("theta_scale_error"),
            "detector_roll_error_rad": geometry_recovery.get("detector_roll_error_rad"),
            "axis_error_rad": geometry_recovery.get("axis_error_rad"),
            "alpha_beta_rmse_rad": geometry_recovery.get("alpha_beta_rmse_rad"),
        },
        "bad_view_detection": bad_view_detection,
        "pose_jump_exclusion": pose_jump_exclusion,
        "object_motion_suspicion": object_motion_suspicion,
        "object_motion_recovery": object_motion_recovery,
        "current_default_comparison": current_default_comparison,
        "backend": backend_payload,
        "failure_labels": failed_gates,
        "benchmark_manifest_criteria": manifest_criteria,
        "benchmark_manifest_evaluation": manifest_evaluation,
        "benchmark_manifest_evaluation_summary": _benchmark_manifest_evaluation_summary(
            manifest_evaluation
        ),
        "geometry_update_volume_source": geometry_update_volume_source,
        "geometry_update_solver": geometry_update_solver,
        "preview_volume_support": preview_volume_support,
        "preview_initialization": preview_initialization,
        "preview_reconstruction_mask_source": preview_reconstruction_mask_source,
        "preview_tv_scale": preview_tv_scale,
        "preview_residual_filter_mode": preview_residual_filter_mode,
        "preview_center_l2_weight": preview_center_l2_weight,
        "stopped_preview_policy": stopped_preview_policy,
    }


def _bad_view_detection_payload(
    *,
    final_volume: jax.Array,
    final_geometry: GeometryState,
    observed: jax.Array,
    mask: jax.Array,
) -> dict[str, object]:
    rows = _view_residual_metric_rows(final_volume, final_geometry, observed, mask)
    rmse = jnp.asarray(
        [float(cast("int | float", row["rmse"])) for row in rows],
        dtype=jnp.float32,
    )
    if rmse.size == 0:
        return {
            "schema": "tomojax.bad_view_detection.v1",
            "method": "robust_view_rmse_mad",
            "flagged_view_indices": [],
            "threshold_rmse": None,
            "median_rmse": None,
            "mad_rmse": None,
        }
    median = jnp.median(rmse)
    mad = jnp.median(jnp.abs(rmse - median))
    robust_sigma = jnp.asarray(1.4826, dtype=jnp.float32) * mad
    threshold = median + jnp.maximum(
        jnp.asarray(2.0, dtype=jnp.float32) * robust_sigma,
        jnp.asarray(0.005, dtype=jnp.float32) * jnp.maximum(
            jnp.abs(median),
            jnp.asarray(1.0, dtype=jnp.float32),
        ),
    )
    flagged_mask = rmse > threshold
    flagged = [int(index) for index, value in enumerate(np.asarray(flagged_mask)) if bool(value)]
    return {
        "schema": "tomojax.bad_view_detection.v1",
        "method": "robust_view_rmse_mad",
        "flagged_view_indices": flagged,
        "flagged_count": len(flagged),
        "threshold_rmse": float(threshold),
        "median_rmse": float(median),
        "mad_rmse": float(mad),
        "max_rmse": float(jnp.max(rmse)),
    }


def _pose_jump_exclusion_payload(
    *,
    true_geometry: GeometryState,
    final_geometry: GeometryState,
) -> dict[str, object]:
    true_canonical = canonicalize_geometry_gauges(true_geometry).state
    final_canonical = canonicalize_geometry_gauges(final_geometry).state
    true_dx = np.asarray(true_canonical.pose.dx_px, dtype=np.float64)
    true_dz = np.asarray(true_canonical.pose.dz_px, dtype=np.float64)
    final_dx = np.asarray(final_canonical.pose.dx_px, dtype=np.float64)
    final_dz = np.asarray(final_canonical.pose.dz_px, dtype=np.float64)
    jump_mask = _pose_jump_neighborhood_mask(true_dx=true_dx, true_dz=true_dz)
    keep = ~jump_mask
    if not bool(np.any(keep)):
        keep = np.ones_like(jump_mask, dtype=bool)
    dx_error = final_dx[keep] - true_dx[keep]
    dz_error = final_dz[keep] - true_dz[keep]
    rmse = float(np.sqrt(np.mean(np.concatenate([dx_error, dz_error]) ** 2)))
    excluded = [int(index) for index, value in enumerate(jump_mask) if bool(value)]
    return {
        "schema": "tomojax.pose_jump_exclusion.v1",
        "method": "truth_pose_dx_dz_derivative_outlier_neighborhood",
        "dx_dz_rmse_px_except_jumps": rmse,
        "excluded_view_indices": excluded,
        "excluded_count": len(excluded),
        "evaluated_view_count": int(np.count_nonzero(keep)),
    }


def _pose_jump_neighborhood_mask(
    *,
    true_dx: np.ndarray,
    true_dz: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    n_views = int(true_dx.shape[0])
    if n_views == 0:
        return np.zeros((0,), dtype=bool)
    deltas = np.hypot(np.diff(true_dx), np.diff(true_dz))
    if deltas.size == 0:
        return np.zeros((n_views,), dtype=bool)
    median = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - median)))
    threshold = median + max(6.0 * 1.4826 * mad, 5.0)
    jump_edges = np.nonzero(deltas > threshold)[0]
    mask = np.zeros((n_views,), dtype=bool)
    for edge in jump_edges:
        start = max(0, int(edge) - radius + 1)
        stop = min(n_views, int(edge) + radius + 2)
        mask[start:stop] = True
    return mask


def _object_motion_suspicion_payload(
    *,
    final_geometry: GeometryState,
    sidecar_readback: Mapping[object, object],
) -> dict[str, object]:
    unsupported = sidecar_readback.get("unsupported_dofs_not_evaluated")
    unsupported_items = cast("list[object]", unsupported) if isinstance(unsupported, list) else []
    unsupported_dofs = [str(item) for item in unsupported_items]
    sidecar_marks_object_motion = "object_motion" in unsupported_dofs
    canonical = canonicalize_geometry_gauges(final_geometry).state
    smooth_pose = _smooth_pose_drift_payload(canonical)
    smooth_pose_suspected = bool(smooth_pose["suspected"])
    evidence_sources: list[str] = []
    if sidecar_marks_object_motion:
        evidence_sources.append("synthetic_sidecar_unsupported_dof")
    if smooth_pose_suspected:
        evidence_sources.append("smooth_pose_drift")
    return {
        "schema": "tomojax.object_motion_suspicion.v1",
        "suspected": sidecar_marks_object_motion or smooth_pose_suspected,
        "evidence_sources": evidence_sources,
        "unsupported_dofs_not_evaluated": unsupported_dofs,
        "smooth_pose_drift": smooth_pose,
    }


def _object_motion_recovery_payload(
    *,
    sidecar_readback: Mapping[object, object],
) -> dict[str, object]:
    truth = sidecar_readback.get("true_object_motion")
    truth_payload: Mapping[object, object] = (
        cast("Mapping[object, object]", truth) if isinstance(truth, Mapping) else {}
    )
    zero_rmse = truth_payload.get("tx_zero_model_rmse_px")
    tx_rmse = float(zero_rmse) if isinstance(zero_rmse, int | float) else None
    return {
        "schema": "tomojax.object_motion_recovery.v1",
        "enabled": False,
        "estimate_source": None,
        "tx_rmse_px": tx_rmse,
        "truth": dict(truth_payload),
        "reason": "object-frame motion solver is not enabled",
    }


def _current_default_comparison_payload(
    *,
    sidecar_readback: Mapping[object, object],
    final_volume_nmse: object,
) -> dict[str, object] | None:
    baseline = sidecar_readback.get("current_default_baseline")
    if not isinstance(baseline, Mapping):
        return None
    baseline_payload = cast("Mapping[object, object]", baseline)
    baseline_nmse = baseline_payload.get("volume_nmse")
    if not isinstance(baseline_nmse, int | float) or not isinstance(
        final_volume_nmse,
        int | float,
    ):
        return {
            "schema": "tomojax.current_default_comparison.v1",
            "baseline": dict(baseline_payload),
            "baseline_volume_nmse": baseline_nmse,
            "candidate_volume_nmse": final_volume_nmse,
            "beats_current_default_nmse": None,
            "reason": "candidate or baseline volume NMSE is not numeric",
        }
    candidate = float(final_volume_nmse)
    baseline_value = float(baseline_nmse)
    return {
        "schema": "tomojax.current_default_comparison.v1",
        "baseline": dict(baseline_payload),
        "baseline_volume_nmse": baseline_value,
        "candidate_volume_nmse": candidate,
        "beats_current_default_nmse": candidate < baseline_value,
        "nmse_delta": candidate - baseline_value,
    }


def _smooth_pose_drift_payload(geometry: GeometryState) -> dict[str, object]:
    dx = np.asarray(geometry.pose.dx_px, dtype=np.float64)
    dz = np.asarray(geometry.pose.dz_px, dtype=np.float64)
    phi = np.asarray(geometry.pose.phi_residual_rad, dtype=np.float64)
    dx_span = _smooth_component_span(dx)
    dz_span = _smooth_component_span(dz)
    phi_span = _smooth_component_span(phi)
    max_shift_span = max(dx_span, dz_span)
    max_rotation_span = phi_span
    suspected = bool(max_shift_span >= 3.0 or max_rotation_span >= np.deg2rad(0.15))
    return {
        "method": "canonical_pose_endpoint_smooth_span",
        "dx_span_px": dx_span,
        "dz_span_px": dz_span,
        "phi_span_rad": phi_span,
        "shift_span_threshold_px": 3.0,
        "phi_span_threshold_rad": float(np.deg2rad(0.15)),
        "suspected": suspected,
    }


def _smooth_component_span(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    trend = np.linspace(float(values[0]), float(values[-1]), values.size)
    return float(np.max(trend) - np.min(trend))


def _selected_jax_device() -> str:
    devices = cast("list[object]", jax.devices())
    if not devices:
        return "unavailable"
    return str(devices[0])


def _backend_fallbacks_from_sidecar(sidecar_readback: Mapping[object, object]) -> list[object]:
    if sidecar_readback.get("detector_grid") != "calibrated_noncanonical":
        return []
    return [
        {
            "reason": "calibrated_noncanonical_detector_grid",
            "requested_policy": "calibrated_grid_fallback_explicit",
            "actual_backend": "core_trilinear_ray",
        }
    ]


def _benchmark_manifest_evaluation(
    *,
    criteria: dict[object, object],
    geometry_recovery: dict[object, object],
    backend: Mapping[str, object] | None = None,
    weak_dof_policy: Mapping[str, object] | None = None,
    bad_view_detection: Mapping[str, object] | None = None,
    pose_jump_exclusion: Mapping[str, object] | None = None,
    object_motion_suspicion: Mapping[str, object] | None = None,
    object_motion_recovery: Mapping[str, object] | None = None,
    current_default_comparison: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        str(name): _criterion_evaluation(
            name=str(name),
            threshold=threshold,
            geometry_recovery=geometry_recovery,
            backend=backend,
            weak_dof_policy=weak_dof_policy,
            bad_view_detection=bad_view_detection,
            pose_jump_exclusion=pose_jump_exclusion,
            object_motion_suspicion=object_motion_suspicion,
            object_motion_recovery=object_motion_recovery,
            current_default_comparison=current_default_comparison,
        )
        for name, threshold in criteria.items()
    }


def _criterion_evaluation(  # noqa: PLR0911 - explicit criterion branches keep reasons auditable.
    *,
    name: str,
    threshold: object,
    geometry_recovery: dict[object, object],
    backend: Mapping[str, object] | None,
    weak_dof_policy: Mapping[str, object] | None,
    bad_view_detection: Mapping[str, object] | None,
    pose_jump_exclusion: Mapping[str, object] | None,
    object_motion_suspicion: Mapping[str, object] | None,
    object_motion_recovery: Mapping[str, object] | None,
    current_default_comparison: Mapping[str, object] | None,
) -> dict[str, object]:
    if name == "backend_policy":
        return _backend_policy_evaluation(threshold=threshold, backend=backend)
    if name == "det_v_policy":
        return _det_v_policy_evaluation(
            threshold=threshold,
            geometry_recovery=geometry_recovery,
            weak_dof_policy=weak_dof_policy,
        )
    if name == "bad_views_flagged":
        return _bad_views_flagged_evaluation(
            threshold=threshold,
            bad_view_detection=bad_view_detection,
        )
    if name == "pose_dx_dz_rmse_px_lt_except_jumps":
        return _pose_dx_dz_except_jumps_evaluation(
            threshold=threshold,
            pose_jump_exclusion=pose_jump_exclusion,
        )
    if name == "core_solver":
        return _core_solver_policy_evaluation(
            threshold=threshold,
            object_motion_suspicion=object_motion_suspicion,
        )
    if name == "object_motion_enabled_tx_rmse_px_lt":
        return _object_motion_enabled_tx_rmse_evaluation(
            threshold=threshold,
            object_motion_recovery=object_motion_recovery,
        )
    if name == "beats_current_default_nmse":
        return _beats_current_default_nmse_evaluation(
            threshold=threshold,
            current_default_comparison=current_default_comparison,
        )
    if name in _MISSING_POLICY_CRITERION_REASONS:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": _MISSING_POLICY_CRITERION_REASONS[name],
        }
    metric_name = _criterion_metric_name(name)
    if metric_name is None:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": "unsupported_dof_not_evaluated",
        }
    value = _criterion_metric_value(metric_name, geometry_recovery)
    if not isinstance(value, int | float) or not isinstance(threshold, int | float):
        return {
            "status": "not_evaluated",
            "value": value,
            "threshold": threshold,
            "reason": "criterion value or threshold is not numeric",
        }
    threshold_value = _criterion_threshold_in_metric_units(name, float(threshold))
    passed = float(value) < threshold_value
    return {
        "status": "passed" if passed else "failed",
        "value": float(value),
        "threshold": threshold_value,
        "reason": "evaluated against smoke geometry recovery metric",
    }


_MISSING_POLICY_CRITERION_REASONS: dict[str, str] = {}


def _bad_views_flagged_evaluation(
    *,
    threshold: object,
    bad_view_detection: Mapping[str, object] | None,
) -> dict[str, object]:
    if threshold is not True:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": "bad-view criterion currently expects boolean true",
        }
    if bad_view_detection is None:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": True,
            "reason": "bad-view detection payload is not in benchmark_result",
        }
    flagged = bad_view_detection.get("flagged_view_indices")
    flagged_count = len(cast("list[object]", flagged)) if isinstance(flagged, list) else 0
    return {
        "status": "passed" if flagged_count > 0 else "failed",
        "value": flagged_count,
        "threshold": True,
        "reason": "evaluated from robust per-view residual outlier detection",
    }


def _pose_dx_dz_except_jumps_evaluation(
    *,
    threshold: object,
    pose_jump_exclusion: Mapping[str, object] | None,
) -> dict[str, object]:
    if pose_jump_exclusion is None:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": "pose jump-exclusion mask is not in benchmark_result",
        }
    value = pose_jump_exclusion.get("dx_dz_rmse_px_except_jumps")
    if not isinstance(value, int | float) or not isinstance(threshold, int | float):
        return {
            "status": "not_evaluated",
            "value": value,
            "threshold": threshold,
            "reason": "pose jump-exclusion value or threshold is not numeric",
        }
    passed = float(value) < float(threshold)
    return {
        "status": "passed" if passed else "failed",
        "value": float(value),
        "threshold": float(threshold),
        "reason": "evaluated from final-vs-true pose dx/dz outside jump neighborhoods",
    }


def _core_solver_policy_evaluation(
    *,
    threshold: object,
    object_motion_suspicion: Mapping[str, object] | None,
) -> dict[str, object]:
    if threshold != "flags_object_motion_suspected":
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": "unknown core_solver policy criterion",
        }
    if object_motion_suspicion is None:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": "object-motion suspicion payload is not in benchmark_result",
        }
    suspected = bool(object_motion_suspicion.get("suspected"))
    evidence_sources = object_motion_suspicion.get("evidence_sources")
    return {
        "status": "passed" if suspected else "failed",
        "value": int(suspected),
        "threshold": threshold,
        "reason": "object-motion suspicion evidence recorded"
        if suspected
        else "no object-motion suspicion evidence recorded",
        "evidence_sources": evidence_sources if isinstance(evidence_sources, list) else [],
    }


def _object_motion_enabled_tx_rmse_evaluation(
    *,
    threshold: object,
    object_motion_recovery: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(threshold, int | float):
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": "object-motion tx RMSE threshold is not numeric",
        }
    if object_motion_recovery is None:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": "object-motion recovery payload is not in benchmark_result",
        }
    value = object_motion_recovery.get("tx_rmse_px")
    enabled = bool(object_motion_recovery.get("enabled"))
    if not isinstance(value, int | float):
        return {
            "status": "failed",
            "value": None,
            "threshold": float(threshold),
            "reason": "object-frame motion solver did not provide tx RMSE",
        }
    passed = enabled and float(value) < float(threshold)
    return {
        "status": "passed" if passed else "failed",
        "value": float(value),
        "threshold": float(threshold),
        "reason": "object-frame motion tx recovery within tolerance"
        if passed
        else "object-frame motion solver is not enabled",
    }


def _beats_current_default_nmse_evaluation(
    *,
    threshold: object,
    current_default_comparison: Mapping[str, object] | None,
) -> dict[str, object]:
    if threshold is not True:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": "beats-current-default criterion currently expects boolean true",
        }
    if current_default_comparison is None:
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": True,
            "reason": "current-default comparison baseline is not in benchmark_result",
        }
    value = current_default_comparison.get("beats_current_default_nmse")
    candidate = current_default_comparison.get("candidate_volume_nmse")
    baseline = current_default_comparison.get("baseline_volume_nmse")
    if not isinstance(value, bool):
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": True,
            "reason": "current-default comparison did not produce a boolean result",
        }
    return {
        "status": "passed" if value else "failed",
        "value": bool(value),
        "threshold": True,
        "candidate_volume_nmse": candidate,
        "baseline_volume_nmse": baseline,
        "reason": "evaluated against explicit current-default baseline artifact",
    }


def _backend_policy_evaluation(
    *,
    threshold: object,
    backend: Mapping[str, object] | None,
) -> dict[str, object]:
    if threshold != "calibrated_grid_fallback_explicit":
        return {
            "status": "not_evaluated",
            "value": None,
            "threshold": threshold,
            "reason": "unknown backend policy criterion",
        }
    fallbacks = None if backend is None else backend.get("fallbacks")
    fallback_count = len(cast("list[object]", fallbacks)) if isinstance(fallbacks, list) else 0
    passed = fallback_count > 0
    return {
        "status": "passed" if passed else "failed",
        "value": fallback_count,
        "threshold": threshold,
        "reason": "explicit backend fallback provenance recorded"
        if passed
        else "expected calibrated-grid fallback provenance but backend fallbacks were empty",
    }


def _det_v_policy_evaluation(
    *,
    threshold: object,
    geometry_recovery: dict[object, object],
    weak_dof_policy: Mapping[str, object] | None,
) -> dict[str, object]:
    recovered = geometry_recovery.get("det_v_realized_rmse_px_passed")
    value = geometry_recovery.get("det_v_realized_rmse_px")
    if recovered is True:
        return {
            "status": "passed",
            "value": value,
            "threshold": threshold,
            "reason": "det_v recovered within geometry recovery tolerance",
        }
    det_v_decision = _weak_dof_decision_from_policy(weak_dof_policy, "det_v_px")
    if det_v_decision is not None and det_v_decision.get("decision") == "keep_frozen":
        return {
            "status": "passed",
            "value": value,
            "threshold": threshold,
            "reason": str(det_v_decision.get("reason", "det_v reported unobservable")),
        }
    return {
        "status": "not_evaluated",
        "value": value,
        "threshold": threshold,
        "reason": (
            "det_v was not recovered and unobservability policy evidence is not "
            "in benchmark_result"
        ),
    }


def _weak_dof_policy_from_schur(schur_result: JointSchurLMResult | None) -> Mapping[str, object]:
    observability = _observability_report_payload(schur_result)
    policy = observability.get("weak_dof_policy")
    return cast("Mapping[str, object]", policy) if isinstance(policy, Mapping) else {}


def _weak_dof_decision_from_policy(
    weak_dof_policy: Mapping[str, object] | None,
    name: str,
) -> Mapping[str, object] | None:
    if weak_dof_policy is None:
        return None
    decisions = weak_dof_policy.get("decisions")
    if not isinstance(decisions, Mapping):
        return None
    decision = cast("Mapping[object, object]", decisions).get(name)
    return cast("Mapping[str, object]", decision) if isinstance(decision, Mapping) else None


def _benchmark_manifest_evaluation_summary(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    counts = {"passed": 0, "failed": 0, "not_evaluated": 0}
    for payload in evaluation.values():
        if not isinstance(payload, dict):
            continue
        status = cast("dict[object, object]", payload).get("status")
        if status in counts:
            counts[cast("str", status)] += 1
    total = sum(counts.values())
    if counts["failed"] > 0:
        status = "failed"
    elif counts["not_evaluated"] > 0:
        status = "partially_evaluated"
    elif counts["passed"] > 0:
        status = "passed"
    else:
        status = "not_evaluated"
    return {
        "status": status,
        "total": total,
        **counts,
    }


def _criterion_metric_name(name: str) -> str | None:
    return {
        "det_u_error_px_lt": "det_u_realized_rmse_px",
        "det_v_error_px_lt": "det_v_realized_rmse_px",
        "axis_error_deg_lt": "axis_error_rad",
        "axis_roll_error_deg_lt": "axis_roll_error_rad",
        "alpha_beta_rmse_deg_lt": "alpha_beta_rmse_rad",
        "detector_roll_error_deg_lt": "detector_roll_error_rad",
        "roll_error_deg_lt": "detector_roll_error_rad",
        "theta_offset_error_deg_lt": "theta_realized_rmse_rad",
        "theta_scale_error_lt": "theta_scale_error",
    }.get(name)


def _criterion_metric_value(
    metric_name: str,
    geometry_recovery: dict[object, object],
) -> object:
    if metric_name != "axis_roll_error_rad":
        return geometry_recovery.get(metric_name)
    axis = geometry_recovery.get("axis_error_rad")
    roll = geometry_recovery.get("detector_roll_error_rad")
    if not isinstance(axis, int | float) or not isinstance(roll, int | float):
        return None
    return max(float(axis), float(roll))


def _criterion_threshold_in_metric_units(name: str, threshold: float) -> float:
    if name in {
        "alpha_beta_rmse_deg_lt",
        "axis_roll_error_deg_lt",
        "axis_error_deg_lt",
        "detector_roll_error_deg_lt",
        "roll_error_deg_lt",
        "theta_offset_error_deg_lt",
    }:
        return float(np.deg2rad(threshold))
    return threshold


def _failed_gate_names(failure_report: Mapping[str, object]) -> list[str]:
    gates = failure_report.get("gates")
    if not isinstance(gates, list):
        return []
    names: list[str] = []
    for gate in cast("list[object]", gates):
        if not isinstance(gate, dict):
            continue
        gate_payload = cast("dict[object, object]", gate)
        if gate_payload.get("passed") is False:
            name = gate_payload.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def _write_config_resolved(
    path: Path,
    schedule: ContinuationSchedule,
    *,
    geometry_update_volume_source: GeometryUpdateVolumeSource,
    geometry_update_solver: str,
    geometry_update_setup_prior_strength: float | None,
    geometry_update_pose_prior_strength: float | None,
    geometry_update_pose_trust_radius: float | None,
    geometry_update_pose_frozen: bool,
    geometry_update_pose_activate_at_level_factor: int | None,
    geometry_update_theta_activate_at_level_factor: int | None,
    geometry_update_active_setup_parameters: tuple[str, ...],
    geometry_update_active_pose_dofs: tuple[str, ...],
    preview_volume_support: str,
    preview_initialization: str,
    preview_reconstruction_mask_source: str,
    preview_tv_scale: float,
    preview_residual_filter_mode: str,
    preview_center_l2_weight: float,
    stopped_preview_policy: str,
    fit_gain_offset_nuisance: bool,
    fit_background_nuisance: bool,
    synthetic_dataset: object,
) -> None:
    lines = [
        f'profile = "{schedule.name}"',
        'align_mode = "auto"',
        'backend_requested = "core_trilinear_ray"',
        'backend_actual = "core_trilinear_ray"',
        f'projection_operator = "{PROJECTION_OPERATOR}"',
        'geometry_model = "parallel_tomography_core_trilinear_ray"',
        f'geometry_update_volume_source = "{geometry_update_volume_source}"',
        f'geometry_update_solver = "{geometry_update_solver}"',
    ]
    if geometry_update_setup_prior_strength is not None:
        lines.append(
            f"geometry_update_setup_prior_strength = {geometry_update_setup_prior_strength}"
        )
    if geometry_update_pose_prior_strength is not None:
        lines.append(f"geometry_update_pose_prior_strength = {geometry_update_pose_prior_strength}")
    if geometry_update_pose_trust_radius is not None:
        lines.append(f"geometry_update_pose_trust_radius = {geometry_update_pose_trust_radius}")
    lines.append(f"geometry_update_pose_frozen = {str(bool(geometry_update_pose_frozen)).lower()}")
    lines.extend(
        _activation_config_lines(
            geometry_update_pose_activate_at_level_factor=(
                geometry_update_pose_activate_at_level_factor
            ),
            geometry_update_theta_activate_at_level_factor=(
                geometry_update_theta_activate_at_level_factor
            ),
        )
    )
    lines.append(
        "geometry_update_active_setup_parameters = "
        f"{json.dumps(list(geometry_update_active_setup_parameters))}"
    )
    lines.append(
        f"geometry_update_active_pose_dofs = {json.dumps(list(geometry_update_active_pose_dofs))}"
    )
    lines.extend(
        _preview_config_lines(
            preview_volume_support=preview_volume_support,
            preview_initialization=preview_initialization,
            preview_reconstruction_mask_source=preview_reconstruction_mask_source,
            preview_tv_scale=preview_tv_scale,
            preview_residual_filter_mode=preview_residual_filter_mode,
            preview_center_l2_weight=preview_center_l2_weight,
            stopped_preview_policy=stopped_preview_policy,
        )
    )
    lines.append(f"fit_gain_offset_nuisance = {str(bool(fit_gain_offset_nuisance)).lower()}")
    lines.append(f"fit_background_nuisance = {str(bool(fit_background_nuisance)).lower()}")
    if isinstance(synthetic_dataset, dict):
        dataset_payload = cast("dict[object, object]", synthetic_dataset)
        name = dataset_payload.get("name")
        artifact_dir = dataset_payload.get("artifact_dir")
        if isinstance(name, str):
            lines.append(f'synthetic_dataset_name = "{name}"')
        if isinstance(artifact_dir, str):
            lines.append(f'synthetic_dataset_artifact_dir = "{artifact_dir}"')
        lines.append(
            "synthetic_dataset_nuisance_applied = "
            f"{str(bool(dataset_payload.get('nuisance_applied_to_projections'))).lower()}"
        )
        sidecar_readback = dataset_payload.get("sidecar_readback")
        if isinstance(sidecar_readback, dict):
            sidecar_payload = cast("dict[object, object]", sidecar_readback)
            validated = sidecar_payload.get("validated")
            n_views = sidecar_payload.get("n_views")
            projections = sidecar_payload.get("projections")
            consistency = sidecar_payload.get("consistency")
            lines.append(f"synthetic_dataset_sidecars_validated = {str(bool(validated)).lower()}")
            if isinstance(n_views, int | float | str):
                lines.append(f"synthetic_dataset_sidecar_views = {int(n_views)}")
            if isinstance(projections, dict):
                projection_payload = cast("dict[object, object]", projections)
                shape = projection_payload.get("shape")
                dtype = projection_payload.get("dtype")
                if isinstance(shape, list):
                    lines.append(f"synthetic_dataset_projection_shape = {shape!r}")
                if isinstance(dtype, str):
                    lines.append(f'synthetic_dataset_projection_dtype = "{dtype}"')
            if isinstance(consistency, dict):
                consistency_payload = cast("dict[object, object]", consistency)
                passed = consistency_payload.get("passed")
                lines.append(
                    f"synthetic_dataset_sidecar_consistency_passed = {str(bool(passed)).lower()}"
                )
    lines.extend((f"level_factors = {list(schedule.level_factors)!r}", ""))
    _ = path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _preview_config_lines(
    *,
    preview_volume_support: str,
    preview_initialization: str,
    preview_reconstruction_mask_source: str,
    preview_tv_scale: float,
    preview_residual_filter_mode: str,
    preview_center_l2_weight: float,
    stopped_preview_policy: str,
) -> list[str]:
    return [
        f'preview_volume_support = "{preview_volume_support}"',
        f'preview_initialization = "{preview_initialization}"',
        f'preview_reconstruction_mask_source = "{preview_reconstruction_mask_source}"',
        f"preview_tv_scale = {float(preview_tv_scale)}",
        f'preview_residual_filter_mode = "{preview_residual_filter_mode}"',
        f"preview_center_l2_weight = {float(preview_center_l2_weight)}",
        f'stopped_preview_policy = "{stopped_preview_policy}"',
    ]


def _activation_config_lines(
    *,
    geometry_update_pose_activate_at_level_factor: int | None,
    geometry_update_theta_activate_at_level_factor: int | None,
) -> list[str]:
    lines: list[str] = []
    if geometry_update_pose_activate_at_level_factor is not None:
        lines.append(
            "geometry_update_pose_activate_at_level_factor = "
            f"{int(geometry_update_pose_activate_at_level_factor)}"
        )
    if geometry_update_theta_activate_at_level_factor is not None:
        lines.append(
            "geometry_update_theta_activate_at_level_factor = "
            f"{int(geometry_update_theta_activate_at_level_factor)}"
        )
    return lines


def _write_final_volume(path: Path, volume: jax.Array) -> None:
    _write_array(path, volume)


def _write_array(path: Path, array: jax.Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.asarray(jax.device_get(array), dtype=np.float32), allow_pickle=False)


def _write_mask_array(path: Path, mask: jax.Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.asarray(jax.device_get(mask), dtype=bool), allow_pickle=False)


def _run_manifest_payload(
    volume: jax.Array,
    projections: jax.Array,
    schedule: ContinuationSchedule,
    *,
    geometry_update_volume_source: GeometryUpdateVolumeSource,
    geometry_update_solver: str,
    preview_volume_support: str,
    preview_initialization: str,
    preview_reconstruction_mask_source: str,
    preview_tv_scale: float,
    preview_residual_filter_mode: str,
    preview_center_l2_weight: float,
    stopped_preview_policy: str,
    fit_gain_offset_nuisance: bool,
    fit_background_nuisance: bool,
    synthetic_dataset: object,
) -> dict[str, object]:
    dataset: dict[str, object] = {
        "source": "tomojax.datasets.make_benchmark_phantom",
        "shape": list(volume.shape),
        "projection_shape": list(projections.shape),
        "projection_dtype": str(projections.dtype),
    }
    if isinstance(synthetic_dataset, dict):
        dataset["synthetic128_benchmark"] = synthetic_dataset
    return {
        "schema": "tomojax.run_manifest.v1",
        "tomojax_version": _tomojax_version(),
        "git_commit": _git_commit(),
        "run_id": f"{schedule.name}-deterministic",
        "started_at": "deterministic-smoke",
        "finished_at": "deterministic-smoke",
        "profile": schedule.name,
        "align_mode": "auto",
        "dataset": dataset,
        "geometry_model": "parallel_tomography_core_trilinear_ray",
        "projection_operator": PROJECTION_OPERATOR,
        "geometry_update_volume_source": geometry_update_volume_source,
        "geometry_update_solver": geometry_update_solver,
        "preview_volume_support": preview_volume_support,
        "preview_initialization": preview_initialization,
        "preview_reconstruction_mask_source": preview_reconstruction_mask_source,
        "preview_tv_scale": preview_tv_scale,
        "preview_residual_filter_mode": preview_residual_filter_mode,
        "preview_center_l2_weight": preview_center_l2_weight,
        "stopped_preview_policy": stopped_preview_policy,
        "fit_gain_offset_nuisance": fit_gain_offset_nuisance,
        "fit_background_nuisance": fit_background_nuisance,
        "continuation": {
            "name": schedule.name,
            "level_factors": list(schedule.level_factors),
        },
        "backend_requested": "core_trilinear_ray",
        "backend_actual": "core_trilinear_ray",
        "status": "passed",
    }


def _tomojax_version() -> str:
    try:
        return version("tomojax")
    except PackageNotFoundError:
        return "0+unknown"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text, encoding="utf-8")
