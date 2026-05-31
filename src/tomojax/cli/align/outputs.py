from __future__ import annotations

# ruff: noqa: D100,D103,TC001,TC002
import logging
import sys
from typing import cast

import jax.numpy as jnp
import numpy as np

from tomojax._typed_arrays import jax_float32_array, numpy_float32_array, object_mapping
from tomojax.align.api import (
    profile_policy_from_config,
    save_alignment_params_csv,
    save_alignment_params_json,
)
from tomojax.cli.manifest import build_manifest, save_manifest
from tomojax.geometry import build_calibrated_geometry_metadata_patch, cylindrical_mask_xy
from tomojax.io import JsonValue, save_projection_payload

from .checkpoint import metadata_json_list, metadata_json_mapping, metadata_list
from .types import AlignCliExecutionResult, AlignCliInfo, AlignCliRunPlan


def _apply_alignment_output_mask(plan: AlignCliRunPlan, x: jnp.ndarray) -> jnp.ndarray:
    if not plan.apply_cyl_mask:
        return x
    try:
        m_xy = cylindrical_mask_xy(plan.recon_grid, plan.detector)
        m = jax_float32_array(m_xy).astype(x.dtype)[:, :, None]
        return x * m
    except Exception:
        m_xy = cylindrical_mask_xy(plan.recon_grid, plan.detector)
        m = numpy_float32_array(m_xy)[:, :, None]
        return jax_float32_array(np.asarray(x) * m)


def _alignment_gauge_metadata(
    plan: AlignCliRunPlan,
    info: AlignCliInfo,
) -> dict[str, JsonValue]:
    mode = cast("object", info.get("gauge_fix", plan.command.gauge_fix))
    dofs = cast("object", info.get("gauge_fix_dofs", []))
    final = cast("object", info.get("gauge_fix_final", {}))
    return {
        "mode": str(mode),
        "dofs": metadata_json_list(dofs),
        "final": metadata_json_mapping(final),
    }


def _write_alignment_result_volume(
    plan: AlignCliRunPlan,
    *,
    x: jnp.ndarray,
    params5_np: np.ndarray,
    gauge_metadata: dict[str, JsonValue],
    geometry_calibration_state: object,
) -> str:
    save_meta = plan.meta.copy_metadata()
    save_meta.grid = plan.recon_grid.to_dict()
    save_meta.volume = np.asarray(x)
    save_meta.align_params = params5_np
    save_meta.align_gauge = gauge_metadata
    if isinstance(geometry_calibration_state, dict):
        calibration_state = object_mapping(cast("object", geometry_calibration_state))
        calibration_patch = build_calibrated_geometry_metadata_patch(
            calibration_state=calibration_state,
            detector=plan.detector.to_dict(),
            geometry_meta=save_meta.geometry_meta or {},
        )
        save_meta.detector = calibration_patch["detector"]
        save_meta.geometry_meta = calibration_patch["geometry_meta"]
        save_meta.geometry_calibration = {
            "calibration_state": calibration_patch["geometry_calibration"]["calibration_state"]
        }
    save_meta.frame = str(plan.meta.sample_name or "sample")
    save_meta.volume_axes_order = plan.command.volume_axes
    save_projection_payload(
        plan.command.out,
        projections=plan.meta.projections,
        metadata=save_meta,
    )
    logging.info("Saved alignment results to %s", plan.command.out)
    return str(save_meta.frame)


def _write_alignment_params_exports(
    plan: AlignCliRunPlan,
    *,
    params5_np: np.ndarray,
    gauge_metadata: dict[str, JsonValue],
) -> None:
    if plan.command.save_params_json is not None:
        save_alignment_params_json(
            plan.command.save_params_json,
            params5_np,
            du=float(plan.detector.du),
            dv=float(plan.detector.dv),
            gauge_metadata=gauge_metadata,
        )
        logging.info("Saved alignment parameter JSON to %s", plan.command.save_params_json)
    if plan.command.save_params_csv is not None:
        save_alignment_params_csv(
            plan.command.save_params_csv,
            params5_np,
            du=float(plan.detector.du),
            dv=float(plan.detector.dv),
        )
        logging.info("Saved alignment parameter CSV to %s", plan.command.save_params_csv)


def _build_alignment_manifest_payload_from_result(
    plan: AlignCliRunPlan,
    execution: AlignCliExecutionResult,
    *,
    x: jnp.ndarray,
    params5_np: np.ndarray,
    gauge_metadata: dict[str, JsonValue],
    geometry_calibration_state: object,
    output_frame: str,
) -> dict[str, object]:
    command = plan.command
    info = execution.info
    loss_values = metadata_list(cast("object", info.get("loss", [])))
    objective_kinds = metadata_list(cast("object", info.get("objective_kinds", [])))
    active_dofs = metadata_list(cast("object", info.get("active_dofs", [])))
    active_pose_dofs = metadata_list(cast("object", info.get("active_pose_dofs", [])))
    active_geometry_dofs = metadata_list(cast("object", info.get("active_geometry_dofs", [])))
    return {
        "input_path": command.data,
        "output_path": command.out,
        "save_params_json": command.save_params_json,
        "save_params_csv": command.save_params_csv,
        "manifest_path": command.save_manifest,
        "config_path": plan.config_metadata["config_path"],
        "config_file_values": plan.config_metadata["config_file_values"],
        "explicit_cli_keys": plan.config_metadata["explicit_cli_keys"],
        "effective_options": plan.config_metadata["effective_options"],
        "geometry_type": str(plan.meta.geometry_type),
        "input_projection_shape": list(plan.meta.projections.shape),
        "reconstruction_grid": plan.recon_grid.to_dict(),
        "detector": plan.detector.to_dict(),
        "align_profile": command.align_profile,
        "profile_policy": profile_policy_from_config(plan.cfg).to_dict(),
        "roi": {
            "requested": command.roi,
            "is_parallel": bool(plan.meta.geometry_type == "parallel"),
            "grid_changed": plan.recon_grid != plan.grid,
            "cylindrical_output_mask": bool(plan.apply_cyl_mask),
        },
        "requested_gather_dtype": command.requested_gather_dtype,
        "gather_dtype": plan.gather_dtype,
        "recon_algo": command.recon_algo,
        "regulariser": command.regulariser,
        "huber_delta": command.huber_delta,
        "views_per_batch": command.views_per_batch,
        "spdhg_seed": command.spdhg_seed,
        "recon_positivity": command.recon_positivity,
        "projector_unroll": command.projector_unroll,
        "projector_backend": command.projector_backend,
        "quality_tier": command.quality_tier,
        "fallback_policy": command.fallback_policy,
        "checkpoint_projector": command.checkpoint_projector,
        "transfer_guard": command.transfer_guard,
        "levels": plan.run_levels,
        "schedule": info.get("schedule", plan.schedule_metadata),
        "used_multires": bool(plan.run_levels is not None and len(plan.run_levels) > 0),
        "checkpoint_path": command.checkpoint,
        "checkpoint_every": command.checkpoint_every,
        "resume_path": command.resume,
        "loss_params": plan.loss_params,
        "loss_spec": plan.loss_config,
        "align_config": plan.cfg,
        "objective_kind": info.get("objective_kind"),
        "objective_kinds": objective_kinds,
        "objective_provenance": info.get("objective_provenance"),
        "backend_provenance": info.get("backend_provenance"),
        "gauge_policy": command.gauge_policy,
        "gauge_decision": info.get("gauge_decision"),
        "active_dofs": active_dofs,
        "active_pose_dofs": active_pose_dofs,
        "active_geometry_dofs": active_geometry_dofs,
        "geometry_calibration_state": geometry_calibration_state,
        "alignment_params_shape": list(params5_np.shape),
        "alignment_gauge": gauge_metadata,
        "volume_shape": list(np.asarray(x).shape),
        "volume_axes": command.volume_axes,
        "frame": output_frame,
        "run_info": {
            "loss_count": len(loss_values),
            "final_loss": loss_values[-1] if len(loss_values) else None,
            "loss_kind": info.get("loss_kind"),
            "stopped_by_observer": info.get("stopped_by_observer"),
            "observer_action": info.get("observer_action"),
        },
    }


def _write_alignment_manifest(
    plan: AlignCliRunPlan,
    execution: AlignCliExecutionResult,
    *,
    x: jnp.ndarray,
    params5_np: np.ndarray,
    gauge_metadata: dict[str, JsonValue],
    geometry_calibration_state: object,
    output_frame: str,
) -> None:
    if plan.command.save_manifest is None:
        return
    payload = _build_alignment_manifest_payload_from_result(
        plan,
        execution,
        x=x,
        params5_np=params5_np,
        gauge_metadata=gauge_metadata,
        geometry_calibration_state=geometry_calibration_state,
        output_frame=output_frame,
    )
    manifest = build_manifest("tomojax align", list(sys.argv), plan.cli_args, payload)
    save_manifest(plan.command.save_manifest, manifest)
    logging.info("Saved reproducibility manifest to %s", plan.command.save_manifest)


def write_alignment_outputs(
    plan: AlignCliRunPlan,
    execution: AlignCliExecutionResult,
) -> None:
    x = _apply_alignment_output_mask(plan, execution.x)
    params5_np = np.asarray(execution.params5)
    gauge_metadata = _alignment_gauge_metadata(plan, execution.info)
    geometry_calibration_state = execution.info.get("geometry_calibration_state")
    output_frame = _write_alignment_result_volume(
        plan,
        x=x,
        params5_np=params5_np,
        gauge_metadata=gauge_metadata,
        geometry_calibration_state=geometry_calibration_state,
    )
    _write_alignment_params_exports(
        plan,
        params5_np=params5_np,
        gauge_metadata=gauge_metadata,
    )
    _write_alignment_manifest(
        plan,
        execution,
        x=x,
        params5_np=params5_np,
        gauge_metadata=gauge_metadata,
        geometry_calibration_state=geometry_calibration_state,
        output_frame=output_frame,
    )
