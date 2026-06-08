from __future__ import annotations

# ruff: noqa: D100,TC001
from collections.abc import Mapping
import json
import logging
import os
from typing import TYPE_CHECKING, cast

from tomojax.align.api import AlignmentLossSchedule, loss_spec_params, normalize_alignment_profile
from tomojax.cli.config import parse_args_with_config
from tomojax.core import log_jax_env, setup_logging
from tomojax.io import JsonValue, normalize_json

from .checkpoint import make_align_cli_checkpoint_callbacks
from .command import build_parser
from .outputs import write_alignment_outputs
from .plan import (
    build_align_cli_run_plan,
    execute_alignment_plan,
    init_jax_compilation_cache,
)
from .types import AlignCliRunPlan

if TYPE_CHECKING:
    from tomojax.align.api import AlignmentLossConfig, AlignmentLossSpec


def _loss_spec_payload(spec: AlignmentLossSpec) -> dict[str, JsonValue]:
    from tomojax.align.api import loss_spec_name

    return {
        "name": loss_spec_name(spec),
        "params": normalize_json(loss_spec_params(spec)),
    }


def _loss_config_payload(loss_config: AlignmentLossConfig) -> dict[str, JsonValue]:
    if isinstance(loss_config, AlignmentLossSchedule):
        return {
            "default": _loss_spec_payload(loss_config.default),
            "by_level": [
                {
                    "level_factor": int(entry.level_factor),
                    "spec": _loss_spec_payload(entry.spec),
                }
                for entry in loss_config.by_level
            ],
        }
    return _loss_spec_payload(loss_config)


def _log_resolved_plan(plan: AlignCliRunPlan) -> None:
    """Emit the effective alignment recipe before any heavy compute starts."""
    command = plan.command
    schedule = plan.schedule_metadata or {}
    stages = cast("object", schedule.get("stages"))
    stage_summary: list[str] = []
    if isinstance(stages, list):
        for stage in cast("list[object]", stages):
            if isinstance(stage, Mapping):
                stage_map = cast("Mapping[str, object]", stage)
                active_dofs = cast("object", stage_map.get("active_dofs", []))
                if not isinstance(active_dofs, list | tuple):
                    active_dofs = []
                dof_items = cast("list[object] | tuple[object, ...]", active_dofs)
                dofs = ",".join(str(item) for item in dof_items)
                stage_name = stage_map.get("stage_name", stage_map.get("name"))
                stage_summary.append(f"{stage_name}[{stage_map.get('optimizer_kind')}:{dofs}]")
    profile = normalize_alignment_profile(command.align_profile)
    logging.info(
        "Resolved alignment plan: mode=%s quality=%s schedule=%s levels=%s "
        "outer_iters=%d recon_iters=%d opt=%s views_per_batch=%d",
        command.mode,
        profile,
        command.schedule,
        plan.run_levels if plan.run_levels is not None else "single",
        command.outer_iters,
        command.recon_iters,
        command.opt_method,
        command.views_per_batch,
    )
    if stage_summary:
        logging.info("Resolved alignment stages: %s", " -> ".join(stage_summary))


def _resolved_plan_payload(plan: AlignCliRunPlan) -> dict[str, JsonValue]:
    """Return the structured public plan printed by --print-plan-json."""
    command = plan.command
    schedule = plan.schedule_metadata or {}
    profile_options = cast("object", plan.config_metadata.get("profile_options", {}))
    payload: dict[str, object] = {
        "mode": command.mode,
        "quality": normalize_alignment_profile(command.align_profile),
        "schedule": command.schedule,
        "levels": plan.run_levels,
        "single_resolution": plan.run_levels is None,
        "stages": normalize_json(schedule.get("stages", [])),
        "active_dofs": normalize_json(schedule.get("active_dofs", [])),
        "active_geometry_dofs": normalize_json(schedule.get("active_geometry_dofs", [])),
        "active_motion_dofs": normalize_json(schedule.get("active_motion_dofs", [])),
        "loss": _loss_config_payload(plan.loss_config),
        "outer_iters": command.outer_iters,
        "recon_iters": command.recon_iters,
        "early_stop": command.early_stop,
        "early_stop_rel": command.early_stop_rel,
        "early_stop_patience": command.early_stop_patience,
        "optimizer": command.opt_method,
        "gauge_policy": command.gauge_policy,
        "pose_model": command.pose_model,
        "projector_backend": command.projector_backend,
        "requested_gather_dtype": command.requested_gather_dtype,
        "gather_dtype": plan.gather_dtype,
        "views_per_batch": command.views_per_batch,
        "checkpoint_projector": command.checkpoint_projector,
        "roi": command.roi,
        "input_path": command.data,
        "output_path": command.out,
        "input_projection_shape": list(plan.projections.shape),
        "grid": plan.grid.to_dict(),
        "reconstruction_grid": plan.recon_grid.to_dict(),
        "detector": plan.detector.to_dict(),
        "apply_cyl_mask": plan.apply_cyl_mask,
        "profile_options": normalize_json(profile_options),
    }
    return cast("dict[str, JsonValue]", normalize_json(payload))


def main() -> None:
    """Run alignment from the public CLI."""
    p = build_parser()
    args, config_metadata = parse_args_with_config(p, required=("data", "out"))

    setup_logging()
    log_jax_env()
    init_jax_compilation_cache()
    if cast("bool", args.progress):
        os.environ["TOMOJAX_PROGRESS"] = "1"
    plan = build_align_cli_run_plan(p, args, config_metadata)
    _log_resolved_plan(plan)
    if plan.command.print_plan_json:
        print(json.dumps(_resolved_plan_payload(plan), indent=2, sort_keys=True))
        return
    checkpoint_callbacks = make_align_cli_checkpoint_callbacks(plan)
    execution = execute_alignment_plan(
        plan,
        single_checkpoint_callback=checkpoint_callbacks.single,
        multires_checkpoint_callback=checkpoint_callbacks.multires,
    )
    write_alignment_outputs(plan, execution)
