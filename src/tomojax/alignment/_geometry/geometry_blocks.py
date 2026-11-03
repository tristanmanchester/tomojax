"""Geometry calibration diagnostics for alignment."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.core.geometry import Geometry


def summarize_geometry_calibration_stats(
    stats: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize geometry calibration updates for run metadata."""
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = {}
    for stat in stats:
        block = str(stat.get("geometry_block") or "")
        active = str(stat.get("geometry_active_dofs") or "")
        level = str(stat.get("level_factor") or "")
        objective = str(stat.get("geometry_objective") or "fixed_volume_gn")
        if not block:
            continue
        grouped.setdefault((block, active, level, objective), []).append(stat)

    blocks: list[dict[str, object]] = []
    for (block, active, level, objective), block_stats in grouped.items():
        attempted = len(block_stats)
        accepted_stats = [s for s in block_stats if bool(s.get("geometry_accepted", False))]
        accepted = len(accepted_stats)
        first = block_stats[0]
        last = block_stats[-1]
        loss_before = _float_or_none(first.get("geometry_loss_before"))
        loss_after = _float_or_none(last.get("geometry_loss_after"))
        final_step = _float_or_none(last.get("geometry_step_norm")) or 0.0
        final_grad = _float_or_none(last.get("geometry_gradient_norm")) or 0.0
        total_step = sum(_float_or_none(s.get("geometry_step_norm")) or 0.0 for s in accepted_stats)
        max_step = max(
            (_float_or_none(s.get("geometry_max_step")) or 0.0 for s in block_stats),
            default=0.0,
        )
        last_loss_before = _float_or_none(last.get("geometry_loss_before"))
        last_loss_after = _float_or_none(last.get("geometry_loss_after"))
        last_loss_drop = (
            float(last_loss_before - last_loss_after)
            if last_loss_before is not None and last_loss_after is not None
            else None
        )
        loss_drop = (
            float(loss_before - loss_after)
            if loss_before is not None and loss_after is not None
            else None
        )
        explicit_status = str(last.get("geometry_status") or "")
        status = explicit_status or _geometry_block_status(
            attempted_updates=attempted,
            accepted_updates=accepted,
            total_step_norm=total_step,
            final_step_norm=final_step,
            final_gradient_norm=final_grad,
            max_step_norm=max_step,
            loss_drop=loss_drop,
            last_loss_drop=last_loss_drop,
        )
        blocks.append(
            {
                "geometry_block": block,
                "geometry_active_dofs": active,
                "level_factor": int(level) if level else None,
                "geometry_objective": objective,
                "attempted_updates": attempted,
                "accepted_updates": accepted,
                "total_step_norm": total_step,
                "final_step_norm": final_step,
                "final_gradient_norm": final_grad,
                "loss_before": loss_before,
                "loss_after": loss_after,
                "loss_drop": loss_drop,
                "status": status,
            }
        )
    return {
        "schema_version": 1,
        "blocks": blocks,
        "overall_status": _overall_geometry_status(blocks),
    }


def add_geometry_acquisition_diagnostics(
    diagnostics: Mapping[str, object],
    geometry: Geometry,
    active_geometry_dofs: Sequence[str],
) -> dict[str, object]:
    """Annotate geometry diagnostics with acquisition-conditioning context."""
    output = dict(diagnostics)
    blocks = [dict(block) for block in output.get("blocks", []) if isinstance(block, Mapping)]
    theta_span = _theta_span_from_geometry(geometry)
    warnings = [str(warning) for warning in output.get("warnings", []) if isinstance(warning, str)]
    axis_active = bool({"axis_rot_x_deg", "axis_rot_y_deg"} & set(active_geometry_dofs))
    if axis_active and theta_span is not None and theta_span < 270.0:
        warning = "axis_direction_sub_full_rotation_acquisition"
        if warning not in warnings:
            warnings.append(warning)
        for block in blocks:
            if block.get("geometry_block") == "axis_direction":
                block["status"] = "ill_conditioned"
                block["acquisition_warning"] = warning
                block["theta_span_deg"] = float(theta_span)
    output["blocks"] = blocks
    output["warnings"] = warnings
    output["overall_status"] = _overall_geometry_status(blocks)
    return output


def _overall_geometry_status(blocks: Sequence[Mapping[str, object]]) -> str:
    latest_blocks: dict[tuple[str, str, str], tuple[float, Mapping[str, object]]] = {}
    fallback_blocks: list[Mapping[str, object]] = []
    for index, block in enumerate(blocks):
        status = block.get("status")
        if not status:
            continue
        level_raw = block.get("level_factor")
        try:
            level = float(level_raw) if level_raw is not None else math.inf
        except Exception:
            level = math.inf
        key = (
            str(block.get("geometry_block") or ""),
            str(block.get("geometry_active_dofs") or ""),
            str(block.get("geometry_objective") or ""),
        )
        if not any(key):
            fallback_blocks.append(block)
            continue
        existing = latest_blocks.get(key)
        order = level if math.isfinite(level) else float(index)
        if existing is None or order <= existing[0]:
            latest_blocks[key] = (order, block)
    selected = [item[1] for item in latest_blocks.values()] or fallback_blocks
    statuses = [str(block.get("status")) for block in selected if block.get("status")]
    if not statuses:
        return "not_run"
    if "ill_conditioned" in statuses:
        return "ill_conditioned"
    if "underconverged" in statuses:
        return "underconverged"
    if all(status == "converged" for status in statuses):
        return "converged"
    return "unknown"


def _theta_span_from_geometry(geometry: Geometry) -> float | None:
    thetas = np.asarray(getattr(geometry, "angles", []), dtype=np.float32).reshape(-1)
    if thetas.size < 2:
        return None
    sorted_thetas = np.sort(np.mod(thetas, 360.0))
    deltas = np.diff(sorted_thetas)
    wrap_delta = float(sorted_thetas[0] + 360.0 - sorted_thetas[-1])
    circular_deltas = np.concatenate([deltas, np.asarray([wrap_delta], dtype=np.float32)])
    largest_gap = float(np.max(circular_deltas))
    span = 360.0 - largest_gap
    return min(span, 360.0)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _geometry_block_status(
    *,
    attempted_updates: int,
    accepted_updates: int,
    total_step_norm: float,
    final_step_norm: float,
    final_gradient_norm: float,
    max_step_norm: float,
    loss_drop: float | None,
    last_loss_drop: float | None,
) -> str:
    if attempted_updates <= 0:
        return "ill_conditioned"
    accepted_ratio = float(accepted_updates) / float(attempted_updates)
    meaningful_step = 0.02 * float(max_step_norm or 1.0)
    tiny_step = abs(float(total_step_norm)) < 1e-5
    tiny_grad = abs(float(final_gradient_norm)) < 1e-8
    total_drop = float(loss_drop or 0.0)
    recent_drop = float(last_loss_drop or 0.0)
    if accepted_updates == 0 and (tiny_grad or tiny_step or total_drop <= 0.0):
        return "ill_conditioned"
    if accepted_ratio <= 0.25 and total_drop <= 0.0:
        return "ill_conditioned"
    if accepted_ratio >= 0.5 and final_step_norm > meaningful_step and recent_drop > 0.0:
        return "underconverged"
    return "converged"
