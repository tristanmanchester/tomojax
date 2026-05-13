"""Geometry calibration state and materialization helpers for alignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from tomojax.align.model.dofs import GEOMETRY_DOF_NAMES, normalize_alignment_dofs
from tomojax.geometry import (
    CalibrationState,
    CalibrationVariable,
    axis_unit_from_rotations,
    detector_grid_from_calibration,
    nominal_axis_unit_from_inputs,
    validate_calibration_gauges,
)
from tomojax.core.geometry import RotationAxisGeometry
from tomojax.core.geometry.lamino import LaminographyGeometry
from tomojax.core.geometry.parallel import ParallelGeometry

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tomojax.core.geometry import Detector, Geometry, Grid


GEOMETRY_DOFS: tuple[str, ...] = GEOMETRY_DOF_NAMES

GEOMETRY_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("detector_center", ("det_u_px", "det_v_px")),
    ("detector_roll", ("detector_roll_deg",)),
    ("axis_direction", ("axis_rot_x_deg", "axis_rot_y_deg")),
)


def normalize_geometry_dofs(
    values: Iterable[str] | None,
    *,
    geometry: Geometry | None = None,
) -> tuple[str, ...]:
    """Normalize detector/instrument geometry DOF names for staged alignment."""
    if values is None:
        return ()
    names = normalize_alignment_dofs(
        values,
        option_name="geometry_dofs",
        geometry=geometry,
    )
    invalid = [name for name in names if name not in GEOMETRY_DOFS]
    if invalid:
        allowed = ", ".join(GEOMETRY_DOFS)
        raise ValueError(f"Unknown geometry DOF {invalid[0]!r}; expected one of: {allowed}")
    return names


def _tilt_alias_for_geometry(geometry: Geometry | None) -> str:
    tilt_about = getattr(geometry, "tilt_about", "x")
    return "axis_rot_y_deg" if str(tilt_about) == "z" else "axis_rot_x_deg"


@dataclass(frozen=True)
class GeometryCalibrationState:
    """Native-resolution detector/instrument geometry state carried across levels."""

    det_u_px: float = 0.0
    det_v_px: float = 0.0
    detector_roll_deg: float = 0.0
    axis_rot_x_deg: float = 0.0
    axis_rot_y_deg: float = 0.0
    nominal_axis_unit: tuple[float, float, float] = (0.0, 0.0, 1.0)
    active_geometry_dofs: tuple[str, ...] = ()

    @classmethod
    def from_geometry(
        cls,
        geometry: Geometry,
        *,
        active_geometry_dofs: Iterable[str] | None = None,
    ) -> GeometryCalibrationState:
        """Create native-resolution calibration state from a geometry."""
        geometry_inputs = geometry_inputs_from_geometry(geometry)
        active = normalize_geometry_dofs(active_geometry_dofs, geometry=geometry)
        nominal = tuple(float(v) for v in nominal_axis_unit_from_inputs(geometry_inputs))
        roll = float(geometry_inputs.get("detector_roll_deg", 0.0))
        state = cls(
            detector_roll_deg=roll,
            nominal_axis_unit=nominal,
            active_geometry_dofs=active,
        )
        validate_calibration_gauges(state.to_calibration_state())
        return state

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, object] | None,
        geometry: Geometry,
        *,
        active_geometry_dofs: Iterable[str] | None = None,
    ) -> GeometryCalibrationState:
        """Restore calibration state values from a checkpoint payload."""
        state = cls.from_geometry(geometry, active_geometry_dofs=active_geometry_dofs)
        if not isinstance(payload, Mapping):
            return state

        values: dict[str, float] = {}
        for section_name in ("detector", "scan"):
            section = payload.get(section_name)
            if not isinstance(section, list | tuple):
                continue
            for item in section:
                if not isinstance(item, Mapping):
                    continue
                name = item.get("name")
                if name in {
                    "det_u_px",
                    "det_v_px",
                    "detector_roll_deg",
                    "axis_rot_x_deg",
                    "axis_rot_y_deg",
                }:
                    raw_value = item.get("value")
                    if isinstance(raw_value, int | float):
                        values[str(name)] = float(raw_value)
        return replace(state, **values)

    def replace_values(self, names: Sequence[str], values: jnp.ndarray) -> GeometryCalibrationState:
        """Return a copy with named calibration values replaced."""
        updates: dict[str, float] = {}
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        for idx, name in enumerate(names):
            updates[str(name)] = float(arr[idx])
        return replace(self, **updates)

    def values_for(self, names: Sequence[str]) -> jnp.ndarray:
        """Return calibration values for named geometry DOFs."""
        return jnp.asarray([float(getattr(self, name)) for name in names], dtype=jnp.float32)

    def to_calibration_state(self) -> CalibrationState:
        """Convert alignment geometry state to the public calibration schema."""
        active = set(self.active_geometry_dofs)
        axis_unit = [float(v) for v in self.axis_unit_lab()]
        return CalibrationState(
            detector=(
                CalibrationVariable(
                    name="det_u_px",
                    value=float(self.det_u_px),
                    unit="native_detector_px",
                    status="estimated" if "det_u_px" in active else "frozen",
                    frame="detector",
                    gauge="detector_ray_grid_center",
                ),
                CalibrationVariable(
                    name="det_v_px",
                    value=float(self.det_v_px),
                    unit="native_detector_px",
                    status="estimated" if "det_v_px" in active else "frozen",
                    frame="detector",
                    gauge="detector_ray_grid_center",
                ),
                CalibrationVariable(
                    name="detector_roll_deg",
                    value=float(self.detector_roll_deg),
                    unit="deg",
                    status="estimated" if "detector_roll_deg" in active else "frozen",
                    frame="detector_plane",
                    gauge="detector_plane_roll",
                ),
            ),
            scan=(
                CalibrationVariable(
                    name="axis_rot_x_deg",
                    value=float(self.axis_rot_x_deg),
                    unit="deg",
                    status="estimated" if "axis_rot_x_deg" in active else "frozen",
                    frame="scan",
                    gauge="rotation_axis_direction",
                ),
                CalibrationVariable(
                    name="axis_rot_y_deg",
                    value=float(self.axis_rot_y_deg),
                    unit="deg",
                    status="estimated" if "axis_rot_y_deg" in active else "frozen",
                    frame="scan",
                    gauge="rotation_axis_direction",
                ),
                CalibrationVariable(
                    name="axis_unit_lab",
                    value=axis_unit,
                    unit="unit_vector",
                    status="derived",
                    frame="scan",
                    gauge="rotation_axis_direction",
                ),
            ),
        )

    def axis_unit_lab(self) -> tuple[float, float, float]:
        """Return the effective lab-frame rotation-axis unit vector."""
        axis = axis_unit_from_rotations(
            self.nominal_axis_unit,
            axis_rot_x_deg=float(self.axis_rot_x_deg),
            axis_rot_y_deg=float(self.axis_rot_y_deg),
        )
        return tuple(float(v) for v in np.asarray(axis))


def geometry_inputs_from_geometry(geometry: Geometry) -> dict[str, object]:
    """Extract calibration input fields from a concrete geometry object."""
    detector = geometry.detector
    grid = geometry.grid
    payload: dict[str, object] = {
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "thetas_deg": np.asarray(geometry.thetas_deg, dtype=np.float32),
        "geometry_type": "parallel",
    }
    if isinstance(geometry, LaminographyGeometry):
        payload["geometry_type"] = "lamino"
        payload["tilt_deg"] = float(geometry.tilt_deg)
        payload["tilt_about"] = str(geometry.tilt_about)
    elif isinstance(geometry, RotationAxisGeometry):
        payload["axis_unit_lab"] = list(geometry.axis_unit_lab)
    if getattr(geometry, "detector_roll_deg", None) is not None:
        payload["detector_roll_deg"] = float(geometry.detector_roll_deg)
    return payload


def geometry_with_axis_state(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    state: GeometryCalibrationState,
) -> Geometry:
    """Build a geometry object with the axis direction from calibration state."""
    thetas = np.asarray(geometry.thetas_deg, dtype=np.float32)
    axis_active = (
        abs(float(state.axis_rot_x_deg)) > 1e-7
        or abs(float(state.axis_rot_y_deg)) > 1e-7
        or isinstance(geometry, RotationAxisGeometry)
    )
    if axis_active:
        return RotationAxisGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            axis_unit_lab=state.axis_unit_lab(),
        )
    if isinstance(geometry, LaminographyGeometry):
        return LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            tilt_deg=float(geometry.tilt_deg),
            tilt_about=str(geometry.tilt_about),
        )
    return ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)


def level_detector_grid(
    detector: Detector,
    *,
    state: GeometryCalibrationState,
    factor: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return detector grid arrays with level-scaled calibration offsets."""
    factor_f = float(max(1, int(factor)))
    return detector_grid_from_calibration(
        detector,
        det_u_px=float(detector.det_center[0]) / float(detector.du)
        + jnp.asarray(state.det_u_px, dtype=jnp.float32) / jnp.float32(factor_f),
        det_v_px=float(detector.det_center[1]) / float(detector.dv)
        + jnp.asarray(state.det_v_px, dtype=jnp.float32) / jnp.float32(factor_f),
        detector_roll_deg=state.detector_roll_deg,
    )


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
    thetas = np.asarray(getattr(geometry, "thetas_deg", []), dtype=np.float32).reshape(-1)
    if thetas.size < 2:
        return None
    sorted_thetas = np.sort(np.mod(thetas, 360.0))
    deltas = np.diff(sorted_thetas)
    wrap_delta = float(sorted_thetas[0] + 360.0 - sorted_thetas[-1])
    circular_deltas = np.concatenate([deltas, np.asarray([wrap_delta], dtype=np.float32)])
    largest_gap_index = int(np.argmax(circular_deltas))
    largest_gap = float(circular_deltas[largest_gap_index])
    span = 360.0 - largest_gap
    if largest_gap_index == int(circular_deltas.size - 1):
        span += float(np.median(circular_deltas))
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
