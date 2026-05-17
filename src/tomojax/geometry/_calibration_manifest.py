"""Calibration manifest construction helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tomojax.geometry._calibration_json import JsonValue, drop_none, normalize_json
from tomojax.geometry._calibration_state import CalibrationState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.geometry._calibration_conventions import ConventionAudit
    from tomojax.geometry._calibration_objectives import ObjectiveCard

CALIBRATION_MANIFEST_SCHEMA_VERSION = 1


def _timestamp_utc(timestamp: datetime | str | None) -> str:
    if timestamp is None:
        timestamp = datetime.now(UTC)
    if isinstance(timestamp, str):
        return timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _tomojax_version() -> str | None:
    try:
        from importlib import metadata

        return metadata.version("tomojax")
    except Exception:
        try:
            import tomojax

            return getattr(tomojax, "__version__", None)
        except Exception:
            return None


def build_calibration_manifest(
    *,
    calibration_state: CalibrationState,
    objective_card: ObjectiveCard | None = None,
    convention_audit: ConventionAudit | None = None,
    calibrated_geometry: Mapping[str, object] | None = None,
    source: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
    timestamp: datetime | str | None = None,
) -> dict[str, JsonValue]:
    """Build a strict JSON-compatible manifest for geometry calibration metadata."""
    return drop_none(
        {
            "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
            "created_at": _timestamp_utc(timestamp),
            "tomojax_version": _tomojax_version(),
            "calibration_state": calibration_state.to_dict(),
            "objective_card": objective_card.to_dict() if objective_card is not None else None,
            "convention_audit": convention_audit.to_dict()
            if convention_audit is not None
            else None,
            "calibrated_geometry": normalize_json(calibrated_geometry),
            "source": normalize_json(source),
            "extra": normalize_json(extra),
        }
    )


def build_calibrated_geometry_metadata_patch(
    *,
    calibration_state: CalibrationState | Mapping[str, object],
    detector: Mapping[str, object],
    geometry_meta: Mapping[str, object] | None = None,
) -> dict[str, JsonValue]:
    """Build detector and geometry metadata updates from a calibration state."""
    state = (
        calibration_state
        if isinstance(calibration_state, CalibrationState)
        else CalibrationState.from_dict(calibration_state)
    )
    variables = state.variables_by_name()
    detector_patch = dict(normalize_json(detector) or {})
    geometry_meta_patch = dict(normalize_json(geometry_meta) or {})

    det_center_raw = detector_patch.get("det_center", [0.0, 0.0])
    det_center = list(det_center_raw) if isinstance(det_center_raw, list) else [0.0, 0.0]
    if len(det_center) < 2:
        det_center = [*det_center, *([0.0] * (2 - len(det_center)))]

    det_u = variables.get("det_u_px")
    det_v = variables.get("det_v_px")
    if det_u is not None:
        det_center[0] = float(det_center[0]) + float(det_u.value) * float(detector_patch["du"])
    if det_v is not None:
        det_center[1] = float(det_center[1]) + float(det_v.value) * float(detector_patch["dv"])
    detector_patch["det_center"] = det_center

    roll = variables.get("detector_roll_deg")
    if roll is not None:
        geometry_meta_patch["detector_roll_deg"] = float(roll.value)

    axis_unit = variables.get("axis_unit_lab")
    if axis_unit is not None and isinstance(axis_unit.value, list):
        geometry_meta_patch["axis_unit_lab"] = [float(value) for value in axis_unit.value]

    return {
        "detector": detector_patch,
        "geometry_meta": geometry_meta_patch,
        "geometry_calibration": {"calibration_state": state.to_dict()},
    }
