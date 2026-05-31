"""Calibration manifest construction helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict, cast

from tomojax.core.geometry.base import DetectorDict
from tomojax.geometry._calibration.json import JsonValue, drop_none, normalize_json
from tomojax.geometry._calibration.state import CalibrationState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.geometry._calibration.conventions import ConventionAudit
    from tomojax.geometry._calibration.objectives import ObjectiveCard

CALIBRATION_MANIFEST_SCHEMA_VERSION = 1

type JsonObject = dict[str, JsonValue]


class GeometryCalibrationPatch(TypedDict):
    """Persisted calibration metadata patch for NXtomo outputs."""

    calibration_state: JsonObject


class CalibratedGeometryMetadataPatch(TypedDict):
    """Typed detector and geometry metadata updates from calibration state."""

    detector: DetectorDict
    geometry_meta: JsonObject
    geometry_calibration: GeometryCalibrationPatch


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
) -> CalibratedGeometryMetadataPatch:
    """Build detector and geometry metadata updates from a calibration state."""
    state = (
        calibration_state
        if isinstance(calibration_state, CalibrationState)
        else CalibrationState.from_dict(calibration_state)
    )
    variables = state.variables_by_name()
    detector_patch = _detector_metadata_patch(detector)
    geometry_meta_patch = _json_object(geometry_meta)

    det_center = _detector_center(detector_patch)

    det_u = variables.get("det_u_px")
    det_v = variables.get("det_v_px")
    if det_u is not None:
        det_center[0] = det_center[0] + _real(det_u.value, field="det_u_px") * detector_patch["du"]
    if det_v is not None:
        det_center[1] = det_center[1] + _real(det_v.value, field="det_v_px") * detector_patch["dv"]
    detector_patch["det_center"] = det_center

    roll = variables.get("detector_roll_deg")
    if roll is not None:
        geometry_meta_patch["detector_roll_deg"] = _real(roll.value, field="detector_roll_deg")

    axis_unit = variables.get("axis_unit_lab")
    if axis_unit is not None and isinstance(axis_unit.value, list):
        geometry_meta_patch["axis_unit_lab"] = [
            _real(value, field="axis_unit_lab") for value in axis_unit.value
        ]

    return {
        "detector": detector_patch,
        "geometry_meta": geometry_meta_patch,
        "geometry_calibration": {"calibration_state": state.to_dict()},
    }


def _detector_metadata_patch(detector: Mapping[str, object]) -> DetectorDict:
    detector_json = _json_object(detector)
    missing = [key for key in ("nu", "nv", "du", "dv") if key not in detector_json]
    if missing:
        raise ValueError(
            "calibrated geometry detector metadata requires "
            f"{', '.join(missing)} before applying calibration offsets"
        )
    return {
        "nu": _integer(detector_json["nu"], field="nu"),
        "nv": _integer(detector_json["nv"], field="nv"),
        "du": _real(detector_json["du"], field="du"),
        "dv": _real(detector_json["dv"], field="dv"),
        "det_center": _detector_center(detector_json),
    }


def _detector_center(detector_patch: Mapping[str, object]) -> list[float]:
    det_center_raw = detector_patch.get("det_center", [0.0, 0.0])
    if isinstance(det_center_raw, list):
        det_center = cast("list[object]", det_center_raw.copy())
    else:
        det_center: list[object] = [0.0, 0.0]
    while len(det_center) < 2:
        det_center.append(0.0)
    return [
        _real(det_center[0], field="det_center[0]"),
        _real(det_center[1], field="det_center[1]"),
    ]


def _json_object(value: Mapping[str, object] | None) -> JsonObject:
    normalized = normalize_json(value)
    return normalized if isinstance(normalized, dict) else {}


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"{field} must be numeric")
    return int(value)


def _real(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"{field} must be numeric")
    return float(value)
