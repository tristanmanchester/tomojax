"""Calibration manifest construction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from tomojax._json import JsonValue, normalize_json
from tomojax._typed_arrays import object_list
from tomojax.core.geometry.base import DetectorDict
from tomojax.core.geometry.cone import ROLL_CONVENTION_KEY
from tomojax.geometry._calibration.state import CalibrationState

if TYPE_CHECKING:
    from collections.abc import Mapping


type JsonObject = dict[str, JsonValue]


class GeometryCalibrationPatch(TypedDict):
    """Persisted calibration metadata patch for NXtomo outputs."""

    calibration_state: JsonObject


class CalibratedGeometryMetadataPatch(TypedDict):
    """Typed detector and geometry metadata updates from calibration state.

    Attributes
    ----------
    detector : dict
        Updated detector dimensions, pitches and physical center.
    geometry_meta : dict
        Updated detector roll and axis metadata.
    geometry_calibration : dict
        Serialized calibration state.
    """

    detector: DetectorDict
    geometry_meta: JsonObject
    geometry_calibration: GeometryCalibrationPatch


def build_calibrated_geometry_metadata_patch(
    *,
    calibration_state: CalibrationState | Mapping[str, object],
    detector: Mapping[str, object],
    geometry_meta: Mapping[str, object] | None = None,
) -> CalibratedGeometryMetadataPatch:
    """Build detector and geometry metadata updates from a calibration state.

    Parameters
    ----------
    calibration_state : CalibrationState or mapping
        Grouped calibration values or their serialized representation.
    detector : mapping
        Original detector metadata with nu, nv, du and dv.
    geometry_meta : mapping or None, optional
        Original geometry metadata to preserve and update.

    Returns
    -------
    CalibratedGeometryMetadataPatch
        Detector offsets converted from pixels to physical lengths, with calibration provenance.

    Raises
    ------
    ValueError
        If required detector metadata or calibration values are invalid.
    TypeError
        If detector dimensions or calibration offsets are not numeric.
    """
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
    shift = (
        0.0 if det_u is None else _real(det_u.value, field="det_u_px") * detector_patch["du"],
        0.0 if det_v is None else _real(det_v.value, field="det_v_px") * detector_patch["dv"],
    )
    detector_patch["det_center"] = [det_center[0] + shift[0], det_center[1] + shift[1]]
    # A segmented geometry's own detectors move with it, keeping their offsets.
    raw_segments = geometry_meta_patch.get("cone_segments") or []
    if not isinstance(raw_segments, list):
        raise TypeError("cone_segments must be a list")
    segments: list[JsonObject] = []
    for entry in raw_segments:
        if not isinstance(entry, dict):
            raise TypeError("cone_segments must contain metadata objects")
        segments.append(entry)
        segment_detector = entry.get("detector")
        if isinstance(segment_detector, dict):
            u, v = _detector_center(segment_detector)
            entry["detector"] = {**segment_detector, "det_center": [u + shift[0], v + shift[1]]}

    roll = variables.get("detector_roll")
    if roll is not None:
        value = _real(roll.value, field="detector_roll")
        beams = [geometry_meta_patch.get("cone_beam")] + [e.get("cone_beam") for e in segments]
        beams = [beam for beam in beams if isinstance(beam, dict)]
        if beams:
            # A cone detector rolls in its beam, in the parallel roll's sense.
            for beam in beams:
                beam["detector_roll"] = value
                beam[ROLL_CONVENTION_KEY] = 1.0
        else:
            geometry_meta_patch["detector_roll"] = value

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
    det_center = object_list(det_center_raw) if isinstance(det_center_raw, list) else []
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
