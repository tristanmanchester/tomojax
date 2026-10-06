"""Geometry artifact serialization."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax.geometry._state import (
    AcquisitionParameters,
    GaugeGroup,
    GeometryState,
    LaminographyTiltAbout,
    PoseParameters,
    ScalarParameter,
    SetupParameters,
)

if TYPE_CHECKING:
    from pathlib import Path

GEOMETRY_STATE_SCHEMA_VERSION = 1
POSE_PARAMS_FIELDS = (
    "view",
    "alpha_rad",
    "beta_rad",
    "theta_nominal_rad",
    "phi_residual_rad",
    "dx_px",
    "dz_px",
)
POSE_DECOMPOSITION_FIELDS = (
    "view",
    "theta_nominal_rad",
    "realized_theta_total_rad",
    "realized_det_u_px",
    "realized_det_v_px",
)


def geometry_state_from_dict(payload: dict[str, object], pose: PoseParameters) -> GeometryState:
    raw_schema_version = payload.get("schema_version", 0)
    if not isinstance(raw_schema_version, int | float | str):
        raise ValueError("geometry schema_version must be numeric")
    schema_version = int(raw_schema_version)
    if schema_version != GEOMETRY_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported geometry schema_version {schema_version}")
    setup_payload = cast("dict[str, object]", payload["setup"])
    return GeometryState(
        setup=SetupParameters(
            det_u_px=_parameter_from_dict(setup_payload["det_u_px"]),
            det_v_px=_parameter_from_dict(setup_payload["det_v_px"]),
            detector_roll_rad=_parameter_from_dict(setup_payload["detector_roll_rad"]),
            axis_rot_x_rad=_parameter_from_dict(setup_payload["axis_rot_x_rad"]),
            axis_rot_y_rad=_parameter_from_dict(setup_payload["axis_rot_y_rad"]),
            theta_offset_rad=_parameter_from_dict(setup_payload["theta_offset_rad"]),
            theta_scale=_parameter_from_dict(setup_payload["theta_scale"]),
        ),
        pose=pose,
        acquisition=_acquisition_from_dict(payload.get("acquisition")),
    )


def _acquisition_from_dict(payload: object) -> AcquisitionParameters:
    if not isinstance(payload, dict):
        return AcquisitionParameters.parallel()
    data = cast("dict[object, object]", payload)
    raw_model = data.get("model", "parallel")
    model = str(raw_model)
    if model == "parallel":
        return AcquisitionParameters.parallel()
    if model != "parallel_laminography":
        raise ValueError(f"unsupported acquisition model {model!r}")
    raw_tilt = data.get("laminography_tilt_rad", 0.0)
    raw_about = str(data.get("laminography_tilt_about", "x"))
    if raw_about not in {"x", "z"}:
        raise ValueError("laminography_tilt_about must be 'x' or 'z'")
    return AcquisitionParameters.parallel_laminography(
        tilt_rad=float(raw_tilt) if isinstance(raw_tilt, int | float | str) else 0.0,
        tilt_about=cast("LaminographyTiltAbout", raw_about),
    )


def read_geometry_json(path: Path, pose: PoseParameters) -> GeometryState:
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    return geometry_state_from_dict(payload, pose)


def read_pose_params_csv(path: Path) -> PoseParameters:
    columns: dict[str, list[float]] = {field: [] for field in POSE_PARAMS_FIELDS if field != "view"}
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for field, values in columns.items():
                values.append(float(row.get(field, 0.0)))
    return PoseParameters(
        alpha_rad=np.asarray(columns["alpha_rad"], dtype=np.float64),
        beta_rad=np.asarray(columns["beta_rad"], dtype=np.float64),
        theta_nominal_rad=np.asarray(columns["theta_nominal_rad"], dtype=np.float64),
        phi_residual_rad=np.asarray(columns["phi_residual_rad"], dtype=np.float64),
        dx_px=np.asarray(columns["dx_px"], dtype=np.float64),
        dz_px=np.asarray(columns["dz_px"], dtype=np.float64),
    )


def _parameter_from_dict(payload: object) -> ScalarParameter:
    data = cast("dict[str, Any]", payload)
    return ScalarParameter(
        name=str(data["name"]),
        value=float(data["value"]),
        unit=str(data["unit"]),
        scale=float(data.get("scale", 1.0)),
        active=bool(data.get("active", True)),
        prior=float(data["prior"]) if data.get("prior") is not None else None,
        trust_radius=float(data["trust_radius"]) if data.get("trust_radius") is not None else None,
        gauge_group=cast("GaugeGroup", data.get("gauge_group", "none")),
    )
