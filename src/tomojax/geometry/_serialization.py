"""Geometry artifact serialization."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax.geometry._state import (
    GaugeGroup,
    GeometryState,
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


def geometry_state_to_dict(state: GeometryState) -> dict[str, object]:
    return {
        "schema_version": GEOMETRY_STATE_SCHEMA_VERSION,
        "setup": {
            "det_u_px": _parameter_to_dict(state.setup.det_u_px),
            "det_v_px": _parameter_to_dict(state.setup.det_v_px),
            "detector_roll_rad": _parameter_to_dict(state.setup.detector_roll_rad),
            "axis_rot_x_rad": _parameter_to_dict(state.setup.axis_rot_x_rad),
            "axis_rot_y_rad": _parameter_to_dict(state.setup.axis_rot_y_rad),
            "theta_offset_rad": _parameter_to_dict(state.setup.theta_offset_rad),
            "theta_scale": _parameter_to_dict(state.setup.theta_scale),
        },
        "pose": {"n_views": state.pose.n_views},
    }


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
    )


def write_geometry_json(path: Path, state: GeometryState) -> None:
    _ = path.write_text(
        json.dumps(geometry_state_to_dict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_geometry_json(path: Path, pose: PoseParameters) -> GeometryState:
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    return geometry_state_from_dict(payload, pose)


def write_pose_params_csv(path: Path, pose: PoseParameters) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=POSE_PARAMS_FIELDS)
        writer.writeheader()
        for view in range(pose.n_views):
            writer.writerow(
                {
                    "view": view,
                    "alpha_rad": float(pose.alpha_rad[view]),
                    "beta_rad": float(pose.beta_rad[view]),
                    "theta_nominal_rad": float(pose.theta_nominal_rad[view]),
                    "phi_residual_rad": float(pose.phi_residual_rad[view]),
                    "dx_px": float(pose.dx_px[view]),
                    "dz_px": float(pose.dz_px[view]),
                }
            )


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


def write_pose_decomposition_csv(path: Path, state: GeometryState) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=POSE_DECOMPOSITION_FIELDS)
        writer.writeheader()
        for view in range(state.pose.n_views):
            writer.writerow(
                {
                    "view": view,
                    "theta_nominal_rad": float(state.pose.theta_nominal_rad[view]),
                    "realized_theta_total_rad": float(state.theta_total_rad()[view]),
                    "realized_det_u_px": state.setup.det_u_px.value + float(state.pose.dx_px[view]),
                    "realized_det_v_px": state.setup.det_v_px.value + float(state.pose.dz_px[view]),
                }
            )


def _parameter_to_dict(parameter: ScalarParameter) -> dict[str, object]:
    return asdict(parameter)


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
