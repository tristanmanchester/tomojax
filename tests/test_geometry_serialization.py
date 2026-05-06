from __future__ import annotations

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import csv
from dataclasses import replace
import json
from typing import Any, cast

import numpy as np

from tomojax.geometry import (
    GEOMETRY_STATE_SCHEMA_VERSION,
    GeometryState,
    PoseParameters,
    SetupParameters,
    read_geometry_json,
    read_pose_params_csv,
    write_geometry_json,
    write_pose_decomposition_csv,
    write_pose_params_csv,
)


def test_geometry_json_and_pose_csv_round_trip_contract_artifacts(tmp_path) -> None:
    state = _example_state()
    geometry_path = tmp_path / "geometry_initial.json"
    pose_path = tmp_path / "pose_params.csv"

    write_geometry_json(geometry_path, state)
    write_pose_params_csv(pose_path, state.pose)

    pose = read_pose_params_csv(pose_path)
    restored = read_geometry_json(geometry_path, pose)

    payload = cast("dict[str, Any]", json.loads(geometry_path.read_text(encoding="utf-8")))
    assert payload["schema_version"] == GEOMETRY_STATE_SCHEMA_VERSION
    assert payload["setup"]["det_u_px"]["unit"] == "px"
    assert payload["setup"]["det_v_px"]["active"] is True
    assert payload["pose"]["n_views"] == 3
    assert restored.setup == state.setup
    np.testing.assert_allclose(restored.pose.alpha_rad, state.pose.alpha_rad)
    np.testing.assert_allclose(restored.pose.beta_rad, state.pose.beta_rad)
    np.testing.assert_allclose(restored.pose.phi_residual_rad, state.pose.phi_residual_rad)
    np.testing.assert_allclose(restored.pose.dx_px, state.pose.dx_px)
    np.testing.assert_allclose(restored.pose.dz_px, state.pose.dz_px)


def test_geometry_final_json_uses_same_state_schema(tmp_path) -> None:
    state = _example_state()
    path = tmp_path / "geometry_final.json"

    write_geometry_json(path, state)

    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    assert payload["setup"]["theta_offset_rad"]["value"] == 0.2
    assert payload["setup"]["theta_scale"]["value"] == 1.0


def test_pose_decomposition_csv_records_realized_setup_plus_pose(tmp_path) -> None:
    state = _example_state()
    path = tmp_path / "pose_decomposition.csv"

    write_pose_decomposition_csv(path, state)

    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 3
    assert set(rows[0]) == {
        "view",
        "realized_theta_offset_rad",
        "realized_det_u_px",
        "realized_det_v_px",
    }
    assert float(rows[0]["realized_theta_offset_rad"]) == 0.25
    assert float(rows[1]["realized_det_u_px"]) == -1.5
    assert float(rows[2]["realized_det_v_px"]) == 2.75


def _example_state() -> GeometryState:
    setup = SetupParameters.defaults()
    setup = setup.replace_parameter("det_u_px", setup.det_u_px.with_value(-2.0))
    setup = setup.replace_parameter(
        "det_v_px",
        replace(setup.det_v_px.with_value(1.5), active=True),
    )
    setup = setup.replace_parameter("theta_offset_rad", setup.theta_offset_rad.with_value(0.2))
    pose = PoseParameters(
        alpha_rad=np.array([0.01, 0.02, 0.03], dtype=np.float64),
        beta_rad=np.array([-0.01, -0.02, -0.03], dtype=np.float64),
        phi_residual_rad=np.array([0.05, 0.0, -0.05], dtype=np.float64),
        dx_px=np.array([0.25, 0.5, 0.75], dtype=np.float64),
        dz_px=np.array([1.0, 1.125, 1.25], dtype=np.float64),
    )
    return GeometryState(setup=setup, pose=pose)
