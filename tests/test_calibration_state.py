from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tomojax.calibration import (
    CalibrationState,
    CalibrationVariable,
    ConventionAudit,
    ConventionEvidence,
    MetricSpec,
    ObjectiveCard,
    build_calibration_manifest,
)


def test_calibration_state_roundtrips_json_and_keeps_sections_separate():
    state = CalibrationState(
        detector=(
            CalibrationVariable(
                name="det_u_px",
                value=-4.0,
                unit="native_detector_px",
                status="estimated",
                frame="detector",
                gauge="detector_ray_grid_center",
            ),
        ),
        object_residual=(
            CalibrationVariable(
                name="object_phi_per_view",
                value=[0.0, 0.01],
                unit="rad",
                status="frozen",
                frame="object",
            ),
        ),
    )

    payload = state.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = CalibrationState.from_dict(payload)

    assert restored.detector[0].name == "det_u_px"
    assert restored.detector[0].status == "estimated"
    assert restored.object_residual[0].status == "frozen"
    assert restored.estimated_names() == {"det_u_px"}


def test_calibration_state_rejects_duplicate_variable_names():
    variable = CalibrationVariable(
        name="det_u_px",
        value=0.0,
        unit="native_detector_px",
        status="estimated",
        frame="detector",
    )

    with pytest.raises(ValueError, match="must be unique"):
        CalibrationState(detector=(variable,), scan=(variable,))


def test_calibration_variable_validates_status_and_frame():
    with pytest.raises(ValueError, match="status"):
        CalibrationVariable(
            name="bad",
            value=0.0,
            unit="px",
            status="free",  # type: ignore[arg-type]
            frame="detector",
        )

    with pytest.raises(ValueError, match="frame"):
        CalibrationVariable(
            name="bad",
            value=0.0,
            unit="px",
            status="estimated",
            frame="lab",  # type: ignore[arg-type]
        )


def test_calibration_manifest_contains_schema_objective_conventions_and_units():
    state = CalibrationState(
        detector=(
            CalibrationVariable(
                name="det_u_px",
                value=-4.0,
                unit="native_detector_px",
                status="estimated",
                frame="detector",
                gauge="detector_ray_grid_center",
                uncertainty={"sigma_px": 0.2},
            ),
            CalibrationVariable(
                name="det_v_px",
                value=0.0,
                unit="native_detector_px",
                status="frozen",
                frame="detector",
                gauge="detector_ray_grid_center",
            ),
        ),
    )
    objective = ObjectiveCard(primary_metric=MetricSpec(name="heldout_projection_mse"))
    convention = ConventionAudit(
        flip_u=True,
        flip_v=False,
        theta_sign=1,
        confidence=0.8,
        evidence=(ConventionEvidence(source="manual", description="flip_u visual check"),),
    )

    manifest = build_calibration_manifest(
        calibration_state=state,
        objective_card=objective,
        convention_audit=convention,
        calibrated_geometry={"detector": {"det_center": [-4.0, 0.0]}},
        source={"dataset": "k11-54014"},
        timestamp=datetime(2026, 4, 24, 10, 0, tzinfo=timezone.utc),
    )

    json.dumps(manifest, allow_nan=False)
    assert manifest["schema_version"] == 1
    assert manifest["created_at"] == "2026-04-24T10:00:00Z"
    assert manifest["calibration_state"]["detector"][0]["name"] == "det_u_px"
    assert manifest["objective_card"]["primary_metric"]["name"] == "heldout_projection_mse"
    assert manifest["convention_audit"]["flip_u"] is True
    assert manifest["convention_audit"]["ambiguous"] is False
