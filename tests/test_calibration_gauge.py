from __future__ import annotations

import pytest

from tomojax.calibration import CalibrationState, CalibrationVariable
from tomojax.calibration.gauge import GaugeValidationError, validate_calibration_gauges


def _var(name: str, status: str) -> CalibrationVariable:
    return CalibrationVariable(
        name=name,
        value=0.0,
        unit="px",
        status=status,  # type: ignore[arg-type]
        frame="detector" if name.startswith("det_") else "world",
    )


@pytest.mark.parametrize(
    ("left", "right", "code"),
    [
        ("det_u_px", "world_dx", "det_u_px_world_dx"),
        ("det_v_px", "world_dz", "det_v_px_world_dz"),
        ("theta0_deg", "object_phi_mean", "theta0_object_phi_mean"),
        (
            "detector_roll_deg",
            "object_in_plane_orientation",
            "detector_roll_object_orientation",
        ),
        ("object_translation_mean", "volume_center", "object_translation_mean_volume_center"),
    ],
)
def test_calibration_gauge_conflicts_fail_when_both_variables_are_estimated(
    left: str, right: str, code: str
):
    with pytest.raises(GaugeValidationError) as exc:
        validate_calibration_gauges({left: "estimated", right: "estimated"})

    assert exc.value.conflicts[0].code == code


def test_calibration_gauge_allows_fixed_detector_center_with_object_residuals():
    state = CalibrationState(
        detector=(_var("det_u_px", "supplied"),),
        world_residual=(_var("world_dx", "estimated"),),
    )

    assert validate_calibration_gauges(state) == ()


def test_calibration_gauge_can_return_structured_conflicts_without_raising():
    conflicts = validate_calibration_gauges(
        {"det_u_px": "estimated", "world_dx": "estimated"},
        hard_fail=False,
    )

    assert conflicts[0].to_dict()["code"] == "det_u_px_world_dx"
    assert conflicts[0].to_dict()["variables"] == ["det_u_px", "world_dx"]
