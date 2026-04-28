from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ._json import JsonValue
from .state import CalibrationState, CalibrationVariable


@dataclass(frozen=True)
class GaugeConflict:
    """A coupled pair of estimated variables that makes a gauge underdetermined."""

    code: str
    variables: tuple[str, str]
    message: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "variables": list(self.variables),
            "message": self.message,
        }


class GaugeValidationError(ValueError):
    """Raised when a calibration state contains underdetermined gauge choices."""

    def __init__(self, conflicts: Iterable[GaugeConflict]):
        self.conflicts = tuple(conflicts)
        codes = ", ".join(conflict.code for conflict in self.conflicts)
        super().__init__(f"Calibration gauge conflicts: {codes}")


_CONFLICT_RULES: tuple[GaugeConflict, ...] = (
    GaugeConflict(
        code="det_u_px_world_dx",
        variables=("det_u_px", "world_dx"),
        message="Estimate detector/ray-grid u-centre or static world dx, not both.",
    ),
    GaugeConflict(
        code="det_u_px_rotation_axis_intercept_u",
        variables=("det_u_px", "rotation_axis_intercept_u_px"),
        message=(
            "Estimate detector/ray-grid u-centre or rotation-axis u-intercept, not both "
            "under the detector-centre gauge."
        ),
    ),
    GaugeConflict(
        code="det_v_px_world_dz",
        variables=("det_v_px", "world_dz"),
        message="Estimate detector/ray-grid v-centre or static world dz, not both.",
    ),
    GaugeConflict(
        code="det_v_px_rotation_axis_intercept_v",
        variables=("det_v_px", "rotation_axis_intercept_v_px"),
        message=(
            "Estimate detector/ray-grid v-centre or rotation-axis v-intercept, not both "
            "under the detector-centre gauge."
        ),
    ),
    GaugeConflict(
        code="axis_rot_x_object_alpha_mean",
        variables=("axis_rot_x_deg", "object_alpha_mean"),
        message="Estimate scanner axis x-tilt or mean object alpha, not both.",
    ),
    GaugeConflict(
        code="axis_rot_y_object_beta_mean",
        variables=("axis_rot_y_deg", "object_beta_mean"),
        message="Estimate scanner axis y-tilt or mean object beta, not both.",
    ),
    GaugeConflict(
        code="axis_unit_lab_tilt_pair",
        variables=("axis_unit_lab", "axis_tilt_deg"),
        message="Estimate axis unit vector or tilt/azimuth parameterization, not both.",
    ),
    GaugeConflict(
        code="theta0_object_phi_mean",
        variables=("theta0_deg", "object_phi_mean"),
        message="Estimate angle zero or mean object phi, not both.",
    ),
    GaugeConflict(
        code="detector_roll_object_orientation",
        variables=("detector_roll_deg", "object_in_plane_orientation"),
        message="Estimate detector roll or global object in-plane orientation, not both.",
    ),
    GaugeConflict(
        code="detector_roll_object_phi_mean",
        variables=("detector_roll_deg", "object_phi_mean"),
        message="Estimate detector roll or mean object phi, not both.",
    ),
    GaugeConflict(
        code="object_translation_mean_volume_center",
        variables=("object_translation_mean", "volume_center"),
        message="Estimate object mean translation or volume centre, not both.",
    ),
)


def _variables_from_input(
    variables: CalibrationState | Iterable[CalibrationVariable] | Mapping[str, object],
) -> dict[str, str]:
    if isinstance(variables, CalibrationState):
        return {variable.name: variable.status for variable in variables.variables()}
    if isinstance(variables, Mapping):
        return {str(name): str(status) for name, status in variables.items()}
    return {variable.name: variable.status for variable in variables}


def validate_calibration_gauges(
    variables: CalibrationState | Iterable[CalibrationVariable] | Mapping[str, object],
    *,
    hard_fail: bool = True,
) -> tuple[GaugeConflict, ...]:
    """Validate hard gauge conflicts for estimated calibration variables."""
    statuses = _variables_from_input(variables)
    estimated = {name for name, status in statuses.items() if status == "estimated"}
    conflicts = tuple(
        rule
        for rule in _CONFLICT_RULES
        if rule.variables[0] in estimated and rule.variables[1] in estimated
    )
    if conflicts and hard_fail:
        raise GaugeValidationError(conflicts)
    return conflicts
