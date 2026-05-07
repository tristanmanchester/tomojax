"""Typed v2 geometry state containers."""
# pyright: reportAny=false

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

GaugeGroup = Literal["detector_u", "detector_v", "rotation", "axis", "none"]


@dataclass(frozen=True)
class ScalarParameter:
    name: str
    value: float
    unit: str
    scale: float = 1.0
    active: bool = True
    prior: float | None = None
    trust_radius: float | None = None
    gauge_group: GaugeGroup = "none"

    def with_value(self, value: float) -> ScalarParameter:
        return replace(self, value=float(value))


@dataclass(frozen=True)
class SetupParameters:
    det_u_px: ScalarParameter
    det_v_px: ScalarParameter
    detector_roll_rad: ScalarParameter
    axis_rot_x_rad: ScalarParameter
    axis_rot_y_rad: ScalarParameter
    theta_offset_rad: ScalarParameter
    theta_scale: ScalarParameter

    @classmethod
    def defaults(cls) -> SetupParameters:
        return cls(
            det_u_px=ScalarParameter("det_u_px", 0.0, "px", gauge_group="detector_u"),
            det_v_px=ScalarParameter("det_v_px", 0.0, "px", active=False, gauge_group="detector_v"),
            detector_roll_rad=ScalarParameter(
                "detector_roll_rad", 0.0, "rad", gauge_group="rotation"
            ),
            axis_rot_x_rad=ScalarParameter("axis_rot_x_rad", 0.0, "rad", gauge_group="axis"),
            axis_rot_y_rad=ScalarParameter("axis_rot_y_rad", 0.0, "rad", gauge_group="axis"),
            theta_offset_rad=ScalarParameter(
                "theta_offset_rad", 0.0, "rad", gauge_group="rotation"
            ),
            theta_scale=ScalarParameter("theta_scale", 1.0, "dimensionless", active=False),
        )

    def replace_parameter(self, name: str, parameter: ScalarParameter) -> SetupParameters:
        return replace(self, **{name: parameter})


@dataclass(frozen=True)
class PoseParameters:
    alpha_rad: NDArray[np.float64]
    beta_rad: NDArray[np.float64]
    theta_nominal_rad: NDArray[np.float64]
    phi_residual_rad: NDArray[np.float64]
    dx_px: NDArray[np.float64]
    dz_px: NDArray[np.float64]

    @classmethod
    def zeros(cls, n_views: int) -> PoseParameters:
        values = np.zeros(int(n_views), dtype=np.float64)
        return cls(
            alpha_rad=values.copy(),
            beta_rad=values.copy(),
            theta_nominal_rad=values.copy(),
            phi_residual_rad=values.copy(),
            dx_px=values.copy(),
            dz_px=values.copy(),
        )

    @property
    def n_views(self) -> int:
        return int(self.dx_px.shape[0])

    def __post_init__(self) -> None:
        shapes = {
            self.alpha_rad.shape,
            self.beta_rad.shape,
            self.theta_nominal_rad.shape,
            self.phi_residual_rad.shape,
            self.dx_px.shape,
            self.dz_px.shape,
        }
        if len(shapes) != 1:
            raise ValueError("all pose parameter arrays must have the same shape")
        if len(next(iter(shapes))) != 1:
            raise ValueError("pose parameter arrays must be one-dimensional")

    def with_updates(
        self,
        *,
        theta_nominal_rad: NDArray[np.float64] | None = None,
        phi_residual_rad: NDArray[np.float64] | None = None,
        dx_px: NDArray[np.float64] | None = None,
        dz_px: NDArray[np.float64] | None = None,
    ) -> PoseParameters:
        return replace(
            self,
            theta_nominal_rad=(
                self.theta_nominal_rad if theta_nominal_rad is None else theta_nominal_rad
            ),
            phi_residual_rad=(
                self.phi_residual_rad if phi_residual_rad is None else phi_residual_rad
            ),
            dx_px=self.dx_px if dx_px is None else dx_px,
            dz_px=self.dz_px if dz_px is None else dz_px,
        )


@dataclass(frozen=True)
class GeometryState:
    setup: SetupParameters
    pose: PoseParameters

    @classmethod
    def zeros(cls, n_views: int) -> GeometryState:
        return cls(setup=SetupParameters.defaults(), pose=PoseParameters.zeros(n_views))

    def theta_total_rad(self) -> NDArray[np.float64]:
        return (
            self.setup.theta_scale.value * self.pose.theta_nominal_rad
            + self.setup.theta_offset_rad.value
            + self.pose.phi_residual_rad
        )
