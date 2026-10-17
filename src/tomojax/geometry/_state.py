"""Typed geometry state containers."""
# pyright: reportAny=false

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

GaugeGroup = Literal["detector_u", "detector_v", "rotation", "axis", "none"]
AcquisitionModel = Literal["parallel", "parallel_laminography"]
LaminographyTiltAbout = Literal["x", "z"]


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
    """Named scanner setup parameters and their optimization metadata.

    Parameters
    ----------
    det_u_px, det_v_px : ScalarParameter
        Detector u/v centre offsets in pixels.
    detector_roll_rad : ScalarParameter
        Detector-plane roll in radians.
    axis_rot_x_rad, axis_rot_y_rad : ScalarParameter
        Lab-axis calibration rotations in radians.
    theta_offset_rad : ScalarParameter
        Global rotation-angle offset in radians.
    theta_scale : ScalarParameter
        Dimensionless scale multiplying nominal view angles.
    """

    det_u_px: ScalarParameter
    det_v_px: ScalarParameter
    detector_roll_rad: ScalarParameter
    axis_rot_x_rad: ScalarParameter
    axis_rot_y_rad: ScalarParameter
    theta_offset_rad: ScalarParameter
    theta_scale: ScalarParameter

    @classmethod
    def defaults(cls) -> SetupParameters:
        """Construct the default uncorrected scanner setup.

        Returns
        -------
        SetupParameters
            Zero offsets and rotations, unit angle scale; v offset and angle scale are inactive.
        """
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
        """Replace one named setup parameter.

        Parameters
        ----------
        name : str
            Setup field to replace.
        parameter : ScalarParameter
            Replacement value and optimization metadata.

        Returns
        -------
        SetupParameters
            Setup with the named parameter replaced.

        Raises
        ------
        TypeError
            If name is not a setup field.
        """
        return replace(self, **{name: parameter})


@dataclass(frozen=True)
class AcquisitionParameters:
    """Acquisition model and nominal laminography tilt.

    Parameters
    ----------
    model : {'parallel', 'parallel_laminography'}
        Acquisition geometry family.
    laminography_tilt_rad : float, optional
        Fixed laminography-axis tilt in radians; defaults to zero.
    laminography_tilt_about : {'x', 'z'}
        Tilt convention, used for parallel laminography.
    """

    model: AcquisitionModel = "parallel"
    laminography_tilt_rad: float = 0.0
    laminography_tilt_about: LaminographyTiltAbout = "x"

    @classmethod
    def parallel(cls) -> AcquisitionParameters:
        """Construct axial parallel-beam acquisition parameters.

        Returns
        -------
        AcquisitionParameters
            Axial parallel-beam model with no fixed tilt.
        """
        return cls()

    @classmethod
    def parallel_laminography(
        cls,
        *,
        tilt_rad: float,
        tilt_about: LaminographyTiltAbout = "x",
    ) -> AcquisitionParameters:
        """Construct tilted parallel-ray acquisition parameters.

        Parameters
        ----------
        tilt_rad : float
            Nominal laminography-axis tilt in radians.
        tilt_about : {'x', 'z'}
            Tilt convention; defaults to x.

        Returns
        -------
        AcquisitionParameters
            Tilted parallel-ray acquisition model.
        """
        return cls(
            model="parallel_laminography",
            laminography_tilt_rad=float(tilt_rad),
            laminography_tilt_about=tilt_about,
        )


@dataclass(frozen=True)
class PoseParameters:
    """Per-view pose parameters, separating nominal angles and corrections.

    Parameters
    ----------
    alpha_rad, beta_rad : numpy.ndarray
        Per-view angular corrections in radians.
    theta_nominal_rad : numpy.ndarray
        Nominal stage angles in radians.
    phi_residual_rad : numpy.ndarray
        Per-view residual rotation about the nominal axis, in radians.
    dx_px, dz_px : numpy.ndarray
        Per-view detector-plane translations in pixels.

    Raises
    ------
    ValueError
        If arrays are not one-dimensional with a common shape.
    """

    alpha_rad: NDArray[np.float64]
    beta_rad: NDArray[np.float64]
    theta_nominal_rad: NDArray[np.float64]
    phi_residual_rad: NDArray[np.float64]
    dx_px: NDArray[np.float64]
    dz_px: NDArray[np.float64]

    @classmethod
    def zeros(cls, n_views: int) -> PoseParameters:
        """Construct zero nominal angles and pose corrections.

        Parameters
        ----------
        n_views : int
            Number of views.

        Returns
        -------
        PoseParameters
            Independent float64 vectors initialized to zero.
        """
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
        """Number of views in the pose table.

        Returns
        -------
        int
            Length of each pose parameter vector.
        """
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
        alpha_rad: NDArray[np.float64] | None = None,
        beta_rad: NDArray[np.float64] | None = None,
        theta_nominal_rad: NDArray[np.float64] | None = None,
        phi_residual_rad: NDArray[np.float64] | None = None,
        dx_px: NDArray[np.float64] | None = None,
        dz_px: NDArray[np.float64] | None = None,
    ) -> PoseParameters:
        """Replace selected pose parameter vectors.

        Parameters
        ----------
        alpha_rad, beta_rad : numpy.ndarray or None, optional
            Per-view angular corrections in radians. None retains the current vector.
        theta_nominal_rad : numpy.ndarray or None, optional
            Nominal stage angles in radians. None retains the current vector.
        phi_residual_rad : numpy.ndarray or None, optional
            Per-view residual rotation about the nominal axis, in radians.
            None retains the current vector.
        dx_px, dz_px : numpy.ndarray or None, optional
            Per-view detector-plane translations in pixels. None retains the current vector.

        Returns
        -------
        PoseParameters
            Pose table with the supplied vectors replaced.

        Raises
        ------
        ValueError
            If updated vectors do not share a one-dimensional shape.
        """
        return replace(
            self,
            alpha_rad=self.alpha_rad if alpha_rad is None else alpha_rad,
            beta_rad=self.beta_rad if beta_rad is None else beta_rad,
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
    """Scanner setup, per-view pose and acquisition model together.

    Parameters
    ----------
    setup : SetupParameters
        Global scanner calibration parameters.
    pose : PoseParameters
        Per-view nominal angles and corrections.
    acquisition : AcquisitionParameters, optional
        Acquisition model; defaults to axial parallel beams.
    """

    setup: SetupParameters
    pose: PoseParameters
    acquisition: AcquisitionParameters = field(default_factory=AcquisitionParameters.parallel)

    @classmethod
    def zeros(cls, n_views: int) -> GeometryState:
        """Construct an uncorrected geometry state.

        Parameters
        ----------
        n_views : int
            Number of views.

        Returns
        -------
        GeometryState
            Default setup and parallel acquisition with zero pose vectors.

        Examples
        --------
        >>> from tomojax.geometry.api import GeometryState
        >>> state = GeometryState.zeros(3)
        >>> state.theta_total_rad().tolist()
        [0.0, 0.0, 0.0]
        """
        return cls(
            setup=SetupParameters.defaults(),
            pose=PoseParameters.zeros(n_views),
            acquisition=AcquisitionParameters.parallel(),
        )

    def theta_total_rad(self) -> NDArray[np.float64]:
        """Return the complete rotation angle of each view.

        Returns
        -------
        numpy.ndarray
            Scaled nominal angles plus global offset and per-view residual, in radians.
        """
        return (
            self.setup.theta_scale.value * self.pose.theta_nominal_rad
            + self.setup.theta_offset_rad.value
            + self.pose.phi_residual_rad
        )
