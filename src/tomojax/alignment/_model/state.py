"""Optimizer-time alignment state containers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import jax.numpy as jnp

from tomojax.alignment._geometry.parametrizations import PoseTranslationFrame
from tomojax.alignment._model.dofs import POSE_WIDTH
from tomojax.geometry.api import CalibrationState, CalibrationVariable, axis_unit_from_rotations


def _scalar(value: object) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.float32).reshape(())


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SetupGeometryState:
    """Optimizer-time setup geometry state.

    Detector centre offsets are always native/full-resolution detector pixels.
    Angular fields are stored in radians; degree-facing names are handled by the
    DOF registry and metadata export.

    Parameters
    ----------
    det_u_px : jax.Array, optional
        Detector-u centre offset in native/full-resolution pixels. Defaults to zero.
    det_v_px : jax.Array, optional
        Detector-v centre offset in native/full-resolution pixels. Defaults to zero.
    detector_roll_rad : jax.Array, optional
        Detector roll in radians. Defaults to zero.
    axis_rot_x_rad : jax.Array, optional
        Rotation-axis x correction in radians. Defaults to zero.
    axis_rot_y_rad : jax.Array, optional
        Rotation-axis y correction in radians. Defaults to zero.
    tilt_rad : jax.Array, optional
        Nominal axis tilt in radians. Defaults to zero.
    nominal_axis_unit : jax.Array, optional
        Nominal lab-frame rotation-axis unit vector of shape (3,); defaults to +z in
        setup state.
    """

    det_u_px: jnp.ndarray = field(default_factory=lambda: _scalar(0.0))
    det_v_px: jnp.ndarray = field(default_factory=lambda: _scalar(0.0))
    detector_roll_rad: jnp.ndarray = field(default_factory=lambda: _scalar(0.0))
    axis_rot_x_rad: jnp.ndarray = field(default_factory=lambda: _scalar(0.0))
    axis_rot_y_rad: jnp.ndarray = field(default_factory=lambda: _scalar(0.0))
    tilt_rad: jnp.ndarray = field(default_factory=lambda: _scalar(0.0))
    nominal_axis_unit: jnp.ndarray = field(
        default_factory=lambda: jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float32)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "det_u_px", _scalar(self.det_u_px))
        object.__setattr__(self, "det_v_px", _scalar(self.det_v_px))
        object.__setattr__(self, "detector_roll_rad", _scalar(self.detector_roll_rad))
        object.__setattr__(self, "axis_rot_x_rad", _scalar(self.axis_rot_x_rad))
        object.__setattr__(self, "axis_rot_y_rad", _scalar(self.axis_rot_y_rad))
        object.__setattr__(self, "tilt_rad", _scalar(self.tilt_rad))
        object.__setattr__(
            self,
            "nominal_axis_unit",
            jnp.asarray(self.nominal_axis_unit, dtype=jnp.float32).reshape((3,)),
        )

    @classmethod
    def from_degrees(
        cls,
        *,
        det_u_px: object = 0.0,
        det_v_px: object = 0.0,
        detector_roll_deg: object = 0.0,
        axis_rot_x_deg: object = 0.0,
        axis_rot_y_deg: object = 0.0,
        tilt_deg: object = 0.0,
        nominal_axis_unit: object = (0.0, 0.0, 1.0),
    ) -> SetupGeometryState:
        """Build setup state from degree-facing metadata values.

        Parameters
        ----------
        det_u_px : float, optional
            Native/full-resolution detector-u pixel offset. Default is zero.
        det_v_px : float, optional
            Native/full-resolution detector-v pixel offset. Default is zero.
        detector_roll_deg : float, optional
            Detector roll in degrees. Default is zero.
        axis_rot_x_deg : float, optional
            Axis x rotation in degrees. Default is zero.
        axis_rot_y_deg : float, optional
            Axis y rotation in degrees. Default is zero.
        tilt_deg : float, optional
            Nominal tilt in degrees. Default is zero.
        nominal_axis_unit : array-like, optional
            Nominal lab-frame unit axis of shape (3,). Default is +z.

        Returns
        -------
        SetupGeometryState
            Setup state with angular fields converted to radians.
        """
        return cls(
            det_u_px=_scalar(det_u_px),
            det_v_px=_scalar(det_v_px),
            detector_roll_rad=jnp.deg2rad(_scalar(detector_roll_deg)),
            axis_rot_x_rad=jnp.deg2rad(_scalar(axis_rot_x_deg)),
            axis_rot_y_rad=jnp.deg2rad(_scalar(axis_rot_y_deg)),
            tilt_rad=jnp.deg2rad(_scalar(tilt_deg)),
            nominal_axis_unit=jnp.asarray(nominal_axis_unit, dtype=jnp.float32),
        )

    def degrees_dict(self) -> dict[str, jnp.ndarray]:
        """Return angular setup fields in degrees.

        Returns
        -------
        dict
            Detector roll, axis rotations and tilt as scalar JAX arrays in degrees.
        """
        return {
            "detector_roll_deg": jnp.rad2deg(self.detector_roll_rad),
            "axis_rot_x_deg": jnp.rad2deg(self.axis_rot_x_rad),
            "axis_rot_y_deg": jnp.rad2deg(self.axis_rot_y_rad),
            "tilt_deg": jnp.rad2deg(self.tilt_rad),
        }

    def axis_unit_lab(self) -> jnp.ndarray:
        """Return the realised lab-frame rotation-axis unit vector.

        Returns
        -------
        jax.Array
            Lab-frame rotation-axis unit vector of shape (3,).
        """
        deg = self.degrees_dict()
        nominal = axis_unit_from_rotations(
            self.nominal_axis_unit,
            axis_rot_x_deg=deg["tilt_deg"],
            axis_rot_y_deg=0.0,
        )
        return axis_unit_from_rotations(
            nominal,
            axis_rot_x_deg=deg["axis_rot_x_deg"],
            axis_rot_y_deg=deg["axis_rot_y_deg"],
        )

    def replace(self, **updates: object) -> SetupGeometryState:
        """Return a copy with selected setup fields replaced.

        Parameters
        ----------
        **updates : object
            Named fields to replace; all other fields retain their values.

        Returns
        -------
        SetupGeometryState
            New state with the requested changes.

        Raises
        ------
        TypeError
            If a keyword does not name a state field.
        """
        values = {
            "det_u_px": self.det_u_px,
            "det_v_px": self.det_v_px,
            "detector_roll_rad": self.detector_roll_rad,
            "axis_rot_x_rad": self.axis_rot_x_rad,
            "axis_rot_y_rad": self.axis_rot_y_rad,
            "tilt_rad": self.tilt_rad,
            "nominal_axis_unit": self.nominal_axis_unit,
        }
        values.update(updates)
        return SetupGeometryState(**values)

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, ...], None]:
        """Flatten this state for JAX pytree handling.

        Returns
        -------
        children : tuple
            Scalar setup fields and the nominal axis vector.
        aux_data : object
            None; no static auxiliary data are needed.
        """
        children = (
            self.det_u_px,
            self.det_v_px,
            self.detector_roll_rad,
            self.axis_rot_x_rad,
            self.axis_rot_y_rad,
            self.tilt_rad,
            self.nominal_axis_unit,
        )
        return children, None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: object,
        children: tuple[jnp.ndarray, ...],
    ) -> SetupGeometryState:
        """Rebuild setup state from JAX pytree children.

        Parameters
        ----------
        aux_data : object
            None; no static auxiliary data are needed.
        children : tuple
            Scalar setup fields and the nominal axis vector.

        Returns
        -------
        SetupGeometryState
            State reconstructed from its pytree representation.
        """
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PoseState:
    """Optimizer-time per-view pose state.

    Parameters
    ----------
    pose_params : jax.Array
        Per-view corrections, columns (alpha, beta, phi, dx, dz, dy); angles are radians
        and translations physical lengths.
    motion_coeffs : jax.Array | None, optional
        Coefficients of a polynomial or spline motion model; None for independent per-view
         parameters.
    translation_frame : PoseTranslationFrame, optional
        Basis for physical translations, "object" or lab-frame "detector". Default is
        'object'.

    Raises
    ------
    ValueError
        If translation_frame is not "object" or "detector".
    """

    pose_params: jnp.ndarray
    motion_coeffs: jnp.ndarray | None = None
    translation_frame: PoseTranslationFrame = "object"

    def __post_init__(self) -> None:
        if self.translation_frame not in {"object", "detector"}:
            raise ValueError("pose translation frame must be 'object' or 'detector'")

    @classmethod
    def zeros(cls, n_views: int) -> PoseState:
        """Return a zero pose table for the requested number of views.

        Parameters
        ----------
        n_views : int
            Number of views in the scan.

        Returns
        -------
        PoseState
            Zero-valued state with a six-column pose table.
        """
        return cls(jnp.zeros((int(n_views), POSE_WIDTH), dtype=jnp.float32))

    def replace(self, **updates: object) -> PoseState:
        """Return a copy with selected pose fields replaced.

        Parameters
        ----------
        **updates : object
            Named fields to replace; all other fields retain their values.

        Returns
        -------
        PoseState
            New state with the requested changes.

        Raises
        ------
        TypeError
            If a keyword does not name a state field.
        """
        values = {
            "pose_params": self.pose_params,
            "motion_coeffs": self.motion_coeffs,
            "translation_frame": self.translation_frame,
        }
        values.update(updates)
        return PoseState(**values)

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, jnp.ndarray | None], PoseTranslationFrame]:
        """Flatten this state for JAX pytree handling.

        Returns
        -------
        children : tuple
            Pose table and optional motion coefficients.
        aux_data : object
            Translation-frame string.
        """
        return (self.pose_params, self.motion_coeffs), self.translation_frame

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: PoseTranslationFrame,
        children: tuple[jnp.ndarray, jnp.ndarray | None],
    ) -> PoseState:
        """Rebuild pose state from JAX pytree children.

        Parameters
        ----------
        aux_data : object
            Translation-frame string.
        children : tuple
            Pose table and optional motion coefficients.

        Returns
        -------
        PoseState
            State reconstructed from its pytree representation.
        """
        pose_params, motion_coeffs = children
        return cls(pose_params=pose_params, motion_coeffs=motion_coeffs, translation_frame=aux_data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AlignmentState:
    """Unified optimizer-time alignment state.

    Parameters
    ----------
    setup : SetupGeometryState
        Global detector and axis calibration state.
    pose : PoseState
        Per-view pose-correction state.
    volume : jax.Array | None, optional
        Optional reconstruction in (z, y, x) order.
    """

    setup: SetupGeometryState
    pose: PoseState
    volume: jnp.ndarray | None = None

    @classmethod
    def zeros(
        cls,
        *,
        n_views: int,
        volume_shape: tuple[int, int, int] | None = None,
    ) -> AlignmentState:
        """Return an alignment state with zero setup, pose, and optional volume.

        Parameters
        ----------
        n_views : int
            Number of views in the scan.
        volume_shape : tuple of int or None, optional
            Optional (nz, ny, nx) shape for a zero volume. None leaves volume absent.

        Returns
        -------
        AlignmentState
            Zero-valued state with a six-column pose table.
        """
        volume = (
            None
            if volume_shape is None
            else jnp.zeros(tuple(int(v) for v in volume_shape), dtype=jnp.float32)
        )
        return cls(setup=SetupGeometryState(), pose=PoseState.zeros(n_views), volume=volume)

    def replace(self, **updates: object) -> AlignmentState:
        """Return a copy with selected alignment fields replaced.

        Parameters
        ----------
        **updates : object
            Named fields to replace; all other fields retain their values.

        Returns
        -------
        AlignmentState
            New state with the requested changes.

        Raises
        ------
        TypeError
            If a keyword does not name a state field.
        """
        values = {"setup": self.setup, "pose": self.pose, "volume": self.volume}
        values.update(updates)
        return AlignmentState(**values)

    def to_calibration_state(
        self,
        *,
        active_dofs: Iterable[str] = (),
    ) -> CalibrationState:
        """Convert optimizer state into persisted calibration metadata.

        Parameters
        ----------
        active_dofs : iterable of str, optional
            Public parameter names to include in persisted metadata.

        Returns
        -------
        CalibrationState
            Persistable setup and per-view pose corrections with explicit units.
        """
        active = {str(name) for name in active_dofs}

        def status(name: str) -> str:
            return "estimated" if name in active else "frozen"

        deg = self.setup.degrees_dict()
        axis_unit = self.setup.axis_unit_lab()
        return CalibrationState(
            detector=(
                CalibrationVariable(
                    name="det_u_px",
                    value=float(self.setup.det_u_px),
                    unit="native_detector_px",
                    status=status("det_u_px"),  # type: ignore[arg-type]
                    frame="detector",
                    gauge="detector_ray_grid_center",
                ),
                CalibrationVariable(
                    name="det_v_px",
                    value=float(self.setup.det_v_px),
                    unit="native_detector_px",
                    status=status("det_v_px"),  # type: ignore[arg-type]
                    frame="detector",
                    gauge="detector_ray_grid_center",
                ),
                CalibrationVariable(
                    name="detector_roll_deg",
                    value=float(deg["detector_roll_deg"]),
                    unit="deg",
                    status=status("detector_roll_deg"),  # type: ignore[arg-type]
                    frame="detector_plane",
                    gauge="detector_plane_roll",
                ),
            ),
            scan=(
                CalibrationVariable(
                    name="axis_rot_x_deg",
                    value=float(deg["axis_rot_x_deg"]),
                    unit="deg",
                    status=status("axis_rot_x_deg"),  # type: ignore[arg-type]
                    frame="scan",
                    gauge="rotation_axis_direction",
                ),
                CalibrationVariable(
                    name="axis_rot_y_deg",
                    value=float(deg["axis_rot_y_deg"]),
                    unit="deg",
                    status=status("axis_rot_y_deg"),  # type: ignore[arg-type]
                    frame="scan",
                    gauge="rotation_axis_direction",
                ),
                CalibrationVariable(
                    name="axis_unit_lab",
                    value=[float(v) for v in axis_unit],
                    unit="unit_vector",
                    status="derived",
                    frame="scan",
                    gauge="rotation_axis_direction",
                ),
            ),
        )

    def tree_flatten(
        self,
    ) -> tuple[tuple[SetupGeometryState, PoseState, jnp.ndarray | None], None]:
        """Flatten this state for JAX pytree handling.

        Returns
        -------
        children : tuple
            Setup state, pose state and optional volume.
        aux_data : object
            None; no static auxiliary data are needed.
        """
        return (self.setup, self.pose, self.volume), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: object,
        children: tuple[SetupGeometryState, PoseState, jnp.ndarray | None],
    ) -> AlignmentState:
        """Rebuild alignment state from JAX pytree children.

        Parameters
        ----------
        aux_data : object
            None; no static auxiliary data are needed.
        children : tuple
            Setup state, pose state and optional volume.

        Returns
        -------
        AlignmentState
            State reconstructed from its pytree representation.
        """
        del aux_data
        setup, pose, volume = children
        return cls(setup=setup, pose=pose, volume=volume)


def alignment_state_from_checkpoint(
    payload: Mapping[str, object] | None,
    *,
    n_views: int,
    volume: jnp.ndarray | None = None,
) -> AlignmentState:
    """Build an `AlignmentState` from optional metadata-like payload values."""
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(n_views), volume=volume)
    if not isinstance(payload, Mapping):
        return state
    setup_values: dict[str, object] = {}
    for section_name in ("detector", "scan"):
        section = payload.get(section_name)
        if not isinstance(section, list | tuple):
            continue
        for item in section:
            if not isinstance(item, Mapping):
                continue
            item = cast("Mapping[str, Any]", item)
            name = str(item.get("name", ""))
            raw_value = item.get("value")
            if raw_value is None:
                continue
            if name in {"det_u_px", "det_v_px"}:
                setup_values[name] = raw_value
            elif name == "detector_roll_deg":
                setup_values["detector_roll_rad"] = jnp.deg2rad(_scalar(raw_value))
            elif name == "axis_rot_x_deg":
                setup_values["axis_rot_x_rad"] = jnp.deg2rad(_scalar(raw_value))
            elif name == "axis_rot_y_deg":
                setup_values["axis_rot_y_rad"] = jnp.deg2rad(_scalar(raw_value))
    return state.replace(setup=state.setup.replace(**setup_values))
