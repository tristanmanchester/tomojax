from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from tomojax.geometry import CalibrationState, CalibrationVariable, axis_unit_from_rotations


def _scalar(value: object) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.float32).reshape(())


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SetupGeometryState:
    """Optimizer-time setup geometry state.

    Detector centre offsets are always native/full-resolution detector pixels.
    Angular fields are stored in radians; degree-facing names are handled by the
    DOF registry and metadata export.
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
        return {
            "detector_roll_deg": jnp.rad2deg(self.detector_roll_rad),
            "axis_rot_x_deg": jnp.rad2deg(self.axis_rot_x_rad),
            "axis_rot_y_deg": jnp.rad2deg(self.axis_rot_y_rad),
            "tilt_deg": jnp.rad2deg(self.tilt_rad),
        }

    def axis_unit_lab(self) -> jnp.ndarray:
        deg = self.degrees_dict()
        return axis_unit_from_rotations(
            self.nominal_axis_unit,
            axis_rot_x_deg=deg["axis_rot_x_deg"],
            axis_rot_y_deg=deg["axis_rot_y_deg"],
        )

    def replace(self, **updates: object) -> SetupGeometryState:
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

    def tree_flatten(self):
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
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PoseState:
    """Optimizer-time per-view pose state."""

    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None = None

    @classmethod
    def zeros(cls, n_views: int) -> PoseState:
        return cls(jnp.zeros((int(n_views), 5), dtype=jnp.float32))

    def replace(self, **updates: object) -> PoseState:
        values = {"params5": self.params5, "motion_coeffs": self.motion_coeffs}
        values.update(updates)
        return PoseState(**values)

    def tree_flatten(self):
        return (self.params5, self.motion_coeffs), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        params5, motion_coeffs = children
        return cls(params5=params5, motion_coeffs=motion_coeffs)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AlignmentState:
    """Unified optimizer-time alignment state."""

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
        volume = (
            None
            if volume_shape is None
            else jnp.zeros(tuple(int(v) for v in volume_shape), dtype=jnp.float32)
        )
        return cls(setup=SetupGeometryState(), pose=PoseState.zeros(n_views), volume=volume)

    def replace(self, **updates: object) -> AlignmentState:
        values = {"setup": self.setup, "pose": self.pose, "volume": self.volume}
        values.update(updates)
        return AlignmentState(**values)

    def to_calibration_state(
        self,
        *,
        active_dofs: Iterable[str] = (),
    ) -> CalibrationState:
        active = set(str(name) for name in active_dofs)

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

    def tree_flatten(self):
        return (self.setup, self.pose, self.volume), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
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
