from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.calibration.axis_geometry import axis_pose_stack, axis_unit_from_rotations
from tomojax.calibration.detector_grid import detector_grid_from_calibration
from tomojax.core.geometry import Detector, Geometry
from tomojax.core.geometry.axis import RotationAxisGeometry
from tomojax.core.geometry.lamino import LaminographyGeometry
from tomojax.core.geometry.views import stack_view_poses

from .parametrizations import se3_from_5d
from ..model.state import AlignmentState, SetupGeometryState


@dataclass(frozen=True, slots=True)
class BaseGeometryArrays:
    """Array/static bundle used by differentiable alignment objectives."""

    thetas_deg: jnp.ndarray
    nominal_pose_stack: jnp.ndarray
    detector: Detector
    nominal_axis_unit: jnp.ndarray
    level_factor: int = 1

    @classmethod
    def from_geometry(
        cls,
        geometry: Geometry,
        detector: Detector,
        *,
        level_factor: int = 1,
    ) -> "BaseGeometryArrays":
        thetas = jnp.asarray(getattr(geometry, "thetas_deg"), dtype=jnp.float32)
        n_views = int(thetas.shape[0])
        return cls(
            thetas_deg=thetas,
            nominal_pose_stack=stack_view_poses(geometry, n_views),
            detector=detector,
            nominal_axis_unit=_nominal_axis_unit_from_geometry(geometry),
            level_factor=max(1, int(level_factor)),
        )


@dataclass(frozen=True, slots=True)
class EffectiveGeometryArrays:
    pose_stack: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]
    axis_unit_lab: jnp.ndarray


def apply_setup_to_detector_grid(
    detector: Detector,
    setup: SetupGeometryState,
    *,
    level_factor: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply native-pixel setup offsets and detector roll to a level detector."""
    factor = jnp.asarray(max(1, int(level_factor)), dtype=jnp.float32)
    det_u_level_px = (
        jnp.asarray(float(detector.det_center[0]) / float(detector.du), dtype=jnp.float32)
        + setup.det_u_px / factor
    )
    det_v_level_px = (
        jnp.asarray(float(detector.det_center[1]) / float(detector.dv), dtype=jnp.float32)
        + setup.det_v_px / factor
    )
    return detector_grid_from_calibration(
        detector,
        det_u_px=det_u_level_px,
        det_v_px=det_v_level_px,
        detector_roll_deg=jnp.rad2deg(setup.detector_roll_rad),
    )


def setup_axis_unit(setup: SetupGeometryState) -> jnp.ndarray:
    """Return effective lab-frame axis unit from radian setup rotations."""
    return axis_unit_from_rotations(
        setup.nominal_axis_unit,
        axis_rot_x_deg=jnp.rad2deg(setup.axis_rot_x_rad),
        axis_rot_y_deg=jnp.rad2deg(setup.axis_rot_y_rad),
    )


def apply_alignment_state(
    base: BaseGeometryArrays,
    state: AlignmentState,
) -> EffectiveGeometryArrays:
    """Apply setup and pose state to base arrays without Python geometry mutation."""
    setup = state.setup
    axis = setup_axis_unit(setup)
    setup_pose = axis_pose_stack(base.thetas_deg, axis)
    pose_delta = jax.vmap(se3_from_5d)(state.pose.params5)
    pose_stack = setup_pose @ pose_delta
    det_grid = apply_setup_to_detector_grid(
        base.detector,
        setup,
        level_factor=base.level_factor,
    )
    return EffectiveGeometryArrays(pose_stack=pose_stack, det_grid=det_grid, axis_unit_lab=axis)


def pose_stack_for_setup(
    base: BaseGeometryArrays,
    setup: SetupGeometryState,
) -> jnp.ndarray:
    axis = setup_axis_unit(setup)
    return axis_pose_stack(base.thetas_deg, axis)


def _nominal_axis_unit_from_geometry(geometry: Geometry) -> jnp.ndarray:
    if isinstance(geometry, RotationAxisGeometry):
        axis = np.asarray(geometry.axis_unit_lab, dtype=np.float32)
    elif isinstance(geometry, LaminographyGeometry):
        axis = np.asarray(geometry._axis_unit(), dtype=np.float32)
    else:
        axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis).all() or norm < 1e-12:
        raise ValueError("geometry nominal axis must be a finite nonzero 3-vector")
    return jnp.asarray(axis / norm, dtype=jnp.float32)


def subset_base_geometry(
    base: BaseGeometryArrays,
    indices: Sequence[int],
) -> BaseGeometryArrays:
    idx = jnp.asarray(indices, dtype=jnp.int32)
    return BaseGeometryArrays(
        thetas_deg=base.thetas_deg[idx],
        nominal_pose_stack=base.nominal_pose_stack[idx],
        detector=base.detector,
        nominal_axis_unit=base.nominal_axis_unit,
        level_factor=base.level_factor,
    )
