"""Apply setup and pose calibration state to geometry arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.axis import RotationAxisGeometry
from tomojax.core.geometry.lamino import LaminographyGeometry
from tomojax.core.geometry.parallel import ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.geometry import (
    axis_pose_stack,
    axis_unit_from_rotations,
    detector_grid_from_calibration,
)

from .parametrizations import se3_from_5d

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.align._model.state import AlignmentState, SetupGeometryState
    from tomojax.core.geometry import Detector, Geometry, Grid


@dataclass(frozen=True, slots=True)
class BaseGeometryArrays:
    """Array/static bundle used by differentiable alignment objectives."""

    thetas_deg: jnp.ndarray
    nominal_pose_stack: jnp.ndarray
    detector: Detector
    nominal_axis_unit: jnp.ndarray
    nominal_tilt_deg: float | None = None
    tilt_about: str | None = None
    level_factor: int = 1

    @classmethod
    def from_geometry(
        cls,
        geometry: Geometry,
        detector: Detector,
        *,
        level_factor: int = 1,
    ) -> BaseGeometryArrays:
        """Build base arrays from a concrete geometry and detector."""
        thetas = jnp.asarray(geometry.thetas_deg, dtype=jnp.float32)
        n_views = int(thetas.shape[0])
        is_lamino = isinstance(geometry, LaminographyGeometry)
        return cls(
            thetas_deg=thetas,
            nominal_pose_stack=stack_view_poses(geometry, n_views),
            detector=detector,
            nominal_axis_unit=_nominal_axis_unit_from_geometry(geometry),
            nominal_tilt_deg=float(geometry.tilt_deg) if is_lamino else None,
            tilt_about=str(geometry.tilt_about) if is_lamino else None,
            level_factor=max(1, int(level_factor)),
        )


@dataclass(frozen=True, slots=True)
class EffectiveGeometryArrays:
    """Pose, detector-grid, and axis arrays after applying alignment state."""

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
    nominal = axis_unit_from_rotations(
        setup.nominal_axis_unit,
        axis_rot_x_deg=jnp.rad2deg(setup.tilt_rad),
        axis_rot_y_deg=0.0,
    )
    return axis_unit_from_rotations(
        nominal,
        axis_rot_x_deg=jnp.rad2deg(setup.axis_rot_x_rad),
        axis_rot_y_deg=jnp.rad2deg(setup.axis_rot_y_rad),
    )


def _laminography_axis_unit_jax(tilt_deg: object, tilt_about: str) -> jnp.ndarray:
    tilt = jnp.deg2rad(jnp.asarray(tilt_deg, dtype=jnp.float32))
    c, s = jnp.cos(tilt), jnp.sin(tilt)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    if str(tilt_about) == "x":
        axis = jnp.stack((zero, s, c))
    elif str(tilt_about) == "z":
        axis = jnp.stack((s, zero, c))
    else:
        raise ValueError("tilt_about must be 'x' or 'z'")
    return axis / jnp.maximum(jnp.linalg.norm(axis), jnp.float32(1e-8))


def _setup_axis_unit_for_base(
    base: BaseGeometryArrays,
    setup: SetupGeometryState,
) -> jnp.ndarray:
    if base.nominal_tilt_deg is None:
        nominal = axis_unit_from_rotations(
            base.nominal_axis_unit,
            axis_rot_x_deg=jnp.rad2deg(setup.tilt_rad),
            axis_rot_y_deg=0.0,
        )
    else:
        nominal = _laminography_axis_unit_jax(
            jnp.asarray(base.nominal_tilt_deg, dtype=jnp.float32) + jnp.rad2deg(setup.tilt_rad),
            str(base.tilt_about),
        )
    return axis_unit_from_rotations(
        nominal,
        axis_rot_x_deg=jnp.rad2deg(setup.axis_rot_x_rad),
        axis_rot_y_deg=jnp.rad2deg(setup.axis_rot_y_rad),
    )


def _setup_axis_unit_for_geometry(
    geometry: Geometry,
    setup: SetupGeometryState,
) -> jnp.ndarray:
    if isinstance(geometry, LaminographyGeometry):
        nominal = _laminography_axis_unit_jax(
            float(geometry.tilt_deg) + jnp.rad2deg(setup.tilt_rad),
            str(geometry.tilt_about),
        )
    else:
        nominal = axis_unit_from_rotations(
            _nominal_axis_unit_from_geometry(geometry),
            axis_rot_x_deg=jnp.rad2deg(setup.tilt_rad),
            axis_rot_y_deg=0.0,
        )
    return axis_unit_from_rotations(
        nominal,
        axis_rot_x_deg=jnp.rad2deg(setup.axis_rot_x_rad),
        axis_rot_y_deg=jnp.rad2deg(setup.axis_rot_y_rad),
    )


def materialize_setup_geometry(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    setup: SetupGeometryState,
    *,
    indices: Sequence[int] | None = None,
) -> Geometry:
    """Build a Python geometry object for setup-state projection/reconstruction calls."""
    thetas = np.asarray(geometry.thetas_deg, dtype=np.float32)
    if indices is not None:
        thetas = thetas[np.asarray(indices, dtype=np.int32)]
    axis = tuple(
        float(v)
        for v in np.asarray(_setup_axis_unit_for_geometry(geometry, setup), dtype=np.float32)
    )
    setup_rotated_axis = (
        abs(float(setup.axis_rot_x_rad)) > 1e-7
        or abs(float(setup.axis_rot_y_rad)) > 1e-7
        or (not isinstance(geometry, LaminographyGeometry) and abs(float(setup.tilt_rad)) > 1e-7)
    )
    if isinstance(geometry, RotationAxisGeometry) or setup_rotated_axis:
        return RotationAxisGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            axis_unit_lab=axis,
        )
    if isinstance(geometry, LaminographyGeometry):
        return LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            tilt_deg=float(geometry.tilt_deg) + float(np.rad2deg(float(setup.tilt_rad))),
            tilt_about=str(geometry.tilt_about),
        )
    return ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)


def apply_alignment_state(
    base: BaseGeometryArrays,
    state: AlignmentState,
) -> EffectiveGeometryArrays:
    """Apply setup and pose state to base arrays without Python geometry mutation."""
    setup = state.setup
    axis = _setup_axis_unit_for_base(base, setup)
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
    """Return the setup-adjusted view pose stack for a base geometry bundle."""
    axis = _setup_axis_unit_for_base(base, setup)
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
    """Return a view-subsetted copy of a base geometry array bundle."""
    idx = jnp.asarray(indices, dtype=jnp.int32)
    return BaseGeometryArrays(
        thetas_deg=base.thetas_deg[idx],
        nominal_pose_stack=base.nominal_pose_stack[idx],
        detector=base.detector,
        nominal_axis_unit=base.nominal_axis_unit,
        nominal_tilt_deg=base.nominal_tilt_deg,
        tilt_about=base.tilt_about,
        level_factor=base.level_factor,
    )
