"""Apply setup and pose calibration state to geometry arrays."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.axis import RotationAxisGeometry
from tomojax.core.geometry.lamino import LaminographyGeometry
from tomojax.core.geometry.parallel import ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.geometry.api import (
    axis_pose_stack,
    axis_unit_from_rotations,
    detector_grid_from_calibration,
)

from .parametrizations import apply_pose_updates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.alignment._model.state import AlignmentState, SetupGeometryState
    from tomojax.core.geometry import Detector, Geometry, Grid


def nominal_setup_geometry(geometry: Geometry) -> Geometry:
    """Remove saved detector roll, which the setup state carries explicitly.

    Keep any pose corrections: those cannot be replaced by an axis direction.
    """
    from tomojax._data.geometry_meta import DetectorRollGeometry

    while isinstance(geometry, DetectorRollGeometry):
        geometry = geometry.base
    return geometry


@dataclass(frozen=True, slots=True)
class BaseGeometryArrays:
    """Array/static bundle used by differentiable alignment objectives.

    Parameters
    ----------
    angles : jax.Array
        View-angle array in degrees.
    nominal_pose_stack : jax.Array
        Nominal object-to-world transforms with shape (views, 4, 4).
    detector : Detector
        Current-level detector dimensions and physical placement.
    nominal_axis_unit : jax.Array
        Nominal lab-frame rotation-axis unit vector of shape (3,); defaults to +z in
        setup state.
    nominal_tilt_deg : float | None, optional
        Nominal laminography tilt in degrees, or None for other geometry.
    tilt_about : str | None, optional
        Laminography tilt axis name, or None for other geometry.
    level_factor : int, optional
        Current coarsening factor relative to the native grid and detector. Default is
        1.
    """

    angles: jnp.ndarray
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
        """Build base arrays from a concrete geometry and detector.

        Parameters
        ----------
        geometry : Geometry
            Nominal geometry with its view angles and poses.
        detector : Detector
            Current-level detector shape and placement.
        level_factor : int, optional
            Coarsening factor relative to the native detector. Default is 1.

        Returns
        -------
        BaseGeometryArrays
            View angles, poses, detector and nominal axis information for the level.
        """
        thetas = jnp.asarray(geometry.angles, dtype=jnp.float32)
        n_views = int(thetas.shape[0])
        is_lamino = isinstance(geometry, LaminographyGeometry)
        return cls(
            angles=thetas,
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
        jnp.asarray(float(detector.center[0]) / float(detector.du), dtype=jnp.float32)
        + setup.det_u_px / factor
    )
    det_v_level_px = (
        jnp.asarray(float(detector.center[1]) / float(detector.dv), dtype=jnp.float32)
        + setup.det_v_px / factor
    )
    return detector_grid_from_calibration(
        detector,
        det_u_px=det_u_level_px,
        det_v_px=det_v_level_px,
        detector_roll_deg=jnp.rad2deg(setup.detector_roll_rad),
    )


def setup_moved_detector(
    detector: Detector,
    setup: SetupGeometryState,
    *,
    level_factor: int = 1,
) -> Detector | None:
    """``detector`` with the setup's native-pixel offsets in its centre, if it has no roll.

    The same detector as :func:`apply_setup_to_detector_grid`'s grid, as a
    :class:`Detector` the batched (CUDA) operators take; None when the setup
    rolls the detector, which only the grid describes.
    """
    if abs(float(setup.detector_roll_rad)) > 0.0:
        return None
    factor = float(max(1, int(level_factor)))
    u, v = detector.center
    moved = (
        u + float(setup.det_u_px) / factor * detector.du,
        v + float(setup.det_v_px) / factor * detector.dv,
    )
    return replace(detector, center=moved)


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
    # These are host degree metadata, not projector arithmetic. Preserve
    # their represented values through view selection and pose generation.
    thetas = np.asarray(geometry.angles, dtype=np.float64)
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
            angles=thetas,
            axis_unit_lab=axis,
        )
    if isinstance(geometry, LaminographyGeometry):
        return LaminographyGeometry(
            grid=grid,
            detector=detector,
            angles=thetas,
            tilt_deg=float(geometry.tilt_deg) + float(np.rad2deg(float(setup.tilt_rad))),
            tilt_about=str(geometry.tilt_about),
        )
    return ParallelGeometry(grid=grid, detector=detector, angles=thetas)


def apply_alignment_state(
    base: BaseGeometryArrays,
    state: AlignmentState,
) -> EffectiveGeometryArrays:
    """Apply setup and pose state to base arrays without Python geometry mutation.

    Parameters
    ----------
    base : BaseGeometryArrays
        Nominal view poses, detector and axis arrays.
    state : AlignmentState
        Setup and pose corrections to apply.

    Returns
    -------
    EffectiveGeometryArrays
        Updated pose stack, detector coordinates and lab-frame axis.
    """
    setup = state.setup
    axis = _setup_axis_unit_for_base(base, setup)
    pose_stack = apply_pose_updates(
        pose_stack_for_setup(base, setup),
        state.pose.pose_params,
        translation_frame=state.pose.translation_frame,
    )
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
    """Adjust the axis without discarding nominal translations or per-view poses."""
    axis = _setup_axis_unit_for_base(base, setup)
    nominal = axis_pose_stack(base.angles, base.nominal_axis_unit)
    adjusted = axis_pose_stack(base.angles, axis)
    # Batched reduced-mantissa products can turn rotations into nonrigid
    # maps, even at zero setup change. Preserve geometry independently of
    # the precision policy a caller chooses for its large data operations.
    rotation = jnp.matmul(
        adjusted[:, :3, :3],
        jnp.swapaxes(nominal[:, :3, :3], 1, 2),
        precision=jax.lax.Precision.HIGHEST,
    )
    moved = jnp.matmul(
        rotation, base.nominal_pose_stack[:, :3, :], precision=jax.lax.Precision.HIGHEST
    )
    return base.nominal_pose_stack.at[:, :3, :].set(moved)


def _nominal_axis_unit_from_geometry(geometry: Geometry) -> jnp.ndarray:
    if isinstance(geometry, RotationAxisGeometry | LaminographyGeometry):
        axis = np.asarray(geometry.axis_unit_lab, dtype=np.float32)
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
        angles=base.angles[idx],
        nominal_pose_stack=base.nominal_pose_stack[idx],
        detector=base.detector,
        nominal_axis_unit=base.nominal_axis_unit,
        nominal_tilt_deg=base.nominal_tilt_deg,
        tilt_about=base.tilt_about,
        level_factor=base.level_factor,
    )
