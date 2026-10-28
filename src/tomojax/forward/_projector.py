"""Forward projection through the core trilinear ray operator."""
# pyright: reportAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp

from tomojax.core.geometry import Detector, Grid
from tomojax.core.projector import project_views
from tomojax.geometry.api import (
    axis_pose_stack,
    axis_unit_from_rotations,
    detector_grid_from_calibration,
)

if TYPE_CHECKING:
    from tomojax.geometry.api import GeometryState

ProjectionOperatorName = Literal["core_trilinear_ray"]

PROJECTION_OPERATOR: ProjectionOperatorName = "core_trilinear_ray"


@dataclass(frozen=True)
class ProjectionArrayGeometryInput:
    """Group pose and setup arrays for unit-spacing reference projection.

    Parameters
    ----------
    theta_rad : jax.Array
        Nominal view angles in radians, shape (views,).
    dx_px, dz_px : jax.Array
        Horizontal and vertical detector shifts in pixels, shape (views,).
        Pose matrices use translations (-dx_px, 0, -dz_px) in the lab frame.
    alpha_rad, beta_rad : jax.Array or float, optional
        Per-view object x and y residual rotations in radians, or scalar values
        broadcast across views; default is zero.
    detector_roll_rad : jax.Array or float, optional
        Global in-plane detector rotation in radians; default is zero.
    axis_rot_x_rad, axis_rot_y_rad : jax.Array or float, optional
        Global rotation-axis corrections in radians; default is zero.
    nominal_axis_unit : array-like, optional
        Nominal rotation axis in lab coordinates, shape (3,), normalized on use.
        Default is (0, 0, 1). This defines the axis, including any nominal tilt.
    acquisition_model : str, optional
        Acquisition label recorded as provenance; default is 'parallel'.
    laminography_tilt_rad : jax.Array or float, optional
        Nominal tilt recorded as provenance, in radians; default is zero.
        Supply its actual axis through nominal_axis_unit.
    laminography_tilt_about : str, optional
        Nominal tilt-axis provenance; default is 'x'.
    detector_shape : tuple of int or None, optional
        Detector (rows, columns). None uses (nz, nx) from the volume shape.
    """

    theta_rad: jax.Array
    dx_px: jax.Array
    dz_px: jax.Array
    alpha_rad: jax.Array | float = 0.0
    beta_rad: jax.Array | float = 0.0
    detector_roll_rad: jax.Array | float = 0.0
    axis_rot_x_rad: jax.Array | float = 0.0
    axis_rot_y_rad: jax.Array | float = 0.0
    nominal_axis_unit: jax.Array | tuple[float, float, float] = (0.0, 0.0, 1.0)
    acquisition_model: str = "parallel"
    laminography_tilt_rad: jax.Array | float = 0.0
    laminography_tilt_about: str = "x"
    detector_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class CoreProjectionGeometry:
    """Store resolved geometry and settings for the reference ray projector.

    Parameters
    ----------
    grid : Grid
        Physical volume grid.
    detector : Detector
        Physical detector grid in the same length unit as grid.
    t_all : jax.Array
        World-from-object transforms of shape (views, 4, 4).
    det_grid : tuple of jax.Array
        Flattened detector-plane u and v coordinates, each of shape (nv * nu,),
        with detector-u varying fastest.
    operator : {'core_trilinear_ray'}, optional
        Recorded projector name.
    step_size : float or None, optional
        Ray sample spacing in physical grid units; None uses the core default.
    n_steps : int or None, optional
        Ray sample count; None uses the core default.
    gather_dtype : str, optional
        Core gather precision; default is 'fp32'.
    checkpoint_projector : bool, optional
        Recompute projector intermediates during differentiation; default is True.
    projector_unroll : int, optional
        Core ray-loop unrolling factor; default is 1.
    detector_roll_rad : object, optional
        Recorded global detector roll in radians; default is zero.
    axis_rot_x_rad, axis_rot_y_rad : object, optional
        Recorded global axis corrections in radians; default is zero.
    alpha_rad_max_abs, beta_rad_max_abs : object, optional
        Recorded maximum absolute residual pose angles in radians; default is zero.
    acquisition_model : str, optional
        Recorded acquisition model; default is 'parallel'.
    laminography_tilt_rad : object, optional
        Recorded nominal tilt in radians; default is zero.
    laminography_tilt_about : str, optional
        Recorded nominal tilt axis; default is 'x'.
    """

    grid: Grid
    detector: Detector
    t_all: jax.Array
    det_grid: tuple[jax.Array, jax.Array]
    operator: ProjectionOperatorName = PROJECTION_OPERATOR
    step_size: float | None = None
    n_steps: int | None = None
    gather_dtype: str = "fp32"
    checkpoint_projector: bool = True
    projector_unroll: int = 1
    detector_roll_rad: object = 0.0
    axis_rot_x_rad: object = 0.0
    axis_rot_y_rad: object = 0.0
    alpha_rad_max_abs: object = 0.0
    beta_rad_max_abs: object = 0.0
    acquisition_model: str = "parallel"
    laminography_tilt_rad: object = 0.0
    laminography_tilt_about: str = "x"

    def provenance(self) -> dict[str, object]:
        """Return serializable operator metadata.

        Returns
        -------
        dict
            Geometry, sampling settings and setup summaries. Traced scalar values
            that cannot be converted to Python floats are recorded as 'dynamic'.
        """
        return {
            "projection_operator": self.operator,
            "grid": self.grid.to_dict(),
            "detector": self.detector.to_dict(),
            "step_size": self.step_size,
            "n_steps": self.n_steps,
            "gather_dtype": self.gather_dtype,
            "checkpoint_projector": self.checkpoint_projector,
            "projector_unroll": self.projector_unroll,
            "detector_roll_rad": _serialisable_scalar(self.detector_roll_rad),
            "axis_rot_x_rad": _serialisable_scalar(self.axis_rot_x_rad),
            "axis_rot_y_rad": _serialisable_scalar(self.axis_rot_y_rad),
            "alpha_rad_max_abs": _serialisable_scalar(self.alpha_rad_max_abs),
            "beta_rad_max_abs": _serialisable_scalar(self.beta_rad_max_abs),
            "acquisition_model": self.acquisition_model,
            "laminography_tilt_rad": _serialisable_scalar(self.laminography_tilt_rad),
            "laminography_tilt_about": self.laminography_tilt_about,
        }


def project_parallel_reference(volume: jax.Array, geometry: GeometryState) -> jax.Array:
    """Project a volume from a geometry state using trilinear ray sampling.

    Parameters
    ----------
    volume : jax.Array
        Three-dimensional volume in (nz, ny, nx) order, converted to float32.
        Voxel and detector spacing are both one in this reference adapter.
    geometry : GeometryState
        Nominal acquisition, setup corrections and per-view poses. Angles use
        radians and detector translations use pixels.

    Returns
    -------
    jax.Array
        Float32 projections of shape (views, nz, nx).

    Raises
    ------
    ValueError
        If the volume is not three-dimensional or pose array shapes disagree.
    """
    vol = jnp.asarray(volume, dtype=jnp.float32)
    if vol.ndim != 3:
        raise ValueError("volume must be 3D")
    theta = jnp.asarray(geometry.theta_total_rad(), dtype=jnp.float32)
    dx = jnp.asarray(geometry.setup.det_u_px.value + geometry.pose.dx_px, dtype=jnp.float32)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    dz = jnp.asarray(dz_setup + geometry.pose.dz_px, dtype=jnp.float32)
    alpha = jnp.asarray(geometry.pose.alpha_rad, dtype=jnp.float32)
    beta = jnp.asarray(geometry.pose.beta_rad, dtype=jnp.float32)
    detector_roll = jnp.asarray(geometry.setup.detector_roll_rad.value, dtype=jnp.float32)
    axis_rot_x = jnp.asarray(geometry.setup.axis_rot_x_rad.value, dtype=jnp.float32)
    axis_rot_y = jnp.asarray(geometry.setup.axis_rot_y_rad.value, dtype=jnp.float32)
    nominal_axis = nominal_axis_unit_from_geometry(geometry)
    return project_parallel_reference_from_input(
        vol,
        ProjectionArrayGeometryInput(
            theta_rad=theta,
            alpha_rad=alpha,
            beta_rad=beta,
            dx_px=dx,
            dz_px=dz,
            detector_roll_rad=detector_roll,
            axis_rot_x_rad=axis_rot_x,
            axis_rot_y_rad=axis_rot_y,
            nominal_axis_unit=nominal_axis,
            acquisition_model=geometry.acquisition.model,
            laminography_tilt_rad=geometry.acquisition.laminography_tilt_rad,
            laminography_tilt_about=geometry.acquisition.laminography_tilt_about,
        ),
    )


def project_parallel_reference_from_input(
    volume: jax.Array,
    geometry_input: ProjectionArrayGeometryInput,
) -> jax.Array:
    """Project a volume from grouped arrays using trilinear ray sampling.

    Parameters
    ----------
    volume : jax.Array
        Volume of shape (nz, ny, nx), converted to float32, with unit voxel spacing.
    geometry_input : ProjectionArrayGeometryInput
        Pose and setup arrays, including optional detector (rows, columns).

    Returns
    -------
    jax.Array
        Float32 projections of shape (views, rows, columns); the default detector
        shape is (nz, nx), with unit pixel spacing.

    Raises
    ------
    ValueError
        If the volume is not three-dimensional or pose array shapes disagree.
    """
    vol = jnp.asarray(volume, dtype=jnp.float32)
    if vol.ndim != 3:
        raise ValueError("volume must be 3D")
    theta = jnp.asarray(geometry_input.theta_rad, dtype=jnp.float32)
    _pose_array_like(geometry_input.alpha_rad, theta, name="alpha_rad")
    _pose_array_like(geometry_input.beta_rad, theta, name="beta_rad")
    dx = jnp.asarray(geometry_input.dx_px, dtype=jnp.float32)
    dz = jnp.asarray(geometry_input.dz_px, dtype=jnp.float32)
    if theta.shape != dx.shape or theta.shape != dz.shape:
        raise ValueError("theta_rad, dx_px, and dz_px must have matching shapes")
    core = core_projection_geometry_from_input(
        _volume_shape(vol.shape),
        geometry_input,
    )

    return project_views(
        core.t_all,
        core.grid,
        core.detector,
        vol,
        step_size=core.step_size,
        n_steps=core.n_steps,
        use_checkpoint=core.checkpoint_projector,
        unroll=core.projector_unroll,
        gather_dtype=core.gather_dtype,
        det_grid=core.det_grid,
    ).astype(jnp.float32)


def core_projection_geometry_from_state(
    volume_shape: tuple[int, int, int],
    geometry: GeometryState,
    *,
    detector_shape: tuple[int, int] | None = None,
    gather_dtype: str = "fp32",
    checkpoint_projector: bool = True,
    projector_unroll: int = 1,
    step_size: float | None = None,
    n_steps: int | None = None,
) -> CoreProjectionGeometry:
    """Resolve a geometry state into unit-spacing core projector geometry.

    Parameters
    ----------
    volume_shape : tuple of int
        Volume dimensions (nz, ny, nx).
    geometry : GeometryState
        Nominal acquisition, setup corrections and per-view poses.
    detector_shape : tuple of int or None, optional
        Detector (rows, columns); None uses (nz, nx).
    gather_dtype : str, optional
        Gather precision of the core projector; default is 'fp32'.
    checkpoint_projector : bool, optional
        Recompute projector intermediates during differentiation; default is True.
    projector_unroll : int, optional
        Core ray-loop unrolling factor; default is 1.
    step_size : float or None, optional
        Ray sample spacing in unit-grid lengths; None uses the core default.
    n_steps : int or None, optional
        Number of ray samples; None uses the core default. With a fixed sample
        count, step_size still controls the spacing.

    Returns
    -------
    CoreProjectionGeometry
        Unit-spacing voxel grid and detector, pose matrices and sampling settings.

    Raises
    ------
    ValueError
        If the volume shape or per-view geometry arrays are inconsistent.
    """
    theta = jnp.asarray(geometry.theta_total_rad(), dtype=jnp.float32)
    alpha = jnp.asarray(geometry.pose.alpha_rad, dtype=jnp.float32)
    beta = jnp.asarray(geometry.pose.beta_rad, dtype=jnp.float32)
    dx = jnp.asarray(geometry.setup.det_u_px.value + geometry.pose.dx_px, dtype=jnp.float32)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    dz = jnp.asarray(dz_setup + geometry.pose.dz_px, dtype=jnp.float32)
    detector_roll = jnp.asarray(geometry.setup.detector_roll_rad.value, dtype=jnp.float32)
    axis_rot_x = jnp.asarray(geometry.setup.axis_rot_x_rad.value, dtype=jnp.float32)
    axis_rot_y = jnp.asarray(geometry.setup.axis_rot_y_rad.value, dtype=jnp.float32)
    nominal_axis = nominal_axis_unit_from_geometry(geometry)
    return _core_projection_geometry_from_arrays(
        volume_shape,
        theta_rad=theta,
        alpha_rad=alpha,
        beta_rad=beta,
        dx_px=dx,
        dz_px=dz,
        detector_roll_rad=detector_roll,
        axis_rot_x_rad=axis_rot_x,
        axis_rot_y_rad=axis_rot_y,
        nominal_axis_unit=nominal_axis,
        acquisition_model=geometry.acquisition.model,
        laminography_tilt_rad=geometry.acquisition.laminography_tilt_rad,
        laminography_tilt_about=geometry.acquisition.laminography_tilt_about,
        detector_shape=detector_shape,
        gather_dtype=gather_dtype,
        checkpoint_projector=checkpoint_projector,
        projector_unroll=projector_unroll,
        step_size=step_size,
        n_steps=n_steps,
    )


def core_projection_geometry_from_input(
    volume_shape: tuple[int, int, int],
    geometry_input: ProjectionArrayGeometryInput,
    *,
    gather_dtype: str = "fp32",
    checkpoint_projector: bool = True,
    projector_unroll: int = 1,
    step_size: float | None = None,
    n_steps: int | None = None,
) -> CoreProjectionGeometry:
    """Resolve grouped arrays into unit-spacing core projector geometry.

    Parameters
    ----------
    volume_shape : tuple of int
        Volume dimensions (nz, ny, nx).
    geometry_input : ProjectionArrayGeometryInput
        Pose and setup arrays and optional detector shape.
    gather_dtype : str, optional
        Gather precision of the core projector; default is 'fp32'.
    checkpoint_projector : bool, optional
        Recompute projector intermediates during differentiation; default is True.
    projector_unroll : int, optional
        Core ray-loop unrolling factor; default is 1.
    step_size : float or None, optional
        Ray sample spacing in unit-grid lengths; None uses the core default.
    n_steps : int or None, optional
        Number of ray samples; None uses the core default. With a fixed sample
        count, step_size still controls the spacing.

    Returns
    -------
    CoreProjectionGeometry
        Unit-spacing voxel grid and detector, pose matrices and sampling settings.

    Raises
    ------
    ValueError
        If the volume shape or per-view geometry arrays are inconsistent.
    """
    return _core_projection_geometry_from_arrays(
        volume_shape,
        theta_rad=geometry_input.theta_rad,
        alpha_rad=geometry_input.alpha_rad,
        beta_rad=geometry_input.beta_rad,
        dx_px=geometry_input.dx_px,
        dz_px=geometry_input.dz_px,
        detector_roll_rad=geometry_input.detector_roll_rad,
        axis_rot_x_rad=geometry_input.axis_rot_x_rad,
        axis_rot_y_rad=geometry_input.axis_rot_y_rad,
        nominal_axis_unit=geometry_input.nominal_axis_unit,
        acquisition_model=geometry_input.acquisition_model,
        laminography_tilt_rad=geometry_input.laminography_tilt_rad,
        laminography_tilt_about=geometry_input.laminography_tilt_about,
        detector_shape=geometry_input.detector_shape,
        gather_dtype=gather_dtype,
        checkpoint_projector=checkpoint_projector,
        projector_unroll=projector_unroll,
        step_size=step_size,
        n_steps=n_steps,
    )


def _core_projection_geometry_from_arrays(
    volume_shape: tuple[int, int, int],
    *,
    theta_rad: jax.Array,
    dx_px: jax.Array,
    dz_px: jax.Array,
    alpha_rad: jax.Array | float = 0.0,
    beta_rad: jax.Array | float = 0.0,
    detector_roll_rad: jax.Array | float = 0.0,
    axis_rot_x_rad: jax.Array | float = 0.0,
    axis_rot_y_rad: jax.Array | float = 0.0,
    nominal_axis_unit: jax.Array | tuple[float, float, float] = (0.0, 0.0, 1.0),
    acquisition_model: str = "parallel",
    laminography_tilt_rad: jax.Array | float = 0.0,
    laminography_tilt_about: str = "x",
    detector_shape: tuple[int, int] | None = None,
    gather_dtype: str = "fp32",
    checkpoint_projector: bool = True,
    projector_unroll: int = 1,
    step_size: float | None = None,
    n_steps: int | None = None,
) -> CoreProjectionGeometry:
    """Build core Grid/Detector/T_all for supported parallel acquisition arrays."""
    nz, ny, nx = _volume_shape(volume_shape)
    theta = jnp.asarray(theta_rad, dtype=jnp.float32)
    alpha = _pose_array_like(alpha_rad, theta, name="alpha_rad")
    beta = _pose_array_like(beta_rad, theta, name="beta_rad")
    dx = jnp.asarray(dx_px, dtype=jnp.float32)
    dz = jnp.asarray(dz_px, dtype=jnp.float32)
    if theta.ndim != 1 or dx.ndim != 1 or dz.ndim != 1:
        raise ValueError("theta_rad, dx_px, and dz_px must be one-dimensional")
    if theta.shape != dx.shape or theta.shape != dz.shape:
        raise ValueError("theta_rad, dx_px, and dz_px must have matching shapes")
    rows, cols = detector_shape or (nz, nx)
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=int(cols), nv=int(rows), du=1.0, dv=1.0)
    roll = jnp.asarray(detector_roll_rad, dtype=jnp.float32)
    det_grid = _detector_grid_for_roll(detector, detector_roll_rad=roll)
    axis_x = jnp.asarray(axis_rot_x_rad, dtype=jnp.float32)
    axis_y = jnp.asarray(axis_rot_y_rad, dtype=jnp.float32)
    nominal_axis = _normalize_axis_unit_jax(nominal_axis_unit)
    return CoreProjectionGeometry(
        grid=grid,
        detector=detector,
        t_all=_stack_core_poses(
            theta,
            alpha,
            beta,
            dx,
            dz,
            axis_rot_x_rad=axis_x,
            axis_rot_y_rad=axis_y,
            nominal_axis_unit=nominal_axis,
        ),
        det_grid=det_grid,
        gather_dtype=gather_dtype,
        checkpoint_projector=bool(checkpoint_projector),
        projector_unroll=int(projector_unroll),
        step_size=step_size,
        n_steps=n_steps,
        detector_roll_rad=roll,
        axis_rot_x_rad=axis_x,
        axis_rot_y_rad=axis_y,
        alpha_rad_max_abs=jnp.max(jnp.abs(alpha)),
        beta_rad_max_abs=jnp.max(jnp.abs(beta)),
        acquisition_model=acquisition_model,
        laminography_tilt_rad=laminography_tilt_rad,
        laminography_tilt_about=laminography_tilt_about,
    )


def _stack_core_poses(
    theta_rad: jax.Array,
    alpha_rad: jax.Array,
    beta_rad: jax.Array,
    dx_px: jax.Array,
    dz_px: jax.Array,
    *,
    axis_rot_x_rad: jax.Array,
    axis_rot_y_rad: jax.Array,
    nominal_axis_unit: jax.Array,
) -> jax.Array:
    axis = axis_unit_from_rotations(
        nominal_axis_unit,
        axis_rot_x_deg=axis_rot_x_rad * jnp.asarray(180.0 / math.pi, dtype=jnp.float32),
        axis_rot_y_deg=axis_rot_y_rad * jnp.asarray(180.0 / math.pi, dtype=jnp.float32),
    )
    t_all = axis_pose_stack(theta_rad * jnp.asarray(180.0 / math.pi, dtype=jnp.float32), axis)
    residual = _pose_residual_rotation_stack(alpha_rad, beta_rad)
    t_all = t_all.at[:, :3, :3].set(t_all[:, :3, :3] @ residual)
    t_all = t_all.at[:, 0, 3].set(-dx_px)
    t_all = t_all.at[:, 2, 3].set(-dz_px)
    return t_all.astype(jnp.float32)


def _pose_residual_rotation_stack(alpha_rad: jax.Array, beta_rad: jax.Array) -> jax.Array:
    alpha = jnp.asarray(alpha_rad, dtype=jnp.float32)
    beta = jnp.asarray(beta_rad, dtype=jnp.float32)

    def one(alpha_i: jax.Array, beta_i: jax.Array) -> jax.Array:
        ca, sa = jnp.cos(alpha_i), jnp.sin(alpha_i)
        cb, sb = jnp.cos(beta_i), jnp.sin(beta_i)
        rx = jnp.asarray(
            [[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]],
            dtype=jnp.float32,
        )
        ry = jnp.asarray(
            [[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]],
            dtype=jnp.float32,
        )
        return ry @ rx

    return jax.vmap(one)(alpha, beta)


def _pose_array_like(value: jax.Array | float, theta: jax.Array, *, name: str) -> jax.Array:
    array = jnp.asarray(value, dtype=jnp.float32)
    if array.ndim == 0:
        return jnp.full_like(theta, array)
    if array.shape != theta.shape:
        raise ValueError(f"{name} must be scalar or match theta_rad shape")
    return array


def nominal_axis_unit_from_geometry(geometry: GeometryState) -> jax.Array:
    """Return the nominal lab-frame rotation axis for a geometry state.

    Parameters
    ----------
    geometry : GeometryState
        Acquisition model and nominal laminography tilt; setup corrections
        and residual per-view rotations are not applied here.

    Returns
    -------
    jax.Array
        Float32 unit vector of shape (3,). Parallel geometry returns (0, 0, 1).
    """
    acquisition = geometry.acquisition
    if acquisition.model == "parallel":
        return jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32)
    tilt = jnp.asarray(acquisition.laminography_tilt_rad, dtype=jnp.float32)
    c, s = jnp.cos(tilt), jnp.sin(tilt)
    if acquisition.laminography_tilt_about == "x":
        return _normalize_axis_unit_jax(jnp.asarray([0.0, s, c], dtype=jnp.float32))
    return _normalize_axis_unit_jax(jnp.asarray([s, 0.0, c], dtype=jnp.float32))


def _normalize_axis_unit_jax(axis: jax.Array | tuple[float, float, float]) -> jax.Array:
    values = jnp.asarray(axis, dtype=jnp.float32)
    if values.shape != (3,):
        raise ValueError("nominal_axis_unit must be a 3-vector")
    return values / jnp.maximum(jnp.linalg.norm(values), jnp.asarray(1e-8, dtype=jnp.float32))


def _volume_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError("volume shape must be 3D")
    return cast("tuple[int, int, int]", tuple(int(axis) for axis in shape))


def _detector_grid_for_roll(
    detector: Detector,
    *,
    detector_roll_rad: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    roll = jnp.asarray(detector_roll_rad, dtype=jnp.float32)
    return detector_grid_from_calibration(
        detector,
        detector_roll_deg=roll * jnp.asarray(180.0 / math.pi, dtype=jnp.float32),
    )


def _serialisable_scalar(value: object) -> object:
    try:
        return float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return "dynamic"
