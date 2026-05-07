"""V2 forward projection through the core trilinear ray operator."""
# pyright: reportAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp

from tomojax.calibration.axis_geometry import axis_pose_stack, axis_unit_from_rotations
from tomojax.calibration.detector_grid import detector_grid_from_calibration
from tomojax.core.geometry import Detector, Grid
from tomojax.core.projector import forward_project_view_T

if TYPE_CHECKING:
    from tomojax.geometry import GeometryState

V2ProjectionOperatorName = Literal["core_trilinear_ray"]

PROJECTION_OPERATOR: V2ProjectionOperatorName = "core_trilinear_ray"


@dataclass(frozen=True)
class CoreProjectionGeometry:
    """Core projector geometry derived from the supported v2 geometry state."""

    grid: Grid
    detector: Detector
    t_all: jax.Array
    det_grid: tuple[jax.Array, jax.Array]
    operator: V2ProjectionOperatorName = PROJECTION_OPERATOR
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

    def provenance(self) -> dict[str, object]:
        """Return serialisable operator metadata for run artifacts."""
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
        }


def project_parallel_reference(volume: jax.Array, geometry: GeometryState) -> jax.Array:
    """Project a 3D volume with the core trilinear ray projector.

    The public v2 name is preserved for API continuity, but the implementation
    is intentionally no longer the old rotate-and-sum approximation.
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
    return project_parallel_reference_arrays(
        vol,
        theta_rad=theta,
        alpha_rad=alpha,
        beta_rad=beta,
        dx_px=dx,
        dz_px=dz,
        detector_roll_rad=detector_roll,
        axis_rot_x_rad=axis_rot_x,
        axis_rot_y_rad=axis_rot_y,
    )


def project_parallel_reference_arrays(
    volume: jax.Array,
    *,
    theta_rad: jax.Array,
    dx_px: jax.Array,
    dz_px: jax.Array,
    alpha_rad: jax.Array | float = 0.0,
    beta_rad: jax.Array | float = 0.0,
    detector_roll_rad: jax.Array | float = 0.0,
    axis_rot_x_rad: jax.Array | float = 0.0,
    axis_rot_y_rad: jax.Array | float = 0.0,
) -> jax.Array:
    """Project a volume from supported v2 pose arrays using core trilinear rays."""
    vol = jnp.asarray(volume, dtype=jnp.float32)
    if vol.ndim != 3:
        raise ValueError("volume must be 3D")
    theta = jnp.asarray(theta_rad, dtype=jnp.float32)
    alpha = _pose_array_like(alpha_rad, theta, name="alpha_rad")
    beta = _pose_array_like(beta_rad, theta, name="beta_rad")
    dx = jnp.asarray(dx_px, dtype=jnp.float32)
    dz = jnp.asarray(dz_px, dtype=jnp.float32)
    if theta.shape != dx.shape or theta.shape != dz.shape:
        raise ValueError("theta_rad, dx_px, and dz_px must have matching shapes")
    core = core_projection_geometry_from_arrays(
        _volume_shape(vol.shape),
        theta_rad=theta,
        alpha_rad=alpha,
        beta_rad=beta,
        dx_px=dx,
        dz_px=dz,
        detector_roll_rad=detector_roll_rad,
        axis_rot_x_rad=axis_rot_x_rad,
        axis_rot_y_rad=axis_rot_y_rad,
    )

    def project_one(t_view: jax.Array) -> jax.Array:
        return forward_project_view_T(
            t_view,
            core.grid,
            core.detector,
            vol,
            step_size=core.step_size,
            n_steps=core.n_steps,
            use_checkpoint=core.checkpoint_projector,
            unroll=core.projector_unroll,
            gather_dtype=core.gather_dtype,
            det_grid=core.det_grid,
        )

    return jax.vmap(project_one)(core.t_all).astype(jnp.float32)


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
    """Adapt supported v2 geometry state to core Grid/Detector/T_all."""
    theta = jnp.asarray(geometry.theta_total_rad(), dtype=jnp.float32)
    alpha = jnp.asarray(geometry.pose.alpha_rad, dtype=jnp.float32)
    beta = jnp.asarray(geometry.pose.beta_rad, dtype=jnp.float32)
    dx = jnp.asarray(geometry.setup.det_u_px.value + geometry.pose.dx_px, dtype=jnp.float32)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    dz = jnp.asarray(dz_setup + geometry.pose.dz_px, dtype=jnp.float32)
    detector_roll = jnp.asarray(geometry.setup.detector_roll_rad.value, dtype=jnp.float32)
    axis_rot_x = jnp.asarray(geometry.setup.axis_rot_x_rad.value, dtype=jnp.float32)
    axis_rot_y = jnp.asarray(geometry.setup.axis_rot_y_rad.value, dtype=jnp.float32)
    return core_projection_geometry_from_arrays(
        volume_shape,
        theta_rad=theta,
        alpha_rad=alpha,
        beta_rad=beta,
        dx_px=dx,
        dz_px=dz,
        detector_roll_rad=detector_roll,
        axis_rot_x_rad=axis_rot_x,
        axis_rot_y_rad=axis_rot_y,
        detector_shape=detector_shape,
        gather_dtype=gather_dtype,
        checkpoint_projector=checkpoint_projector,
        projector_unroll=projector_unroll,
        step_size=step_size,
        n_steps=n_steps,
    )


def core_projection_geometry_from_arrays(
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
    detector_shape: tuple[int, int] | None = None,
    gather_dtype: str = "fp32",
    checkpoint_projector: bool = True,
    projector_unroll: int = 1,
    step_size: float | None = None,
    n_steps: int | None = None,
) -> CoreProjectionGeometry:
    """Build core Grid/Detector/T_all for supported parallel tomography arrays."""
    nx, ny, nz = _volume_shape(volume_shape)
    theta = jnp.asarray(theta_rad, dtype=jnp.float32)
    alpha = _pose_array_like(alpha_rad, theta, name="alpha_rad")
    beta = _pose_array_like(beta_rad, theta, name="beta_rad")
    dx = jnp.asarray(dx_px, dtype=jnp.float32)
    dz = jnp.asarray(dz_px, dtype=jnp.float32)
    if theta.ndim != 1 or dx.ndim != 1 or dz.ndim != 1:
        raise ValueError("theta_rad, dx_px, and dz_px must be one-dimensional")
    if theta.shape != dx.shape or theta.shape != dz.shape:
        raise ValueError("theta_rad, dx_px, and dz_px must have matching shapes")
    rows, cols = detector_shape or (nx, nz)
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=int(cols), nv=int(rows), du=1.0, dv=1.0)
    roll = jnp.asarray(detector_roll_rad, dtype=jnp.float32)
    det_grid = _detector_grid_for_roll(detector, detector_roll_rad=roll)
    axis_x = jnp.asarray(axis_rot_x_rad, dtype=jnp.float32)
    axis_y = jnp.asarray(axis_rot_y_rad, dtype=jnp.float32)
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
) -> jax.Array:
    axis = axis_unit_from_rotations(
        (0.0, 0.0, 1.0),
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
