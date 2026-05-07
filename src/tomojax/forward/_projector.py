"""V2 forward projection through the core trilinear ray operator."""
# pyright: reportAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp

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
        }


def project_parallel_reference(volume: jax.Array, geometry: GeometryState) -> jax.Array:
    """Project a 3D volume with the core trilinear ray projector.

    The public v2 name is preserved for API continuity, but the implementation
    is intentionally no longer the old rotate-and-sum approximation.
    """
    vol = jnp.asarray(volume, dtype=jnp.float32)
    if vol.ndim != 3:
        raise ValueError("volume must be 3D")
    _raise_for_unsupported_dofs(geometry)
    theta = jnp.asarray(geometry.theta_total_rad(), dtype=jnp.float32)
    dx = jnp.asarray(geometry.setup.det_u_px.value + geometry.pose.dx_px, dtype=jnp.float32)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    dz = jnp.asarray(dz_setup + geometry.pose.dz_px, dtype=jnp.float32)
    detector_roll = jnp.asarray(geometry.setup.detector_roll_rad.value, dtype=jnp.float32)
    return project_parallel_reference_arrays(
        vol,
        theta_rad=theta,
        dx_px=dx,
        dz_px=dz,
        detector_roll_rad=detector_roll,
    )


def project_parallel_reference_arrays(
    volume: jax.Array,
    *,
    theta_rad: jax.Array,
    dx_px: jax.Array,
    dz_px: jax.Array,
    detector_roll_rad: jax.Array | float = 0.0,
) -> jax.Array:
    """Project a volume from supported v2 pose arrays using core trilinear rays."""
    vol = jnp.asarray(volume, dtype=jnp.float32)
    if vol.ndim != 3:
        raise ValueError("volume must be 3D")
    theta = jnp.asarray(theta_rad, dtype=jnp.float32)
    dx = jnp.asarray(dx_px, dtype=jnp.float32)
    dz = jnp.asarray(dz_px, dtype=jnp.float32)
    if theta.shape != dx.shape or theta.shape != dz.shape:
        raise ValueError("theta_rad, dx_px, and dz_px must have matching shapes")
    core = core_projection_geometry_from_arrays(
        _volume_shape(vol.shape),
        theta_rad=theta,
        dx_px=dx,
        dz_px=dz,
        detector_roll_rad=detector_roll_rad,
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
    _raise_for_unsupported_dofs(geometry)
    theta = jnp.asarray(geometry.theta_total_rad(), dtype=jnp.float32)
    dx = jnp.asarray(geometry.setup.det_u_px.value + geometry.pose.dx_px, dtype=jnp.float32)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    dz = jnp.asarray(dz_setup + geometry.pose.dz_px, dtype=jnp.float32)
    detector_roll = jnp.asarray(geometry.setup.detector_roll_rad.value, dtype=jnp.float32)
    return core_projection_geometry_from_arrays(
        volume_shape,
        theta_rad=theta,
        dx_px=dx,
        dz_px=dz,
        detector_roll_rad=detector_roll,
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
    detector_roll_rad: jax.Array | float = 0.0,
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
    return CoreProjectionGeometry(
        grid=grid,
        detector=detector,
        t_all=_stack_core_poses(theta, dx, dz),
        det_grid=det_grid,
        gather_dtype=gather_dtype,
        checkpoint_projector=bool(checkpoint_projector),
        projector_unroll=int(projector_unroll),
        step_size=step_size,
        n_steps=n_steps,
        detector_roll_rad=roll,
    )


def _stack_core_poses(theta_rad: jax.Array, dx_px: jax.Array, dz_px: jax.Array) -> jax.Array:
    cos_t = jnp.cos(theta_rad)
    sin_t = jnp.sin(theta_rad)
    zeros = jnp.zeros_like(theta_rad)
    ones = jnp.ones_like(theta_rad)
    row0 = jnp.stack((cos_t, -sin_t, zeros, -dx_px), axis=1)
    row1 = jnp.stack((sin_t, cos_t, zeros, zeros), axis=1)
    row2 = jnp.stack((zeros, zeros, ones, -dz_px), axis=1)
    row3 = jnp.broadcast_to(
        jnp.asarray((0.0, 0.0, 0.0, 1.0), dtype=jnp.float32),
        row0.shape,
    )
    return jnp.stack((row0, row1, row2, row3), axis=1).astype(jnp.float32)


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


def _raise_for_unsupported_dofs(geometry: GeometryState) -> None:
    unsupported = {
        "axis_rot_x_rad": geometry.setup.axis_rot_x_rad,
        "axis_rot_y_rad": geometry.setup.axis_rot_y_rad,
    }
    for name, parameter in unsupported.items():
        if parameter.active and abs(float(parameter.value)) > 0.0:
            raise ValueError(f"{name} is not supported by the v2 core_trilinear_ray adapter yet")
    if jnp.any(jnp.asarray(geometry.pose.alpha_rad) != 0.0) or jnp.any(
        jnp.asarray(geometry.pose.beta_rad) != 0.0
    ):
        raise ValueError(
            "alpha_rad and beta_rad are not supported by the v2 core_trilinear_ray adapter yet"
        )
