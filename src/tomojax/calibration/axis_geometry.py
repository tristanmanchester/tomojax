from __future__ import annotations

from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.lamino import laminography_axis_unit


AXIS_DIRECTION_DOFS: tuple[str, ...] = ("axis_rot_x_deg", "axis_rot_y_deg")


def nominal_axis_unit_from_inputs(geometry_inputs: Mapping[str, object]) -> np.ndarray:
    """Resolve the nominal lab-frame rotation axis from persisted geometry metadata."""
    if geometry_inputs.get("axis_unit_lab") is not None:
        axis = np.asarray(geometry_inputs["axis_unit_lab"], dtype=np.float64)
    elif str(geometry_inputs.get("geometry_type", "parallel")).lower() in {
        "lamino",
        "laminography",
    }:
        axis = laminography_axis_unit(
            float(geometry_inputs.get("tilt_deg", 30.0)),
            str(geometry_inputs.get("tilt_about", "x")),
        )
    else:
        axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("nominal axis_unit_lab must be a finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("nominal axis_unit_lab must be non-zero")
    return axis / norm


def default_active_axis_dofs(geometry_inputs: Mapping[str, object]) -> tuple[str, ...]:
    if str(geometry_inputs.get("geometry_type", "parallel")).lower() in {
        "lamino",
        "laminography",
    } and str(geometry_inputs.get("tilt_about", "x")) == "x":
        return ("axis_rot_x_deg",)
    return AXIS_DIRECTION_DOFS


def axis_values_from_rotations(
    *,
    active_names: Sequence[str],
    axis_rot_x_deg: float,
    axis_rot_y_deg: float,
) -> jnp.ndarray:
    values = []
    for name in active_names:
        if name == "axis_rot_x_deg":
            values.append(float(axis_rot_x_deg))
        elif name == "axis_rot_y_deg":
            values.append(float(axis_rot_y_deg))
    return jnp.asarray(values, dtype=jnp.float32)


def axis_rotations_from_active(
    active_values: jnp.ndarray,
    *,
    active_names: Sequence[str],
    fixed_axis_rot_x_deg: float,
    fixed_axis_rot_y_deg: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    values = jnp.asarray(active_values, dtype=jnp.float32)
    rot_x = jnp.asarray(fixed_axis_rot_x_deg, dtype=jnp.float32)
    rot_y = jnp.asarray(fixed_axis_rot_y_deg, dtype=jnp.float32)
    for idx, name in enumerate(active_names):
        if name == "axis_rot_x_deg":
            rot_x = values[idx]
        elif name == "axis_rot_y_deg":
            rot_y = values[idx]
    return rot_x, rot_y


def axis_unit_from_rotations(
    nominal_axis_unit: object,
    *,
    axis_rot_x_deg: object,
    axis_rot_y_deg: object,
) -> jnp.ndarray:
    axis = jnp.asarray(nominal_axis_unit, dtype=jnp.float32)
    rx = jnp.deg2rad(jnp.asarray(axis_rot_x_deg, dtype=jnp.float32))
    ry = jnp.deg2rad(jnp.asarray(axis_rot_y_deg, dtype=jnp.float32))
    cx, sx = jnp.cos(rx), jnp.sin(rx)
    cy, sy = jnp.cos(ry), jnp.sin(ry)
    r_x = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]],
        dtype=jnp.float32,
    )
    r_y = jnp.asarray(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=jnp.float32,
    )
    candidate = r_y @ (r_x @ axis)
    return candidate / jnp.maximum(jnp.linalg.norm(candidate), jnp.float32(1e-8))


def axis_unit_from_active(
    active_values: jnp.ndarray,
    *,
    active_names: Sequence[str],
    nominal_axis_unit: object,
    fixed_axis_rot_x_deg: float,
    fixed_axis_rot_y_deg: float,
) -> jnp.ndarray:
    rot_x, rot_y = axis_rotations_from_active(
        active_values,
        active_names=active_names,
        fixed_axis_rot_x_deg=fixed_axis_rot_x_deg,
        fixed_axis_rot_y_deg=fixed_axis_rot_y_deg,
    )
    return axis_unit_from_rotations(
        nominal_axis_unit,
        axis_rot_x_deg=rot_x,
        axis_rot_y_deg=rot_y,
    )


def _skew(v: jnp.ndarray) -> jnp.ndarray:
    x, y, z = v
    return jnp.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=jnp.float32)


def _align_ez_to_axis(axis_unit: jnp.ndarray) -> jnp.ndarray:
    ez = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32)
    axis = axis_unit / jnp.maximum(jnp.linalg.norm(axis_unit), jnp.float32(1e-8))
    cos_theta = jnp.clip(jnp.dot(ez, axis), -1.0, 1.0)

    def aligned() -> jnp.ndarray:
        # Rodrigues' formula using the unnormalised cross product avoids the
        # zero-axis singularity at ez. This keeps the pose map differentiable
        # at the nominal parallel-CT axis, where axis calibration starts.
        cross = jnp.cross(ez, axis)
        K = _skew(cross)
        return jnp.eye(3, dtype=jnp.float32) + K + (K @ K) / jnp.maximum(
            jnp.float32(1.0) + cos_theta,
            jnp.float32(1e-8),
        )

    def near_parallel() -> jnp.ndarray:
        return jnp.diag(jnp.asarray([1.0, -1.0, -1.0], dtype=jnp.float32))

    return jax.lax.cond(
        cos_theta > jnp.float32(-1.0 + 1e-6),
        lambda _: aligned(),
        lambda _: near_parallel(),
        operand=None,
    )


def axis_pose_stack(
    thetas_deg: object,
    axis_unit_lab: object,
) -> jnp.ndarray:
    """Build a traced-safe stack of world-from-object poses for an axis direction."""
    thetas = jnp.asarray(thetas_deg, dtype=jnp.float32)
    axis = jnp.asarray(axis_unit_lab, dtype=jnp.float32)
    S = _align_ez_to_axis(axis)

    def one(theta_deg: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.deg2rad(theta_deg)
        c, s = jnp.cos(theta), jnp.sin(theta)
        Rz = jnp.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)
        T = jnp.eye(4, dtype=jnp.float32)
        T = T.at[:3, :3].set(S @ Rz)
        return T

    return jax.vmap(one)(thetas)
