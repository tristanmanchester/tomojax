"""Rotation-axis calibration helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.lamino import laminography_axis_unit

if TYPE_CHECKING:
    from collections.abc import Mapping

AXIS_DIRECTION_DOFS: tuple[str, ...] = ("axis_rot_x_deg", "axis_rot_y_deg")


def nominal_axis_unit_from_inputs(geometry_inputs: Mapping[str, object]) -> np.ndarray:
    """Resolve the nominal lab-frame rotation axis from persisted geometry metadata.

    Parameters
    ----------
    geometry_inputs : mapping
        Persisted axis vector or geometry type and laminography tilt.

    Returns
    -------
    numpy.ndarray
        Normalized three-component lab-frame axis.

    Raises
    ------
    ValueError
        If the axis is not finite and nonzero, or the tilt convention is invalid.
    """
    if geometry_inputs.get("axis_unit_lab") is not None:
        axis = np.asarray(geometry_inputs["axis_unit_lab"], dtype=np.float64)
    elif str(geometry_inputs.get("geometry_type", "parallel")).lower() in {
        "lamino",
        "laminography",
    }:
        if "tilt_deg" not in geometry_inputs:
            raise ValueError("Missing laminography tilt_deg or axis_unit_lab")
        axis = laminography_axis_unit(
            float(geometry_inputs["tilt_deg"]),
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


def axis_unit_from_rotations(
    nominal_axis_unit: object,
    *,
    axis_rot_x_deg: object,
    axis_rot_y_deg: object,
) -> jnp.ndarray:
    """Apply x/y calibration rotations to a nominal lab-frame axis.

    Parameters
    ----------
    nominal_axis_unit : array-like
        Three-component nominal lab-frame unit axis.
    axis_rot_x_deg, axis_rot_y_deg : array-like
        Scalar x and y calibration rotations in degrees, applied in that order.

    Returns
    -------
    jax.Array
        Calibrated lab-frame unit axis, in float32.
    """
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
    candidate = jnp.matmul(
        r_y,
        jnp.matmul(r_x, axis, precision=jax.lax.Precision.HIGHEST),
        precision=jax.lax.Precision.HIGHEST,
    )
    return candidate / jnp.maximum(jnp.linalg.norm(candidate), jnp.float32(1e-8))


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
        return (
            jnp.eye(3, dtype=jnp.float32)
            + K
            + jnp.matmul(K, K, precision=jax.lax.Precision.HIGHEST)
            / jnp.maximum(
                jnp.float32(1.0) + cos_theta,
                jnp.float32(1e-8),
            )
        )

    def near_antiparallel() -> jnp.ndarray:
        # Keep the same minimum-angle rotation as the host geometries. A
        # helper-vector basis maps ez correctly but changes the in-plane
        # frame, creating an artificial pose jump as the axis is calibrated.
        # Normalised Rodrigues avoids cancellation in 1 + cos(theta) here.
        cross = jnp.cross(ez, axis)
        sin2 = jnp.dot(cross, cross)

        def tilted(_: None) -> jnp.ndarray:
            sine = jnp.sqrt(sin2)
            x, y = axis[:2] / sine
            xy = (cos_theta - 1) * x * y
            # Component form keeps the requested axis as the third column,
            # rather than recovering it through near-half-turn cancellation.
            return jnp.asarray(
                [
                    [y * y + cos_theta * x * x, xy, axis[0]],
                    [xy, x * x + cos_theta * y * y, axis[1]],
                    [-axis[0], -axis[1], cos_theta],
                ],
                dtype=jnp.float32,
            )

        # At exactly -ez the minimum rotation is not unique. Match the host
        # geometry's half-turn about x without differentiating a zero norm.
        return jax.lax.cond(
            sin2 > 0,
            tilted,
            lambda _: jnp.diag(jnp.asarray([1.0, -1.0, -1.0], dtype=jnp.float32)),
            operand=None,
        )

    return jax.lax.cond(
        cos_theta >= 0,
        lambda _: aligned(),
        lambda _: near_antiparallel(),
        operand=None,
    )


@jax.jit
def axis_pose_stack(
    angles: object,
    axis_unit_lab: object,
) -> jnp.ndarray:
    """Build a traced-safe stack of world-from-object poses for an axis direction.

    Cache the compiled pose map so eager alignment calls do not repeatedly
    trace the nested axis-alignment conditionals or dispatch its small ops.

    Parameters
    ----------
    angles : array-like
        One-dimensional view angles in degrees.
    axis_unit_lab : array-like
        Three-component lab-frame rotation axis.

    Returns
    -------
    jax.Array
        Float32 world-from-object transforms, shape (views, 4, 4).
    """
    thetas = jnp.asarray(angles, dtype=jnp.float32)
    axis = jnp.asarray(axis_unit_lab, dtype=jnp.float32)
    S = _align_ez_to_axis(axis)

    def one(theta_deg: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.deg2rad(theta_deg)
        c, s = jnp.cos(theta), jnp.sin(theta)
        Rz = jnp.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)
        T = jnp.eye(4, dtype=jnp.float32)
        # Rigid poses need FP32 products even if the caller permits reduced
        # mantissa precision for large image/volume matrix multiplications.
        return T.at[:3, :3].set(jnp.matmul(S, Rz, precision=jax.lax.Precision.HIGHEST))

    return jax.vmap(one)(thetas)
