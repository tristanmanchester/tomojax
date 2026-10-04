"""Pose parametrization helpers for alignment transforms."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

type PoseTranslationFrame = Literal["object", "detector"]


def rot_x(a: jnp.ndarray) -> jnp.ndarray:
    """Return a right-handed x-axis rotation matrix."""
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=jnp.float32)


def rot_y(b: jnp.ndarray) -> jnp.ndarray:
    """Return a right-handed y-axis rotation matrix."""
    c, s = jnp.cos(b), jnp.sin(b)
    return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=jnp.float32)


def rot_z(p: jnp.ndarray) -> jnp.ndarray:
    """Return a right-handed z-axis rotation matrix."""
    c, s = jnp.cos(p), jnp.sin(p)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)


def compose_R(alpha: jnp.ndarray, beta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Compose the alignment Euler-like rotation matrix."""
    # Match the original repo: R = R_y(beta) R_x(alpha) R_z(phi).
    return jnp.matmul(
        jnp.matmul(rot_y(beta), rot_x(alpha), precision=jax.lax.Precision.HIGHEST),
        rot_z(phi),
        precision=jax.lax.Precision.HIGHEST,
    )


def se3_from_5d(params5: jnp.ndarray) -> jnp.ndarray:
    """Build a 4x4 transform from 5-DOF [alpha, beta, phi, dx, dz].

    Translations are (dx, 0, dz) in world/object units.
    """
    alpha, beta, phi, dx, dz = params5
    R = compose_R(alpha, beta, phi)
    T = jnp.eye(4, dtype=jnp.float32)
    T = T.at[:3, :3].set(R)
    return T.at[:3, 3].set(jnp.array([dx, 0.0, dz], dtype=jnp.float32))


def apply_pose_update(
    nominal: jnp.ndarray,
    params5: jnp.ndarray,
    *,
    translation_frame: PoseTranslationFrame = "object",
) -> jnp.ndarray:
    """Apply object-frame rotation and explicitly framed physical translations.

    ``object`` preserves ``nominal @ se3_from_5d(params5)``. ``detector`` adds
    ``(dx, 0, dz)`` to the nominal lab translation after composing rotations;
    positive dx/dz move the projected object along lab detector x/z. These
    remain physical lengths, not pixels, and detector roll does not rotate
    their lab-frame basis. Beam-direction nominal translation is preserved.
    """
    delta = se3_from_5d(params5)
    combined = jnp.matmul(nominal, delta, precision=jax.lax.Precision.HIGHEST)
    if translation_frame == "object":
        return combined
    if translation_frame == "detector":
        return combined.at[:3, 3].set(nominal[:3, 3] + delta[:3, 3])
    raise ValueError("pose translation frame must be 'object' or 'detector'")


def apply_pose_updates(
    nominal: jnp.ndarray,
    params5: jnp.ndarray,
    *,
    translation_frame: PoseTranslationFrame = "object",
) -> jnp.ndarray:
    """Apply the same explicit translation convention to a stack of views."""
    return jax.vmap(
        lambda pose, params: apply_pose_update(pose, params, translation_frame=translation_frame)
    )(nominal, params5)
