"""Pose parametrization helpers for alignment transforms."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

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


def se3_from_pose_params(pose_params: jnp.ndarray) -> jnp.ndarray:
    """Build a 4x4 transform from ``[alpha, beta, phi, dx, dz]`` or with ``dy`` appended.

    Translations are (dx, dy, dz) in world/object units; five values mean dy = 0.
    """
    alpha, beta, phi, dx, dz = (
        pose_params[0],
        pose_params[1],
        pose_params[2],
        pose_params[3],
        pose_params[4],
    )
    dy = pose_params[5] if pose_params.shape[0] > 5 else jnp.zeros((), pose_params.dtype)
    R = compose_R(alpha, beta, phi)
    T = jnp.eye(4, dtype=jnp.float32)
    T = T.at[:3, :3].set(R)
    return T.at[:3, 3].set(jnp.stack([dx, dy, dz]).astype(jnp.float32))


def pad_pose_params(params: object) -> np.ndarray:
    """Return an ``(n_views, 6)`` FP32 pose table, adding dy = 0 to five-column tables."""
    arr = np.asarray(params, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] not in (5, 6):
        raise ValueError(f"pose parameters must have shape (n_views, 5 or 6), got {arr.shape}")
    if arr.shape[1] == 5:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), np.float32)], axis=1)
    return arr


def apply_pose_update(
    nominal: jnp.ndarray,
    pose_params: jnp.ndarray,
    *,
    translation_frame: PoseTranslationFrame = "object",
) -> jnp.ndarray:
    """Apply object-frame rotation and explicitly framed physical translations.

    ``object`` preserves ``nominal @ se3_from_pose_params(pose_params)``. ``detector`` adds
    ``(dx, dy, dz)`` to the nominal lab translation after composing rotations;
    positive dx/dz move the projected object along lab detector x/z. These
    remain physical lengths, not pixels, and detector roll does not rotate
    their lab-frame basis. Beam-direction nominal translation is preserved.
    """
    delta = se3_from_pose_params(pose_params)
    combined = jnp.matmul(nominal, delta, precision=jax.lax.Precision.HIGHEST)
    if translation_frame == "object":
        return combined
    if translation_frame == "detector":
        return combined.at[:3, 3].set(nominal[:3, 3] + delta[:3, 3])
    raise ValueError("pose translation frame must be 'object' or 'detector'")


def apply_pose_updates(
    nominal: jnp.ndarray,
    pose_params: jnp.ndarray,
    *,
    translation_frame: PoseTranslationFrame = "object",
) -> jnp.ndarray:
    """Apply the same explicit translation convention to a stack of views."""
    return jax.vmap(
        lambda pose, params: apply_pose_update(pose, params, translation_frame=translation_frame)
    )(nominal, pose_params)
