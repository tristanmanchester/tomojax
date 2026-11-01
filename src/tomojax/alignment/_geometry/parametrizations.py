"""Pose parametrization helpers for alignment transforms."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from tomojax.core.geometry.transforms import rot_x, rot_y, rot_z

type PoseTranslationFrame = Literal["object", "detector"]


def compose_R(alpha: jnp.ndarray, beta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Compose the alignment Euler-like rotation matrix."""
    # Match the original repo: R = R_y(beta) R_x(alpha) R_z(phi).
    return jnp.matmul(
        jnp.matmul(rot_y(beta, xp=jnp), rot_x(alpha, xp=jnp), precision=jax.lax.Precision.HIGHEST),
        rot_z(phi, xp=jnp),
        precision=jax.lax.Precision.HIGHEST,
    )


def se3_from_pose_params(pose_params: jnp.ndarray) -> jnp.ndarray:
    """Build a 4x4 transform from ``[alpha, beta, phi, dx, dz, dy]``.

    Translations are (dx, dy, dz) in world/object units.

    Parameters
    ----------
    pose_params : array-like
        One pose vector (alpha, beta, phi, dx, dz, dy); angles in radians,
        translations physical lengths.

    Returns
    -------
    jax.Array
        Float32 rigid transform of shape (4, 4), with rotation Ry(beta) Rx(alpha)
        Rz(phi).
    """
    alpha, beta, phi, dx, dz, dy = pose_params
    R = compose_R(alpha, beta, phi)
    T = jnp.eye(4, dtype=jnp.float32)
    T = T.at[:3, :3].set(R)
    return T.at[:3, 3].set(jnp.stack([dx, dy, dz]).astype(jnp.float32))


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

    Parameters
    ----------
    nominal : array-like
        Nominal object-to-world transform with shape (4, 4).
    pose_params : array-like
        One correction vector (alpha, beta, phi, dx, dz, dy), in radians and physical
        lengths.
    translation_frame : {"object", "detector"}, optional
        Basis for physical translations; detector means lab x/y/z axes, not pixels.
        Default is 'object'.

    Returns
    -------
    jax.Array
        Updated object-to-world transform with shape (4, 4).

    Raises
    ------
    ValueError
        If translation_frame is not "object" or "detector".
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
    """Apply the same explicit translation convention to a stack of views.

    Parameters
    ----------
    nominal : array-like
        Nominal object-to-world transforms with shape (views, 4, 4).
    pose_params : array-like
        Per-view corrections with columns (alpha, beta, phi, dx, dz, dy); angles are
        radians and translations physical lengths.
    translation_frame : {"object", "detector"}, optional
        Basis for physical translations; detector means lab x/y/z axes, not pixels.
        Default is 'object'.

    Returns
    -------
    jax.Array
        Updated object-to-world transforms with shape (views, 4, 4).

    Raises
    ------
    ValueError
        If translation_frame is not "object" or "detector".
    """
    return jax.vmap(
        lambda pose, params: apply_pose_update(pose, params, translation_frame=translation_frame)
    )(nominal, pose_params)
