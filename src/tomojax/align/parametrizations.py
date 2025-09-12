from __future__ import annotations

import jax.numpy as jnp


def rot_x(a: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=jnp.float32)


def rot_y(b: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(b), jnp.sin(b)
    return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=jnp.float32)


def rot_z(p: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(p), jnp.sin(p)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)


def compose_R(alpha: jnp.ndarray, beta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    # Match the original repo: R = R_y(β) R_x(α) R_z(φ)
    return rot_y(beta) @ rot_x(alpha) @ rot_z(phi)


def se3_from_5d(params5: jnp.ndarray) -> jnp.ndarray:
    """Build a 4x4 transform from 5-DOF [alpha, beta, phi, dx, dz].

    Translations are (dx, 0, dz) in world/object units.
    """
    alpha, beta, phi, dx, dz = params5
    R = compose_R(alpha, beta, phi)
    T = jnp.eye(4, dtype=jnp.float32)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(jnp.array([dx, 0.0, dz], dtype=jnp.float32))
    return T

