from __future__ import annotations

import math
from typing import Literal, cast

import jax.numpy as jnp


Regulariser = Literal["tv", "huber_tv"]


def grad3(u: jnp.ndarray):
    dx = jnp.concatenate(
        [jnp.diff(u, axis=0), jnp.zeros((1, u.shape[1], u.shape[2]), u.dtype)],
        axis=0,
    )
    dy = jnp.concatenate(
        [jnp.diff(u, axis=1), jnp.zeros((u.shape[0], 1, u.shape[2]), u.dtype)],
        axis=1,
    )
    dz = jnp.concatenate(
        [jnp.diff(u, axis=2), jnp.zeros((u.shape[0], u.shape[1], 1), u.dtype)],
        axis=2,
    )
    return dx, dy, dz


def div3(px: jnp.ndarray, py: jnp.ndarray, pz: jnp.ndarray):
    """Discrete divergence matching the TV updates: ``div3 = -grad3*``."""
    if px.shape[0] == 1:
        dx = jnp.zeros_like(px)
    else:
        first_x = px[0:1, :, :]
        if px.shape[0] > 2:
            mid_x = px[1:-1, :, :] - px[0:-2, :, :]
            dx = jnp.concatenate([first_x, mid_x, -px[-2:-1, :, :]], axis=0)
        else:
            dx = jnp.concatenate([first_x, -px[-2:-1, :, :]], axis=0)

    if py.shape[1] == 1:
        dy = jnp.zeros_like(py)
    else:
        first_y = py[:, 0:1, :]
        if py.shape[1] > 2:
            mid_y = py[:, 1:-1, :] - py[:, 0:-2, :]
            dy = jnp.concatenate([first_y, mid_y, -py[:, -2:-1, :]], axis=1)
        else:
            dy = jnp.concatenate([first_y, -py[:, -2:-1, :]], axis=1)

    if pz.shape[2] == 1:
        dz = jnp.zeros_like(pz)
    else:
        first_z = pz[:, :, 0:1]
        if pz.shape[2] > 2:
            mid_z = pz[:, :, 1:-1] - pz[:, :, 0:-2]
            dz = jnp.concatenate([first_z, mid_z, -pz[:, :, -2:-1]], axis=2)
        else:
            dz = jnp.concatenate([first_z, -pz[:, :, -2:-1]], axis=2)
    return dx + dy + dz


def validate_regulariser(
    regulariser: str,
    huber_delta: float,
    *,
    context: str,
) -> Regulariser:
    """Validate a regulariser config and return its normalized name."""
    reg = str(regulariser).lower()
    if reg not in ("tv", "huber_tv"):
        raise ValueError(
            f"{context}: regulariser must be 'tv' or 'huber_tv'; got {regulariser!r}"
        )
    if reg == "huber_tv":
        delta = float(huber_delta)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"{context}: huber_delta must be positive for huber_tv")
    return cast(Regulariser, reg)


def _gradient_norm_sq(u: jnp.ndarray) -> jnp.ndarray:
    gx, gy, gz = grad3(u)
    return gx * gx + gy * gy + gz * gz


def isotropic_tv_value(u: jnp.ndarray, eps: float = 0.0) -> jnp.ndarray:
    """Return isotropic 3D TV using the repo's forward-difference operator."""
    sq = _gradient_norm_sq(u)
    if eps == 0.0:
        return jnp.sum(jnp.sqrt(sq))
    return jnp.sum(jnp.sqrt(sq + jnp.asarray(eps, dtype=u.dtype)))


def huber_tv_value(u: jnp.ndarray, delta: float) -> jnp.ndarray:
    """Return isotropic Huber-TV with a quadratic basin of radius ``delta``."""
    delta_arr = jnp.asarray(delta, dtype=u.dtype)
    sq = _gradient_norm_sq(u)
    tiny = jnp.asarray(jnp.finfo(u.dtype).tiny, dtype=u.dtype)
    norm = jnp.sqrt(jnp.maximum(sq, tiny))
    quadratic = 0.5 * sq / delta_arr
    linear = norm - 0.5 * delta_arr
    return jnp.sum(jnp.where(norm <= delta_arr, quadratic, linear))


def huber_tv_grad(u: jnp.ndarray, delta: float) -> jnp.ndarray:
    """Return the gradient of isotropic Huber-TV under ``grad3``/``div3``."""
    delta_arr = jnp.asarray(delta, dtype=u.dtype)
    gx, gy, gz = grad3(u)
    norm = jnp.sqrt(gx * gx + gy * gy + gz * gz)
    denom = jnp.maximum(norm, delta_arr)
    qx = gx / denom
    qy = gy / denom
    qz = gz / denom
    return -div3(qx, qy, qz)


def prox_huber_tv_conj(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    p3: jnp.ndarray,
    *,
    sigma: float | jnp.ndarray,
    lam: float | jnp.ndarray,
    delta: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prox for the conjugate of ``lam * huber_tv`` on vector-gradient duals."""
    sigma_arr = jnp.asarray(sigma, dtype=p1.dtype)
    lam_arr = jnp.asarray(lam, dtype=p1.dtype)
    delta_arr = jnp.asarray(delta, dtype=p1.dtype)
    scale = jnp.where(
        lam_arr > 0.0,
        lam_arr / jnp.maximum(lam_arr + sigma_arr * delta_arr, jnp.finfo(p1.dtype).tiny),
        0.0,
    )
    q1 = p1 * scale
    q2 = p2 * scale
    q3 = p3 * scale
    radius = jnp.maximum(lam_arr, 0.0)
    norm = jnp.sqrt(q1 * q1 + q2 * q2 + q3 * q3)
    denom = jnp.maximum(radius, jnp.asarray(jnp.finfo(p1.dtype).eps, dtype=p1.dtype))
    shrink = jnp.maximum(1.0, norm / denom)
    return q1 / shrink, q2 / shrink, q3 / shrink
