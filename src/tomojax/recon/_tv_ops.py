"""Discrete isotropic-TV values, derivatives, and constrained proximal maps."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .types import Regulariser


def grad3(u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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


def gradient_norm_squared(shape: tuple[int, ...]) -> float:
    """Exact squared norm of the Neumann forward differences on a finite grid.

    An n-voxel axis contributes ``2 + 2*cos(pi/n)`` (zero for a singleton).
    The axis normal matrices form a Kronecker sum, so their maxima add.
    """
    return sum(2 + 2 * math.cos(math.pi / n) for n in shape if n > 1)


def div3(px: jnp.ndarray, py: jnp.ndarray, pz: jnp.ndarray) -> jnp.ndarray:
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
        raise ValueError(f"{context}: regulariser must be 'tv' or 'huber_tv'; got {regulariser!r}")
    if reg == "huber_tv":
        delta = float(huber_delta)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"{context}: huber_delta must be positive for huber_tv")
    return cast("Regulariser", reg)


def _gradient_norm_sq(u: jnp.ndarray) -> jnp.ndarray:
    gx, gy, gz = grad3(u)
    return gx * gx + gy * gy + gz * gz


def _norm_with_zero_subgradient(squared: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a norm safely inside the smooth regions of Huber formulas.

    The enclosing gradient/prox is differentiable at zero even though the
    Euclidean norm alone is not. Mask before sqrt: masking only its result
    leaves an infinite derivative that can produce NaNs in JVPs and VJPs.
    Nonzero norms and nonfinite inputs retain their ordinary values.
    """
    zero = squared == 0
    safe = jnp.where(zero, jnp.ones_like(squared), squared)
    return jnp.where(zero, jnp.zeros_like(squared), jnp.sqrt(safe))


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
    norm = _norm_with_zero_subgradient(gx * gx + gy * gy + gz * gz)
    denom = jnp.maximum(norm, delta_arr)
    qx = gx / denom
    qy = gy / denom
    qz = gz / denom
    return -div3(qx, qy, qz)


def project_tv_duals(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    p3: jnp.ndarray,
    *,
    radius: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project vector fields onto their requested ball, without an absolute floor.

    Normalize before squaring to avoid norm underflow/overflow at small/large
    physical signal scales. A zero radius projects exactly to zero; positive
    radii retain the identity derivative at zero.
    """
    radius = jnp.maximum(jnp.asarray(radius, dtype=p1.dtype), 0.0)
    peak = jnp.maximum(jnp.maximum(jnp.abs(p1), jnp.abs(p2)), jnp.abs(p3))
    scale = jnp.maximum(peak, radius)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q1, q2, q3 = p1 / scale, p2 / scale, p3 / scale
    norm = _norm_with_zero_subgradient(q1 * q1 + q2 * q2 + q3 * q3)
    norm = jnp.where(norm == 0, jnp.ones_like(norm), norm)
    inside = norm <= radius / scale
    length = radius / norm
    return (
        jnp.where(inside, p1, q1 * length),
        jnp.where(inside, p2, q2 * length),
        jnp.where(inside, p3, q3 * length),
    )


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
    return project_tv_duals(q1, q2, q3, radius=lam_arr)


def tv_proximal(
    x: jnp.ndarray,
    lam_over_L: float | jax.Array,
    iterations: int = 20,
    *,
    nonnegative: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> jnp.ndarray:
    """Approximate the isotropic TV proximal by projected gradient on its dual.

    Solve ``min_u 0.5 ||u - x||^2 + lam TV(u)`` inside the optional box through
    the dual field ``p`` with ``|p| <= lam`` pointwise. The minimizing primal is
    ``u = project_box(x + div p)``; its gradient drives every dual update.
    Clipping only after an unconstrained solve is not the joint isotropic-TV
    prox: clipping can change the direction of a coupled gradient vector.
    Box projection is nonexpansive, so the step ``1 / ||grad3||^2`` remains
    valid. Use the exact finite-grid norm, not its asymptotic 3D bound of 12.
    Only three dual components persist between iterations.
    """
    lam = jnp.asarray(lam_over_L, dtype=x.dtype)
    tau = jnp.asarray(1.0 / (gradient_norm_squared(x.shape) or 1.0), dtype=x.dtype)

    project = functools.partial(
        project_box,
        nonnegative=nonnegative,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    def prox_impl(lam_val: jnp.ndarray) -> jnp.ndarray:
        def body(
            p: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _: object
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
            gx, gy, gz = grad3(project(x + div3(*p)))
            q1, q2, q3 = p[0] + tau * gx, p[1] + tau * gy, p[2] + tau * gz
            return project_tv_duals(q1, q2, q3, radius=lam_val), None

        zeros = jnp.zeros_like(x)
        p, _ = jax.lax.scan(body, (zeros, zeros, zeros), None, length=int(iterations))
        return project(x + div3(*p))

    return jax.lax.cond(lam > 0, prox_impl, lambda _: project(x), lam)


def project_box(
    x: jnp.ndarray,
    *,
    nonnegative: bool,
    lower_bound: float | None,
    upper_bound: float | None,
) -> jnp.ndarray:
    """Project onto optional elementwise physical constraints."""
    if lower_bound is not None:
        x = jnp.maximum(x, jnp.asarray(lower_bound, dtype=x.dtype))
    if nonnegative:
        x = jnp.maximum(x, jnp.asarray(0.0, dtype=x.dtype))
    if upper_bound is not None:
        x = jnp.minimum(x, jnp.asarray(upper_bound, dtype=x.dtype))
    return x
