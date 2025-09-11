from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp

from ..core.geometry import Geometry, Grid, Detector
from ..core.projector import forward_project_view


def _grad3(u: jnp.ndarray):
    dx = jnp.concatenate([jnp.diff(u, axis=0), jnp.zeros((1, u.shape[1], u.shape[2]), u.dtype)], axis=0)
    dy = jnp.concatenate([jnp.diff(u, axis=1), jnp.zeros((u.shape[0], 1, u.shape[2]), u.dtype)], axis=1)
    dz = jnp.concatenate([jnp.diff(u, axis=2), jnp.zeros((u.shape[0], u.shape[1], 1), u.dtype)], axis=2)
    return dx, dy, dz


def _div3(px: jnp.ndarray, py: jnp.ndarray, pz: jnp.ndarray):
    dx = px - jnp.pad(px[:-1, :, :], ((1, 0), (0, 0), (0, 0)))
    dy = py - jnp.pad(py[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    dz = pz - jnp.pad(pz[:, :, :-1], ((0, 0), (0, 0), (1, 0)))
    return dx + dy + dz


@dataclass
class FistaConfig:
    iters: int = 50
    lambda_tv: float = 0.005
    L_init: float = 1.0
    power_iters: int = 10
    checkpoint: bool = True


def grad_data_term(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    x: jnp.ndarray,
) -> Tuple[jnp.ndarray, float]:
    """Compute ∇(1/2 Σ_i ||A_i x - y_i||^2) and loss value."""
    n_views = int(projections.shape[0])
    loss = 0.0
    grad = jnp.zeros_like(x)

    for i in range(n_views):
        def fwd(vol):
            return forward_project_view(
                geometry=geometry,
                grid=grid,
                detector=detector,
                volume=vol,
                view_index=i,
                use_checkpoint=True,
            ).ravel()

        y_i = projections[i].ravel()
        Ax = fwd(x)
        r = Ax - y_i
        loss = loss + 0.5 * jnp.vdot(r, r).real
        _, vjp = jax.vjp(fwd, x)
        grad_i = vjp(r)[0]
        grad = grad + grad_i

    return grad, float(loss)


def power_method_L(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections_shape: Tuple[int, int, int],
    *,
    iters: int = 10,
) -> float:
    """Estimate Lipschitz constant of ∇f(x) ≈ ||A||^2 via power method on AᵀA."""
    n_views, nv, nu = projections_shape
    x = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    v = jnp.ones_like(x)
    v = v / jnp.linalg.norm(v.ravel())
    for _ in range(iters):
        g, _ = grad_data_term(geometry, grid, detector, jnp.zeros((n_views, nv, nu), dtype=jnp.float32), v)
        v = g / (jnp.linalg.norm(g.ravel()) + 1e-12)
    # Rayleigh quotient
    g, _ = grad_data_term(geometry, grid, detector, jnp.zeros((n_views, nv, nu), dtype=jnp.float32), v)
    L = float(jnp.vdot(v, g).real)
    return max(L, 1e-6)


def tv_proximal(x: jnp.ndarray, lam_over_L: float, iters: int = 20) -> jnp.ndarray:
    """Isotropic TV proximal via Chambolle's algorithm (3D generalization)."""
    p1 = jnp.zeros_like(x)
    p2 = jnp.zeros_like(x)
    p3 = jnp.zeros_like(x)
    tau = 0.25

    u = x
    for _ in range(iters):
        # Compute gradient of divergence term
        div_p = _div3(p1, p2, p3)
        w = u - lam_over_L * div_p
        gx, gy, gz = _grad3(w)
        # Update dual variables with projection onto unit ball
        denom = jnp.maximum(1.0, jnp.sqrt(gx * gx + gy * gy + gz * gz))
        p1 = (p1 + tau * gx) / denom
        p2 = (p2 + tau * gy) / denom
        p3 = (p3 + tau * gz) / denom
    return u - lam_over_L * _div3(p1, p2, p3)


def fista_tv(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    iters: int = 50,
    lambda_tv: float = 0.005,
    L: float | None = None,
    callback: Callable[[int, float], None] | None = None,
    init_x: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Run FISTA with TV regularization for parallel-ray geometry.

    Returns (x, info) where info contains loss history.
    """
    n_views, nv, nu = projections.shape
    x = (
        jnp.asarray(init_x, dtype=jnp.float32)
        if init_x is not None
        else jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    )
    z = x
    t = 1.0
    if L is None:
        L = power_method_L(geometry, grid, detector, projections.shape, iters=5)
    loss_hist = []

    for k in range(iters):
        g, data_loss = grad_data_term(geometry, grid, detector, projections, z)
        y = z - (1.0 / L) * g
        x_new = tv_proximal(y, lambda_tv / L, iters=10)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        gx, gy, gz = _grad3(x)
        tv_norm = jnp.sum(jnp.sqrt(gx * gx + gy * gy + gz * gz + 1e-8))
        obj = float(data_loss + lambda_tv * tv_norm)
        loss_hist.append(obj)
        if callback is not None:
            callback(k, obj)

    info = {"loss": loss_hist, "L": L}
    return x, info
