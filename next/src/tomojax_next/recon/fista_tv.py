from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp

from ..core.geometry import Geometry, Grid, Detector
from ..core.projector import forward_project_view, forward_project_view_T
from ..utils.logging import progress_iter


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
    """Compute ∇(1/2 Σ_i ||A_i x - y_i||^2) and loss via vmapped projector."""
    n_views = int(projections.shape[0])
    T_all = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )

    vm_project = jax.vmap(
        lambda T, vol: forward_project_view_T(T, grid, detector, vol, use_checkpoint=True),
        in_axes=(0, None),
    )

    def data_loss(vol):
        pred = vm_project(T_all, vol)
        resid = (pred - projections).astype(jnp.float32)
        return 0.5 * jnp.vdot(resid, resid).real

    val, grad = jax.value_and_grad(data_loss)(x)
    return grad, float(val)


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
    """Isotropic TV proximal via Chambolle's algorithm using lax.scan."""
    tau = jnp.float32(0.25)

    def body(carry, _):
        u, p1, p2, p3 = carry
        div_p = _div3(p1, p2, p3)
        w = u - lam_over_L * div_p
        gx, gy, gz = _grad3(w)
        denom = jnp.maximum(1.0, jnp.sqrt(gx * gx + gy * gy + gz * gz))
        p1_n = (p1 + tau * gx) / denom
        p2_n = (p2 + tau * gy) / denom
        p3_n = (p3 + tau * gz) / denom
        return (u, p1_n, p2_n, p3_n), None

    p1 = jnp.zeros_like(x)
    p2 = jnp.zeros_like(x)
    p3 = jnp.zeros_like(x)
    (u, p1, p2, p3), _ = jax.lax.scan(body, (x, p1, p2, p3), None, length=int(iters))
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
    # Precompute nominal poses and set up jitted loss/grad and prox
    n_views = int(projections.shape[0])
    T_all = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )
    vm_project = jax.vmap(
        lambda T, vol: forward_project_view_T(T, grid, detector, vol, use_checkpoint=True),
        in_axes=(0, None),
    )

    def data_loss(vol):
        pred = vm_project(T_all, vol)
        resid = (pred - projections).astype(jnp.float32)
        return 0.5 * jnp.vdot(resid, resid).real

    val_and_grad = jax.jit(jax.value_and_grad(data_loss))
    tv_prox_jit = jax.jit(tv_proximal, static_argnames=("iters",))

    def step(carry, k):
        x_c, z_c, t_c, loss_arr = carry
        data_loss_val, g = val_and_grad(z_c)
        y = z_c - (1.0 / L) * g
        x_new = tv_prox_jit(y, lambda_tv / L, iters=10)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_c * t_c))
        z_new = x_new + ((t_c - 1.0) / t_new) * (x_new - x_c)
        gx, gy, gz = _grad3(x_new)
        tv_norm = jnp.sum(jnp.sqrt(gx * gx + gy * gy + gz * gz + 1e-8))
        obj = data_loss_val + lambda_tv * tv_norm
        loss_arr = loss_arr.at[k].set(obj.astype(jnp.float32))
        return (x_new, z_new, t_new, loss_arr), None

    loss_arr0 = jnp.zeros((iters,), dtype=jnp.float32)
    (x_f, _, _, loss_arr), _ = jax.lax.scan(step, (x, z, t, loss_arr0), jnp.arange(iters))

    # Fire optional callbacks on endpoints only (avoid per-iter host sync)
    if callback is not None:
        callback(0, float(loss_arr[0]))
        callback(iters - 1, float(loss_arr[-1]))

    info = {"loss": [float(v) for v in list(loss_arr)], "L": L}
    return x_f, info
