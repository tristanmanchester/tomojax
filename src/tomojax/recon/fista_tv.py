from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Literal

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
    *,
    views_per_batch: int | None = None,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    grad_mode: Literal["auto", "batched", "stream"] = "auto",
) -> Tuple[jnp.ndarray, float]:
    """Compute ∇(1/2 Σ_i ||A_i x - y_i||^2) and loss.

    Two execution modes:
    - batched: vmap over a chunk of views. Fast but higher peak memory.
    - stream: process one view at a time via lax.scan and per-view VJP. Low peak memory.

    When grad_mode="auto", selects stream if the effective batch is 1, else batched.
    """
    n_views = int(projections.shape[0])
    T_all = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )

    def batched_loss(vol):
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T,
                grid,
                detector,
                v,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
            ),
            in_axes=(0, None),
        )
        n = T_all.shape[0]
        b = int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else n
        loss = jnp.float32(0.0)
        for s in range(0, n, b):
            pred = vm_project(T_all[s : s + b], vol)
            resid = (pred - projections[s : s + b]).astype(jnp.float32)
            loss = loss + 0.5 * jnp.vdot(resid, resid).real
        return loss

    def stream_loss_and_grad(vol):
        def one_view(carry, i):
            loss_acc, g_acc = carry
            T_i = T_all[i]
            y_i = projections[i]
            def fwd(v):
                return forward_project_view_T(
                    T_i,
                    grid,
                    detector,
                    v,
                    use_checkpoint=checkpoint_projector,
                    unroll=int(projector_unroll),
                    gather_dtype=gather_dtype,
                )
            pred_i = fwd(vol)
            resid_i = (pred_i - y_i).astype(jnp.float32)
            loss_i = 0.5 * jnp.vdot(resid_i, resid_i).real
            # Vectorize VJP via ravel to a single 1D cotangent
            _, vjp = jax.vjp(lambda vv: fwd(vv).ravel(), vol)
            g_i = vjp(resid_i.ravel())[0]
            return (loss_acc + loss_i, g_acc + g_i), None
        init = (jnp.float32(0.0), jnp.zeros_like(vol))
        (loss_tot, g_tot), _ = jax.lax.scan(one_view, init, jnp.arange(T_all.shape[0]))
        return loss_tot, g_tot

    # Select execution mode
    eff_b = int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else T_all.shape[0]
    mode = grad_mode
    if grad_mode == "auto":
        mode = "stream" if eff_b <= 1 else "batched"

    if mode == "stream":
        loss_val, grad = stream_loss_and_grad(x)
        return grad, loss_val
    else:
        val, grad = jax.value_and_grad(batched_loss)(x)
        return grad, val


def power_method_L(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections_shape: Tuple[int, int, int],
    *,
    iters: int = 10,
    views_per_batch: int | None = None,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    grad_mode: Literal["auto", "batched", "stream"] = "auto",
) -> float:
    """Estimate Lipschitz constant of ∇f(x) ≈ ||A||^2 via power method on AᵀA."""
    n_views, nv, nu = projections_shape
    x = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    v = jnp.ones_like(x)
    v = v / jnp.linalg.norm(v.ravel())
    for _ in range(iters):
        g, _ = grad_data_term(
            geometry,
            grid,
            detector,
            jnp.zeros((n_views, nv, nu), dtype=jnp.float32),
            v,
            views_per_batch=views_per_batch,
            projector_unroll=projector_unroll,
            checkpoint_projector=checkpoint_projector,
            gather_dtype=gather_dtype,
            grad_mode=grad_mode,
        )
        v = g / (jnp.linalg.norm(g.ravel()) + 1e-12)
    # Rayleigh quotient
    g, _ = grad_data_term(
        geometry,
        grid,
        detector,
        jnp.zeros((n_views, nv, nu), dtype=jnp.float32),
        v,
        views_per_batch=views_per_batch,
        projector_unroll=projector_unroll,
        checkpoint_projector=checkpoint_projector,
        gather_dtype=gather_dtype,
        grad_mode=grad_mode,
    )
    L = float(jnp.vdot(v, g).real)
    return max(L, 1e-6)


def tv_proximal(x: jnp.ndarray, lam_over_L: float, iters: int = 20) -> jnp.ndarray:
    """Isotropic TV proximal via Chambolle's dual ascent."""
    lam = jnp.asarray(lam_over_L, dtype=x.dtype)
    tau = jnp.asarray(1.0 / (2.0 * x.ndim), dtype=x.dtype)
    eps = jnp.asarray(jnp.finfo(x.dtype).eps, dtype=x.dtype)

    def prox_impl(lam_val):
        lam_safe = jnp.maximum(lam_val, eps)

        def body(carry, _):
            p1, p2, p3 = carry
            div_p = _div3(p1, p2, p3)
            u = x - div_p
            gx, gy, gz = _grad3(u)
            norm = jnp.sqrt(gx * gx + gy * gy + gz * gz)
            denom = 1.0 + (tau / lam_safe) * norm
            p1_n = (p1 - tau * gx) / denom
            p2_n = (p2 - tau * gy) / denom
            p3_n = (p3 - tau * gz) / denom
            return (p1_n, p2_n, p3_n), None

        init = (jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x))
        (p1, p2, p3), _ = jax.lax.scan(body, init, None, length=int(iters))
        div_p = _div3(p1, p2, p3)
        return x - div_p

    return jax.lax.cond(lam > 0, prox_impl, lambda _: x, lam)


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
    views_per_batch: int | None = None,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    grad_mode: Literal["auto", "batched", "stream"] = "auto",
    tv_prox_iters: int = 10,
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
        # Use streamed gradient in power-method to avoid batched VJP memory spikes
        L = power_method_L(
            geometry,
            grid,
            detector,
            projections.shape,
            iters=5,
            views_per_batch=views_per_batch,
            projector_unroll=projector_unroll,
            checkpoint_projector=checkpoint_projector,
            gather_dtype=gather_dtype,
            grad_mode="stream",
        )
    # Precompute jitted loss/grad using the chunked grad_data_term
    def val_and_grad_fn(z):
        g, v = grad_data_term(
            geometry,
            grid,
            detector,
            projections,
            z,
            views_per_batch=views_per_batch,
            projector_unroll=projector_unroll,
            checkpoint_projector=checkpoint_projector,
            gather_dtype=gather_dtype,
            grad_mode=grad_mode,
        )
        return v, g
    val_and_grad = jax.jit(val_and_grad_fn, donate_argnums=(0,))
    tv_prox_jit = jax.jit(tv_proximal, static_argnames=("iters",))

    def step(carry, k):
        x_c, z_c, t_c, loss_arr = carry
        data_loss_val, g = val_and_grad(z_c)
        y = z_c - (1.0 / L) * g
        x_new = tv_prox_jit(y, lambda_tv / L, iters=int(tv_prox_iters))
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
