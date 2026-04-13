from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Tuple, Optional

import jax
import jax.numpy as jnp

from ..core.geometry import Geometry, Grid, Detector
from ..core.projector import forward_project_view_T, get_detector_grid_device



def _grad3(u: jnp.ndarray):
    dx = jnp.concatenate([jnp.diff(u, axis=0), jnp.zeros((1, u.shape[1], u.shape[2]), u.dtype)], axis=0)
    dy = jnp.concatenate([jnp.diff(u, axis=1), jnp.zeros((u.shape[0], 1, u.shape[2]), u.dtype)], axis=1)
    dz = jnp.concatenate([jnp.diff(u, axis=2), jnp.zeros((u.shape[0], u.shape[1], 1), u.dtype)], axis=2)
    return dx, dy, dz


def _div3(px: jnp.ndarray, py: jnp.ndarray, pz: jnp.ndarray):
    """Discrete divergence matching the TV updates: ``_div3 = -_grad3*``."""
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
    T_all: jnp.ndarray | None = None,
    vol_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, float]:
    """Compute ∇(1/2 Σ_i ||A_i x - y_i||^2) and loss.

    Two execution modes:
    - batched: vmap over a chunk of views. Fast but higher peak memory.
    - stream: process one view at a time via lax.scan and per-view VJP. Low peak memory.

    When grad_mode="auto", selects stream if the effective batch is 1, else batched.
    """
    n_views = int(projections.shape[0])
    nv = int(projections.shape[1])
    nu = int(projections.shape[2])
    if T_all is None:
        T_all = jnp.stack(
            [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
            axis=0,
        )

    det_grid = get_detector_grid_device(detector)

    def batched_loss(vol):
        """Loss over views batched in chunks, using a scan to keep jaxpr compact.

        We pad the last chunk to size ``b`` and mask it out in the reduction so the
        compiled graph is a single-sized loop body regardless of the number of chunks.
        """
        masked_vol = vol * vol_mask if vol_mask is not None else vol
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T,
                grid,
                detector,
                v,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            ),
            in_axes=(0, None),
        )
        # Use Python ints for sizes to keep them static at trace time
        n = int(T_all.shape[0])
        nv = int(projections.shape[1])
        nu = int(projections.shape[2])
        b = int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else n
        b = min(b, n)
        m = (n + b - 1) // b

        def body(loss_acc, i):
            i = jnp.int32(i)
            start = i * jnp.int32(b)
            remaining = jnp.maximum(0, jnp.int32(n) - start)
            valid = jnp.minimum(jnp.int32(b), remaining)
            # Shift start so that we always take a full-size window of length b ending at start+valid
            shift = jnp.int32(b) - valid
            start_shifted = jnp.maximum(0, start - shift)
            # Slice fixed-size (b, ...) window
            T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
            y_chunk = jax.lax.dynamic_slice(projections, (start_shifted, 0, 0), (b, nv, nu))
            pred = vm_project(T_chunk, masked_vol)
            # Keep only the last `valid` rows in the chunk
            idx = jnp.arange(b)
            mask = (idx >= (jnp.int32(b) - valid))[:, None, None]
            resid = (pred - y_chunk).astype(jnp.float32) * mask
            loss_batch = 0.5 * jnp.vdot(resid, resid).real
            return (loss_acc + loss_batch, None)

        loss0 = jnp.float32(0.0)
        loss_tot, _ = jax.lax.scan(body, loss0, jnp.arange(m))
        return loss_tot

    def stream_loss_and_grad(vol):
        masked_vol = vol * vol_mask if vol_mask is not None else vol

        def one_view(carry, i):
            loss_acc, g_acc = carry
            # Use dynamic slices in jitted context instead of array indexing
            T_i = jax.lax.dynamic_slice(T_all, (i, 0, 0), (1, 4, 4))[0]
            y_i = jax.lax.dynamic_slice(projections, (i, 0, 0), (1, nv, nu))[0]

            def fwd(masked_v):
                return forward_project_view_T(
                    T_i,
                    grid,
                    detector,
                    masked_v,
                    use_checkpoint=checkpoint_projector,
                    unroll=int(projector_unroll),
                    gather_dtype=gather_dtype,
                    det_grid=det_grid,
                )

            pred_i = fwd(masked_vol)
            resid_i = (pred_i - y_i).astype(jnp.float32)
            loss_i = 0.5 * jnp.vdot(resid_i, resid_i).real
            # Vectorize VJP via ravel to a single 1D cotangent
            _, vjp = jax.vjp(lambda vv: fwd(vv).ravel(), masked_vol)
            g_i = vjp(resid_i.ravel())[0]
            if vol_mask is not None:
                g_i = g_i * vol_mask
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


def data_term_value(
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
    T_all: jnp.ndarray | None = None,
    vol_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute the data term ``1/2 Σ_i ||A_i x - y_i||^2`` without its gradient."""
    n_views = int(projections.shape[0])
    nv = int(projections.shape[1])
    nu = int(projections.shape[2])
    if T_all is None:
        T_all = jnp.stack(
            [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
            axis=0,
        )

    det_grid = get_detector_grid_device(detector)

    def batched_loss(vol):
        masked_vol = vol * vol_mask if vol_mask is not None else vol
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T,
                grid,
                detector,
                v,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            ),
            in_axes=(0, None),
        )
        n = int(T_all.shape[0])
        b = int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else n
        b = min(b, n)
        m = (n + b - 1) // b

        def body(loss_acc, i):
            i = jnp.int32(i)
            start = i * jnp.int32(b)
            remaining = jnp.maximum(0, jnp.int32(n) - start)
            valid = jnp.minimum(jnp.int32(b), remaining)
            shift = jnp.int32(b) - valid
            start_shifted = jnp.maximum(0, start - shift)
            T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
            y_chunk = jax.lax.dynamic_slice(projections, (start_shifted, 0, 0), (b, nv, nu))
            pred = vm_project(T_chunk, masked_vol)
            idx = jnp.arange(b)
            mask = (idx >= (jnp.int32(b) - valid))[:, None, None]
            resid = (pred - y_chunk).astype(jnp.float32) * mask
            loss_batch = 0.5 * jnp.vdot(resid, resid).real
            return (loss_acc + loss_batch, None)

        loss0 = jnp.float32(0.0)
        loss_tot, _ = jax.lax.scan(body, loss0, jnp.arange(m))
        return loss_tot

    def stream_loss(vol):
        masked_vol = vol * vol_mask if vol_mask is not None else vol

        def one_view(loss_acc, i):
            T_i = jax.lax.dynamic_slice(T_all, (i, 0, 0), (1, 4, 4))[0]
            y_i = jax.lax.dynamic_slice(projections, (i, 0, 0), (1, nv, nu))[0]

            pred_i = forward_project_view_T(
                T_i,
                grid,
                detector,
                masked_vol,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            )
            resid_i = (pred_i - y_i).astype(jnp.float32)
            loss_i = 0.5 * jnp.vdot(resid_i, resid_i).real
            return loss_acc + loss_i, None

        loss0 = jnp.float32(0.0)
        loss_tot, _ = jax.lax.scan(one_view, loss0, jnp.arange(T_all.shape[0]))
        return loss_tot

    eff_b = int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else T_all.shape[0]
    mode = grad_mode
    if grad_mode == "auto":
        mode = "stream" if eff_b <= 1 else "batched"

    if mode == "stream":
        return stream_loss(x)
    return batched_loss(x)


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
    T_all: jnp.ndarray | None = None,
    vol_mask: Optional[jnp.ndarray] = None,
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
            T_all=T_all,
            vol_mask=vol_mask,
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
        T_all=T_all,
        vol_mask=vol_mask,
    )
    L = float(jnp.vdot(v, g).real)
    return max(L, 1e-6)


def tv_proximal(x: jnp.ndarray, lam_over_L: float, iters: int = 20) -> jnp.ndarray:
    """Apply the isotropic TV proximal via a PDHG (Chambolle-Pock) update."""
    lam = jnp.asarray(lam_over_L, dtype=x.dtype)
    tau = jnp.asarray(0.25, dtype=x.dtype)
    sigma = jnp.asarray(0.25, dtype=x.dtype)
    theta = jnp.asarray(1.0, dtype=x.dtype)
    eps = jnp.asarray(jnp.finfo(x.dtype).eps, dtype=x.dtype)

    def prox_impl(lam_val):
        lam_safe = jnp.maximum(lam_val, eps)

        def body(carry, _):
            u, u_bar, p1, p2, p3 = carry
            gx, gy, gz = _grad3(u_bar)
            p1_n = p1 + sigma * gx
            p2_n = p2 + sigma * gy
            p3_n = p3 + sigma * gz
            norm = jnp.maximum(1.0, jnp.sqrt(p1_n * p1_n + p2_n * p2_n + p3_n * p3_n) / lam_safe)
            p1_n = p1_n / norm
            p2_n = p2_n / norm
            p3_n = p3_n / norm
            div_p = _div3(p1_n, p2_n, p3_n)
            u_prev = u
            u_minus = u + tau * div_p
            u_n = (u_minus + tau * x) / (1.0 + tau)
            u_bar_n = u_n + theta * (u_n - u_prev)
            return (u_n, u_bar_n, p1_n, p2_n, p3_n), None

        init = (
            x,
            x,
            jnp.zeros_like(x),
            jnp.zeros_like(x),
            jnp.zeros_like(x),
        )
        (u, _, _, _, _), _ = jax.lax.scan(body, init, None, length=int(iters))
        return u

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
    views_per_batch: int | None = 1,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    grad_mode: Literal["auto", "batched", "stream"] = "auto",
    tv_prox_iters: int = 10,
    recon_rel_tol: float | None = None,
    recon_patience: int = 0,
    vol_mask: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, dict]:
    """Run FISTA with TV regularization for parallel-ray geometry.

    Args:
        recon_rel_tol: Relative tolerance on successive objectives for early termination. ``None`` disables.
        recon_patience: Number of consecutive tolerance hits required before stopping early.

    Returns:
        Tuple of reconstructed volume and metadata. ``info`` includes the full loss history, the
        measured/assumed Lipschitz constant, whether early stopping triggered, and the effective
        iteration count.
    """
    x = (
        jnp.asarray(init_x, dtype=jnp.float32)
        if init_x is not None
        else jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    )
    z = x
    t = 1.0
    # Precompute poses once and thread them through the projector calls to avoid
    # repeatedly stacking in inner loops.
    n_views = int(projections.shape[0])
    T_all = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )

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
            T_all=T_all,
            vol_mask=vol_mask,
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
            T_all=T_all,
            vol_mask=vol_mask,
        )
        return v, g
    val_and_grad = jax.jit(val_and_grad_fn, donate_argnums=(0,))

    def data_value_fn(x):
        return data_term_value(
            geometry,
            grid,
            detector,
            projections,
            x,
            views_per_batch=views_per_batch,
            projector_unroll=projector_unroll,
            checkpoint_projector=checkpoint_projector,
            gather_dtype=gather_dtype,
            grad_mode=grad_mode,
            T_all=T_all,
            vol_mask=vol_mask,
        )

    data_value = jax.jit(data_value_fn, donate_argnums=(0,))
    tv_prox_jit = jax.jit(tv_proximal, static_argnames=("iters",))

    use_early_stop = (
        (recon_rel_tol is not None)
        and float(recon_rel_tol) > 0.0
        and int(recon_patience) > 0
    )
    tol = jnp.float32(float(recon_rel_tol) if use_early_stop else 0.0)
    patience = jnp.int32(int(recon_patience) if use_early_stop else 0)
    early_flag = jnp.bool_(use_early_stop)

    def step(carry, k):
        x_c, z_c, t_c, loss_arr, prev_obj, streak, done, has_prev, last_obj, iters_done = carry

        def run_active(state):
            (
                x_a,
                z_a,
                t_a,
                loss_a,
                prev_a,
                streak_a,
                done_a,
                has_prev_a,
                last_a,
                iters_a,
            ) = state
            _, g = val_and_grad(z_a)
            y = z_a - (1.0 / L) * g
            x_new = tv_prox_jit(y, lambda_tv / L, iters=int(tv_prox_iters))
            t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_a * t_a))
            z_new = x_new + ((t_a - 1.0) / t_new) * (x_new - x_a)
            data_loss_val = data_value(x_new)
            gx, gy, gz = _grad3(x_new)
            tv_norm = jnp.sum(jnp.sqrt(gx * gx + gy * gy + gz * gz + 1e-8))
            obj = data_loss_val + lambda_tv * tv_norm
            obj32 = obj.astype(jnp.float32)
            rel_change = jnp.abs(obj - prev_a) / jnp.maximum(jnp.abs(prev_a), 1e-6)
            small = jnp.logical_and(
                early_flag, jnp.logical_and(has_prev_a, rel_change <= tol)
            )
            streak_next = jnp.where(
                early_flag,
                jnp.where(small, jnp.minimum(streak_a + 1, patience), jnp.int32(0)),
                streak_a,
            )
            done_next = jnp.logical_or(done_a, jnp.logical_and(early_flag, streak_next >= patience))
            loss_next = loss_a.at[k].set(obj32)
            return (
                x_new,
                z_new,
                t_new,
                loss_next,
                obj,
                streak_next,
                done_next,
                jnp.bool_(True),
                obj,
                iters_a + jnp.int32(1),
            ), None

        def run_skip(state):
            (
                x_s,
                z_s,
                t_s,
                loss_s,
                prev_s,
                streak_s,
                done_s,
                has_prev_s,
                last_s,
                iters_s,
            ) = state
            loss_next = loss_s.at[k].set(last_s.astype(jnp.float32))
            return (
                x_s,
                z_s,
                t_s,
                loss_next,
                prev_s,
                streak_s,
                done_s,
                has_prev_s,
                last_s,
                iters_s,
            ), None

        new_state, _ = jax.lax.cond(
            done,
            run_skip,
            run_active,
            (x_c, z_c, t_c, loss_arr, prev_obj, streak, done, has_prev, last_obj, iters_done),
        )
        return new_state, None

    loss_arr0 = jnp.zeros((iters,), dtype=jnp.float32)
    init_carry = (
        x,
        z,
        t,
        loss_arr0,
        jnp.float32(0.0),
        jnp.int32(0),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.float32(0.0),
        jnp.int32(0),
    )
    carry_final, _ = jax.lax.scan(step, init_carry, jnp.arange(iters))
    (
        x_f,
        _,
        _,
        loss_arr,
        _,
        _,
        done_flag,
        _,
        _,
        iters_done,
    ) = carry_final

    # Fire optional callbacks on endpoints only (avoid per-iter host sync)
    if callback is not None:
        callback(0, float(loss_arr[0]))
        callback(iters - 1, float(loss_arr[-1]))

    info = {
        "loss": [float(v) for v in list(loss_arr)],
        "L": L,
        "effective_iters": int(iters_done),
        "early_stop": bool(done_flag),
    }
    return x_f, info
