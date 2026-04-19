from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal, Tuple, Optional

import jax
import jax.numpy as jnp

from ..core.geometry.base import Geometry, Grid, Detector
from ..core.geometry.views import stack_view_poses
from ..core.projector import (
    backproject_view_T,
    forward_project_view_T,
    get_detector_grid_device,
    sum_backproject_views_T,
)
from ..core.validation import (
    validate_grid,
    validate_optional_broadcastable_shape,
    validate_optional_same_shape,
    validate_pose_stack,
    validate_projection_shape,
    validate_projection_stack,
    validate_volume,
)
from ._callbacks import LossCallback, emit_loss_callback_endpoints
from ._tv_ops import div3, grad3


GradMode = Literal["auto", "batched", "stream"]


@dataclass
class FistaConfig:
    iters: int = 50
    lambda_tv: float = 0.005
    L: float | None = None
    views_per_batch: int | None = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    grad_mode: GradMode = "auto"
    tv_prox_iters: int = 10
    recon_rel_tol: float | None = None
    recon_patience: int = 0
    power_iters: int = 5
    support: jnp.ndarray | None = None
    positivity: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None


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
    grad_mode: GradMode = "auto",
    T_all: jnp.ndarray | None = None,
    vol_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, float]:
    """Compute ∇(1/2 Σ_i ||A_i x - y_i||^2) and loss.

    Two execution modes:
    - batched: vmap over a chunk of views. Fast but higher peak memory.
    - stream: process one view at a time via lax.scan and explicit adjoint. Low peak memory.

    When grad_mode="auto", selects stream if the effective batch is 1, else batched.
    """
    validate_grid(grid, "grad_data_term grid")
    n_views, nv, nu = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="grad_data_term projections",
    )
    validate_volume(x, grid, context="grad_data_term", name="x")
    validate_optional_broadcastable_shape(
        vol_mask,
        (grid.nx, grid.ny, grid.nz),
        context="grad_data_term support",
        name="vol_mask",
        fix="use a volume mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if T_all is None:
        T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="grad_data_term geometry")

    det_grid = get_detector_grid_device(detector)
    mask_arr = None if vol_mask is None else jnp.asarray(vol_mask, dtype=jnp.float32)

    def apply_mask(vol):
        return vol * mask_arr if mask_arr is not None else vol

    def adjoint(resid, T_i):
        grad_i = backproject_view_T(
            T_i,
            grid,
            detector,
            resid,
            unroll=int(projector_unroll),
            gather_dtype=gather_dtype,
            det_grid=det_grid,
        )
        return grad_i if mask_arr is None else grad_i * mask_arr

    def batched_loss_and_grad(vol):
        """Loss/grad over views batched in chunks, using a scan to keep jaxpr compact.

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
        n = int(T_all.shape[0])
        nv = int(projections.shape[1])
        nu = int(projections.shape[2])
        b = (
            int(views_per_batch)
            if (views_per_batch is not None and int(views_per_batch) > 0)
            else n
        )
        b = min(b, n)
        m = (n + b - 1) // b

        def body(carry, i):
            loss_acc, grad_acc = carry
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
            grad_batch = sum_backproject_views_T(
                T_chunk,
                grid,
                detector,
                resid,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            )
            if mask_arr is not None:
                grad_batch = grad_batch * mask_arr
            return ((loss_acc + loss_batch, grad_acc + grad_batch), None)

        init = (jnp.float32(0.0), jnp.zeros_like(vol))
        (loss_tot, grad_tot), _ = jax.lax.scan(body, init, jnp.arange(m))
        return loss_tot, grad_tot

    def stream_loss_and_grad(vol):
        masked_vol = vol * vol_mask if vol_mask is not None else vol

        def one_view(carry, i):
            loss_acc, g_acc = carry
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
            g_i = adjoint(resid_i, T_i)
            return (loss_acc + loss_i, g_acc + g_i), None

        init = (jnp.float32(0.0), jnp.zeros_like(vol))
        (loss_tot, g_tot), _ = jax.lax.scan(one_view, init, jnp.arange(T_all.shape[0]))
        return loss_tot, g_tot

    # Select execution mode
    eff_b = (
        int(views_per_batch)
        if (views_per_batch is not None and int(views_per_batch) > 0)
        else T_all.shape[0]
    )
    mode = grad_mode
    if grad_mode == "auto":
        mode = "stream" if eff_b <= 1 else "batched"

    if mode == "stream":
        loss_val, grad = stream_loss_and_grad(x)
        return grad, loss_val
    else:
        loss_val, grad = batched_loss_and_grad(x)
        return grad, loss_val


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
    grad_mode: GradMode = "auto",
    T_all: jnp.ndarray | None = None,
    vol_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute the data term ``1/2 Σ_i ||A_i x - y_i||^2`` without its gradient."""
    validate_grid(grid, "data_term_value grid")
    n_views, nv, nu = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="data_term_value projections",
    )
    validate_volume(x, grid, context="data_term_value", name="x")
    validate_optional_broadcastable_shape(
        vol_mask,
        (grid.nx, grid.ny, grid.nz),
        context="data_term_value support",
        name="vol_mask",
        fix="use a volume mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if T_all is None:
        T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="data_term_value geometry")

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
        b = (
            int(views_per_batch)
            if (views_per_batch is not None and int(views_per_batch) > 0)
            else n
        )
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

    eff_b = (
        int(views_per_batch)
        if (views_per_batch is not None and int(views_per_batch) > 0)
        else T_all.shape[0]
    )
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
    grad_mode: GradMode = "auto",
    T_all: jnp.ndarray | None = None,
    vol_mask: Optional[jnp.ndarray] = None,
) -> float:
    """Estimate Lipschitz constant of ∇f(x) ≈ ||A||^2 via power method on AᵀA."""
    validate_grid(grid, "power_method_L grid")
    n_views, nv, nu = validate_projection_shape(
        projections_shape,
        detector,
        geometry=geometry,
        context="power_method_L projections_shape",
    )
    validate_optional_broadcastable_shape(
        vol_mask,
        (grid.nx, grid.ny, grid.nz),
        context="power_method_L support",
        name="vol_mask",
        fix="use a volume mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if T_all is None:
        T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="power_method_L geometry")
    zero_proj = jnp.zeros((n_views, nv, nu), dtype=jnp.float32)
    num_iters = max(1, int(iters))

    def ata_apply(v):
        g, _ = grad_data_term(
            geometry,
            grid,
            detector,
            zero_proj,
            v,
            views_per_batch=views_per_batch,
            projector_unroll=projector_unroll,
            checkpoint_projector=checkpoint_projector,
            gather_dtype=gather_dtype,
            grad_mode=grad_mode,
            T_all=T_all,
            vol_mask=vol_mask,
        )
        return g

    def normalize(v):
        return v / (jnp.linalg.norm(v.ravel()) + 1e-12)

    ata_apply_jit = jax.jit(ata_apply)
    v0 = jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    v = normalize(v0)
    for _ in range(num_iters):
        v = normalize(ata_apply_jit(v))
    g = ata_apply_jit(v)
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
            gx, gy, gz = grad3(u_bar)
            p1_n = p1 + sigma * gx
            p2_n = p2 + sigma * gy
            p3_n = p3 + sigma * gz
            norm = jnp.maximum(1.0, jnp.sqrt(p1_n * p1_n + p2_n * p2_n + p3_n * p3_n) / lam_safe)
            p1_n = p1_n / norm
            p2_n = p2_n / norm
            p3_n = p3_n / norm
            div_p = div3(p1_n, p2_n, p3_n)
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


def _normalize_constraint_config(cfg: FistaConfig) -> tuple[bool, float | None, float | None]:
    """Validate FISTA feasibility constraints and return scalar bounds."""
    lower = None if cfg.lower_bound is None else float(cfg.lower_bound)
    upper = None if cfg.upper_bound is None else float(cfg.upper_bound)

    if lower is not None and not math.isfinite(lower):
        raise ValueError("fista_tv constraints: lower_bound must be finite when provided")
    if upper is not None and not math.isfinite(upper):
        raise ValueError("fista_tv constraints: upper_bound must be finite when provided")

    effective_lower = lower
    if bool(cfg.positivity):
        effective_lower = max(0.0, lower) if lower is not None else 0.0

    if upper is not None and effective_lower is not None and upper < effective_lower:
        raise ValueError(
            "fista_tv constraints: upper_bound must be greater than or equal to "
            "the effective lower bound"
        )

    return bool(cfg.positivity), lower, upper


def _project_constraints(
    x: jnp.ndarray,
    *,
    positivity: bool,
    lower_bound: float | None,
    upper_bound: float | None,
) -> jnp.ndarray:
    """Project a volume onto optional elementwise physical constraints."""
    if lower_bound is not None:
        x = jnp.maximum(x, jnp.asarray(lower_bound, dtype=x.dtype))
    if positivity:
        x = jnp.maximum(x, jnp.asarray(0.0, dtype=x.dtype))
    if upper_bound is not None:
        x = jnp.minimum(x, jnp.asarray(upper_bound, dtype=x.dtype))
    return x


def fista_tv(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    init_x: jnp.ndarray | None = None,
    config: FistaConfig = FistaConfig(),
    callback: LossCallback | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Run FISTA with TV regularization using an explicit solver configuration.

    If ``callback`` is provided, it fires on the first recorded loss sample and on
    the final recorded loss sample. The callback arguments are ``(step, loss)``,
    where ``step`` is the zero-based iteration index that produced ``loss``. When
    early stopping truncates active iterations, the final callback reports the
    last active iteration rather than the repeated padded tail entry.
    """
    cfg = config
    vol_mask = cfg.support
    positivity, lower_bound, upper_bound = _normalize_constraint_config(cfg)
    constraints_enabled = positivity or lower_bound is not None or upper_bound is not None
    validate_grid(grid, "fista_tv grid")
    n_views, _, _ = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="fista_tv projections",
    )
    validate_optional_broadcastable_shape(
        vol_mask,
        (grid.nx, grid.ny, grid.nz),
        context="fista_tv support",
        name="support",
        fix="use a support mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if init_x is not None:
        validate_volume(init_x, grid, context="fista_tv init_x", name="init_x")
    x = (
        jnp.asarray(init_x, dtype=jnp.float32)
        if init_x is not None
        else jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    )
    if constraints_enabled:
        x = _project_constraints(
            x,
            positivity=positivity,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    z = x
    t = 1.0
    # Precompute poses once and thread them through the projector calls to avoid
    # repeatedly stacking in inner loops.
    T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="fista_tv geometry")

    L = cfg.L
    if L is None:
        # Use streamed gradient in power-method to avoid batched VJP memory spikes
        L = power_method_L(
            geometry,
            grid,
            detector,
            projections.shape,
            iters=cfg.power_iters,
            views_per_batch=cfg.views_per_batch,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
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
            views_per_batch=cfg.views_per_batch,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            grad_mode=cfg.grad_mode,
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
            views_per_batch=cfg.views_per_batch,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            grad_mode=cfg.grad_mode,
            T_all=T_all,
            vol_mask=vol_mask,
        )

    data_value = jax.jit(data_value_fn, donate_argnums=(0,))
    tv_prox_jit = jax.jit(tv_proximal, static_argnames=("iters",))

    use_early_stop = (
        (cfg.recon_rel_tol is not None)
        and float(cfg.recon_rel_tol) > 0.0
        and int(cfg.recon_patience) > 0
    )
    tol = jnp.float32(float(cfg.recon_rel_tol) if use_early_stop else 0.0)
    patience = jnp.int32(int(cfg.recon_patience) if use_early_stop else 0)
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
            x_new = tv_prox_jit(y, cfg.lambda_tv / L, iters=int(cfg.tv_prox_iters))
            x_new = _project_constraints(
                x_new,
                positivity=positivity,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t_a * t_a))
            z_new = x_new + ((t_a - 1.0) / t_new) * (x_new - x_a)
            z_new = _project_constraints(
                z_new,
                positivity=positivity,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            data_loss_val = data_value(x_new)
            gx, gy, gz = grad3(x_new)
            tv_norm = jnp.sum(jnp.sqrt(gx * gx + gy * gy + gz * gz + 1e-8))
            obj = data_loss_val + cfg.lambda_tv * tv_norm
            obj32 = obj.astype(jnp.float32)
            rel_change = jnp.abs(obj - prev_a) / jnp.maximum(jnp.abs(prev_a), 1e-6)
            small = jnp.logical_and(early_flag, jnp.logical_and(has_prev_a, rel_change <= tol))
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

    loss_arr0 = jnp.zeros((int(cfg.iters),), dtype=jnp.float32)
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
    carry_final, _ = jax.lax.scan(step, init_carry, jnp.arange(int(cfg.iters)))
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

    if int(cfg.iters) > 0:
        final_step = max(int(iters_done) - 1, 0)
        emit_loss_callback_endpoints(
            callback,
            (
                (0, float(loss_arr[0])),
                (final_step, float(loss_arr[final_step])),
            ),
        )

    info = {
        "loss": [float(v) for v in list(loss_arr)],
        "L": L,
        "effective_iters": int(iters_done),
        "early_stop": bool(done_flag),
    }
    return x_f, info
