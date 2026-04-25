from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp

from ..core.geometry.base import Geometry, Grid, Detector
from ..core.geometry.views import stack_view_poses
from ..core.projector import (
    forward_project_view_T,
    get_detector_grid_device,
    sum_backproject_views_T,
)
from ..core.validation import (
    validate_grid,
    validate_optional_broadcastable_shape,
    validate_optional_same_shape,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)
from ._callbacks import LossCallback, emit_loss_callback_endpoints
from ._tv_ops import (
    Regulariser,
    div3,
    grad3,
    huber_tv_value,
    isotropic_tv_value,
    prox_huber_tv_conj,
    validate_regulariser,
)


# --------- config ----------
@dataclass
class SPDHGConfig:
    iters: int = 400
    lambda_tv: float = 5e-3
    regulariser: Regulariser = "tv"
    huber_delta: float = 1e-2
    theta: float = 1.0  # extrapolation for xbar
    views_per_batch: int = 16  # size of a stochastic block
    seed: int = 0

    # step sizes (set to None => auto from operator norms)
    tau: float | None = None
    sigma_data: float | None = None
    sigma_tv: float | None = None

    # projector / memory knobs
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"

    # constraints
    positivity: bool = True
    support: jnp.ndarray | None = None  # 0/1 mask in volume space

    # logging
    log_every: int = 10  # minibatch objective estimator every k steps


# --------- helpers ----------
def _estimate_norm_A2(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections_shape: Tuple[int, int, int],
    T_all: jnp.ndarray,
    *,
    views_per_batch: int,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
    key: jax.Array | None = None,
    power_iters: int = 20,
    safety: float = 1.05,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> float:
    """Reuse your power method to estimate ||A||^2 (i.e., Lipschitz of ∇(1/2||Ax||^2))."""
    n_views, nv, nu = projections_shape
    det_grid = get_detector_grid_device(detector) if det_grid is None else det_grid

    def A_apply(vol, T_chunk):
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
        return vm_project(T_chunk, vol)

    b = int(max(1, min(views_per_batch, n_views)))
    m = (n_views + b - 1) // b
    num_iters = max(1, int(power_iters))

    def AtranA(v):
        # iterate over contiguous blocks with masking of the last chunk
        def body(g_acc, i):
            i = jnp.int32(i)
            start = i * jnp.int32(b)
            remaining = jnp.maximum(0, jnp.int32(n_views) - start)
            valid = jnp.minimum(jnp.int32(b), remaining)
            shift = jnp.int32(b) - valid
            start_shifted = jnp.maximum(0, start - shift)

            T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
            pred_fun = lambda vol: A_apply(vol, T_chunk)
            proj = pred_fun(v)
            idx = jnp.arange(b)
            mask = (idx >= (jnp.int32(b) - valid))[:, None, None]
            proj = proj * mask  # zero padded rows

            g_chunk = sum_backproject_views_T(
                T_chunk,
                grid,
                detector,
                proj,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            )
            return g_acc + g_chunk, None

        g0 = jnp.zeros_like(v)
        g_final, _ = jax.lax.scan(body, g0, jnp.arange(m))
        return g_final

    def normalize(v):
        return v / (jnp.linalg.norm(v) + 1e-12)

    AtranA_jit = jax.jit(AtranA)

    if key is None:
        key = jax.random.key(0)
    v0 = jax.random.normal(key, (grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    v = normalize(v0)
    for _ in range(num_iters):
        v = normalize(AtranA_jit(v))
    Aw = AtranA_jit(v)
    L = float(jnp.vdot(v, Aw).real) * float(safety**2)  # ~||A||^2 with margin
    return max(L, 1e-6)


def _proj_pos_support(x: jnp.ndarray, positivity: bool, support: jnp.ndarray | None):
    if support is not None:
        x = x * support
    if positivity:
        x = jnp.maximum(x, 0)
    return x


def _prox_fstar_l2(u: jnp.ndarray, sigma: float, y_meas: jnp.ndarray, w: jnp.ndarray):
    """prox_{σ f*}(u) for f(z) = 1/2 ||W^{1/2}(z - y)||^2. Elementwise:
    if w>0: (u - σ y) * w / (σ + w); if w==0: 0  (domain of f*).
    """
    sigma = jnp.asarray(sigma, dtype=u.dtype)
    denom = sigma + w
    v = (u - sigma * y_meas) * w / jnp.maximum(denom, 1e-12)
    return jnp.where(w > 0, v, 0.0).astype(u.dtype)


# --------- main algorithm ----------
def spdhg_tv(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    weights: jnp.ndarray | None = None,  # same shape as projections; 0 for unmeasured
    init_x: jnp.ndarray | None = None,
    config: SPDHGConfig = SPDHGConfig(),
    callback: LossCallback | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, dict]:
    """
    SPDHG (stochastic Chambolle-Pock) with weighted L2 data term and TV-like regularization.

    If ``callback`` is provided, it fires on the first logged objective sample and
    on the final logged objective sample. The callback arguments are ``(step,
    loss)``, where ``step`` is the zero-based iteration index that produced
    ``loss``. Only iterations whose objective was recorded under ``config.log_every``
    are eligible for callbacks; if a single logged sample exists, the callback
    fires once.
    """
    regulariser = validate_regulariser(
        config.regulariser,
        config.huber_delta,
        context="spdhg_tv config",
    )
    huber_delta = float(config.huber_delta)

    validate_grid(grid, "spdhg_tv grid")
    n_views, nv, nu = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="spdhg_tv projections",
    )
    expected_proj_shape = (n_views, nv, nu)
    validate_optional_same_shape(
        weights,
        expected_proj_shape,
        context="spdhg_tv weights",
        name="weights",
        fix="use weights with the same shape as projections.",
    )
    validate_optional_broadcastable_shape(
        config.support,
        (grid.nx, grid.ny, grid.nz),
        context="spdhg_tv support",
        name="support",
        fix="use a support mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if init_x is not None:
        validate_volume(init_x, grid, context="spdhg_tv init_x", name="init_x")
    y_meas = jnp.asarray(projections, dtype=jnp.float32)
    W = jnp.ones_like(y_meas) if weights is None else jnp.asarray(weights, dtype=jnp.float32)

    # precompute per-view poses once (like your FISTA)
    T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="spdhg_tv geometry")
    det_grid = get_detector_grid_device(detector) if det_grid is None else det_grid

    # batched projector over a chunk of views
    def project_chunk(T_chunk, vol):
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T,
                grid,
                detector,
                v,
                use_checkpoint=config.checkpoint_projector,
                unroll=int(config.projector_unroll),
                gather_dtype=config.gather_dtype,
                det_grid=det_grid,
            ),
            in_axes=(0, None),
        )
        return vm_project(T_chunk, vol)

    # Step sizes and optional operator norm estimate
    L_grad = float(np.sqrt(12.0))  # 3D forward/backward diffs (safe upper bound)
    L_A: float | None = None
    b = int(max(1, min(config.views_per_batch, n_views)))
    if config.tau is None or config.sigma_data is None or config.sigma_tv is None:
        L_A2 = _estimate_norm_A2(
            geometry,
            grid,
            detector,
            y_meas.shape,
            T_all,
            views_per_batch=max(1, config.views_per_batch),
            projector_unroll=config.projector_unroll,
            checkpoint_projector=config.checkpoint_projector,
            gather_dtype=config.gather_dtype,
            key=jax.random.key(config.seed),
            power_iters=20,
            safety=1.05,
            det_grid=det_grid,
        )
        L_A = float(np.sqrt(L_A2))
        rho = 0.99
        tau = rho / (L_A + L_grad)
        sigma_data_base = rho / max(L_A, 1e-6)
        sigma_tv = rho / L_grad
    else:
        tau = float(config.tau)
        sigma_data_base = float(config.sigma_data)
        sigma_tv = float(config.sigma_tv)

    # stochastic block schedule = random permutation of contiguous blocks per epoch
    m = (n_views + b - 1) // b
    p_prob = 1.0 / float(max(m, 1))
    sigma_data_eff = sigma_data_base
    rng = np.random.default_rng(config.seed)
    epochs = (config.iters + m - 1) // m
    block_ids = []
    for e in range(epochs):
        perm = rng.permutation(m)
        block_ids.extend(list(perm))
    block_ids = jnp.asarray(block_ids[: config.iters], dtype=jnp.int32)

    # init variables
    x0 = (
        jnp.asarray(init_x, dtype=jnp.float32)
        if init_x is not None
        else jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    )
    x = x0
    # Ensure distinct buffers to avoid double-donation aliasing under JIT
    x_bar = jnp.array(x0, copy=True)
    y_data = jnp.zeros_like(y_meas)
    p1 = jnp.zeros_like(x)
    p2 = jnp.zeros_like(x)
    p3 = jnp.zeros_like(x)
    # accumulator s = A^T y_data - div(p)  (K^T with K=∇ is -div)
    s = jnp.zeros_like(x)

    support = None if config.support is None else jnp.asarray(config.support, dtype=jnp.float32)
    lam = jnp.asarray(config.lambda_tv, dtype=jnp.float32)

    # allocate logging
    losses = jnp.zeros((config.iters,), dtype=jnp.float32)

    def one_step(carry, t):
        (x, x_bar, y_data, p1, p2, p3, s, losses) = carry
        block = block_ids[t]
        start = block * jnp.int32(b)
        remaining = jnp.maximum(0, jnp.int32(n_views) - start)
        valid = jnp.minimum(jnp.int32(b), remaining)
        shift = jnp.int32(b) - valid
        start_shifted = jnp.maximum(0, start - shift)

        # slice a fixed-size window; mask out padded rows
        T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
        y_chunk = jax.lax.dynamic_slice(y_meas, (start_shifted, 0, 0), (b, nv, nu))
        w_chunk = jax.lax.dynamic_slice(W, (start_shifted, 0, 0), (b, nv, nu))
        y_dual_old = jax.lax.dynamic_slice(y_data, (start_shifted, 0, 0), (b, nv, nu))

        idx = jnp.arange(b)
        row_mask = (idx >= (jnp.int32(b) - valid))[:, None, None]
        row_mask = row_mask.astype(jnp.float32)

        # DATA DUAL UPDATE
        sigma_eff = jnp.asarray(sigma_data_eff, dtype=x_bar.dtype)

        pred = project_chunk(T_chunk, x_bar)
        u = y_dual_old + sigma_eff * pred
        y_dual_new = _prox_fstar_l2(u, sigma_eff, y_chunk, w_chunk)
        # keep padded rows unchanged
        y_dual_new = row_mask * y_dual_new + (1.0 - row_mask) * y_dual_old

        # delta dual and its contribution to s via A^T (only changed rows)
        delta_y = (y_dual_new - y_dual_old) * row_mask

        g_block = sum_backproject_views_T(
            T_chunk,
            grid,
            detector,
            delta_y,
            unroll=int(config.projector_unroll),
            gather_dtype=config.gather_dtype,
            det_grid=det_grid,
        )

        # TV DUAL UPDATE (global)
        gx, gy, gz = grad3(x_bar)
        p1_u = p1 + sigma_tv * gx
        p2_u = p2 + sigma_tv * gy
        p3_u = p3 + sigma_tv * gz
        if regulariser == "huber_tv":
            p1_new, p2_new, p3_new = prox_huber_tv_conj(
                p1_u,
                p2_u,
                p3_u,
                sigma=sigma_tv,
                lam=lam,
                delta=huber_delta,
            )
        else:
            norm = jnp.maximum(
                1.0,
                jnp.sqrt(p1_u * p1_u + p2_u * p2_u + p3_u * p3_u)
                / jnp.maximum(lam, 1e-12),
            )
            p1_new = p1_u / norm
            p2_new = p2_u / norm
            p3_new = p3_u / norm
        # TV contributes with -div(p) in the primal gradient; track the increment
        delta_div = div3(p1_new - p1, p2_new - p2, p3_new - p3)

        # update accumulator s
        # Update accumulator: A^T y + (-div p)
        s_new = s + g_block - delta_div

        # PRIMAL UPDATE
        x_minus = x - tau * s_new
        x_new = _proj_pos_support(x_minus, config.positivity, support)

        # EXTRAGRAD
        x_bar_candidate = x_new + jnp.asarray(config.theta, x_new.dtype) * (x_new - x)
        x_bar_new = _proj_pos_support(x_bar_candidate, config.positivity, support)

        # write back data dual window
        y_data_new = jax.lax.dynamic_update_slice(y_data, y_dual_new, (start_shifted, 0, 0))

        # minibatch objective estimate for logging (optional)
        do_log = (config.log_every > 0) & ((t + 1) % config.log_every == 0)

        def _log_step():
            resid = (pred - y_chunk) * jnp.sqrt(w_chunk) * row_mask
            data_est = (
                0.5
                * jnp.vdot(resid, resid).real
                * (float(n_views) / jnp.maximum(valid.astype(jnp.float32), 1.0))
            )
            if regulariser == "huber_tv":
                reg_value = huber_tv_value(x_new, huber_delta)
            else:
                reg_value = isotropic_tv_value(x_new)
            obj = (data_est + lam * reg_value).astype(jnp.float32)
            return losses.at[t].set(obj)

        def _no_log():
            return losses

        losses_new = jax.lax.cond(do_log, _log_step, _no_log)

        return (x_new, x_bar_new, y_data_new, p1_new, p2_new, p3_new, s_new, losses_new), None

    scan_init = (x, x_bar, y_data, p1, p2, p3, s, losses)
    (x_f, xbar_f, ydata_f, p1_f, p2_f, p3_f, s_f, losses_f), _ = jax.jit(
        lambda carry: jax.lax.scan(one_step, carry, jnp.arange(config.iters)), donate_argnums=(0,)
    )(scan_init)

    logged_steps: list[int] = []
    if config.log_every > 0:
        logged_steps = [
            int(step)
            for step in np.flatnonzero((np.arange(config.iters) + 1) % config.log_every == 0)
        ]
    if logged_steps:
        losses_host = np.asarray(losses_f)
        emit_loss_callback_endpoints(
            callback,
            (
                (logged_steps[0], float(losses_host[logged_steps[0]])),
                (logged_steps[-1], float(losses_host[logged_steps[-1]])),
            ),
        )

    info = {
        "loss": [float(v) for v in list(losses_f)],
        "tau": float(tau),
        "sigma_data": float(sigma_data_eff),
        "sigma_data_base": float(sigma_data_base),
        "sigma_tv": float(sigma_tv),
        "lambda_tv": float(config.lambda_tv),
        "views_per_batch": int(b),
        "num_blocks": int(m),
        "A_norm": (float(L_A) if L_A is not None else None),
        "grad_norm": float(L_grad),
        "selection_prob": float(p_prob),
        "regulariser": regulariser,
        "huber_delta": float(config.huber_delta),
    }
    return x_f, info
