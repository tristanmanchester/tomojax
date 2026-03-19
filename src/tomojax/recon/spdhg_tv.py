from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from ..core.geometry import Geometry, Grid, Detector
from ..core.projector import forward_project_view_T, get_detector_grid_device


# --------- finite differences (match your FISTA helpers) ----------
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


# --------- config ----------
@dataclass
class SPDHGConfig:
    iters: int = 400
    lambda_tv: float = 5e-3
    theta: float = 1.0           # extrapolation for xbar
    views_per_batch: int = 16    # size of a stochastic block
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
) -> float:
    """Reuse your power method to estimate ||A||^2 (i.e., Lipschitz of ∇(1/2||Ax||^2))."""
    # Minimal reimplementation: apply A^T A via VJP in streamed fashion over all views
    n_views, nv, nu = projections_shape
    det_grid = get_detector_grid_device(detector)

    def A_apply(vol, T_chunk):
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T, grid, detector, v,
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

            # backproject via VJP
            _, vjp = jax.vjp(lambda vv: pred_fun(vv).ravel(), v)
            g_chunk = vjp(proj.ravel())[0]
            return g_acc + g_chunk, None

        g0 = jnp.zeros_like(v)
        g_final, _ = jax.lax.scan(body, g0, jnp.arange(m))
        return g_final

    if key is None:
        key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    v = v / (jnp.linalg.norm(v) + 1e-12)
    for _ in range(max(1, power_iters)):
        w = AtranA(v)
        v = w / (jnp.linalg.norm(w) + 1e-12)
    Aw = AtranA(v)
    L = float(jnp.vdot(v, Aw).real) * float(safety ** 2)  # ~||A||^2 with margin
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
    weights: jnp.ndarray | None = None,     # same shape as projections; 0 for unmeasured
    init_x: jnp.ndarray | None = None,
    config: SPDHGConfig = SPDHGConfig(),
    callback: Callable[[int, float], None] | None = None,
) -> tuple[jnp.ndarray, dict]:
    """
    SPDHG (stochastic Chambolle–Pock) with weighted L2 data term and isotropic TV.
    Returns (reconstruction, info).
    """
    y_meas = jnp.asarray(projections, dtype=jnp.float32)
    n_views, nv, nu = int(y_meas.shape[0]), int(y_meas.shape[1]), int(y_meas.shape[2])
    W = jnp.ones_like(y_meas) if weights is None else jnp.asarray(weights, dtype=jnp.float32)

    # precompute per-view poses once (like your FISTA)
    T_all = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )
    det_grid = get_detector_grid_device(detector)

    # batched projector over a chunk of views
    def project_chunk(T_chunk, vol):
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T, grid, detector, v,
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
            geometry, grid, detector, y_meas.shape, T_all,
            views_per_batch=max(1, config.views_per_batch),
            projector_unroll=config.projector_unroll,
            checkpoint_projector=config.checkpoint_projector,
            gather_dtype=config.gather_dtype,
            key=jax.random.PRNGKey(config.seed),
            power_iters=20,
            safety=1.05,
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
    x0 = jnp.asarray(init_x, dtype=jnp.float32) if init_x is not None else jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    x = x0
    # Ensure distinct buffers to avoid double-donation aliasing under JIT
    x_bar = jnp.array(x0, copy=True)
    y_data = jnp.zeros_like(y_meas)
    p1 = jnp.zeros_like(x); p2 = jnp.zeros_like(x); p3 = jnp.zeros_like(x)
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

        # backproject this block's delta via a single batched VJP (faster, higher peak memory)
        def pred_fun(vol):
            return project_chunk(T_chunk, vol)
        _, vjp = jax.vjp(lambda vv: pred_fun(vv).ravel(), x_bar)
        g_block = vjp(delta_y.ravel())[0]

        # TV DUAL UPDATE (global)
        gx, gy, gz = _grad3(x_bar)
        p1_u = p1 + sigma_tv * gx
        p2_u = p2 + sigma_tv * gy
        p3_u = p3 + sigma_tv * gz
        norm = jnp.maximum(1.0, jnp.sqrt(p1_u*p1_u + p2_u*p2_u + p3_u*p3_u) / jnp.maximum(lam, 1e-12))
        p1_new = p1_u / norm; p2_new = p2_u / norm; p3_new = p3_u / norm
        # TV contributes with -div(p) in the primal gradient; track the increment
        delta_div = _div3(p1_new - p1, p2_new - p2, p3_new - p3)

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
            data_est = 0.5 * jnp.vdot(resid, resid).real * (float(n_views) / jnp.maximum(valid.astype(jnp.float32), 1.0))
            gx2, gy2, gz2 = _grad3(x_new)
            tv = jnp.sum(jnp.sqrt(gx2*gx2 + gy2*gy2 + gz2*gz2 + 1e-8))
            obj = (data_est + lam * tv).astype(jnp.float32)
            return losses.at[t].set(obj)
        def _no_log():
            return losses
        losses_new = jax.lax.cond(do_log, _log_step, _no_log)

        return (x_new, x_bar_new, y_data_new, p1_new, p2_new, p3_new, s_new, losses_new), None

    scan_init = (x, x_bar, y_data, p1, p2, p3, s, losses)
    (x_f, xbar_f, ydata_f, p1_f, p2_f, p3_f, s_f, losses_f), _ = jax.jit(
        lambda carry: jax.lax.scan(one_step, carry, jnp.arange(config.iters)),
        donate_argnums=(0,)
    )(scan_init)

    # optional callback with the last logged loss
    if callback is not None:
        last_logged = int(np.where(np.array((np.arange(config.iters)+1) % config.log_every == 0))[0][-1]) if config.log_every>0 and config.iters>=config.log_every else config.iters-1
        try:
            callback(last_logged, float(losses_f[last_logged]))
        except Exception:
            pass

    info = {
        "loss": [float(v) for v in list(losses_f)],
        "tau": float(tau),
        "sigma_data": float(sigma_data_eff),
        "sigma_data_base": float(sigma_data_base),
        "sigma_tv": float(sigma_tv),
        "lambda_tv": float(config.lambda_tv),
        "views_per_batch": int(b), "num_blocks": int(m),
        "A_norm": (float(L_A) if L_A is not None else None),
        "grad_norm": float(L_grad),
        "selection_prob": float(p_prob),
    }
    return x_f, info
