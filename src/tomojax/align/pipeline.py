from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import math
import time
from typing import Callable, Iterable, Literal, TypedDict

import jax
import jax.numpy as jnp

from ..core.geometry.base import Geometry, Grid, Detector
from ..core.geometry.views import stack_view_poses
from ..core.projector import forward_project_view_T, get_detector_grid_device
from ..recon.fista_tv import FistaConfig, fista_tv
from ..utils.logging import progress_iter, format_duration
from .parametrizations import se3_from_5d
from .losses import L2OtsuLossSpec, AlignmentLossSpec, build_loss_adapter, loss_is_within_relative_tolerance
from ..utils.fov import cylindrical_mask_xy


ObserverAction = Literal["continue", "advance_level", "stop_run"]
type OuterStatValue = float | int | bool | str | None
type OuterStat = dict[str, OuterStatValue]
ObserverCallback = Callable[[jnp.ndarray, jnp.ndarray, OuterStat], ObserverAction | bool]


class AlignInfo(TypedDict):
    loss: list[float]
    L: float | None
    outer_stats: list[OuterStat]
    stopped_by_observer: bool
    observer_action: ObserverAction
    wall_time_total: float


class AlignMultiresInfo(TypedDict):
    loss: list[float]
    factors: list[int]
    stopped_by_observer: bool
    observer_action: ObserverAction
    total_outer_iters: int
    wall_time_total: float


class MultiresLevel(TypedDict):
    factor: int
    grid: Grid
    detector: Detector
    projections: jnp.ndarray


def _normalize_observer_action(
    action: ObserverAction | str | bool | None,
) -> ObserverAction:
    if action is False or action is None:
        return "continue"
    if action is True:
        return "stop_run"
    if isinstance(action, str):
        lowered = action.strip().lower()
        if lowered in {"continue", "advance_level", "stop_run"}:
            return lowered  # type: ignore[return-value]
    raise ValueError(f"Unsupported observer action: {action!r}")


def _should_prefer_gn_candidate(
    loss_before: float,
    current_loss: float,
    candidate_loss: float,
    rel_tol: float,
) -> bool:
    """Accept tolerated GN candidates only when they improve the current best step."""
    candidate_ok = candidate_loss < loss_before or loss_is_within_relative_tolerance(
        loss_before, candidate_loss, rel_tol
    )
    return candidate_ok and candidate_loss < current_loss


def _second_difference_gram(n: int) -> jnp.ndarray:
    if n < 3:
        return jnp.zeros((n, n), dtype=jnp.float32)
    d2 = jnp.zeros((n - 2, n), dtype=jnp.float32)
    rows = jnp.arange(n - 2, dtype=jnp.int32)
    d2 = d2.at[rows, rows].set(1.0)
    d2 = d2.at[rows, rows + 1].set(-2.0)
    d2 = d2.at[rows, rows + 2].set(1.0)
    return d2.T @ d2


def _smooth_gn_candidate(
    params5: jnp.ndarray,
    smoothness_gram: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Project a per-view GN candidate through the quadratic curvature prior."""
    n_views = int(params5.shape[0])
    if n_views < 3:
        return params5

    eye = jnp.eye(n_views, dtype=jnp.float32)

    def solve_one_dim(rhs: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.cond(
            weight > 0.0,
            lambda _: jnp.linalg.solve(eye + 2.0 * weight * smoothness_gram, rhs),
            lambda _: rhs,
            operand=None,
        )

    return jax.vmap(solve_one_dim, in_axes=(1, 0), out_axes=1)(params5, weights)


def _select_gn_candidate(
    params5_prev: jnp.ndarray,
    dp_all: jnp.ndarray,
    *,
    loss_before: float,
    eval_loss: Callable[[jnp.ndarray], float],
    gn_accept_tol: float,
    smooth_candidate: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    light_smoothness_weights_sq: jnp.ndarray | None = None,
    medium_smoothness_weights_sq: jnp.ndarray | None = None,
    smoothness_weights_sq: jnp.ndarray | None = None,
    trans_only_smoothness_weights_sq: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, float]:
    """Pick a GN candidate using a small hierarchical full-loss search."""

    def _accepts(candidate_loss: float, current_best_loss: float = float("inf")) -> bool:
        return _should_prefer_gn_candidate(
            loss_before,
            current_best_loss,
            candidate_loss,
            gn_accept_tol,
        )

    raw_params = params5_prev + dp_all
    raw_loss = eval_loss(raw_params)
    if _accepts(raw_loss):
        return raw_params, raw_loss

    half_params = params5_prev + jnp.float32(0.5) * dp_all
    half_loss = eval_loss(half_params)
    if _accepts(half_loss):
        return half_params, half_loss

    base_params = raw_params if raw_loss <= half_loss else half_params

    def _has_active_weights(weights: jnp.ndarray | None) -> bool:
        return weights is not None and bool(jnp.any(weights > 0.0))

    if smooth_candidate is None:
        return params5_prev, loss_before

    smooth_weights = []
    for weights in (
        light_smoothness_weights_sq,
        medium_smoothness_weights_sq,
        smoothness_weights_sq,
        trans_only_smoothness_weights_sq,
    ):
        if _has_active_weights(weights):
            smooth_weights.append(weights)

    if not smooth_weights:
        return params5_prev, loss_before

    best_params = params5_prev
    best_loss = float("inf")
    accepted = False
    for weights in smooth_weights:
        candidate_params = smooth_candidate(base_params, weights)
        candidate_loss = eval_loss(candidate_params)
        if _accepts(candidate_loss, best_loss):
            best_params = candidate_params
            best_loss = candidate_loss
            accepted = True

    if accepted:
        return best_params, best_loss
    return params5_prev, loss_before


_EXPECTED_ALIGN_EVAL_FAILURE_SNIPPETS = (
    "allocator",
    "cholesky",
    "failed to converge",
    "inf",
    "nan",
    "non-finite",
    "not positive definite",
    "out of memory",
    "resource_exhausted",
    "singular",
    "svd",
)


def _is_expected_align_eval_failure(exc: Exception) -> bool:
    if isinstance(exc, FloatingPointError):
        return True
    msg = str(exc).lower()
    return any(snippet in msg for snippet in _EXPECTED_ALIGN_EVAL_FAILURE_SNIPPETS)


def _evaluate_align_loss(
    eval_loss: Callable[[], float | jnp.ndarray],
    *,
    fallback: float | None,
    context: str,
) -> float | None:
    try:
        return float(eval_loss())
    except Exception as exc:
        if _is_expected_align_eval_failure(exc):
            logging.warning("%s after expected numeric failure: %s", context, exc)
            return fallback
        raise


@dataclass
class AlignConfig:
    outer_iters: int = 5
    recon_iters: int = 10
    lambda_tv: float = 0.005
    tv_prox_iters: int = 10
    # Reconstruction stopping criteria
    recon_rel_tol: float | None = None
    recon_patience: int = 2
    # Alignment step sizes
    lr_rot: float = 1e-3  # radians
    lr_trans: float = 1e-1  # world units
    # Memory/throughput knobs (hidden defaults)
    views_per_batch: int = 1  # stream one view at a time
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    # Solver and regularization
    opt_method: str = "gn"
    gn_damping: float = 1e-6
    w_rot: float = 0.0
    w_trans: float = 0.0
    seed_translations: bool = False
    # Volume masking before forward projection (modeling for ROI/truncation)
    # Options: "off" (default), "cyl" (cylindrical mask in x–y broadcast along z)
    mask_vol: str = "off"
    # Logging
    log_summary: bool = False
    log_compact: bool = True  # print one compact line per outer when log_summary is enabled
    # Reconstruction Lipschitz (optional override to skip power-method)
    recon_L: float | None = None
    # Early stopping across outers (alignment phase)
    early_stop: bool = True
    early_stop_rel_impr: float = 1e-3  # stop if (before-after)/before < this
    early_stop_patience: int = 2
    # Accept GN steps only when they improve the loss, up to gn_accept_tol.
    gn_accept_only_improving: bool = True
    gn_accept_tol: float = 0.0  # allow tiny increases if >0 (as fraction of before)
    # Data term / similarity
    loss: AlignmentLossSpec = field(default_factory=L2OtsuLossSpec)


def align(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,  # (n_views, nv, nu)
    *,
    cfg: AlignConfig | None = None,
    init_x: jnp.ndarray | None = None,
    init_params5: jnp.ndarray | None = None,
    observer: ObserverCallback | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignInfo]:
    """Alternating reconstruction + per-view alignment (5-DOF) on small cases.

    Returns (x, params5, info) with loss history and optional metrics.
    """
    if cfg is None:
        cfg = AlignConfig()
    n_views = int(projections.shape[0])
    # Initialize volume and params
    x = (
        jnp.asarray(init_x, dtype=jnp.float32)
        if init_x is not None
        else jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    )
    params5 = (
        jnp.asarray(init_params5, dtype=jnp.float32)
        if init_params5 is not None
        else jnp.zeros((n_views, 5), dtype=jnp.float32)
    )

    loss_hist = []
    stopped_by_observer = False
    observer_action: ObserverAction = "continue"

    # Precompute nominal poses once
    n_views = int(projections.shape[0])
    T_nom_all = stack_view_poses(geometry, n_views)

    # Precompute detector grid once (device arrays) to avoid repeated transfers/logging
    det_grid = get_detector_grid_device(detector)

    # Vmapped projector across views (pose-aware). Closure captures unroll as a static constant.
    def _project_batch(T_batch, vol):
        f = lambda T: forward_project_view_T(
            T,
            grid,
            detector,
            vol,
            use_checkpoint=cfg.checkpoint_projector,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
            det_grid=det_grid,
        )
        return jax.vmap(f, in_axes=0)(T_batch)

    # Static smoothness weights to avoid rebuilding inside jitted loss
    W_weights = jnp.array(
        [cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans], dtype=jnp.float32
    )
    smoothness_gram = _second_difference_gram(n_views)
    smoothness_weights_sq = W_weights * W_weights
    medium_smoothness_weights_sq = smoothness_weights_sq * jnp.float32(0.4)
    trans_only_smoothness_weights_sq = smoothness_weights_sq.at[:3].set(0.0)
    light_smoothness_weights_sq = smoothness_weights_sq * jnp.float32(0.25)

    # Optional static volume mask before projection
    vol_mask = None
    try:
        if str(getattr(cfg, "mask_vol", "off")).lower() in ("cyl", "cylindrical"):
            m_xy = cylindrical_mask_xy(grid, detector)
            vol_mask = jnp.asarray(m_xy, dtype=jnp.float32)[:, :, None]
    except Exception:
        vol_mask = None

    # Build per-view loss once (may precompute masks on targets)
    loss_adapter = build_loss_adapter(cfg.loss, projections)
    per_view_loss_fn = loss_adapter.per_view_loss
    loss_state = loss_adapter.state

    nv = int(projections.shape[1])
    nu = int(projections.shape[2])
    chunk_size = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n_views
    chunk_size = min(chunk_size, n_views)
    num_chunks = (n_views + chunk_size - 1) // chunk_size
    loss_mask = getattr(loss_state, "mask", None)
    has_loss_mask = loss_mask is not None
    ls_like = loss_adapter.supports_gauss_newton

    def _chunk_schedule(i: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        i = jnp.asarray(i, dtype=jnp.int32)
        start = i * jnp.int32(chunk_size)
        remaining = jnp.maximum(0, jnp.int32(n_views) - start)
        valid = jnp.minimum(jnp.int32(chunk_size), remaining)
        shift = jnp.int32(chunk_size) - valid
        start_shifted = jnp.maximum(0, start - shift)
        idx = jnp.arange(chunk_size, dtype=jnp.int32)
        vmask = (idx >= (jnp.int32(chunk_size) - valid)).astype(jnp.float32)
        return start_shifted, vmask, start_shifted + idx

    def _apply_vol_mask(vol: jnp.ndarray) -> jnp.ndarray:
        return vol * vol_mask if vol_mask is not None else vol

    if has_loss_mask:

        def _loss_chunk_values(
            pred: jnp.ndarray,
            y_chunk: jnp.ndarray,
            start_shifted: jnp.ndarray,
            view_idx_chunk: jnp.ndarray,
        ) -> jnp.ndarray:
            mask_chunk = jax.lax.dynamic_slice(
                loss_mask, (start_shifted, 0, 0), (chunk_size, nv, nu)
            )
            return per_view_loss_fn(
                pred,
                y_chunk,
                mask_chunk,
                view_indices=view_idx_chunk,
            )
    else:

        def _loss_chunk_values(
            pred: jnp.ndarray,
            y_chunk: jnp.ndarray,
            start_shifted: jnp.ndarray,
            view_idx_chunk: jnp.ndarray,
        ) -> jnp.ndarray:
            del start_shifted
            return per_view_loss_fn(
                pred,
                y_chunk,
                None,
                view_indices=view_idx_chunk,
            )

    def align_loss(params5, vol):
        # Compose augmented poses
        # Current convention: per-view misalignment parameters act in the object
        # frame and are post-multiplied: T_world_from_obj_aug = T_nom @ T_delta.
        # This is consistent across parallel CT and laminography sample-frame.
        T_aug = T_nom_all @ jax.vmap(se3_from_5d)(params5)  # (n_views, 4, 4)
        masked_vol = _apply_vol_mask(vol)

        def body(loss_acc, i):
            start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
            T_chunk = jax.lax.dynamic_slice(T_aug, (start_shifted, 0, 0), (chunk_size, 4, 4))
            y_chunk = jax.lax.dynamic_slice(
                projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
            )
            pred = _project_batch(T_chunk, masked_vol)
            lvec = _loss_chunk_values(pred, y_chunk, start_shifted, view_idx_chunk)
            loss_batch = jnp.sum(lvec * vmask)
            return (loss_acc + loss_batch, None)

        loss0 = jnp.float32(0.0)
        loss_tot, _ = jax.lax.scan(body, loss0, jnp.arange(num_chunks, dtype=jnp.int32))

        # Smoothness prior across views (2nd difference)
        loss = loss_tot
        if int(params5.shape[0]) >= 3:
            d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
            loss = loss + jnp.sum((d2 * W_weights) ** 2)
        return loss

    # Value function for whole batch (forward only) kept for logging and line search
    align_loss_jit = jax.jit(align_loss)

    # Memory-safe gradient over fixed-size chunks.
    if has_loss_mask:

        def _one_view_loss_masked(p5_i, T_nom_i, y_i, masked_vol, mask_i, view_idx):
            T_i = T_nom_i @ se3_from_5d(p5_i)
            pred_i = forward_project_view_T(
                T_i,
                grid,
                detector,
                masked_vol,
                use_checkpoint=cfg.checkpoint_projector,
                unroll=int(cfg.projector_unroll),
                gather_dtype=cfg.gather_dtype,
                det_grid=det_grid,
            )
            view_indices = jnp.expand_dims(jnp.asarray(view_idx, dtype=jnp.int32), axis=0)
            lvec = per_view_loss_fn(
                pred_i[None, ...],
                y_i[None, ...],
                mask_i[None, ...],
                view_indices=view_indices,
            )
            return lvec[0]

        one_view_val_and_grad_batch = jax.jit(
            jax.vmap(
                jax.value_and_grad(_one_view_loss_masked),
                in_axes=(0, 0, 0, None, 0, 0),
            )
        )

        def loss_and_grad_manual(params5, vol):
            masked_vol = _apply_vol_mask(vol)

            def body(carry, i):
                total, g = carry
                start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
                params_chunk = jax.lax.dynamic_slice(
                    params5, (start_shifted, 0), (chunk_size, params5.shape[1])
                )
                T_nom_chunk = jax.lax.dynamic_slice(
                    T_nom_all, (start_shifted, 0, 0), (chunk_size, 4, 4)
                )
                y_chunk = jax.lax.dynamic_slice(
                    projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                mask_chunk = jax.lax.dynamic_slice(
                    loss_mask, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                lvec, g_chunk = one_view_val_and_grad_batch(
                    params_chunk,
                    T_nom_chunk,
                    y_chunk,
                    masked_vol,
                    mask_chunk,
                    view_idx_chunk,
                )
                total = total + jnp.sum(lvec * vmask)
                g = g.at[view_idx_chunk].add(g_chunk * vmask[:, None])
                return (total, g), None

            init = (jnp.float32(0.0), jnp.zeros_like(params5))
            (total, g), _ = jax.lax.scan(body, init, jnp.arange(num_chunks, dtype=jnp.int32))
            if int(params5.shape[0]) >= 3:
                d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
                w = jnp.array(
                    [cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans],
                    jnp.float32,
                )
                total = total + jnp.sum((d2 * w) ** 2)
                ww = (w**2) * 2.0
                g = g.at[1:-1].add(-2.0 * d2 * ww)
                g = g.at[0:-2].add(1.0 * d2 * ww)
                g = g.at[2:].add(1.0 * d2 * ww)
            return total, g
    else:

        def _one_view_loss_unmasked(p5_i, T_nom_i, y_i, masked_vol, view_idx):
            T_i = T_nom_i @ se3_from_5d(p5_i)
            pred_i = forward_project_view_T(
                T_i,
                grid,
                detector,
                masked_vol,
                use_checkpoint=cfg.checkpoint_projector,
                unroll=int(cfg.projector_unroll),
                gather_dtype=cfg.gather_dtype,
                det_grid=det_grid,
            )
            view_indices = jnp.expand_dims(jnp.asarray(view_idx, dtype=jnp.int32), axis=0)
            lvec = per_view_loss_fn(
                pred_i[None, ...],
                y_i[None, ...],
                None,
                view_indices=view_indices,
            )
            return lvec[0]

        one_view_val_and_grad_batch = jax.jit(
            jax.vmap(
                jax.value_and_grad(_one_view_loss_unmasked),
                in_axes=(0, 0, 0, None, 0),
            )
        )

        def loss_and_grad_manual(params5, vol):
            masked_vol = _apply_vol_mask(vol)

            def body(carry, i):
                total, g = carry
                start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
                params_chunk = jax.lax.dynamic_slice(
                    params5, (start_shifted, 0), (chunk_size, params5.shape[1])
                )
                T_nom_chunk = jax.lax.dynamic_slice(
                    T_nom_all, (start_shifted, 0, 0), (chunk_size, 4, 4)
                )
                y_chunk = jax.lax.dynamic_slice(
                    projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                lvec, g_chunk = one_view_val_and_grad_batch(
                    params_chunk,
                    T_nom_chunk,
                    y_chunk,
                    masked_vol,
                    view_idx_chunk,
                )
                total = total + jnp.sum(lvec * vmask)
                g = g.at[view_idx_chunk].add(g_chunk * vmask[:, None])
                return (total, g), None

            init = (jnp.float32(0.0), jnp.zeros_like(params5))
            (total, g), _ = jax.lax.scan(body, init, jnp.arange(num_chunks, dtype=jnp.int32))
            if int(params5.shape[0]) >= 3:
                d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
                w = jnp.array(
                    [cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans],
                    jnp.float32,
                )
                total = total + jnp.sum((d2 * w) ** 2)
                ww = (w**2) * 2.0
                g = g.at[1:-1].add(-2.0 * d2 * ww)
                g = g.at[0:-2].add(1.0 * d2 * ww)
                g = g.at[2:].add(1.0 * d2 * ww)
            return total, g

    loss_and_grad_manual = jax.jit(loss_and_grad_manual)

    # Gauss–Newton (Levenberg–Marquardt) single-view update
    def _pred_flat(T_i, masked_vol):
        return forward_project_view_T(
            T_i,
            grid,
            detector,
            masked_vol,
            use_checkpoint=cfg.checkpoint_projector,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
            det_grid=det_grid,
        ).ravel()

    def _gn_update_one(p5_i, T_nom_i, y_i, vol, w_i):
        def f(p5):
            T_i = T_nom_i @ se3_from_5d(p5)
            r = _pred_flat(T_i, vol) - y_i.ravel()
            return w_i.ravel() * r

        # J^T r
        r = f(p5_i)
        _, vjp = jax.vjp(f, p5_i)
        g = vjp(r)[0]
        # J^T J via 5 JVPs
        eye5 = jnp.eye(5, dtype=jnp.float32)

        def jvp_col(v):
            return jax.jvp(f, (p5_i,), (v,))[1]

        cols = jax.vmap(jvp_col)(eye5)
        H = cols @ cols.T
        lam = jnp.float32(cfg.gn_damping)
        dp = jnp.linalg.solve(H + lam * jnp.eye(5, dtype=H.dtype), -g)
        return dp

    _gn_update_batch = jax.jit(jax.vmap(_gn_update_one, in_axes=(0, 0, 0, None, 0)))

    if has_loss_mask:

        def _ls_weight_chunk(y_chunk: jnp.ndarray, mask_chunk: jnp.ndarray) -> jnp.ndarray:
            return loss_adapter.gauss_newton_weights(y_chunk, mask_chunk)

        def _gn_update_all(params5, vol):
            masked_vol = _apply_vol_mask(vol)

            def body(dp_acc, i):
                start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
                params_chunk = jax.lax.dynamic_slice(
                    params5, (start_shifted, 0), (chunk_size, params5.shape[1])
                )
                T_chunk = jax.lax.dynamic_slice(
                    T_nom_all, (start_shifted, 0, 0), (chunk_size, 4, 4)
                )
                y_chunk = jax.lax.dynamic_slice(
                    projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                mask_chunk = jax.lax.dynamic_slice(
                    loss_mask, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                w_chunk = _ls_weight_chunk(y_chunk, mask_chunk)
                dp_chunk = _gn_update_batch(
                    params_chunk,
                    T_chunk,
                    y_chunk,
                    masked_vol,
                    w_chunk,
                )
                dp_acc = dp_acc.at[view_idx_chunk].add(dp_chunk * vmask[:, None])
                return dp_acc, None

            dp0 = jnp.zeros_like(params5)
            dp_all, _ = jax.lax.scan(body, dp0, jnp.arange(num_chunks, dtype=jnp.int32))
            return dp_all
    else:

        def _ls_weight_chunk(y_chunk: jnp.ndarray) -> jnp.ndarray:
            return loss_adapter.gauss_newton_weights(y_chunk, None)

        def _gn_update_all(params5, vol):
            masked_vol = _apply_vol_mask(vol)

            def body(dp_acc, i):
                start_shifted, vmask, view_idx_chunk = _chunk_schedule(i)
                params_chunk = jax.lax.dynamic_slice(
                    params5, (start_shifted, 0), (chunk_size, params5.shape[1])
                )
                T_chunk = jax.lax.dynamic_slice(
                    T_nom_all, (start_shifted, 0, 0), (chunk_size, 4, 4)
                )
                y_chunk = jax.lax.dynamic_slice(
                    projections, (start_shifted, 0, 0), (chunk_size, nv, nu)
                )
                w_chunk = _ls_weight_chunk(y_chunk)
                dp_chunk = _gn_update_batch(
                    params_chunk,
                    T_chunk,
                    y_chunk,
                    masked_vol,
                    w_chunk,
                )
                dp_acc = dp_acc.at[view_idx_chunk].add(dp_chunk * vmask[:, None])
                return dp_acc, None

            dp0 = jnp.zeros_like(params5)
            dp_all, _ = jax.lax.scan(body, dp0, jnp.arange(num_chunks, dtype=jnp.int32))
            return dp_all

    _gn_update_all = jax.jit(_gn_update_all)

    # Reuse measured Lipschitz across outer iterations to avoid repeated power-method
    L_prev = cfg.recon_L
    small_impr_streak = 0
    opt_mode = str(cfg.opt_method).lower()
    outer_stats: list[OuterStat] = []
    wall_start = time.perf_counter()

    def _log_outer_summary(stat: OuterStat) -> None:
        outer_idx = int(stat.get("outer_idx", 0))
        total_iters = int(cfg.outer_iters)
        total_time = format_duration(stat.get("outer_time"))
        elapsed = format_duration(stat.get("cumulative_time"))
        if cfg.log_compact:
            # Build compact one-liner with key fields
            parts: list[str] = [f"Outer {outer_idx}/{total_iters}"]
            # Recon summary
            rbits: list[str] = []
            rt = stat.get("recon_time")
            if rt is not None:
                rbits.append(f"{format_duration(rt)}")
            if stat.get("recon_retry"):
                rbits.append("retry")
            lm = stat.get("L_meas")
            ln = stat.get("L_next")
            if (lm is not None) and (ln is not None):
                rbits.append(f"L {lm:.2e}->{ln:.2e}")
            ff = stat.get("fista_first")
            fl = stat.get("fista_last")
            fm = stat.get("fista_min")
            if (ff is not None) and (fl is not None):
                if fm is not None:
                    rbits.append(f"loss {ff:.2e}->{fl:.2e} (min {fm:.2e})")
                else:
                    rbits.append(f"loss {ff:.2e}->{fl:.2e}")
            if rbits:
                parts.append("recon " + " ".join(rbits))
            # Align summary
            abits: list[str] = []
            at = stat.get("align_time")
            if at is not None:
                abits.append(f"{format_duration(at)}")
            sk = stat.get("step_kind")
            if sk == "gn":
                rm = stat.get("rot_mean")
                tm = stat.get("trans_mean")
                if rm is not None:
                    abits.append(f"|drot| {rm:.2e}")
                if tm is not None:
                    abits.append(f"|dtrans| {tm:.2e}")
            elif sk == "gd":
                rr = stat.get("rot_rms")
                tr = stat.get("trans_rms")
                if rr is not None:
                    abits.append(f"rotRMS {rr:.2e}")
                if tr is not None:
                    abits.append(f"transRMS {tr:.2e}")
            lb = stat.get("loss_before")
            la = stat.get("loss_after")
            ld = stat.get("loss_delta")
            rp = stat.get("loss_rel_pct")
            if (lb is not None) and (la is not None):
                rel = f" {rp:+.2f}%" if rp is not None else ""
                abits.append(f"loss {lb:.2e}->{la:.2e} (Δ {ld:+.2e}{rel})")
            if abits:
                parts.append("align " + " ".join(abits))
            parts.append(f"elapsed {elapsed}")
            logging.info(" | ".join(parts))
            return
        logging.info(
            "Outer %d/%d | total %s | elapsed %s",
            outer_idx,
            total_iters,
            total_time,
            elapsed,
        )

        recon_parts: list[str] = []
        recon_time = stat.get("recon_time")
        if recon_time is not None:
            recon_parts.append(f"time {format_duration(recon_time)}")
        if stat.get("recon_retry"):
            recon_parts.append("fallback retry")
        l_meas = stat.get("L_meas")
        l_next = stat.get("L_next")
        if (l_meas is not None) and (l_next is not None):
            recon_parts.append(f"L {l_meas:.3e}->{l_next:.3e}")
        f_first = stat.get("fista_first")
        f_last = stat.get("fista_last")
        f_min = stat.get("fista_min")
        if (f_first is not None) and (f_last is not None):
            if f_min is not None:
                recon_parts.append(f"loss {f_first:.3e}->{f_last:.3e} (min {f_min:.3e})")
            else:
                recon_parts.append(f"loss {f_first:.3e}->{f_last:.3e}")
        logging.info("  Recon | %s", " | ".join(recon_parts) if recon_parts else "-")

        align_parts: list[str] = []
        align_time = stat.get("align_time")
        if align_time is not None:
            align_parts.append(f"time {format_duration(align_time)}")
        step_kind = stat.get("step_kind")
        if step_kind == "gn":
            rot_mean = stat.get("rot_mean")
            trans_mean = stat.get("trans_mean")
            if rot_mean is not None:
                align_parts.append(f"|drot|_mean {rot_mean:.3e} rad")
            if trans_mean is not None:
                align_parts.append(f"|dtrans|_mean {trans_mean:.3e}")
        elif step_kind == "gd":
            rot_rms = stat.get("rot_rms")
            trans_rms = stat.get("trans_rms")
            if rot_rms is not None:
                align_parts.append(f"rot RMS {rot_rms:.3e}")
            if trans_rms is not None:
                align_parts.append(f"trans RMS {trans_rms:.3e}")
        loss_before = stat.get("loss_before")
        loss_after = stat.get("loss_after")
        loss_delta = stat.get("loss_delta")
        rel_pct = stat.get("loss_rel_pct")
        if (loss_before is not None) and (loss_after is not None):
            rel_str = f", {rel_pct:+.2f}%" if rel_pct is not None else ""
            align_parts.append(
                f"loss {loss_before:.3e}->{loss_after:.3e} (Δ {loss_delta:+.3e}{rel_str})"
            )
        logging.info("  Align | %s", " | ".join(align_parts) if align_parts else "-")

    for it in progress_iter(
        range(cfg.outer_iters), total=cfg.outer_iters, desc="Align: outer iters"
    ):
        outer_idx = it + 1
        stat: OuterStat = {"outer_idx": outer_idx}
        outer_start = time.perf_counter()

        # Reconstruction step
        class _GAll:
            def pose_for_view(self, i):
                T_nom = jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32)
                T_al = se3_from_5d(params5[i])
                return tuple(map(tuple, T_nom @ T_al))

            def rays_for_view(self, i):
                return geometry.rays_for_view(i)

        def _run_fista_safe(vpb: int | None, unroll: int, gather: str, gm: str):
            fista_cfg = FistaConfig(
                iters=cfg.recon_iters,
                lambda_tv=cfg.lambda_tv,
                L=L_prev,
                views_per_batch=vpb,
                projector_unroll=int(unroll),
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=gather,
                grad_mode=gm,
                tv_prox_iters=int(cfg.tv_prox_iters),
                recon_rel_tol=cfg.recon_rel_tol,
                recon_patience=(
                    int(cfg.recon_patience) if cfg.recon_patience is not None else 0
                ),
            )
            return fista_tv(
                _GAll(),
                grid,
                detector,
                projections,
                init_x=x,
                config=fista_cfg,
            )

        vpb0 = cfg.views_per_batch if cfg.views_per_batch > 0 else None
        recon_retry = False
        recon_start = time.perf_counter()
        try:
            x, info_rec = _run_fista_safe(vpb0, int(cfg.projector_unroll), cfg.gather_dtype, "auto")
        except Exception as e:
            msg = str(e)
            if ("RESOURCE_EXHAUSTED" in msg) or ("Out of memory" in msg) or ("Allocator" in msg):
                logging.warning(
                    "FISTA OOM detected; retrying with safer settings (vpb=1, unroll=1, stream)"
                )
                try:
                    recon_retry = True
                    x, info_rec = _run_fista_safe(1, 1, cfg.gather_dtype, "stream")
                except Exception as e2:
                    msg2 = str(e2)
                    if (
                        ("RESOURCE_EXHAUSTED" in msg2)
                        or ("Out of memory" in msg2)
                        or ("Allocator" in msg2)
                    ):
                        logging.error(
                            "FISTA still OOM at finest level. Reduce memory pressure (smaller problem size or lower internal batching), or provide --recon-L to skip power-method."
                        )
                    raise
            else:
                raise
        # Ensure device work is finished before timing recon.
        jax.block_until_ready(x)
        recon_time = time.perf_counter() - recon_start
        stat["recon_time"] = recon_time
        stat["recon_retry"] = recon_retry
        # Capture and reuse measured L next iteration (with small safety margin)
        try:
            L_meas = float(info_rec.get("L", 0.0))
            if L_meas > 0.0:
                L_prev = 1.2 * L_meas
                stat["L_meas"] = L_meas
                stat["L_next"] = L_prev
        except Exception:
            pass
        if info_rec and "loss" in info_rec and info_rec["loss"]:
            try:
                lhist = info_rec["loss"]
                stat["fista_first"] = float(lhist[0])
                stat["fista_last"] = float(lhist[-1])
                stat["fista_min"] = float(min(lhist))
            except Exception:
                pass

        # Alignment step: Gauss–Newton, LBFGS, or gradient descent
        # Evaluate alignment loss before update (needed for GN acceptance / early stop)
        align_start = time.perf_counter()
        loss_before = _evaluate_align_loss(
            lambda: align_loss_jit(params5, x),
            fallback=None,
            context="Skipping pre-step alignment loss evaluation",
        )
        stat["loss_before"] = loss_before
        if opt_mode == "gn" and ls_like:
            step_kind = "gn"
        else:
            step_kind = "gd"
        loss_after = None
        if step_kind == "gn":
            params5_prev = params5
            dp_all = _gn_update_all(params5_prev, x)
            if cfg.gn_accept_only_improving and (loss_before is not None):
                smooth_candidate = None
                if int(params5.shape[0]) >= 3:
                    smooth_candidate = lambda candidate, weights: _smooth_gn_candidate(
                        candidate,
                        smoothness_gram,
                        weights,
                    )
                params5, loss_after = _select_gn_candidate(
                    params5_prev,
                    dp_all,
                    loss_before=loss_before,
                    eval_loss=lambda candidate: float(
                        _evaluate_align_loss(
                            lambda: align_loss_jit(candidate, x),
                            fallback=math.inf,
                            context="Treating GN candidate as rejected during alignment loss evaluation",
                        )
                    ),
                    gn_accept_tol=cfg.gn_accept_tol,
                    smooth_candidate=smooth_candidate,
                    light_smoothness_weights_sq=light_smoothness_weights_sq,
                    medium_smoothness_weights_sq=medium_smoothness_weights_sq,
                    smoothness_weights_sq=smoothness_weights_sq,
                    trans_only_smoothness_weights_sq=trans_only_smoothness_weights_sq,
                )
            else:
                params5 = params5_prev + dp_all
                candidate_loss = _evaluate_align_loss(
                    lambda: align_loss_jit(params5, x),
                    fallback=math.inf,
                    context="Treating GN step as rejected during alignment loss evaluation",
                )
                if candidate_loss is not None and math.isfinite(candidate_loss):
                    loss_after = candidate_loss
                else:
                    params5 = params5_prev
                    loss_after = loss_before
            # Log step stats
            try:
                stat["rot_mean"] = float(jnp.mean(jnp.abs(dp_all[:, :3])))
                stat["trans_mean"] = float(jnp.mean(jnp.abs(dp_all[:, 3:])))
            except Exception:
                pass

        else:
            scales = jnp.array(
                [cfg.lr_rot, cfg.lr_rot, cfg.lr_rot, cfg.lr_trans, cfg.lr_trans], dtype=jnp.float32
            )
            # Keep a copy for line search; donated arg may be reused internally
            p5_in = params5
            _, g_params = loss_and_grad_manual(params5, x)
            rms = jnp.sqrt(jnp.mean(jnp.square(g_params), axis=0)) + 1e-6
            eff_scales = scales / rms
            # Simple 2-point line search on step factor to improve single-iter progress
            best_params = p5_in - g_params * eff_scales
            best_loss = _evaluate_align_loss(
                lambda: align_loss_jit(best_params, x),
                fallback=math.inf,
                context="Treating GD base candidate as rejected during alignment loss evaluation",
            )
            cand_params = p5_in - 2.0 * g_params * eff_scales
            cand_loss = _evaluate_align_loss(
                lambda: align_loss_jit(cand_params, x),
                fallback=math.inf,
                context="Treating GD doubled-step candidate as rejected during alignment loss evaluation",
            )
            best_loss_f = float(best_loss) if best_loss is not None else math.inf
            cand_loss_f = float(cand_loss) if cand_loss is not None else math.inf
            if not math.isfinite(best_loss_f) and not math.isfinite(cand_loss_f):
                params5 = p5_in
                loss_after = loss_before
            else:
                params5 = cand_params if cand_loss_f < best_loss_f else best_params
                chosen_loss = min(best_loss_f, cand_loss_f)
                loss_after = float(chosen_loss) if math.isfinite(chosen_loss) else loss_before
            try:
                stat["rot_rms"] = float(jnp.mean(rms[:3]))
                stat["trans_rms"] = float(jnp.mean(rms[3:]))
            except Exception:
                pass
        stat["step_kind"] = step_kind
        stat["loss_after_step"] = loss_after
        # Ensure device work from alignment step is finished before timing.
        jax.block_until_ready(params5)
        stat["align_time"] = time.perf_counter() - align_start

        # Track overall data loss
        final_loss_fallback = loss_after
        if final_loss_fallback is None:
            final_loss_fallback = loss_before
        if final_loss_fallback is None and loss_hist:
            final_loss_fallback = loss_hist[-1]
        total_loss_eval = _evaluate_align_loss(
            lambda: align_loss_jit(params5, x),
            fallback=final_loss_fallback,
            context="Using fallback for final alignment loss bookkeeping",
        )
        total_loss = float(total_loss_eval) if total_loss_eval is not None else math.nan
        loss_hist.append(total_loss)
        stat["loss_after"] = total_loss
        if loss_before is not None:
            delta = total_loss - loss_before
            stat["loss_delta"] = delta
            if math.isfinite(loss_before) and abs(loss_before) > 1e-12:
                stat["loss_rel_pct"] = (delta / loss_before) * 100.0
            else:
                stat["loss_rel_pct"] = None
            if math.isfinite(loss_before) and math.isfinite(total_loss):
                denom = max(abs(loss_before), 1e-12)
                rel_impr = (loss_before - total_loss) / denom
            else:
                rel_impr = None
        else:
            stat["loss_delta"] = None
            stat["loss_rel_pct"] = None
            rel_impr = None
        stat["rel_impr"] = rel_impr

        outer_time = time.perf_counter() - outer_start
        stat["outer_time"] = outer_time
        stat["cumulative_time"] = time.perf_counter() - wall_start
        outer_stats.append(stat)

        if cfg.log_summary:
            _log_outer_summary(stat)

        if observer is not None:
            observer_action = _normalize_observer_action(observer(x, params5, dict(stat)))
            stat["observer_action"] = observer_action
            stat["observer_stop"] = observer_action != "continue"
            if observer_action != "continue":
                stopped_by_observer = observer_action == "stop_run"
                break

        # Early stopping based on alignment improvement during GN/GD step
        if cfg.early_stop and (rel_impr is not None):
            rel_for_patience = rel_impr
            if (not math.isfinite(rel_for_patience)) or (rel_for_patience < 0.0):
                rel_for_patience = 0.0
            if rel_for_patience < float(cfg.early_stop_rel_impr):
                small_impr_streak += 1
            else:
                small_impr_streak = 0
            if small_impr_streak >= int(cfg.early_stop_patience):
                if cfg.log_summary:
                    logging.info(
                        "Early stop after %d outer iters (%s elapsed): rel_impr=%.3e < %.3e for %d consecutive outers",
                        outer_idx,
                        format_duration(stat.get("cumulative_time")),
                        float(rel_impr),
                        float(cfg.early_stop_rel_impr),
                        int(cfg.early_stop_patience),
                    )
                break
        elif cfg.early_stop:
            small_impr_streak = 0

    if cfg.log_summary and outer_stats:
        recon_total = sum(
            float(s.get("recon_time", 0.0)) for s in outer_stats if s.get("recon_time") is not None
        )
        align_total = sum(
            float(s.get("align_time", 0.0)) for s in outer_stats if s.get("align_time") is not None
        )
        wall_total = time.perf_counter() - wall_start
        logging.info(
            "Alignment completed in %s (recon %s, align %s over %d outer iters)",
            format_duration(wall_total),
            format_duration(recon_total),
            format_duration(align_total),
            len(outer_stats),
        )
        first_loss = outer_stats[0].get("loss_before") if outer_stats else None
        final_loss = outer_stats[-1].get("loss_after") if outer_stats else None
        if (first_loss is not None) and (final_loss is not None):
            total_delta = final_loss - first_loss
            rel_pct = (total_delta / first_loss) * 100.0 if abs(first_loss) > 1e-12 else None
            rel_str = f", {rel_pct:+.2f}%" if rel_pct is not None else ""
            logging.info(
                "  Loss %s -> %s (Δ %s%s)",
                f"{first_loss:.3e}",
                f"{final_loss:.3e}",
                f"{total_delta:+.3e}",
                rel_str,
            )
        best_loss = min(
            (s.get("loss_after") for s in outer_stats if s.get("loss_after") is not None),
            default=None,
        )
        if best_loss is not None and final_loss is not None and best_loss < final_loss:
            logging.info("  Best loss observed: %.3e", best_loss)

    # Provide last measured/reused L for potential reuse across levels
    wall_total = time.perf_counter() - wall_start
    info = {
        "loss": loss_hist,
        "L": (float(L_prev) if L_prev is not None else None),
        "outer_stats": outer_stats,
        "stopped_by_observer": stopped_by_observer,
        "observer_action": observer_action,
        "wall_time_total": float(wall_total),
    }
    return x, params5, info


def align_multires(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    factors: Iterable[int] = (2, 1),
    cfg: AlignConfig | None = None,
    observer: ObserverCallback | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignMultiresInfo]:
    """Coarse-to-fine alignment using simple binning for speed and robustness.

    Carries alignment parameters across levels and downsamples/upsamples volume.
    """
    from ..recon.multires import scale_grid, scale_detector, bin_projections, upsample_volume

    if cfg is None:
        cfg = AlignConfig()

    levels: list[MultiresLevel] = []
    for f in factors:
        levels.append(
            {
                "factor": int(f),
                "grid": scale_grid(grid, int(f)),
                "detector": scale_detector(detector, int(f)),
                "projections": bin_projections(projections, int(f)),
            }
        )

    x_init = None
    params5 = None
    prev_factor: int | None = None
    loss_hist: list[float] = []
    stopped_by_observer = False
    final_observer_action: ObserverAction = "continue"
    global_outer_idx = 0
    global_elapsed_offset = 0.0
    executed_outer_iters = 0

    for li, lvl in enumerate(levels):
        g = lvl["grid"]
        d = lvl["detector"]
        y = lvl["projections"]
        if x_init is not None and prev_factor is not None:
            # Upsample previous x to current level as init
            f_up = prev_factor // lvl["factor"]
            x0 = upsample_volume(x_init, f_up, (g.nx, g.ny, g.nz))
        else:
            x0 = None

        # Optional translation seeding at the coarsest level via phase correlation
        params0 = params5
        if li == 0 and cfg.seed_translations:
            # quick seed recon to project nominal poses
            seed_cfg = FistaConfig(
                iters=max(3, cfg.recon_iters // 2),
                lambda_tv=cfg.lambda_tv,
                projector_unroll=int(cfg.projector_unroll),
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=cfg.gather_dtype,
                recon_rel_tol=cfg.recon_rel_tol,
                recon_patience=(
                    int(cfg.recon_patience) if cfg.recon_patience is not None else 0
                ),
            )
            x_seed, _ = fista_tv(
                geometry,
                g,
                d,
                y,
                init_x=x0,
                config=seed_cfg,
            )
            T_nom = stack_view_poses(geometry, y.shape[0])
            from ..utils.phasecorr import phase_corr_shift

            vm_pred = jax.vmap(
                lambda T: forward_project_view_T(
                    T,
                    g,
                    d,
                    x_seed,
                    use_checkpoint=cfg.checkpoint_projector,
                    gather_dtype=cfg.gather_dtype,
                ),
                in_axes=0,
            )
            preds = vm_pred(T_nom)
            shift_uv = jax.vmap(phase_corr_shift)(preds, y)  # returns (du, dv)
            shifts = jnp.stack(shift_uv, axis=1).astype(jnp.float32)  # (n, 2)
            # Convert pixel shifts to world units using detector spacing
            dx = shifts[:, 0] * jnp.float32(d.du)
            dz = shifts[:, 1] * jnp.float32(d.dv)
            params0 = jnp.zeros((y.shape[0], 5), dtype=jnp.float32)
            params0 = params0.at[:, 3].set(dx)
            params0 = params0.at[:, 4].set(dz)

        # Run alignment at this level
        # Re-estimate L at each level using a fresh (streamed) power-method for stability
        cfg_level = replace(cfg, recon_L=None)

        def _level_observer(x_obs, params_obs, stat_obs):
            nonlocal global_outer_idx, stopped_by_observer
            global_outer_idx += 1
            enriched = dict(stat_obs)
            enriched["level_factor"] = int(lvl["factor"])
            enriched["level_index"] = int(li)
            enriched["global_outer_idx"] = int(global_outer_idx)
            level_elapsed = stat_obs.get("cumulative_time")
            try:
                level_elapsed_f = float(level_elapsed) if level_elapsed is not None else None
            except Exception:
                level_elapsed_f = None
            enriched["level_elapsed_seconds"] = level_elapsed_f
            enriched["global_elapsed_seconds"] = (
                float(global_elapsed_offset + level_elapsed_f)
                if level_elapsed_f is not None
                else None
            )
            if observer is None:
                return "continue"
            return _normalize_observer_action(observer(x_obs, params_obs, enriched))

        x_lvl, params5, info = align(
            geometry,
            g,
            d,
            y,
            cfg=cfg_level,
            init_x=x0,
            init_params5=params0,
            observer=_level_observer if observer is not None else None,
        )
        loss_hist.extend(info.get("loss", []))
        executed_outer_iters += len(info.get("outer_stats", []))
        x_init = x_lvl
        prev_factor = lvl["factor"]
        try:
            global_elapsed_offset += float(info.get("wall_time_total") or 0.0)
        except Exception:
            pass
        level_action = _normalize_observer_action(info.get("observer_action"))
        final_observer_action = level_action
        if level_action == "stop_run":
            stopped_by_observer = True
            break
        if level_action == "advance_level":
            continue

    # Always return a full-resolution-compatible final volume.
    if x_init is None:
        x_final = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    elif prev_factor is not None and prev_factor != 1:
        x_final = upsample_volume(x_init, prev_factor, (grid.nx, grid.ny, grid.nz))
    else:
        x_final = x_init

    return (
        x_final,
        params5 if params5 is not None else jnp.zeros((projections.shape[0], 5), jnp.float32),
        {
            "loss": loss_hist,
            "factors": list(factors),
            "stopped_by_observer": stopped_by_observer,
            "observer_action": final_observer_action,
            "total_outer_iters": int(executed_outer_iters),
            "wall_time_total": float(global_elapsed_offset),
        },
    )
