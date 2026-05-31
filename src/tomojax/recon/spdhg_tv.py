"""SPDHG/TV reconstruction routine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import (
    forward_project_view_T,
    get_detector_grid_device,
    sum_backproject_views_T,
)
from tomojax.core.validation import (
    validate_grid,
    validate_optional_broadcastable_shape,
    validate_optional_same_shape,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)

from ._callbacks import LossCallback, emit_loss_callback_endpoints
from ._tv_ops import (
    div3,
    grad3,
    huber_tv_value,
    isotropic_tv_value,
    prox_huber_tv_conj,
    validate_regulariser,
)

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from .types import Regulariser

_SPDHGProjectChunk = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


# --------- config ----------


@dataclass
class SPDHGConfig:
    """Configuration for stochastic primal-dual TV reconstruction."""

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


class _SPDHGScanState(NamedTuple):
    x: jnp.ndarray
    x_bar: jnp.ndarray
    y_data: jnp.ndarray
    p1: jnp.ndarray
    p2: jnp.ndarray
    p3: jnp.ndarray
    s: jnp.ndarray
    losses: jnp.ndarray


@dataclass(frozen=True)
class _SPDHGStepSizes:
    tau: float
    sigma_data_base: float
    sigma_data_eff: float
    sigma_tv: float
    data_norm: float | None
    grad_norm: float


@dataclass(frozen=True)
class _SPDHGSchedule:
    views_per_batch: int
    num_blocks: int
    selection_prob: float
    block_ids: jnp.ndarray


@dataclass(frozen=True)
class _SPDHGRuntime:
    config: SPDHGConfig
    regulariser: Regulariser
    huber_delta: float
    y_meas: jnp.ndarray
    weights: jnp.ndarray
    poses: jnp.ndarray
    detector_grid: tuple[jnp.ndarray, jnp.ndarray]
    support: jnp.ndarray | None
    lambda_tv: jnp.ndarray
    step_sizes: _SPDHGStepSizes
    schedule: _SPDHGSchedule
    initial_state: _SPDHGScanState


@dataclass(frozen=True)
class _SPDHGResult:
    volume: jnp.ndarray
    losses: jnp.ndarray
    step_sizes: _SPDHGStepSizes
    schedule: _SPDHGSchedule
    regulariser: Regulariser
    huber_delta: float
    lambda_tv: float

    def info(self) -> dict[str, object]:
        step_sizes = self.step_sizes
        schedule = self.schedule
        return {
            "loss": [float(v) for v in list(self.losses)],
            "tau": float(step_sizes.tau),
            "sigma_data": float(step_sizes.sigma_data_eff),
            "sigma_data_base": float(step_sizes.sigma_data_base),
            "sigma_tv": float(step_sizes.sigma_tv),
            "lambda_tv": float(self.lambda_tv),
            "views_per_batch": int(schedule.views_per_batch),
            "num_blocks": int(schedule.num_blocks),
            "A_norm": (float(step_sizes.data_norm) if step_sizes.data_norm is not None else None),
            "grad_norm": float(step_sizes.grad_norm),
            "selection_prob": float(schedule.selection_prob),
            "regulariser": self.regulariser,
            "huber_delta": float(self.huber_delta),
        }


# --------- helpers ----------


def _estimate_norm_A2(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections_shape: tuple[int, int, int],
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
    """Estimate the squared projection-operator norm by power iteration."""
    del geometry
    n_views, _nv, _nu = projections_shape
    det_grid = get_detector_grid_device(detector) if det_grid is None else det_grid

    def A_apply(vol: jnp.ndarray, T_chunk: jnp.ndarray) -> jnp.ndarray:
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

    def AtranA(v: jnp.ndarray) -> jnp.ndarray:
        # iterate over contiguous blocks with masking of the last chunk
        def body(
            g_acc: jnp.ndarray,
            i: jnp.ndarray,
        ) -> tuple[jnp.ndarray, None]:
            i = jnp.int32(i)
            start = i * jnp.int32(b)
            remaining = jnp.maximum(0, jnp.int32(n_views) - start)
            valid = jnp.minimum(jnp.int32(b), remaining)
            shift = jnp.int32(b) - valid
            start_shifted = jnp.maximum(0, start - shift)

            T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))

            def pred_fun(vol: jnp.ndarray) -> jnp.ndarray:
                return A_apply(vol, T_chunk)

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

    def normalize(v: jnp.ndarray) -> jnp.ndarray:
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


def _proj_pos_support(
    x: jnp.ndarray,
    positivity: bool,
    support: jnp.ndarray | None,
) -> jnp.ndarray:
    if support is not None:
        x = x * support
    if positivity:
        x = jnp.maximum(x, 0)
    return x


def _prox_fstar_l2(
    u: jnp.ndarray,
    sigma: float,
    y_meas: jnp.ndarray,
    w: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the weighted L2 dual proximal.

    Elementwise: if w > 0, return ``(u - sigma * y) * w / (sigma + w)``;
    otherwise return zero for the domain of the conjugate.
    """
    sigma = jnp.asarray(sigma, dtype=u.dtype)
    denom = sigma + w
    v = (u - sigma * y_meas) * w / jnp.maximum(denom, 1e-12)
    return jnp.where(w > 0, v, 0.0).astype(u.dtype)


def _resolve_spdhg_step_sizes(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    y_meas: jnp.ndarray,
    poses: jnp.ndarray,
    config: SPDHGConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
) -> _SPDHGStepSizes:
    grad_norm = float(np.sqrt(12.0))
    data_norm: float | None = None
    if config.tau is None or config.sigma_data is None or config.sigma_tv is None:
        data_norm_sq = _estimate_norm_A2(
            geometry,
            grid,
            detector,
            y_meas.shape,
            poses,
            views_per_batch=max(1, config.views_per_batch),
            projector_unroll=config.projector_unroll,
            checkpoint_projector=config.checkpoint_projector,
            gather_dtype=config.gather_dtype,
            key=jax.random.key(config.seed),
            power_iters=20,
            safety=1.05,
            det_grid=det_grid,
        )
        data_norm = float(np.sqrt(data_norm_sq))
        rho = 0.99
        tau = rho / (data_norm + grad_norm)
        sigma_data_base = rho / max(data_norm, 1e-6)
        sigma_tv = rho / grad_norm
    else:
        tau = float(config.tau)
        sigma_data_base = float(config.sigma_data)
        sigma_tv = float(config.sigma_tv)

    return _SPDHGStepSizes(
        tau=tau,
        sigma_data_base=sigma_data_base,
        sigma_data_eff=sigma_data_base,
        sigma_tv=sigma_tv,
        data_norm=data_norm,
        grad_norm=grad_norm,
    )


def _build_spdhg_schedule(n_views: int, config: SPDHGConfig) -> _SPDHGSchedule:
    views_per_batch = int(max(1, min(config.views_per_batch, n_views)))
    num_blocks = (n_views + views_per_batch - 1) // views_per_batch
    rng = np.random.default_rng(config.seed)
    epochs = (config.iters + num_blocks - 1) // num_blocks
    block_ids: list[int] = []
    for _ in range(epochs):
        block_ids.extend(int(block) for block in rng.permutation(num_blocks))

    return _SPDHGSchedule(
        views_per_batch=views_per_batch,
        num_blocks=num_blocks,
        selection_prob=1.0 / float(max(num_blocks, 1)),
        block_ids=jnp.asarray(block_ids[: config.iters], dtype=jnp.int32),
    )


def _initial_spdhg_state(
    grid: Grid,
    y_meas: jnp.ndarray,
    init_x: jnp.ndarray | None,
    *,
    iters: int,
) -> _SPDHGScanState:
    x0 = (
        jnp.asarray(init_x, dtype=jnp.float32)
        if init_x is not None
        else jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    )
    return _SPDHGScanState(
        x=x0,
        x_bar=jnp.array(x0, copy=True),
        y_data=jnp.zeros_like(y_meas),
        p1=jnp.zeros_like(x0),
        p2=jnp.zeros_like(x0),
        p3=jnp.zeros_like(x0),
        s=jnp.zeros_like(x0),
        losses=jnp.zeros((iters,), dtype=jnp.float32),
    )


def _prepare_spdhg_runtime(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    weights: jnp.ndarray | None,
    init_x: jnp.ndarray | None,
    config: SPDHGConfig | None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> _SPDHGRuntime:
    cfg = SPDHGConfig() if config is None else config
    regulariser = validate_regulariser(
        cfg.regulariser,
        cfg.huber_delta,
        context="spdhg_tv config",
    )
    huber_delta = float(cfg.huber_delta)

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
        cfg.support,
        (grid.nx, grid.ny, grid.nz),
        context="spdhg_tv support",
        name="support",
        fix="use a support mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if init_x is not None:
        validate_volume(init_x, grid, context="spdhg_tv init_x", name="init_x")

    y_meas = jnp.asarray(projections, dtype=jnp.float32)
    weights_arr = (
        jnp.ones_like(y_meas) if weights is None else jnp.asarray(weights, dtype=jnp.float32)
    )
    poses = stack_view_poses(geometry, n_views)
    validate_pose_stack(poses, n_views, context="spdhg_tv geometry")
    resolved_det_grid = get_detector_grid_device(detector) if det_grid is None else det_grid
    step_sizes = _resolve_spdhg_step_sizes(
        geometry,
        grid,
        detector,
        y_meas,
        poses,
        cfg,
        resolved_det_grid,
    )

    return _SPDHGRuntime(
        config=cfg,
        regulariser=regulariser,
        huber_delta=huber_delta,
        y_meas=y_meas,
        weights=weights_arr,
        poses=poses,
        detector_grid=resolved_det_grid,
        support=None if cfg.support is None else jnp.asarray(cfg.support, dtype=jnp.float32),
        lambda_tv=jnp.asarray(cfg.lambda_tv, dtype=jnp.float32),
        step_sizes=step_sizes,
        schedule=_build_spdhg_schedule(n_views, cfg),
        initial_state=_initial_spdhg_state(grid, y_meas, init_x, iters=cfg.iters),
    )


def _spdhg_logged_steps(config: SPDHGConfig) -> list[int]:
    if config.log_every <= 0:
        return []
    return [
        int(step) for step in np.flatnonzero((np.arange(config.iters) + 1) % config.log_every == 0)
    ]


def _emit_spdhg_callback(
    callback: LossCallback | None,
    result: _SPDHGResult,
    config: SPDHGConfig,
) -> None:
    logged_steps = _spdhg_logged_steps(config)
    if not logged_steps:
        return
    losses_host = np.asarray(result.losses)
    emit_loss_callback_endpoints(
        callback,
        (
            (logged_steps[0], float(losses_host[logged_steps[0]])),
            (logged_steps[-1], float(losses_host[logged_steps[-1]])),
        ),
    )


# --------- main algorithm ----------


def _make_spdhg_project_chunk(
    grid: Grid,
    detector: Detector,
    runtime: _SPDHGRuntime,
) -> _SPDHGProjectChunk:
    cfg = runtime.config

    def project_chunk(T_chunk: jnp.ndarray, vol: jnp.ndarray) -> jnp.ndarray:
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T,
                grid,
                detector,
                v,
                use_checkpoint=cfg.checkpoint_projector,
                unroll=int(cfg.projector_unroll),
                gather_dtype=cfg.gather_dtype,
                det_grid=runtime.detector_grid,
            ),
            in_axes=(0, None),
        )
        return vm_project(T_chunk, vol)

    return project_chunk


def _run_spdhg_scan(  # noqa: PLR0915
    grid: Grid,
    detector: Detector,
    runtime: _SPDHGRuntime,
) -> _SPDHGResult:
    cfg = runtime.config
    schedule = runtime.schedule
    step_sizes = runtime.step_sizes
    nv = runtime.y_meas.shape[1]
    nu = runtime.y_meas.shape[2]
    project_chunk = _make_spdhg_project_chunk(grid, detector, runtime)

    def one_step(state: _SPDHGScanState, t: jnp.ndarray) -> tuple[_SPDHGScanState, None]:
        block = schedule.block_ids[t]
        start = block * jnp.int32(schedule.views_per_batch)
        remaining = jnp.maximum(0, jnp.int32(runtime.y_meas.shape[0]) - start)
        valid = jnp.minimum(jnp.int32(schedule.views_per_batch), remaining)
        shift = jnp.int32(schedule.views_per_batch) - valid
        start_shifted = jnp.maximum(0, start - shift)

        T_chunk = jax.lax.dynamic_slice(
            runtime.poses,
            (start_shifted, 0, 0),
            (schedule.views_per_batch, 4, 4),
        )
        y_chunk = jax.lax.dynamic_slice(
            runtime.y_meas,
            (start_shifted, 0, 0),
            (schedule.views_per_batch, nv, nu),
        )
        w_chunk = jax.lax.dynamic_slice(
            runtime.weights,
            (start_shifted, 0, 0),
            (schedule.views_per_batch, nv, nu),
        )
        y_dual_old = jax.lax.dynamic_slice(
            state.y_data,
            (start_shifted, 0, 0),
            (schedule.views_per_batch, nv, nu),
        )

        idx = jnp.arange(schedule.views_per_batch)
        row_mask = (idx >= (jnp.int32(schedule.views_per_batch) - valid))[:, None, None]
        row_mask = row_mask.astype(jnp.float32)

        sigma_eff = jnp.asarray(step_sizes.sigma_data_eff, dtype=state.x_bar.dtype)
        pred = project_chunk(T_chunk, state.x_bar)
        u = y_dual_old + sigma_eff * pred
        y_dual_new = _prox_fstar_l2(u, sigma_eff, y_chunk, w_chunk)
        y_dual_new = row_mask * y_dual_new + (1.0 - row_mask) * y_dual_old
        delta_y = (y_dual_new - y_dual_old) * row_mask

        g_block = sum_backproject_views_T(
            T_chunk,
            grid,
            detector,
            delta_y,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
            det_grid=runtime.detector_grid,
        )

        gx, gy, gz = grad3(state.x_bar)
        p1_u = state.p1 + step_sizes.sigma_tv * gx
        p2_u = state.p2 + step_sizes.sigma_tv * gy
        p3_u = state.p3 + step_sizes.sigma_tv * gz
        if runtime.regulariser == "huber_tv":
            p1_new, p2_new, p3_new = prox_huber_tv_conj(
                p1_u,
                p2_u,
                p3_u,
                sigma=step_sizes.sigma_tv,
                lam=runtime.lambda_tv,
                delta=runtime.huber_delta,
            )
        else:
            norm = jnp.maximum(
                1.0,
                jnp.sqrt(p1_u * p1_u + p2_u * p2_u + p3_u * p3_u)
                / jnp.maximum(runtime.lambda_tv, 1e-12),
            )
            p1_new = p1_u / norm
            p2_new = p2_u / norm
            p3_new = p3_u / norm

        delta_div = div3(p1_new - state.p1, p2_new - state.p2, p3_new - state.p3)
        s_new = state.s + g_block - delta_div
        x_new = _proj_pos_support(state.x - step_sizes.tau * s_new, cfg.positivity, runtime.support)
        x_bar_candidate = x_new + jnp.asarray(cfg.theta, x_new.dtype) * (x_new - state.x)
        x_bar_new = _proj_pos_support(x_bar_candidate, cfg.positivity, runtime.support)
        y_data_new = jax.lax.dynamic_update_slice(
            state.y_data,
            y_dual_new,
            (start_shifted, 0, 0),
        )

        do_log = (cfg.log_every > 0) & ((t + 1) % cfg.log_every == 0)

        def log_step() -> jnp.ndarray:
            resid = (pred - y_chunk) * jnp.sqrt(w_chunk) * row_mask
            data_est = (
                0.5
                * jnp.vdot(resid, resid).real
                * (float(runtime.y_meas.shape[0]) / jnp.maximum(valid.astype(jnp.float32), 1.0))
            )
            if runtime.regulariser == "huber_tv":
                reg_value = huber_tv_value(x_new, runtime.huber_delta)
            else:
                reg_value = isotropic_tv_value(x_new)
            obj = (data_est + runtime.lambda_tv * reg_value).astype(jnp.float32)
            return state.losses.at[t].set(obj)

        losses_new = jax.lax.cond(do_log, log_step, lambda: state.losses)
        return _SPDHGScanState(
            x=x_new,
            x_bar=x_bar_new,
            y_data=y_data_new,
            p1=p1_new,
            p2=p2_new,
            p3=p3_new,
            s=s_new,
            losses=losses_new,
        ), None

    final_state, _ = jax.jit(
        lambda state: jax.lax.scan(one_step, state, jnp.arange(cfg.iters)),
        donate_argnums=(0,),
    )(runtime.initial_state)

    return _SPDHGResult(
        volume=final_state.x,
        losses=final_state.losses,
        step_sizes=step_sizes,
        schedule=schedule,
        regulariser=runtime.regulariser,
        huber_delta=runtime.huber_delta,
        lambda_tv=float(cfg.lambda_tv),
    )


def spdhg_tv(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    weights: jnp.ndarray | None = None,  # same shape as projections; 0 for unmeasured
    init_x: jnp.ndarray | None = None,
    config: SPDHGConfig | None = None,
    callback: LossCallback | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, dict[str, object]]:
    """SPDHG (stochastic Chambolle-Pock) with weighted L2 data term and TV-like regularization.

    If ``callback`` is provided, it fires on the first logged objective sample and
    on the final logged objective sample. The callback arguments are ``(step,
    loss)``, where ``step`` is the zero-based iteration index that produced
    ``loss``. Only iterations whose objective was recorded under ``config.log_every``
    are eligible for callbacks; if a single logged sample exists, the callback
    fires once.
    """
    runtime = _prepare_spdhg_runtime(
        geometry,
        grid,
        detector,
        projections,
        weights=weights,
        init_x=init_x,
        config=config,
        det_grid=det_grid,
    )
    result = _run_spdhg_scan(grid, detector, runtime)
    _emit_spdhg_callback(callback, result, runtime.config)
    return result.volume, result.info()
