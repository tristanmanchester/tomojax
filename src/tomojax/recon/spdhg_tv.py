"""SPDHG/TV reconstruction routine."""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import dataclass, field, fields, replace
import functools
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.cone import is_cone_beam
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.operator_norm import estimate_normal_norm
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
from tomojax.recon._projection import (
    ConeModel,
    ProjectorBackend,
    ProjectorModel,
    normal_operator_norm,
    projection_operators,
    resolve_geometry_projector,
    resolve_projector,
)

from ._callbacks import LossCallback, emit_loss_callback_endpoints
from ._host_stream import (
    host_buffer,
    host_source,
    read_block,
    read_views,
    should_stream,
    write_block,
)
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
    """Configuration for stochastic primal-dual TV reconstruction.

    ``projector_model`` and ``projector_backend`` choose the projection operator as
    in :class:`FistaConfig`: ``"auto"`` uses Joseph plane sampling with Pallas on
    CUDA unless an explicit detector grid or the exact ray integrator requires
    the ray-model reference path. ``gather_dtype``, ``projector_unroll`` and
    ``checkpoint_projector`` apply to the reference path only.

    ``stream_projections`` keeps NumPy or memmap projections, any weights and
    the sinogram-sized dual variable in host memory; each iteration reads and
    writes only its block of views. ``None`` streams when the projections would
    take more than 40% of free device memory. Streaming needs the batched
    operators.
    """

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
    ray_integrator: str = "sampled"
    projector_model: ProjectorModel = "auto"
    projector_backend: ProjectorBackend = "auto"

    # constraints
    positivity: bool = True
    support: jnp.ndarray | None = None  # 0/1 mask in volume space

    # logging
    log_every: int = 10  # minibatch objective estimator every k steps

    stream_projections: bool | None = None


jax.tree_util.register_dataclass(
    SPDHGConfig,
    data_fields=["support"],
    meta_fields=[field.name for field in fields(SPDHGConfig) if field.name != "support"],
)


class _SPDHGScanState(NamedTuple):
    x: jnp.ndarray
    x_bar: jnp.ndarray
    y_data: jnp.ndarray
    p1: jnp.ndarray
    p2: jnp.ndarray
    p3: jnp.ndarray
    # Sum of A_i^T y_i over data blocks; the TV part, -div p, is recomputed.
    s_data: jnp.ndarray
    losses: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _SPDHGStepSizes:
    tau: float
    sigma_data_base: float
    sigma_data_eff: float
    sigma_tv: float
    data_norm: float | None
    grad_norm: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _SPDHGSchedule:
    views_per_batch: int = field(metadata={"static": True})
    num_blocks: int = field(metadata={"static": True})
    selection_prob: float = field(metadata={"static": True})
    block_ids: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _SPDHGRuntime:
    config: SPDHGConfig
    regulariser: Regulariser = field(metadata={"static": True})
    huber_delta: float = field(metadata={"static": True})
    # None when streaming: data, weights and duals then live in host memory.
    y_meas: jnp.ndarray | None
    # None means unit weights; no sinogram of ones is stored.
    weights: jnp.ndarray | None
    poses: jnp.ndarray
    detector_grid: tuple[jnp.ndarray, jnp.ndarray]
    support: jnp.ndarray | None
    lambda_tv: jnp.ndarray
    step_sizes: _SPDHGStepSizes
    schedule: _SPDHGSchedule
    # None starts from zeros created inside the compiled solve.
    init_x: jnp.ndarray | None
    # (model, backend) for batched operators, or None for the ray reference path.
    projector: tuple[str, str] | None = field(metadata={"static": True})
    data_shape: tuple[int, int, int] = field(metadata={"static": True})
    weighted: bool = field(default=False, metadata={"static": True})
    # Host keys (measured views, duals, weights) when streaming.
    host_keys: jnp.ndarray | None = None


@dataclass(frozen=True)
class _SPDHGResult:
    volume: jnp.ndarray
    losses: jnp.ndarray
    step_sizes: _SPDHGStepSizes
    schedule: _SPDHGSchedule
    regulariser: Regulariser = field(metadata={"static": True})
    huber_delta: float = field(metadata={"static": True})
    lambda_tv: float

    def info(self) -> dict[str, object]:
        step_sizes = self.step_sizes
        schedule = self.schedule
        return {
            "loss": np.asarray(self.losses).tolist(),
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
    ray_integrator: str = "sampled",
    key: jax.Array | None = None,
    power_iters: int = 20,
    safety: float = 1.05,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> float:
    """Estimate the squared projection-operator norm by power iteration."""
    del geometry
    n_views = projections_shape[0]
    if key is None:
        key = jax.random.key(0)
    initial = jax.random.normal(key, (grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    norm_squared = estimate_normal_norm(
        T_all,
        initial,
        det_grid,
        None,
        grid=grid,
        detector=detector,
        batch_size=max(1, min(views_per_batch, n_views)),
        iters=int(power_iters),
        unroll=int(projector_unroll),
        checkpoint=checkpoint_projector,
        gather_dtype=gather_dtype,
        ray_integrator=ray_integrator,
    )
    return max(float(norm_squared) * float(safety**2), 1e-6)


def _proj_pos_support(
    x: jnp.ndarray,
    support: jnp.ndarray | None,
    *,
    positivity: bool,
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
    w: jnp.ndarray | None,
) -> jnp.ndarray:
    """Apply the weighted L2 dual proximal.

    Elementwise: if w > 0, return ``(u - sigma * y) * w / (sigma + w)``;
    otherwise return zero for the domain of the conjugate. ``None`` is w = 1.
    """
    sigma = jnp.asarray(sigma, dtype=u.dtype)
    if w is None:
        return ((u - sigma * y_meas) / (sigma + 1)).astype(u.dtype)
    denom = sigma + w
    v = (u - sigma * y_meas) * w / jnp.maximum(denom, 1e-12)
    return jnp.where(w > 0, v, 0.0).astype(u.dtype)


def _batched_projector(
    config: SPDHGConfig,
    det_grid: object,
    geometry: object = None,
    detector: Detector | None = None,
) -> tuple[str | ConeModel, str] | None:
    """Choose batched operators, or None for the ray-model reference path."""
    model, backend = config.projector_model, config.projector_backend
    if is_cone_beam(geometry):
        if config.ray_integrator == "exact":
            raise ValueError(
                "spdhg_tv: cone-beam geometry uses Joseph sampling, not exact integration"
            )
        assert detector is not None
        return resolve_geometry_projector(
            geometry, model, backend, detector=detector, det_grid=det_grid, context="spdhg_tv"
        )
    if det_grid is not None or config.ray_integrator != "sampled":
        if model == "joseph" or backend == "pallas":
            raise ValueError(
                "spdhg_tv: explicit detector grids and exact integration require "
                "projector_model='ray' and projector_backend='jax'"
            )
        return None
    model, backend = resolve_projector(model, backend, context="spdhg_tv")
    return None if (model, backend) == ("ray", "jax") else (model, backend)


@functools.partial(jax.jit, static_argnames=("grid", "detector", "projector", "iters"))
def _batched_norm_squared(
    poses: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    projector: tuple[str, str],
    iters: int,
) -> jnp.ndarray:
    model, backend = projector
    batch = min(64, int(poses.shape[0]))
    forward, adjoint = projection_operators(poses, grid, detector, None, backend, batch, model)
    return normal_operator_norm(forward, adjoint, (grid.nx, grid.ny, grid.nz), iters=iters)


def _resolve_spdhg_step_sizes(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data_shape: tuple[int, int, int],
    poses: jnp.ndarray,
    config: SPDHGConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    projector: tuple[str, str] | None = None,
) -> _SPDHGStepSizes:
    grad_norm = float(np.sqrt(12.0))
    if config.tau is not None and config.sigma_data is not None and config.sigma_tv is not None:
        return _SPDHGStepSizes(
            tau=float(config.tau),
            sigma_data_base=float(config.sigma_data),
            sigma_data_eff=float(config.sigma_data),
            sigma_tv=float(config.sigma_tv),
            data_norm=None,
            grad_norm=grad_norm,
        )
    if projector is not None:
        norm_sq = _batched_norm_squared(
            poses, grid=grid, detector=detector, projector=projector, iters=20
        )
        data_norm_sq = max(float(norm_sq) * 1.05**2, 1e-6)
    else:
        data_norm_sq = _estimate_norm_A2(
            geometry,
            grid,
            detector,
            data_shape,
            poses,
            views_per_batch=max(1, config.views_per_batch),
            projector_unroll=config.projector_unroll,
            checkpoint_projector=config.checkpoint_projector,
            gather_dtype=config.gather_dtype,
            key=jax.random.key(config.seed),
            power_iters=20,
            safety=1.05,
            det_grid=det_grid,
            ray_integrator=config.ray_integrator,
        )
    data_norm = float(np.sqrt(data_norm_sq))
    rho = 0.99
    tau = rho / (data_norm + grad_norm)
    sigma_data_base = rho / max(data_norm, 1e-6)
    sigma_tv = rho / grad_norm
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
    y_meas: jnp.ndarray | None,
    init_x: jnp.ndarray | None,
    *,
    iters: int,
) -> _SPDHGScanState:
    # Called inside the compiled solve: zero states are never input buffers.
    x0 = init_x if init_x is not None else jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    return _SPDHGScanState(
        x=x0,
        x_bar=x0,
        # Streaming keeps the duals on the host; a placeholder stands in here.
        y_data=jnp.zeros((1, 1, 1), jnp.float32) if y_meas is None else jnp.zeros_like(y_meas),
        p1=jnp.zeros_like(x0),
        p2=jnp.zeros_like(x0),
        p3=jnp.zeros_like(x0),
        s_data=jnp.zeros_like(x0),
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

    poses = stack_view_poses(geometry, n_views)
    validate_pose_stack(poses, n_views, context="spdhg_tv geometry")
    resolved_det_grid = get_detector_grid_device(detector) if det_grid is None else det_grid
    projector = _batched_projector(cfg, det_grid, geometry, detector)
    stream = not isinstance(projections, jax.Array) and (
        bool(cfg.stream_projections)
        if cfg.stream_projections is not None
        else should_stream(projections)
    )
    if stream and projector is None:
        raise ValueError("spdhg_tv: stream_projections requires the batched projection operators")
    y_meas = None if stream else jnp.asarray(projections, dtype=jnp.float32)
    weights_arr = None if weights is None or stream else jnp.asarray(weights, dtype=jnp.float32)
    step_sizes = _resolve_spdhg_step_sizes(
        geometry,
        grid,
        detector,
        expected_proj_shape,
        poses,
        cfg,
        resolved_det_grid,
        projector,
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
        init_x=None if init_x is None else jnp.asarray(init_x, dtype=jnp.float32),
        projector=projector,
        data_shape=expected_proj_shape,
        weighted=weights is not None,
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
    if runtime.projector is not None:
        model, backend = runtime.projector

        def project_batched(T_chunk: jnp.ndarray, vol: jnp.ndarray) -> jnp.ndarray:
            forward, _ = projection_operators(
                T_chunk, grid, detector, None, backend, T_chunk.shape[0], model
            )
            return forward(vol)

        return project_batched

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
                ray_integrator=cfg.ray_integrator,
            ),
            in_axes=(0, None),
        )
        return vm_project(T_chunk, vol)

    return project_chunk


def _make_spdhg_backproject_chunk(
    grid: Grid,
    detector: Detector,
    runtime: _SPDHGRuntime,
) -> _SPDHGProjectChunk:
    cfg = runtime.config

    def backproject(T_chunk: jnp.ndarray, images: jnp.ndarray, total: jnp.ndarray) -> jnp.ndarray:
        """Return ``total`` plus the chunk transpose; CUDA Joseph adds in place."""
        if runtime.projector is not None:
            model, backend = runtime.projector
            _, adjoint = projection_operators(
                T_chunk, grid, detector, None, backend, T_chunk.shape[0], model
            )
            return adjoint(images, accumulate=total)
        return total + sum_backproject_views_T(
            T_chunk,
            grid,
            detector,
            images,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
            det_grid=runtime.detector_grid,
            ray_integrator=cfg.ray_integrator,
        )

    return backproject


@functools.partial(jax.jit, static_argnames=("grid", "detector"))
def _run_spdhg_scan(  # noqa: PLR0915
    grid: Grid,
    detector: Detector,
    runtime: _SPDHGRuntime,
) -> _SPDHGScanState:
    cfg = runtime.config
    schedule = runtime.schedule
    step_sizes = runtime.step_sizes
    n_views, nv, nu = runtime.data_shape
    block_shape = (schedule.views_per_batch, nv, nu)
    keys = runtime.host_keys
    project_chunk = _make_spdhg_project_chunk(grid, detector, runtime)
    backproject_chunk = _make_spdhg_backproject_chunk(grid, detector, runtime)

    def load(state: _SPDHGScanState, start: jnp.ndarray) -> tuple:
        # Measured views, weights and duals of one block, from host or device.
        if keys is not None:
            weights = read_views(keys[2], start, block_shape) if runtime.weighted else None
            return (
                read_views(keys[0], start, block_shape),
                weights,
                read_block(keys[1], start, block_shape),
            )
        weights = None
        if runtime.weights is not None:
            weights = jax.lax.dynamic_slice(runtime.weights, (start, 0, 0), block_shape)
        return (
            jax.lax.dynamic_slice(runtime.y_meas, (start, 0, 0), block_shape),
            weights,
            jax.lax.dynamic_slice(state.y_data, (start, 0, 0), block_shape),
        )

    def store(state: _SPDHGScanState, start: jnp.ndarray, duals: jnp.ndarray) -> jnp.ndarray:
        if keys is not None:
            write_block(keys[1], start, duals)
            return state.y_data
        return jax.lax.dynamic_update_slice(state.y_data, duals, (start, 0, 0))

    def one_step(state: _SPDHGScanState, t: jnp.ndarray) -> tuple[_SPDHGScanState, None]:
        block = schedule.block_ids[t]
        start = block * jnp.int32(schedule.views_per_batch)
        remaining = jnp.maximum(0, jnp.int32(n_views) - start)
        valid = jnp.minimum(jnp.int32(schedule.views_per_batch), remaining)
        shift = jnp.int32(schedule.views_per_batch) - valid
        start_shifted = jnp.maximum(0, start - shift)

        T_chunk = jax.lax.dynamic_slice(
            runtime.poses,
            (start_shifted, 0, 0),
            (schedule.views_per_batch, 4, 4),
        )
        y_chunk, w_chunk, y_dual_old = load(state, start_shifted)

        idx = jnp.arange(schedule.views_per_batch)
        row_mask = (idx >= (jnp.int32(schedule.views_per_batch) - valid))[:, None, None]
        row_mask = row_mask.astype(jnp.float32)

        sigma_eff = jnp.asarray(step_sizes.sigma_data_eff, dtype=state.x_bar.dtype)
        pred = project_chunk(T_chunk, state.x_bar)
        u = y_dual_old + sigma_eff * pred
        y_dual_new = _prox_fstar_l2(u, sigma_eff, y_chunk, w_chunk)
        y_dual_new = row_mask * y_dual_new + (1.0 - row_mask) * y_dual_old
        delta_y = (y_dual_new - y_dual_old) * row_mask

        s_data_new = backproject_chunk(T_chunk, delta_y, state.s_data)

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

        # s = sum_i A_i^T y_i - div p. Recomputing div p from the updated duals,
        # rather than differencing old and new duals, lets them update in place.
        s_new = s_data_new - div3(p1_new, p2_new, p3_new)
        x_new = _proj_pos_support(
            state.x - step_sizes.tau * s_new, runtime.support, positivity=cfg.positivity
        )
        x_bar_candidate = x_new + jnp.asarray(cfg.theta, x_new.dtype) * (x_new - state.x)
        x_bar_new = _proj_pos_support(x_bar_candidate, runtime.support, positivity=cfg.positivity)
        y_data_new = store(state, start_shifted, y_dual_new)

        do_log = (cfg.log_every > 0) & ((t + 1) % cfg.log_every == 0)

        def log_step() -> jnp.ndarray:
            resid = (pred - y_chunk) * row_mask
            if w_chunk is not None:
                resid = resid * jnp.sqrt(w_chunk)
            data_est = (
                0.5
                * jnp.vdot(resid, resid).real
                * (float(n_views) / jnp.maximum(valid.astype(jnp.float32), 1.0))
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
            s_data=s_data_new,
            losses=losses_new,
        ), None

    initial = _initial_spdhg_state(grid, runtime.y_meas, runtime.init_x, iters=cfg.iters)
    final_state, _ = jax.lax.scan(one_step, initial, jnp.arange(cfg.iters))
    return final_state


def spdhg_tv(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
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
    if runtime.y_meas is None:
        with contextlib.ExitStack() as stack:
            keys = [
                stack.enter_context(host_source(np.asarray(projections))),
                stack.enter_context(host_buffer(runtime.data_shape)),
                stack.enter_context(host_source(np.asarray(weights)))
                if weights is not None
                else jnp.int32(-1),
            ]
            runtime = replace(runtime, host_keys=jnp.stack(keys))
            final = _run_spdhg_scan(grid, detector, runtime)
            final.x.block_until_ready()
    else:
        final = _run_spdhg_scan(grid, detector, runtime)
    result = _SPDHGResult(
        volume=final.x,
        losses=final.losses,
        step_sizes=runtime.step_sizes,
        schedule=runtime.schedule,
        regulariser=runtime.regulariser,
        huber_delta=runtime.huber_delta,
        lambda_tv=float(runtime.config.lambda_tv),
    )
    _emit_spdhg_callback(callback, result, runtime.config)
    return result.volume, result.info()
