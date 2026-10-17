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
    chunk_operators,
    resolve_geometry_projector,
    resolve_projector,
)

from ._callbacks import LossCallback, emit_loss_callback_endpoints
from ._data_prox import weighted_l2_conjugate
from ._host_arrays import validate_host_views
from ._host_stream import (
    host_buffer,
    host_source,
    read_block,
    read_views,
    should_stream,
    write_block,
)
from ._solver_validation import validate_spdhg_config, validate_spdhg_weights
from ._spdhg_steps import SPDHGStepSizes, resolve_spdhg_step_sizes
from ._tv_ops import (
    div3,
    grad3,
    huber_tv_value,
    isotropic_tv_value,
    project_tv_duals,
    prox_huber_tv_conj,
    validate_regulariser,
)

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from .types import Regulariser

_SPDHGProjectChunk = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
_SPDHGBackprojectChunk = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


# --------- config ----------


@dataclass(kw_only=True)
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

    ``tau``, ``sigma_data`` and ``sigma_tv`` independently override the primal,
    data-dual and TV-dual step sizes. ``None`` estimates only that missing size;
    explicit sizes must be finite and positive. Automatic sizes bound every
    block's squared norm divided by its selection probability. Batched
    positive-kernel operators use three upper-bound passes per block;
    reference rays use a one-pass row-sum bound per block. ``info["A_norm"]``
    is the square root of this probability-scaled bound, also bounding the
    full-scan operator norm.
    Supplying overrides makes the caller responsible for the combined step
    sizes' stability.
    """

    iterations: int = 400
    tv_weight: float = 5e-3
    regulariser: Regulariser = "tv"
    huber_delta: float = 1e-2
    theta: float = 1.0  # extrapolation of the dual gradient
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
    nonnegative: bool = False
    support: jnp.ndarray | None = None  # 0/1 mask in volume space

    # logging
    log_every: int = 10  # sampled objective at the current primal every k steps

    stream_projections: bool | None = None


jax.tree_util.register_dataclass(
    SPDHGConfig,
    data_fields=["support"],
    meta_fields=[field.name for field in fields(SPDHGConfig) if field.name != "support"],
)


class _SPDHGScanState(NamedTuple):
    x: jnp.ndarray
    y_data: jnp.ndarray
    p1: jnp.ndarray
    p2: jnp.ndarray
    p3: jnp.ndarray
    # Sum of A_i^T y_i over data blocks; the TV part, -div p, is recomputed.
    s_data: jnp.ndarray
    losses: jnp.ndarray


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
    tv_weight: jnp.ndarray
    step_sizes: SPDHGStepSizes
    schedule: _SPDHGSchedule
    # None starts from zeros created inside the compiled solve.
    init_x: jnp.ndarray | None
    # (model, backend) for batched operators, or None for the ray reference path.
    projector: tuple[str | ConeModel, str] | None = field(metadata={"static": True})
    data_shape: tuple[int, int, int] = field(metadata={"static": True})
    weighted: bool = field(default=False, metadata={"static": True})
    # Host keys (measured views, duals, weights) when streaming.
    host_keys: jnp.ndarray | None = None


@dataclass(frozen=True)
class _SPDHGResult:
    volume: jnp.ndarray
    losses: jnp.ndarray
    step_sizes: SPDHGStepSizes
    schedule: _SPDHGSchedule
    regulariser: Regulariser = field(metadata={"static": True})
    huber_delta: float = field(metadata={"static": True})
    tv_weight: float

    def info(self) -> dict[str, object]:
        step_sizes = self.step_sizes
        schedule = self.schedule
        return {
            "loss": np.asarray(self.losses).tolist(),
            "tau": float(step_sizes.tau),
            "sigma_data": float(step_sizes.sigma_data_eff),
            "sigma_data_base": float(step_sizes.sigma_data_base),
            "sigma_tv": float(step_sizes.sigma_tv),
            "tv_weight": float(self.tv_weight),
            "views_per_batch": int(schedule.views_per_batch),
            "num_blocks": int(schedule.num_blocks),
            "A_norm": (float(step_sizes.data_norm) if step_sizes.data_norm is not None else None),
            "grad_norm": float(step_sizes.grad_norm),
            "selection_prob": float(schedule.selection_prob),
            "regulariser": self.regulariser,
            "huber_delta": float(self.huber_delta),
        }


# --------- helpers ----------


def _proj_pos_support(
    x: jnp.ndarray,
    support: jnp.ndarray | None,
    *,
    nonnegative: bool,
) -> jnp.ndarray:
    if support is not None:
        x = x * support
    if nonnegative:
        x = jnp.maximum(x, 0)
    return x


def _prox_fstar_l2(
    u: jnp.ndarray,
    sigma: float | jnp.ndarray,
    y_meas: jnp.ndarray,
    w: jnp.ndarray | None,
) -> jnp.ndarray:
    """Apply the weighted L2 dual proximal.

    Elementwise: if w > 0, return ``(u - sigma * y) * w / (sigma + w)``;
    otherwise return zero for the domain of the conjugate. ``None`` is w = 1.
    """
    sigma = jnp.asarray(sigma, dtype=u.dtype)
    w = jnp.ones_like(u) if w is None else jnp.asarray(w, dtype=u.dtype)
    y_meas = jnp.where(w > 0, y_meas, 0.0)
    return weighted_l2_conjugate(u, sigma, y_meas, w).astype(u.dtype)


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


def _build_spdhg_schedule(n_views: int, config: SPDHGConfig) -> _SPDHGSchedule:
    views_per_batch = int(max(1, min(config.views_per_batch, n_views)))
    num_blocks = (n_views + views_per_batch - 1) // views_per_batch
    rng = np.random.default_rng(config.seed)
    # SPDHG's inverse-probability extrapolation assumes independent draws,
    # not a permutation whose remaining blocks depend on earlier draws.
    block_ids = rng.integers(0, num_blocks, size=config.iterations, dtype=np.int32)

    return _SPDHGSchedule(
        views_per_batch=views_per_batch,
        num_blocks=num_blocks,
        selection_prob=1.0 / float(max(num_blocks, 1)),
        block_ids=jnp.asarray(block_ids),
    )


def _initial_spdhg_state(
    grid: Grid,
    runtime: _SPDHGRuntime,
) -> _SPDHGScanState:
    # Called inside the compiled solve: zero states are never input buffers.
    cfg = runtime.config
    x0 = runtime.init_x
    if x0 is None:
        x0 = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    x0 = _proj_pos_support(x0, runtime.support, nonnegative=cfg.nonnegative)
    tv_zero = jnp.zeros_like(x0) if cfg.tv_weight > 0 else jnp.zeros((0,), jnp.float32)
    return _SPDHGScanState(
        x=x0,
        # Streaming keeps the duals on the host; a placeholder stands in here.
        y_data=(
            jnp.zeros((1, 1, 1), jnp.float32)
            if runtime.y_meas is None
            else jnp.zeros_like(runtime.y_meas)
        ),
        p1=tv_zero,
        p2=tv_zero,
        p3=tv_zero,
        s_data=jnp.zeros_like(x0),
        losses=jnp.zeros((cfg.iterations,), dtype=jnp.float32),
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
    validate_spdhg_config(cfg)
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
    validate_spdhg_weights(weights)
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
    step_sizes = resolve_spdhg_step_sizes(
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
        tv_weight=jnp.asarray(cfg.tv_weight, dtype=jnp.float32),
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
    return list(range(config.log_every - 1, config.iterations, config.log_every))


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


def _make_spdhg_chunk_operators(
    grid: Grid,
    detector: Detector,
    runtime: _SPDHGRuntime,
) -> tuple[_SPDHGProjectChunk, _SPDHGBackprojectChunk]:
    """Slice full-scan coefficients along with poses, retaining each arrangement."""
    cfg = runtime.config
    if runtime.projector is not None:
        model, backend = runtime.projector
        return chunk_operators(
            runtime.poses, grid, detector, backend, runtime.schedule.views_per_batch, model
        )

    def project_chunk(vol: jnp.ndarray, _start: jnp.ndarray, T_chunk: jnp.ndarray) -> jnp.ndarray:
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

    def backproject_chunk(
        images: jnp.ndarray, _start: jnp.ndarray, T_chunk: jnp.ndarray, total: jnp.ndarray
    ) -> jnp.ndarray:
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

    return project_chunk, backproject_chunk


def _spdhg_data_residual(
    pred: jnp.ndarray,
    measured: jnp.ndarray,
    weights: jnp.ndarray | None,
    active: jnp.ndarray,
) -> jnp.ndarray:
    if weights is not None:
        active = active & (weights > 0)
    residual = jnp.where(active, pred, 0.0) - jnp.where(active, measured, 0.0)
    return residual if weights is None else residual * jnp.sqrt(weights)


def _update_spdhg_tv_duals(
    state: _SPDHGScanState, x: jnp.ndarray, runtime: _SPDHGRuntime
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    step_sizes = runtime.step_sizes
    gx, gy, gz = grad3(x)
    p1_u = state.p1 + step_sizes.sigma_tv * gx
    p2_u = state.p2 + step_sizes.sigma_tv * gy
    p3_u = state.p3 + step_sizes.sigma_tv * gz
    if runtime.regulariser == "huber_tv":
        p1_new, p2_new, p3_new = prox_huber_tv_conj(
            p1_u,
            p2_u,
            p3_u,
            sigma=step_sizes.sigma_tv,
            lam=runtime.tv_weight,
            delta=runtime.huber_delta,
        )
    else:
        p1_new, p2_new, p3_new = project_tv_duals(p1_u, p2_u, p3_u, radius=runtime.tv_weight)
    return p1_new, p2_new, p3_new


def _extrapolate_spdhg_gradient(
    state: _SPDHGScanState,
    runtime: _SPDHGRuntime,
    x: jnp.ndarray,
    s_data_new: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update deterministic TV and extrapolate each dual increment by its probability."""
    cfg = runtime.config
    gradient_bar = s_data_new + (cfg.theta / runtime.schedule.selection_prob) * (
        s_data_new - state.s_data
    )
    if cfg.tv_weight == 0:
        return state.p1, state.p2, state.p3, gradient_bar
    p1, p2, p3 = _update_spdhg_tv_duals(state, x, runtime)
    tv_adjoint = -div3(p1, p2, p3)
    # TV is selected on every iteration, so its probability is one.
    gradient_bar += tv_adjoint + cfg.theta * (tv_adjoint + div3(state.p1, state.p2, state.p3))
    return p1, p2, p3, gradient_bar


def _spdhg_primal_successor(
    x: jnp.ndarray,
    gradient: jnp.ndarray,
    step: jnp.ndarray,
    runtime: _SPDHGRuntime,
) -> jnp.ndarray:
    """Prepare the next primal, retaining the projected image on the terminal step."""
    candidate = _proj_pos_support(
        x - runtime.step_sizes.tau * gradient,
        runtime.support,
        nonnegative=runtime.config.nonnegative,
    )
    return jnp.where(step + 1 < runtime.config.iterations, candidate, x)


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
    project_chunk, backproject_chunk = _make_spdhg_chunk_operators(grid, detector, runtime)

    def load(state: _SPDHGScanState, start: jnp.ndarray) -> tuple:
        # Measured views, weights and duals of one block, from host or device.
        if keys is not None:
            weights = read_views(keys[2], start, block_shape) if runtime.weighted else None
            return (
                read_views(
                    keys[0], start, block_shape, weights_key=keys[2] if runtime.weighted else None
                ),
                weights,
                read_block(keys[1], start, block_shape),
            )
        weights = None
        if runtime.weights is not None:
            weights = jax.lax.dynamic_slice_in_dim(runtime.weights, start, block_shape[0])
        assert runtime.y_meas is not None
        return (
            jax.lax.dynamic_slice_in_dim(runtime.y_meas, start, block_shape[0]),
            weights,
            jax.lax.dynamic_slice_in_dim(state.y_data, start, block_shape[0]),
        )

    def store(state: _SPDHGScanState, start: jnp.ndarray, duals: jnp.ndarray) -> jnp.ndarray:
        if keys is not None:
            write_block(keys[1], start, duals)
            return state.y_data
        return jax.lax.dynamic_update_slice_in_dim(state.y_data, duals, start, axis=0)

    def one_step(state: _SPDHGScanState, t: jnp.ndarray) -> tuple[_SPDHGScanState, None]:
        block = schedule.block_ids[t]
        start = block * jnp.int32(schedule.views_per_batch)
        remaining = jnp.maximum(0, jnp.int32(n_views) - start)
        valid = jnp.minimum(jnp.int32(schedule.views_per_batch), remaining)
        start_shifted = jnp.maximum(0, start - (jnp.int32(schedule.views_per_batch) - valid))

        T_chunk = jax.lax.dynamic_slice_in_dim(
            runtime.poses, start_shifted, schedule.views_per_batch
        )
        y_chunk, w_chunk, y_dual_old = load(state, start_shifted)

        idx = jnp.arange(schedule.views_per_batch)
        row_mask = (idx >= (jnp.int32(schedule.views_per_batch) - valid))[:, None, None]
        row_mask = row_mask.astype(jnp.float32)

        # The state holds the primal for this projection. Its successor is formed
        # after the duals update, so the extrapolated gradient and the primal
        # proximal share one pointwise program and no volume-sized buffer is carried.
        sigma_eff = jnp.asarray(step_sizes.sigma_data_eff, dtype=state.x.dtype)
        pred = project_chunk(state.x, start_shifted, T_chunk)
        u = y_dual_old + sigma_eff * pred
        y_dual_new = _prox_fstar_l2(u, sigma_eff, y_chunk, w_chunk)
        y_dual_new = row_mask * y_dual_new + (1.0 - row_mask) * y_dual_old
        delta_y = (y_dual_new - y_dual_old) * row_mask

        s_data_new = backproject_chunk(delta_y, start_shifted, T_chunk, state.s_data)

        p1_new, p2_new, p3_new, gradient_bar_new = _extrapolate_spdhg_gradient(
            state, runtime, state.x, s_data_new
        )
        x_new = _spdhg_primal_successor(state.x, gradient_bar_new, t, runtime)
        y_data_new = store(state, start_shifted, y_dual_new)

        do_log = (cfg.log_every > 0) & ((t + 1) % cfg.log_every == 0)

        def log_step() -> jnp.ndarray:
            resid = _spdhg_data_residual(pred, y_chunk, w_chunk, row_mask != 0)
            # Blocks are selected uniformly, not in proportion to their view
            # counts. The final partial block has the same selection weight.
            data_est = 0.5 * jnp.vdot(resid, resid).real / schedule.selection_prob
            objective = data_est
            if cfg.tv_weight != 0:
                if runtime.regulariser == "huber_tv":
                    reg_value = huber_tv_value(state.x, runtime.huber_delta)
                else:
                    reg_value = isotropic_tv_value(state.x)
                objective = objective + runtime.tv_weight * reg_value
            obj = objective.astype(jnp.float32)
            return state.losses.at[t].set(obj)

        losses_new = jax.lax.cond(do_log, log_step, lambda: state.losses)
        return _SPDHGScanState(
            x=x_new,
            y_data=y_data_new,
            p1=p1_new,
            p2=p2_new,
            p3=p3_new,
            s_data=s_data_new,
            losses=losses_new,
        ), None

    initial = _initial_spdhg_state(grid, runtime)
    if cfg.iterations == 0:
        return initial
    return jax.lax.scan(one_step, initial, jnp.arange(cfg.iterations))[0]


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

    ``weights`` must be real, nonnegative and finite when stored in FP32;
    zero weights exclude unmeasured samples, including nonfinite measurements.
    Invalid weights fail before solver setup.
    Boolean/integer masks, standard NumPy floating dtypes and bfloat16 are supported.

    The supplied ``init_x`` obeys the configured positivity and support map
    before the first projection, including when ``iterations=0``.

    Logged losses estimate the full objective at the primal image used
    for that iteration's data projection, with inverse block-selection weights.
    They are samples, not full-scan objectives of the returned primal image.

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
        y_host = np.asarray(projections)
        w_host = None if weights is None else np.asarray(weights)
        validate_host_views(y_host, weights=w_host)
        with contextlib.ExitStack() as stack:
            keys = [
                stack.enter_context(host_source(y_host)),
                stack.enter_context(host_buffer(runtime.data_shape)),
                stack.enter_context(host_source(w_host)) if w_host is not None else jnp.int32(-1),
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
        tv_weight=float(runtime.config.tv_weight),
    )
    _emit_spdhg_callback(callback, result, runtime.config)
    return result.volume, result.info()
