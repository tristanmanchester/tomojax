"""SPDHG/TV reconstruction routine."""

from collections.abc import Callable
import contextlib
from dataclasses import dataclass, field, fields, replace
import functools
import math
from typing import ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import progress
from tomojax._parameters import ProjectorBackend, check_backend
from tomojax.backends import device_free_memory_bytes
from tomojax.backends.host_stream import (
    host_buffer,
    host_source,
    read_block,
    read_views,
    should_stream,
    write_block,
)
from tomojax.core.devices import announce
from tomojax.core.geometry.base import Detector, Geometry, Grid
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import (
    validate_optional_broadcastable_shape,
    validate_optional_same_shape,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)
from tomojax.recon._projection import (
    resolve_geometry_projector,
)
from tomojax.recon._volume_storage import preview_planes
from tomojax.wire.metrics import Metric

from ._chunks import Chunked, fixed_updates
from ._data_term import project_residual
from ._host_arrays import validate_host_views
from ._noise import detector_error_scale
from ._noise_probe import probe_gradient_rms
from ._solver_validation import validate_spdhg_config, validate_spdhg_weights
from ._spdhg_steps import SPDHGStepSizes, make_spdhg_chunk_operators, resolve_spdhg_step_sizes
from ._tv_metric import TVMetric, resolve_tv_metric, voxel_noise_weight
from ._tv_ops import (
    backward_difference,
    forward_difference,
    huber_tv_value,
    isotropic_tv_value,
    project_tv_duals,
    prox_huber_tv_conj,
    validate_regularizer,
)
from ._volume_storage import Model, pack_volume, resident_model, restore_volume, volume_shape
from .types import DataTerm, Regularizer

# --------- config ----------


@dataclass(kw_only=True)
class SPDHGConfig:
    """Configuration for stochastic primal-dual TV reconstruction.

    ``backend`` chooses where the projection runs, as in :class:`FISTAConfig`:
    ``"auto"`` is CUDA on CUDA.

    ``stream`` keeps NumPy or memmap projections, any weights and
    the sinogram-sized dual variable in host memory; each iteration reads and
    writes only its block of views. ``None`` streams when the projections would
    take more than 40% of free device memory. Streaming needs the batched
    operators.

    ``tau``, ``sigma_data`` and ``sigma_tv`` independently override the primal,
    data-dual and TV-dual step sizes. ``None`` estimates only that missing size;
    explicit sizes must be finite and positive. Automatic sizes bound every
    block's squared norm divided by its selection probability. One forward
    and adjoint pass per block compute row/column-sum Schur bounds. ``info["A_norm"]``
    is the square root of this probability-scaled bound, also bounding the
    full-scan operator norm.
    Supplying overrides makes the caller responsible for the combined step
    sizes' stability.

    Each update samples one data block. For a full-data solve the default budget
    is 200. For B equal-probability blocks it grows as the square root of B.
    The effective count and step sizes are recorded with the reconstruction.

    Parameters
    ----------
    iterations : int or None, optional
        Number of sampled data-block updates, each with a TV update. None uses
        ceil(200 * sqrt(number of blocks)). An explicit count fixes the budget.
    tv_weight : float | None, optional
        Dimensionless physical-TV strength. None estimates the score of detector
        noise from the measured pixels and calibrates the penalty per isolated
        voxel against that score. An explicit zero disables TV.
    data_term : {"absolute", "per_view_contrast"}, optional
        Absolute least squares by default. Use per_view_contrast when material
        lies outside the field of view (ROI scans, most laminography, K11).
        It discards each view's baseline and gives up the absolute density level.
        Keep absolute fitting for contained objects: removing their view means
        discards measured information and can worsen the reconstruction.
    regularizer : Regularizer, optional
        Regularization model: "tv" or "huber_tv". Default is 'tv'.
    huber_delta : float, optional
        Positive dimensionless Huber transition scale when regularizer is "huber_tv". Default is
        0.01.
    theta : float, optional
        Extrapolation factor for the dual gradient. Default is 1.0.
    views_per_block : int, optional
        Number of views per stochastic update block; changing this changes the
        stochastic algorithm as well as temporary storage. Default is 16.
    seed : int, optional
        Random seed for stochastic block selection. Default is 0.
    tau : float | None, optional
        Positive primal step size, or None to estimate it.
    sigma_data : float | None, optional
        Positive data-dual step size, or None to estimate it.
    sigma_tv : float | None, optional
        Positive TV-dual step size, or None to estimate it.
    backend : ProjectorBackend, optional
        Execution backend: "auto", "jax" or "cuda". An explicit accelerator
        request never silently falls back. Default is 'auto'.
    lower_bound : float | None, optional
        Lower voxel-value bound (set ``0.0`` for nonnegativity), or None for no
        lower bound.
    support : array-like | None, optional
        Boolean volume support mask, broadcastable to the grid. False (zero)
        voxels remain zero; None uses the entire grid.
    log_every : int, optional
        Number of iterations between logged objective samples. Default is 10.
    stream : bool | None, optional
        Read host projections one view batch at a time. None selects streaming from
        available device memory; False keeps them resident.
    """

    method: ClassVar[Literal["spdhg"]] = "spdhg"

    iterations: int | None = None
    tv_weight: float | None = None
    data_term: DataTerm = "absolute"
    regularizer: Regularizer = "tv"
    huber_delta: float = 1e-2
    theta: float = 1.0  # extrapolation of the dual gradient
    views_per_block: int = 16  # size of a stochastic block
    seed: int = 0

    # Step sizes (None resolves the probability-aware physical Schur sizes).
    tau: float | None = None
    sigma_data: float | None = None
    sigma_tv: float | None = None

    # projector / memory knobs
    backend: ProjectorBackend = "auto"

    # constraints
    lower_bound: float | None = None
    support: jnp.ndarray | None = None  # boolean mask in volume space

    # logging
    log_every: int = 10  # sampled objective at the current primal every k steps

    stream: bool | None = None

    def __post_init__(self) -> None:
        check_backend(self.backend, "spdhg")
        validate_spdhg_config(self)
        _ = validate_regularizer(self.regularizer, self.huber_delta, context="spdhg_tv config")


_ = jax.tree_util.register_dataclass(
    SPDHGConfig,
    data_fields=["support"],
    meta_fields=[field.name for field in fields(SPDHGConfig) if field.name != "support"],
)


class _SPDHGScanState(NamedTuple):
    x: jnp.ndarray
    y_data: jnp.ndarray
    # The TV dual's three components, empty when TV is off.
    p: tuple[jax.Array, ...]
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
    regularizer: Regularizer = field(metadata={"static": True})
    huber_delta: float
    # None when streaming: data, weights and duals then live in host memory.
    y_meas: jnp.ndarray | None
    # None means unit weights; no sinogram of ones is stored.
    weights: jnp.ndarray | None
    poses: jnp.ndarray
    detector_grid: tuple[jnp.ndarray, jnp.ndarray] | None
    support: jnp.ndarray | None
    metric: TVMetric
    resolved_tv_weight: float
    step_sizes: SPDHGStepSizes
    schedule: _SPDHGSchedule
    # None starts from zeros created inside the compiled solve.
    init_x: jnp.ndarray | None
    # (model, backend) of the batched operators.
    projector: tuple[Model, str] = field(metadata={"static": True})
    data_shape: tuple[int, int, int] = field(metadata={"static": True})
    weighted: bool = field(default=False, metadata={"static": True})
    # Host keys (measured views, duals, weights) when streaming.
    host_keys: jnp.ndarray | None = None

    @property
    def storage_scales(self) -> tuple[float, float, float]:
        """Metric coefficients along fast, middle and slow storage axes."""
        scales = self.metric.gradient_scale_xyz
        return (scales[2], scales[1], scales[0]) if self.projector[1] == "cuda" else scales


@dataclass(frozen=True)
class _SPDHGResult:
    volume: jnp.ndarray
    losses: jnp.ndarray
    step_sizes: SPDHGStepSizes
    schedule: _SPDHGSchedule
    regularizer: Regularizer = field(metadata={"static": True})
    huber_delta: float
    tv_weight: float | None
    resolved_tv_weight: float
    effective_tv_weight: float

    def info(self) -> dict[str, object]:
        step_sizes = self.step_sizes
        schedule = self.schedule
        return {
            "loss": np.asarray(self.losses).tolist(),
            "effective_iterations": len(self.losses),
            "termination": "iteration_limit",
            "early_stop": False,
            "tau": float(step_sizes.tau),
            "sigma_data": float(step_sizes.sigma_data),
            "sigma_tv": float(step_sizes.sigma_tv),
            "tv_weight": self.tv_weight,
            "resolved_tv_weight": self.resolved_tv_weight,
            "effective_tv_weight": self.effective_tv_weight,
            "views_per_batch": int(schedule.views_per_batch),
            "num_blocks": int(schedule.num_blocks),
            "A_norm": (float(step_sizes.data_norm) if step_sizes.data_norm is not None else None),
            "grad_norm": float(step_sizes.grad_norm),
            "selection_prob": float(schedule.selection_prob),
            "regularizer": self.regularizer,
            "huber_delta": float(self.huber_delta),
        }


# --------- helpers ----------


def _proj_pos_support(
    x: jnp.ndarray,
    support: jnp.ndarray | None,
    *,
    lower_bound: float | None,
) -> jnp.ndarray:
    if lower_bound is not None:
        x = jnp.maximum(x, jnp.asarray(lower_bound, x.dtype))
    if support is not None:
        x = jnp.where(support, x, 0.0)
    return x


def _prox_fstar_l2(
    u: jnp.ndarray,
    sigma: float | jnp.ndarray,
    y_meas: jnp.ndarray,
    w: jnp.ndarray | None,
    data_term: DataTerm = "absolute",
) -> jnp.ndarray:
    """Apply the weighted L2 dual proximal.

    Elementwise: if w > 0, return ``(u - sigma * y) * w / (sigma + w)``;
    otherwise return zero for the domain of the conjugate. ``None`` is w = 1.
    """
    sigma = jnp.asarray(sigma, dtype=u.dtype)
    w = jnp.ones_like(u) if w is None else jnp.asarray(w, dtype=u.dtype)
    y_meas = jnp.where(w > 0, y_meas, 0.0)
    dual = jnp.where(w > 0, (u - sigma * y_meas) * (w / (sigma + w)), 0.0)
    if data_term == "per_view_contrast":
        # The profiled conjugate is restricted to sum(dual)=0 per view.
        # Its metric is sigma*w/(sigma+w), not the observation weights.
        gain = jnp.where(w > 0, sigma * (w / (sigma + w)), 0)
        total = jnp.sum(gain, axis=(-2, -1), keepdims=True)
        dual -= gain * jnp.sum(dual, axis=(-2, -1), keepdims=True) / jnp.where(total > 0, total, 1)
    return dual.astype(u.dtype)


def _batched_projector(
    config: SPDHGConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    geometry: object,
    detector: Detector,
) -> tuple[Model, str]:
    """The batched operators and their volume layout."""
    model, backend = resolve_geometry_projector(
        geometry, config.backend, detector=detector, det_grid=det_grid, context="spdhg_tv"
    )
    return resident_model(model, backend), backend


def _build_spdhg_schedule(n_views: int, config: SPDHGConfig) -> _SPDHGSchedule:
    views_per_batch = int(max(1, min(config.views_per_block, n_views)))
    num_blocks = (n_views + views_per_batch - 1) // views_per_batch
    rng = np.random.default_rng(config.seed)
    # SPDHG's inverse-probability extrapolation assumes independent draws,
    # not a permutation whose remaining blocks depend on earlier draws.
    count = (
        config.iterations
        if config.iterations is not None
        else math.ceil(200 * math.sqrt(num_blocks))
    )
    block_ids = rng.integers(0, num_blocks, size=count, dtype=np.int32)

    return _SPDHGSchedule(
        views_per_batch=views_per_batch,
        num_blocks=num_blocks,
        selection_prob=1.0 / float(max(num_blocks, 1)),
        block_ids=jnp.asarray(block_ids),
    )


@functools.partial(jax.jit, static_argnames=("shape",))
def _initial_spdhg_state(runtime: _SPDHGRuntime, shape: tuple[int, int, int]) -> _SPDHGScanState:
    cfg = runtime.config
    zeros = jnp.zeros(shape, jnp.float32)
    x0 = zeros if runtime.init_x is None else runtime.init_x
    x0 = _proj_pos_support(x0, runtime.support, lower_bound=cfg.lower_bound)
    return _SPDHGScanState(
        x=x0,
        # Streaming keeps the duals on the host; a placeholder stands in here.
        y_data=(
            jnp.zeros((1, 1, 1), jnp.float32)
            if runtime.y_meas is None
            else jnp.zeros_like(runtime.y_meas)
        ),
        p=tuple(jnp.zeros(shape if cfg.tv_weight != 0 else (0,), jnp.float32) for _ in range(3)),
        s_data=zeros,
        losses=jnp.zeros((runtime.schedule.block_ids.shape[0],), dtype=jnp.float32),
    )


@functools.partial(
    jax.jit,
    static_argnames=("grid", "detector", "projector", "views_per_batch"),
)
def _spdhg_noise_rms(
    grid: Grid,
    detector: Detector,
    poses: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    weights: jnp.ndarray | None,
    weights_key: jax.Array | None,
    support: jnp.ndarray | None,
    *,
    projector: tuple[Model, str],
    views_per_batch: int,
) -> jnp.ndarray:
    """A bounded noise transpose through the same weighted data operator."""
    _, backproject = make_spdhg_chunk_operators(
        grid,
        detector,
        poses,
        det_grid,
        projector=projector,
        views_per_batch=views_per_batch,
    )

    def read_weight_block(start: jax.Array, size: int) -> jax.Array:
        if weights_key is not None:
            return read_views(weights_key, start, (size, detector.nv, detector.nu))
        assert weights is not None
        return jax.lax.dynamic_slice_in_dim(weights, start, size)

    return probe_gradient_rms(
        poses,
        grid,
        detector,
        backproject,
        batch_size=views_per_batch,
        shape=volume_shape(projector[0], grid),
        support=support,
        read_weights=read_weight_block if weights_key is not None or weights is not None else None,
    )


def _stream_data(
    projections: jax.Array | np.ndarray,
    weights: jax.Array | None,
    cfg: SPDHGConfig,
    shape: tuple[int, int, int],
) -> bool:
    """Place measurements and data duals after accounting for the volume state."""
    # The iterate, the accumulated backprojection and, with TV, the dual's three components.
    state_bytes = 4 * (2 + 3 * (cfg.tv_weight != 0)) * math.prod(shape)
    free = device_free_memory_bytes()
    if free is not None and state_bytes > free:
        raise MemoryError(
            f"spdhg needs {state_bytes / 2**30:.1f} GiB of volume state for this "
            f"{math.prod(shape):,}-voxel grid, with {free / 2**30:.1f} GiB free. "
            "fista, cgls and sirt keep large volumes on the host; or bin the scan or "
            "reconstruct a smaller region"
        )
    # Leave room for the recurrence, slab workspace and a batch of projector scratch.
    reserve, copies = state_bytes + 4 * math.prod(shape), 2 + (weights is not None)
    return not isinstance(projections, jax.Array) and (
        bool(cfg.stream)
        if cfg.stream is not None
        else should_stream(projections, reserve_bytes=reserve, copies=copies)
    )


def _prepare_spdhg_runtime(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    weights: jnp.ndarray | None,
    init_x: jnp.ndarray | None,
    config: SPDHGConfig | None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> _SPDHGRuntime:
    cfg = SPDHGConfig() if config is None else config
    validate_spdhg_config(cfg)
    regularizer = validate_regularizer(cfg.regularizer, cfg.huber_delta, context="spdhg_tv config")

    expected_proj_shape = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="spdhg_tv projections",
    )
    n_views, nv, nu = expected_proj_shape
    _ = validate_optional_same_shape(
        weights,
        expected_proj_shape,
        context="spdhg_tv weights",
        name="weights",
        fix="use weights with the same shape as projections.",
    )
    validate_spdhg_weights(weights)
    _ = validate_optional_broadcastable_shape(
        cfg.support,
        (grid.nz, grid.ny, grid.nx),
        context="spdhg_tv support",
        name="support",
        fix="use a support mask broadcastable to shape (grid.nz, grid.ny, grid.nx).",
    )
    if init_x is not None:
        _ = validate_volume(init_x, grid, context="spdhg_tv init_x", name="init_x")

    announce(array=projections)
    poses = stack_view_poses(geometry, n_views)
    _ = validate_pose_stack(poses, n_views, context="spdhg_tv geometry")
    projector = _batched_projector(cfg, det_grid, geometry, detector)
    model, _ = projector
    stream = _stream_data(projections, weights, cfg, volume_shape(model, grid))
    y_meas = None if stream else jnp.asarray(projections, dtype=jnp.float32)
    weights_arr = None if weights is None or stream else jnp.asarray(weights, dtype=jnp.float32)
    metric = resolve_tv_metric(
        geometry,
        grid,
        detector,
        projections,
        weights=weights,
        tv_weight=1.0 if cfg.tv_weight is None else cfg.tv_weight,
        huber_delta=cfg.huber_delta,
    )
    support = None if cfg.support is None else jnp.asarray(cfg.support, dtype=bool)
    if support is not None:
        support = pack_volume(support.reshape((1,) * (3 - support.ndim) + support.shape), model)
    resolved_tv_weight = cfg.tv_weight
    if resolved_tv_weight is None:
        initial_factor = float(metric.weight_over_contrast)
        rms = 0.0
        if nv * nu > 1:
            source = (
                host_source(np.asarray(weights))
                if stream and weights is not None
                else contextlib.nullcontext(None)
            )
            with source as key:
                rms = float(
                    _spdhg_noise_rms(
                        grid,
                        detector,
                        poses,
                        det_grid,
                        weights_arr,
                        key,
                        None if support is None else support.astype(jnp.float32),
                        projector=projector,
                        views_per_batch=cfg.views_per_block,
                    )
                ) * float(detector_error_scale(projections, weights=weights))
        metric = voxel_noise_weight(metric, rms, grid.shape, cfg.support)
        resolved_tv_weight = float(metric.weight_over_contrast) / (initial_factor or 1)
    # A zero TV weight is identically zero, including a flat input's zero delta.
    huber_delta = float(metric.delta) if float(metric.weight) > 0 else 1.0
    step_sizes = resolve_spdhg_step_sizes(
        grid, detector, poses, cfg, det_grid, projector, metric=metric
    )

    return _SPDHGRuntime(
        config=cfg,
        regularizer=regularizer,
        huber_delta=huber_delta,
        y_meas=y_meas,
        weights=weights_arr,
        poses=poses,
        detector_grid=det_grid,
        support=support,
        metric=metric,
        resolved_tv_weight=float(resolved_tv_weight),
        step_sizes=step_sizes,
        schedule=_build_spdhg_schedule(n_views, cfg),
        init_x=pack_volume(None if init_x is None else jnp.asarray(init_x, jnp.float32), model),
        projector=projector,
        data_shape=expected_proj_shape,
        weighted=weights is not None,
    )


# --------- main algorithm ----------


def _spdhg_data_residual(
    pred: jnp.ndarray,
    measured: jnp.ndarray,
    weights: jnp.ndarray | None,
    active: jnp.ndarray,
    data_term: DataTerm,
) -> jnp.ndarray:
    if weights is not None:
        active = active & (weights > 0)
    residual = jnp.where(active, pred, 0.0) - jnp.where(active, measured, 0.0)
    residual = project_residual(residual, data_term, weights=weights)
    return residual if weights is None else residual * jnp.sqrt(weights)


def _tv_dual_step(
    p: tuple[jnp.ndarray, ...],
    gradient: tuple[jnp.ndarray, ...],
    runtime: _SPDHGRuntime,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The TV duals' ascent and proximal step, pointwise in the duals and the gradient."""
    step_sizes = runtime.step_sizes
    p1_u, p2_u, p3_u = (q + step_sizes.sigma_tv * g for q, g in zip(p, gradient, strict=True))
    if runtime.regularizer == "huber_tv":
        return prox_huber_tv_conj(
            p1_u,
            p2_u,
            p3_u,
            sigma=step_sizes.sigma_tv,
            lam=runtime.metric.weight,
            delta=runtime.huber_delta,
        )
    return project_tv_duals(p1_u, p2_u, p3_u, radius=runtime.metric.weight)


def _sweep(
    state: _SPDHGScanState, runtime: _SPDHGRuntime, iteration: jax.Array
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Overwrite each plane after its forward differences and outgoing flux are known.

    Carry the old and new normal flux across slab boundaries. Fold the old
    data-adjoint term into the primal here so its only remaining use is as the
    in-place initializer of the next transpose.
    """
    cfg, x = runtime.config, state.x
    nz, ny, nx = x.shape
    tv = cfg.tv_weight != 0
    # Bound intermediate fields to a MiB each or one complete plane.
    depth = min(max(1, 2**18 // (ny * nx)), nz)
    full = (nz - 1) // depth  # slabs whose next plane exists
    cx, cy, cz = runtime.storage_scales

    def divergence(p: tuple[jax.Array, ...], below: jnp.ndarray) -> jnp.ndarray:
        # A Neumann edge starts at zero. Its forward difference is always zero,
        # and the isotropic proximal scales it, so the outer normal flux stays zero.
        z = p[2] - jnp.concatenate([below, p[2][:-1]])
        return cx * backward_difference(p[0], 2) + cy * backward_difference(p[1], 1) + cz * z

    def update(carry: tuple, start: jax.Array | int, size: int, after: jnp.ndarray) -> tuple:
        p, x, below_old, below_new = carry
        u = jax.lax.dynamic_slice_in_dim(x, start, size)
        s_old = jax.lax.dynamic_slice_in_dim(state.s_data, start, size)
        gradient_bar = -(cfg.theta / runtime.schedule.selection_prob) * s_old
        if tv:
            old = tuple(jax.lax.dynamic_slice_in_dim(q, start, size) for q in p)
            previous = divergence(old, below_old)
            gradient = (
                cx * forward_difference(u, 2),
                cy * forward_difference(u, 1),
                cz * (after - u),
            )
            new = _tv_dual_step(old, gradient, runtime)
            tv_adjoint = -divergence(new, below_new)
            # TV is selected on every iteration, so its probability is one.
            gradient_bar += tv_adjoint + cfg.theta * (tv_adjoint + previous)
            p = tuple(
                jax.lax.dynamic_update_slice_in_dim(q, n, start, 0)
                for q, n in zip(p, new, strict=True)
            )
            below_old, below_new = old[2][-1:], new[2][-1:]
        candidate = u - runtime.step_sizes.tau * gradient_bar
        kept = jnp.where(iteration < runtime.schedule.block_ids.size, candidate, u)
        return p, jax.lax.dynamic_update_slice_in_dim(x, kept, start, 0), below_old, below_new

    def slab(i: jax.Array, carry: tuple) -> tuple:
        start = i * depth
        return update(carry, start, depth, jax.lax.dynamic_slice_in_dim(carry[1], start + 1, depth))

    edge = jnp.zeros((1, ny, nx), x.dtype)
    carry = (state.p, x, edge, edge)
    carry = jax.lax.fori_loop(0, full, slab, carry)
    # The remaining planes; the last is its own successor, a zero z difference (Neumann).
    start = full * depth
    after = jnp.concatenate([carry[1][start + 1 :], carry[1][-1:]])
    p, x, _, _ = update(carry, start, nz - start, after)
    return p, x


def _sampled_loss(
    runtime: _SPDHGRuntime,
    x: jax.Array,
    pred: jax.Array,
    data: jax.Array,
    weights: jax.Array | None,
    mask: jax.Array,
) -> jax.Array:
    """Unbiased data-block objective at the current primal, before its update."""
    residual = _spdhg_data_residual(pred, data, weights, mask, runtime.config.data_term)
    objective = 0.5 * jnp.vdot(residual, residual).real / runtime.schedule.selection_prob
    if runtime.config.tv_weight != 0:
        scales = runtime.storage_scales
        regularizer = (
            huber_tv_value(x, runtime.huber_delta, gradient_scale_xyz=scales)
            if runtime.regularizer == "huber_tv"
            else isotropic_tv_value(x, gradient_scale_xyz=scales)
        )
        objective = objective + runtime.metric.weight * regularizer
    return objective.astype(jnp.float32)


@functools.partial(jax.jit, static_argnames=("grid", "detector"), donate_argnames=("state",))
def _advance_spdhg(
    grid: Grid,
    detector: Detector,
    runtime: _SPDHGRuntime,
    state: _SPDHGScanState,
    first: int,
    end: int,
) -> _SPDHGScanState:
    cfg, schedule, step_sizes = runtime.config, runtime.schedule, runtime.step_sizes
    n_views, nv, nu = runtime.data_shape
    block_shape = (schedule.views_per_batch, nv, nu)
    keys = runtime.host_keys
    project_chunk, backproject_chunk = make_spdhg_chunk_operators(
        grid,
        detector,
        runtime.poses,
        runtime.detector_grid,
        projector=runtime.projector,
        views_per_batch=schedule.views_per_batch,
    )

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
            _ = write_block(keys[1], start, duals)
            return state.y_data
        return jax.lax.dynamic_update_slice_in_dim(state.y_data, duals, start, axis=0)

    def one_step(iteration: jnp.ndarray, state: _SPDHGScanState) -> _SPDHGScanState:
        size = schedule.views_per_batch
        start = schedule.block_ids[iteration - 1] * size
        start_shifted = jnp.minimum(start, n_views - size)

        T_chunk = jax.lax.dynamic_slice_in_dim(
            runtime.poses, start_shifted, schedule.views_per_batch
        )
        y_chunk, w_chunk, y_dual_old = load(state, start_shifted)

        row_mask = (start_shifted + jnp.arange(size) >= start)[:, None, None]

        sigma_eff = jnp.asarray(step_sizes.sigma_data, dtype=state.x.dtype)
        pred = project_chunk(state.x, start_shifted, T_chunk)
        u = y_dual_old + sigma_eff * pred
        y_dual_new = _prox_fstar_l2(u, sigma_eff, y_chunk, w_chunk, cfg.data_term)
        # Round the stored dual before taking its increment in either placement.
        y_dual_new = jax.lax.optimization_barrier(jnp.where(row_mask, y_dual_new, y_dual_old))
        delta_y = (y_dual_new - y_dual_old) * row_mask

        y_data_new = store(state, start_shifted, y_dual_new)

        do_log = (cfg.log_every > 0) & (iteration % cfg.log_every == 0)

        def log_step() -> jnp.ndarray:
            obj = _sampled_loss(runtime, state.x, pred, y_chunk, w_chunk, row_mask)
            return state.losses.at[iteration - 1].set(obj)

        # The objective at the current primal, before the sweep overwrites it.
        losses_new = jax.lax.cond(do_log, log_step, lambda: state.losses)
        p, partial = _sweep(state, runtime, iteration)
        # Consume the old accumulator before the matched transpose overwrites it.
        partial, acc = jax.lax.optimization_barrier((partial, state.s_data))
        s_data_new = backproject_chunk(delta_y, start_shifted, T_chunk, acc)
        candidate = _proj_pos_support(
            partial - (1 + cfg.theta / schedule.selection_prob) * step_sizes.tau * s_data_new,
            runtime.support,
            lower_bound=cfg.lower_bound,
        )
        x_new = jnp.where(iteration < runtime.schedule.block_ids.size, candidate, partial)
        return _SPDHGScanState(
            x=x_new,
            y_data=y_data_new,
            p=p,
            s_data=s_data_new,
            losses=losses_new,
        )

    # Count completed updates directly: the sampled loss and final primal both use this
    # count. Keeping it as the loop induction variable lets XLA overwrite x in the sweep.
    return jax.lax.fori_loop(first + 1, end + 1, one_step, state)


def _spdhg_program(
    grid: Grid, detector: Detector, runtime: _SPDHGRuntime, *, drive: Callable
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...], jax.Array]:
    cfg = runtime.config
    initial = _initial_spdhg_state(runtime, volume_shape(runtime.projector[0], grid))
    count = runtime.schedule.block_ids.size

    def metrics(state: _SPDHGScanState, done: int) -> dict[str, float | None]:
        if cfg.log_every and done and done % cfg.log_every == 0:
            return {Metric.loss.name: float(state.losses[done - 1])}
        return {Metric.loss.name: None}

    advance = functools.partial(_advance_spdhg, grid, detector, runtime)
    final = (
        fixed_updates(
            initial,
            count,
            advance,
            drive,
            lambda s: preview_planes(s.x, runtime.projector[0], grid),
            metrics,
        )
        if count
        else initial
    )
    # Return every large carry: a carry left out of the outputs becomes a temporary that
    # XLA keeps beside the loop's own copy, one more volume or data stack at the peak.
    return final.x, final.losses, final.p, final.y_data


_run_spdhg_scan = Chunked(_spdhg_program, static_argnames=("grid", "detector"))


@progress.reports_iterations(200)
def spdhg(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    weights: jnp.ndarray | None = None,  # same shape as projections; 0 for unmeasured
    init_x: jnp.ndarray | None = None,
    config: SPDHGConfig | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, dict[str, object]]:
    """Reconstruct with stochastic primal-dual weighted L2 and TV updates.

    ``weights`` must be real, nonnegative and finite when stored in FP32;
    zero weights exclude unmeasured samples, including nonfinite measurements.
    Invalid weights fail before solver setup.
    Boolean/integer masks, standard NumPy floating dtypes and bfloat16 are supported.

    The supplied ``init_x`` obeys the configured positivity and support map
    before the first projection, including when ``iterations=0``.

    Logged losses estimate the full objective at the primal image used
    for that iteration's data projection, with inverse block-selection weights.
    They are samples, not full-scan objectives of the returned primal image.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry, including view angles and any per-view poses.
    grid : Grid
        Reconstruction grid in ``(x, y, z)`` order, with physical voxel spacing.
    detector : Detector
        Detector dimensions and placement in the same length unit as the grid.
    projections : array-like
        Measured line integrals with shape ``(views, detector.nv, detector.nu)``.
    weights : array-like or None, optional
        Real, nonnegative weights matching the projection shape; zero excludes
        unmeasured samples. None weights all samples equally.
    init_x : array-like or None, optional
        Initial volume of shape ``(nz, ny, nx)``; None starts from zero.
    config : SPDHGConfig or None, optional
        Solver settings. None uses SPDHGConfig defaults.
    det_grid : tuple of array-like or None, optional
        Explicit detector-coordinate arrays ``(u, v)``, each of shape ``(nv, nu)``.
        None uses the canonical detector grid.

    Returns
    -------
    volume : jax.Array
        Float32 reconstruction of shape ``(nz, ny, nx)``.
    info : dict
        Solver diagnostics and effective work; a small residual or iteration
        count does not establish reconstruction accuracy.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.

    Examples
    --------
    >>> import numpy as np
    >>> import tomojax as tj
    >>> from tomojax.recon import SPDHGConfig, spdhg
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = tj.project(geometry, np.ones((1, 4, 4), np.float32))
    >>> volume, info = spdhg(
    ...     geometry,
    ...     geometry.grid,
    ...     geometry.detector,
    ...     data,
    ...     config=SPDHGConfig(iterations=1),
    ... )
    >>> volume.shape
    (1, 4, 4)
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
    progress.remaining("iterations", runtime.schedule.block_ids.size)
    with contextlib.ExitStack() as stack:
        if runtime.y_meas is None:
            y_host = np.asarray(projections)
            w_host = None if weights is None else np.asarray(weights)
            validate_host_views(y_host, weights=w_host)
            keys = [
                stack.enter_context(host_source(y_host)),
                stack.enter_context(host_buffer(runtime.data_shape)),
                stack.enter_context(host_source(w_host)) if w_host is not None else jnp.int32(-1),
            ]
            runtime = replace(runtime, host_keys=jnp.stack(keys))
        if runtime.schedule.block_ids.size and progress.for_unit("iterations") is not None:
            progress.check_canceled()
            # XLA can use a volume of scratch during compilation. Release it
            # before materializing the watched primal and dual arrays.
            shape = volume_shape(runtime.projector[0], grid)
            with progress.preparing("compile spdhg updates"):
                state = _initial_spdhg_state.eval_shape(runtime, shape)
                _ = _advance_spdhg.lower(grid, detector, runtime, state, 0, 1).compile()
        x, losses = _run_spdhg_scan(grid, detector, runtime)[:2]
        if runtime.y_meas is None:
            _ = x.block_until_ready()
    result = _SPDHGResult(
        volume=restore_volume(x, runtime.projector[0]),
        losses=losses,
        step_sizes=runtime.step_sizes,
        schedule=runtime.schedule,
        regularizer=runtime.regularizer,
        huber_delta=runtime.huber_delta,
        tv_weight=None if config is None else config.tv_weight,
        resolved_tv_weight=runtime.resolved_tv_weight,
        effective_tv_weight=float(runtime.metric.weight),
    )
    return result.volume, result.info()
