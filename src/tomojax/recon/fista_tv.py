"""Public FISTA/TV reconstruction adapter."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import functools
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends.host_stream import host_source, should_stream
from tomojax.core.devices import ViewSplit, as_devices, refuse_streaming, view_split
from tomojax.core.geometry.cone import is_cone_beam, require_parallel_beam
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.operator_norm import estimate_normal_norm
from tomojax.core.projector import (
    RAY_INTEGRATORS,
    backproject_view_T,
    forward_project_view_T,
    get_detector_grid_device,
    sum_backproject_views_T,
)
from tomojax.core.validation import (
    validate_grid,
    validate_optional_broadcastable_shape,
    validate_pose_stack,
    validate_projection_shape,
    validate_projection_stack,
    validate_volume,
)

from ._callbacks import LossCallback, emit_loss_callback_endpoints
from ._fista_initialization import (
    first_gradient as estimate_first_gradient,
    initial_view_order,
    ordered_start,
    power_start,
    validate_initialization,
)
from ._fista_owned import FistaScanState, run_owned_fista, use_owned_fista
from ._projection import (
    ConeModel,
    ProjectorBackend,
    ProjectorModel,
    least_squares_operators,
    resolve_geometry_projector,
    resolve_projector,
)
from ._solver_validation import validate_fista_config
from ._tv_ops import (
    gradient_norm_squared,
    huber_tv_grad,
    huber_tv_value,
    isotropic_tv_value,
    project_box,
    tv_proximal,
    validate_regulariser,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tomojax._typed_arrays import Device
    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from .types import Regulariser

GradMode = Literal["auto", "batched", "stream"]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["nonnegative", "lower_bound", "upper_bound"],
)
@dataclass(frozen=True)
class _FistaConstraints:
    nonnegative: bool
    lower_bound: float | None
    upper_bound: float | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FistaRuntime:
    """Resolved solver state shared with the initialization module."""

    config: FistaConfig
    regulariser: Regulariser = field(metadata={"static": True})
    huber_delta: float = field(metadata={"static": True})
    constraints: _FistaConstraints
    # None starts from zeros created inside the compiled solve, so the start
    # is not a separate argument buffer alongside the iterates.
    x0: jnp.ndarray | None
    poses: jnp.ndarray
    # None: estimated inside the compiled solve (batched operators), see _run_fista_scan.
    lipschitz: float | None
    volume_mask: jnp.ndarray | None
    detector_grid: tuple[jnp.ndarray, jnp.ndarray] | None
    # (model, backend, views per batch) for the batched operators, or None for
    # the ray-model reference path.
    projector: tuple[str, str, int] | None = field(metadata={"static": True})
    # The data argument is then a host_source key, read one batch at a time.
    stream: bool = field(default=False, metadata={"static": True})
    # The views shared among devices, the data placed accordingly; None for one.
    split: ViewSplit | None = field(default=None, metadata={"static": True})
    initial_order: jax.Array | None = None


@dataclass(frozen=True)
class _FistaResult:
    volume: jnp.ndarray
    losses: jnp.ndarray
    lipschitz: float
    effective_iterations: int
    early_stop: bool
    regulariser: Regulariser = field(metadata={"static": True})
    huber_delta: float = field(metadata={"static": True})

    def info(self) -> dict[str, object]:
        return {
            "loss": np.asarray(self.losses).tolist(),
            "lipschitz": self.lipschitz,
            "effective_iterations": self.effective_iterations,
            "early_stop": self.early_stop,
            "regulariser": self.regulariser,
            "huber_delta": self.huber_delta,
        }


def _effective_view_chunk_size(n_views: int, views_per_batch: int | None) -> int:
    requested = (
        int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else 1
    )
    return max(1, min(requested, int(n_views)))


def _view_chunk_schedule(
    i: jnp.ndarray,
    *,
    n_views: int,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    i = jnp.asarray(i, dtype=jnp.int32)
    b = jnp.int32(chunk_size)
    start = i * b
    remaining = jnp.maximum(0, jnp.int32(n_views) - start)
    valid = jnp.minimum(b, remaining)
    shift = b - valid
    start_shifted = jnp.maximum(0, start - shift)
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    valid_mask = idx >= (b - valid)
    return start_shifted, valid_mask, start_shifted + idx


@dataclass(kw_only=True)
class FistaConfig:
    """Configuration for public FISTA/TV reconstruction.

    ``projector_model``/``projector_backend`` follow :class:`CGLSConfig`: ``"auto"``
    uses batched Joseph (also for ``ray_integrator="joseph"``), with CUDA on CUDA.
    Explicit grids/exact integration use the reference path, alone honouring
    ``gather_dtype``, ``projector_unroll``, ``checkpoint_projector`` and ``grad_mode``.
    ``views_per_batch=None`` batches 64 views, or one on the reference path.

    ``stream_projections`` reads NumPy or memmap projections from host memory
    one view batch at a time, so they never occupy the device whole. ``None``
    streams when they would take more than 40% of free device memory; ``True``
    always streams host arrays. Streaming needs the batched operators.

    ``lipschitz`` is the data term's Lipschitz constant; ``None`` estimates it
    with ``power_iterations`` power iterations, started from the backprojected data.
    Huber-TV's finite-grid gradient bound is added automatically.

    ``initial_epochs`` optionally initializes nonnegative least squares with
    accelerated, interleaved ordered subsets before the full-data iterations.
    Each epoch interleaves nominal angle order into ``initial_subsets`` groups; more groups save
    time but increase gradient error. This needs resident, unmasked parallel or
    laminographic Joseph data, ``tv_weight=0``, and at least one final iteration.
    It is not a convergence or image-quality guarantee. Defaults leave it off.
    Its norm bounds are reused unless ``lipschitz`` is supplied. Unmasked resident
    nonnegative Joseph volumes larger than the data reuse two owned buffers
    for zero or ordered starts; this changes FP32 summation order.

    ``devices`` (two or more JAX devices) shares the views among them, each
    holding the whole volume, and sums their backprojections; one device, or
    ``None``, runs on the default device. It needs the batched operators and
    projections held in device memory (no streaming).

    Parameters
    ----------
    iterations : int, optional
        Maximum number of solver updates. For SIRT and MLEM, an update is a complete
        subset sweep. Default is 50.
    initial_epochs : int, optional
        Number of ordered-subset initialization sweeps; zero disables
        initialization. Default is 0.
    initial_subsets : int, optional
        Number of angle-interleaved groups per initialization sweep. Default is 16.
    tv_weight : float, optional
        Nonnegative regularization weight. Default is 0.005.
    regulariser : Regulariser, optional
        Regularization model: "tv" or "huber_tv". Default is 'tv'.
    huber_delta : float, optional
        Positive Huber transition scale when regulariser is "huber_tv". Default is
        0.01.
    lipschitz : float | None, optional
        Positive data-gradient Lipschitz constant, or None to estimate it by power
        iteration.
    views_per_batch : int | None, optional
        View batch size. None uses 64 for batched operators or one on the reference
        path.
    projector_unroll : int, optional
        Loop unrolling for the explicit-grid or ray-reference projector. Default is
        1.
    checkpoint_projector : bool, optional
        Recompute intermediate projector values during differentiation on the
        reference path. Default is True.
    gather_dtype : str, optional
        Gather precision on the reference path: "auto", "fp32", "fp16" or "bf16". Default is 'fp32'.
    grad_mode : GradMode, optional
        Reference gradient evaluation: "auto", "batched" or "stream". Default is
        'auto'.
    tv_prox_iterations : int, optional
        Iterations of the TV proximal solve per FISTA update. Default is 10.
    recon_rel_tol : float | None, optional
        Relative objective-change tolerance for stopping reconstruction early; None disables
        it.
    recon_patience : int, optional
        Number of successive small-change iterations required for early stopping.
        Default is 0.
    power_iterations : int, optional
        Number of power iterations used to estimate the data-gradient Lipschitz
        constant. Default is 3.
    support : array-like | None, optional
        Volume support mask. Excluded voxels remain zero; None uses the entire grid.
    nonnegative : bool, optional
        Constrain the reconstruction to nonnegative values. Default is False.
    lower_bound : float | None, optional
        Lower voxel-value bound, or None for no explicit lower bound.
    upper_bound : float | None, optional
        Upper voxel-value bound, or None for no upper bound.
    ray_integrator : str, optional
        Reference ray integration rule; "sampled" is the default, "exact" uses exact
        integration, and "joseph" selects Joseph sampling. Default is 'sampled'.
    projector_model : ProjectorModel, optional
        Projection model: "auto", "ray" or "joseph". Auto selects Joseph unless an
        explicit detector grid requires rays. Default is 'auto'.
    projector_backend : ProjectorBackend, optional
        Execution backend: "auto", "jax" or "cuda". An explicit accelerator
        request never silently falls back. Default is 'auto'.
    stream_projections : bool | None, optional
        Read host projections one view batch at a time. None selects streaming from
        available device memory; False keeps them resident.
    devices : jax.Device or sequence of jax.Device or None, optional
        JAX device or sequence of devices. None uses the current default; multiple
        devices divide the views and each hold the full volume.
    """

    iterations: int = 50
    initial_epochs: int = 0
    initial_subsets: int = 16
    tv_weight: float = 0.005
    regulariser: Regulariser = "tv"
    huber_delta: float = 1e-2
    lipschitz: float | None = None
    views_per_batch: int | None = None
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    grad_mode: GradMode = "auto"
    tv_prox_iterations: int = 10
    recon_rel_tol: float | None = None
    recon_patience: int = 0
    power_iterations: int = 3
    support: jnp.ndarray | None = None
    nonnegative: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    ray_integrator: str = "sampled"
    projector_model: ProjectorModel = "auto"
    projector_backend: ProjectorBackend = "auto"
    stream_projections: bool | None = None
    devices: Device | Sequence[Device] | None = None  # kept as a tuple

    def __post_init__(self) -> None:
        self.devices = as_devices(self.devices)


jax.tree_util.register_dataclass(
    FistaConfig,
    data_fields=["support"],
    meta_fields=[field.name for field in fields(FistaConfig) if field.name != "support"],
)


def grad_data_term(  # noqa: PLR0915
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
    ray_integrator: str = "sampled",
    T_all: jnp.ndarray | None = None,
    vol_mask: jnp.ndarray | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, float]:
    """Compute grad(1/2 sum_i ||A_i x - y_i||^2) and loss.

    Two execution modes:
    - batched: vmap over a chunk of views. Fast but higher peak memory.
    - stream: process one view at a time via lax.scan and explicit adjoint. Low peak memory.

    When grad_mode="auto", selects stream if the effective batch is 1, else batched.
    """
    require_parallel_beam(geometry, "grad_data_term")
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

    det_grid = get_detector_grid_device(detector) if det_grid is None else det_grid
    mask_arr = None if vol_mask is None else jnp.asarray(vol_mask, dtype=jnp.float32)

    def apply_mask(vol: jnp.ndarray) -> jnp.ndarray:
        return vol * mask_arr if mask_arr is not None else vol

    def adjoint(resid: jnp.ndarray, T_i: jnp.ndarray) -> jnp.ndarray:
        grad_i = backproject_view_T(
            T_i,
            grid,
            detector,
            resid,
            unroll=int(projector_unroll),
            gather_dtype=gather_dtype,
            det_grid=det_grid,
            ray_integrator=ray_integrator,
        )
        return grad_i if mask_arr is None else grad_i * mask_arr

    def batched_loss_and_grad(vol: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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
                ray_integrator=ray_integrator,
            ),
            in_axes=(0, None),
        )
        n = int(T_all.shape[0])
        b = _effective_view_chunk_size(n, views_per_batch)
        m = (n + b - 1) // b

        def body(
            carry: tuple[jnp.ndarray, jnp.ndarray],
            i: jnp.ndarray,
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
            loss_acc, grad_acc = carry
            start_shifted, valid_mask, _view_idx = _view_chunk_schedule(
                i,
                n_views=n,
                chunk_size=b,
            )
            T_chunk = jax.lax.dynamic_slice_in_dim(T_all, start_shifted, b)
            y_chunk = jax.lax.dynamic_slice_in_dim(projections, start_shifted, b)
            pred = vm_project(T_chunk, masked_vol)
            mask = valid_mask[:, None, None]
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
                ray_integrator=ray_integrator,
            )
            if mask_arr is not None:
                grad_batch = grad_batch * mask_arr
            return ((loss_acc + loss_batch, grad_acc + grad_batch), None)

        init = (jnp.float32(0.0), jnp.zeros_like(vol))
        (loss_tot, grad_tot), _ = jax.lax.scan(body, init, jnp.arange(m))
        return loss_tot, grad_tot

    def stream_loss_and_grad(vol: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        masked_vol = vol * vol_mask if vol_mask is not None else vol

        def one_view(
            carry: tuple[jnp.ndarray, jnp.ndarray],
            i: jnp.ndarray,
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
            loss_acc, g_acc = carry
            T_i = jax.lax.dynamic_slice_in_dim(T_all, i, 1)[0]
            y_i = jax.lax.dynamic_slice_in_dim(projections, i, 1)[0]
            pred_i = forward_project_view_T(
                T_i,
                grid,
                detector,
                masked_vol,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
                ray_integrator=ray_integrator,
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
        int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else 1
    )
    mode = grad_mode
    if grad_mode == "auto":
        mode = "stream" if eff_b <= 1 else "batched"

    if mode == "stream":
        loss_val, grad = stream_loss_and_grad(x)
        return grad, loss_val
    loss_val, grad = batched_loss_and_grad(x)
    return grad, loss_val


def power_method_L(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections_shape: tuple[int, int, int],
    *,
    iterations: int = 10,
    views_per_batch: int | None = None,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    grad_mode: GradMode = "auto",
    ray_integrator: str = "sampled",
    T_all: jnp.ndarray | None = None,
    vol_mask: jnp.ndarray | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    initial: jnp.ndarray | None = None,
) -> float:
    """Estimate the Lipschitz constant of the data gradient by power iteration.

    ``initial`` is the start (default a constant volume); see :func:`power_start`.
    """
    require_parallel_beam(geometry, "power_method_L")
    validate_grid(grid, "power_method_L grid")
    n_views, _, _ = validate_projection_shape(
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
    batch_size = (
        1 if grad_mode == "stream" else _effective_view_chunk_size(n_views, views_per_batch)
    )
    if initial is None:
        initial = jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    return float(
        estimate_normal_norm(
            T_all,
            initial,
            det_grid,
            vol_mask,
            grid=grid,
            detector=detector,
            batch_size=batch_size,
            iterations=int(iterations),
            unroll=int(projector_unroll),
            checkpoint=checkpoint_projector,
            gather_dtype=gather_dtype,
            ray_integrator=ray_integrator,
        )
    )


def _prepare_fista_runtime(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    init_x: jnp.ndarray | None,
    config: FistaConfig | None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> FistaRuntime:
    cfg = FistaConfig() if config is None else config
    nonnegative, lower_bound, upper_bound = validate_fista_config(cfg)
    if cfg.ray_integrator not in RAY_INTEGRATORS:
        raise ValueError("ray_integrator must be sampled or exact")
    volume_mask = cfg.support
    regulariser = validate_regulariser(
        cfg.regulariser,
        cfg.huber_delta,
        context="fista_tv config",
    )
    constraints = _FistaConstraints(
        nonnegative=nonnegative,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    validate_grid(grid, "fista_tv grid")
    n_views, _, _ = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="fista_tv projections",
    )
    validate_optional_broadcastable_shape(
        volume_mask,
        (grid.nx, grid.ny, grid.nz),
        context="fista_tv support",
        name="support",
        fix="use a support mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if init_x is not None:
        validate_volume(init_x, grid, context="fista_tv init_x", name="init_x")

    x0 = None if init_x is None else jnp.asarray(init_x, dtype=jnp.float32)
    poses = stack_view_poses(geometry, n_views)
    validate_pose_stack(poses, n_views, context="fista_tv geometry")
    projector = _batched_projector(cfg, n_views, det_grid, geometry, detector)
    split = view_split(cfg.devices, n_views)
    refuse_streaming(split, cfg.stream_projections, "fista_tv")
    if split is not None and projector is None:
        raise ValueError("fista_tv: devices requires the batched projection operators")
    stream = (
        split is None
        and not isinstance(projections, jax.Array)
        and (
            bool(cfg.stream_projections)
            if cfg.stream_projections is not None
            else should_stream(projections)
        )
    )
    if stream and projector is None:
        raise ValueError("fista_tv: stream_projections requires the batched projection operators")

    validate_initialization(cfg, projector, n_views, stream=stream, split=split)
    lipschitz = cfg.lipschitz
    if lipschitz is None and projector is None:
        # The gradient at zero is minus the backprojected data: start from it.
        at_zero, _ = grad_data_term(
            geometry, grid, detector, projections, jnp.zeros((grid.nx, grid.ny, grid.nz)),
            views_per_batch=cfg.views_per_batch, projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector, gather_dtype=cfg.gather_dtype,
            grad_mode="stream", T_all=poses, vol_mask=volume_mask, det_grid=det_grid,
            ray_integrator=cfg.ray_integrator,
        )  # fmt: skip
        lipschitz = power_method_L(
            geometry,
            grid,
            detector,
            projections.shape,
            iterations=cfg.power_iterations,
            initial=power_start(at_zero),
            views_per_batch=cfg.views_per_batch,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            grad_mode="stream",
            T_all=poses,
            vol_mask=volume_mask,
            det_grid=det_grid,
            ray_integrator=cfg.ray_integrator,
        )
    if lipschitz is not None:
        lipschitz = float(_with_huber(float(lipschitz), cfg, regulariser, grid))

    return FistaRuntime(
        config=cfg,
        regulariser=regulariser,
        huber_delta=float(cfg.huber_delta),
        constraints=constraints,
        x0=x0,
        poses=poses,
        lipschitz=lipschitz,
        volume_mask=volume_mask,
        detector_grid=det_grid,
        projector=projector,
        stream=stream,
        split=split,
        initial_order=initial_view_order(cfg, geometry),
    )


def _batched_projector(
    cfg: FistaConfig,
    n_views: int,
    det_grid: object,
    geometry: object = None,
    detector: Detector | None = None,
) -> tuple[str | ConeModel, str, int] | None:
    """Choose batched operators, or None for the ray-model reference path."""
    model, backend = cfg.projector_model, cfg.projector_backend
    requested = 64 if cfg.views_per_batch is None else int(cfg.views_per_batch)
    if is_cone_beam(geometry):
        if cfg.ray_integrator == "exact":
            raise ValueError(
                "fista_tv: cone-beam geometry uses Joseph sampling, not exact integration"
            )
        assert detector is not None
        cone, backend = resolve_geometry_projector(
            geometry, model, backend, detector=detector, det_grid=det_grid, context="fista_tv"
        )
        return cone, backend, max(1, min(requested, n_views))
    if det_grid is None and cfg.ray_integrator == "joseph" and model == "auto":
        model = "joseph"
    elif det_grid is not None or cfg.ray_integrator != "sampled":
        if model == "joseph" or backend == "cuda":
            raise ValueError(
                "fista_tv: explicit detector grids and exact integration require "
                "projector_model='ray' and projector_backend='jax'"
            )
        return None
    model, backend = resolve_projector(model, backend, context="fista_tv")
    if model == "ray" and backend == "jax":
        return None
    return model, backend, max(1, min(requested, n_views))


def huber_tv_lipschitz(tv_weight: float, huber_delta: float, grid: Grid) -> float:
    """The Lipschitz constant of ``tv_weight`` times Huber-TV's gradient on ``grid``."""
    shape = (grid.nx, grid.ny, grid.nz)
    return float(tv_weight) * gradient_norm_squared(shape) / float(huber_delta)


def _with_huber(
    lipschitz: jax.Array | float, cfg: FistaConfig, regulariser: str, grid: Grid
) -> jax.Array | float:
    """``lipschitz`` plus the Huber-TV gradient's own Lipschitz constant, if it has one."""
    if regulariser == "huber_tv" and float(cfg.tv_weight) != 0.0:
        lipschitz = lipschitz + huber_tv_lipschitz(cfg.tv_weight, cfg.huber_delta, grid)
    # A zero data operator (e.g. empty support) needs no step-size restriction.
    # Keep the update defined rather than dividing a zero gradient by zero.
    if isinstance(lipschitz, jax.Array):
        return jnp.where(lipschitz == 0, 1e-6, lipschitz)
    return 1e-6 if lipschitz == 0 else lipschitz


def _data_term(
    grid: Grid, detector: Detector, projections: jnp.ndarray, runtime: FistaRuntime
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Return the data term's value-and-gradient function."""
    cfg = runtime.config
    regulariser = runtime.regulariser
    huber_delta = runtime.huber_delta
    mask = None if runtime.volume_mask is None else jnp.asarray(runtime.volume_mask, jnp.float32)
    if runtime.projector is not None:
        model, backend, batch = runtime.projector
        least_squares = least_squares_operators(
            runtime.poses, grid, detector, backend, batch, model, stream=runtime.stream,
            split=runtime.split,
        )  # fmt: skip

        def batched_value_and_grad(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            v, g = least_squares(z if mask is None else z * mask, projections)
            return v, g if mask is None else g * mask

    def val_and_grad_fn(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if runtime.projector is not None:
            v, g = batched_value_and_grad(z)
            if regulariser == "huber_tv" and float(cfg.tv_weight) != 0.0:
                g = g + jnp.asarray(cfg.tv_weight, dtype=z.dtype) * huber_tv_grad(z, huber_delta)
            return v, g
        g, v = grad_data_term(
            None,
            grid,
            detector,
            projections,
            z,
            views_per_batch=cfg.views_per_batch,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            grad_mode=cfg.grad_mode,
            T_all=runtime.poses,
            vol_mask=runtime.volume_mask,
            det_grid=runtime.detector_grid,
            ray_integrator=cfg.ray_integrator,
        )
        if regulariser == "huber_tv" and float(cfg.tv_weight) != 0.0:
            g = g + jnp.asarray(cfg.tv_weight, dtype=z.dtype) * huber_tv_grad(z, huber_delta)
        return v, g

    return val_and_grad_fn


def _initial_fista_state(grid: Grid, runtime: FistaRuntime) -> FistaScanState:
    """Build the feasible start and empty diagnostics inside the compiled solve."""
    x0 = runtime.x0
    if x0 is None:
        x0 = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    x0 = _feasible_volume(x0, runtime)
    return FistaScanState(
        x=x0,
        z=x0,
        t=jnp.float32(1.0),
        loss=jnp.zeros((runtime.config.iterations,), dtype=jnp.float32),
        prev_obj=jnp.float32(0.0),
        streak=jnp.int32(0),
        done=jnp.bool_(False),
        has_prev=jnp.bool_(False),
        last_obj=jnp.float32(0.0),
        iters_done=jnp.int32(0),
    )


def _feasible_volume(x: jax.Array, runtime: FistaRuntime) -> jax.Array:
    constraints = runtime.constraints
    return project_box(
        x,
        nonnegative=constraints.nonnegative,
        lower_bound=constraints.lower_bound,
        upper_bound=constraints.upper_bound,
    )


def _fista_proximal(
    runtime: FistaRuntime, lipschitz: jax.Array | float
) -> Callable[[jax.Array], jax.Array]:
    """The joint regularizer/feasibility prox for the selected FISTA objective."""
    cfg, constraints = runtime.config, runtime.constraints
    if runtime.regulariser == "huber_tv" or cfg.tv_weight == 0.0:
        return functools.partial(_feasible_volume, runtime=runtime)
    return functools.partial(
        tv_proximal,
        lam_over_L=cfg.tv_weight / lipschitz,
        iterations=int(cfg.tv_prox_iterations),
        nonnegative=constraints.nonnegative,
        lower_bound=constraints.lower_bound,
        upper_bound=constraints.upper_bound,
    )


@functools.partial(jax.jit, static_argnames=("grid", "detector"))
def _run_fista_scan(
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    runtime: FistaRuntime,
) -> tuple[FistaScanState, jax.Array]:
    """Run the iterations; returns the final state and the Lipschitz constant used."""
    cfg = runtime.config
    regulariser = runtime.regulariser
    huber_delta = runtime.huber_delta
    L, first_value, first_gradient = runtime.lipschitz, None, None
    if cfg.initial_epochs:
        initial, bound = ordered_start(grid, detector, projections, runtime)
        runtime = replace(runtime, x0=initial)
        L = bound if L is None else L
    if L is None:
        L, first_value, first_gradient = estimate_first_gradient(
            grid, detector, projections, runtime
        )
        L = _with_huber(L, cfg, regulariser, grid)

    val_and_grad_fn = _data_term(grid, detector, projections, runtime)
    val_and_grad = jax.jit(val_and_grad_fn)
    proximal = _fista_proximal(runtime, L)

    def regulariser_value_fn(x: jnp.ndarray) -> jnp.ndarray:
        if regulariser == "huber_tv":
            return huber_tv_value(x, huber_delta)
        return isotropic_tv_value(x)

    use_early_stop = (
        (cfg.recon_rel_tol is not None)
        and float(cfg.recon_rel_tol) > 0.0
        and int(cfg.recon_patience) > 0
    )
    tol = jnp.float32(float(cfg.recon_rel_tol) if use_early_stop else 0.0)
    patience = jnp.int32(int(cfg.recon_patience) if use_early_stop else 0)
    early_flag = jnp.bool_(use_early_stop)

    def step(state: FistaScanState, k: jnp.ndarray) -> tuple[FistaScanState, None]:
        def run_active(active_state: FistaScanState) -> FistaScanState:
            # The objective is tracked at z, where the gradient's residual already
            # gives the data term: evaluating it at x would cost a projection.
            if first_gradient is None:
                data_loss_val, g = val_and_grad(active_state.z)
            else:  # the start, zero: first_gradient already has it
                data_loss_val, g = jax.lax.cond(
                    k == 0, lambda _: (first_value, first_gradient), val_and_grad, active_state.z
                )
            reg_value = 0.0 if cfg.tv_weight == 0 else regulariser_value_fn(active_state.z)
            y = active_state.z - (1.0 / L) * g
            x_new = proximal(y)
            t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * active_state.t * active_state.t))
            z_new = x_new + ((active_state.t - 1.0) / t_new) * (x_new - active_state.x)
            z_new = _feasible_volume(z_new, runtime)
            obj = data_loss_val + cfg.tv_weight * reg_value
            obj32 = obj.astype(jnp.float32)
            rel_change = jnp.abs(obj - active_state.prev_obj) / jnp.maximum(
                jnp.abs(active_state.prev_obj),
                1e-6,
            )
            small = jnp.logical_and(
                early_flag,
                jnp.logical_and(active_state.has_prev, rel_change <= tol),
            )
            streak_next = jnp.where(
                early_flag,
                jnp.where(small, jnp.minimum(active_state.streak + 1, patience), jnp.int32(0)),
                active_state.streak,
            )
            done_next = jnp.logical_or(
                active_state.done,
                jnp.logical_and(early_flag, streak_next >= patience),
            )
            return active_state._replace(
                x=x_new,
                z=z_new,
                t=t_new,
                loss=active_state.loss.at[k].set(obj32),
                prev_obj=obj,
                streak=streak_next,
                done=done_next,
                has_prev=jnp.bool_(True),
                last_obj=obj,
                iters_done=active_state.iters_done + jnp.int32(1),
            )

        def run_skip(skip_state: FistaScanState) -> FistaScanState:
            return skip_state._replace(
                loss=skip_state.loss.at[k].set(skip_state.last_obj.astype(jnp.float32))
            )

        return jax.lax.cond(state.done, run_skip, run_active, state), None

    init_carry = _initial_fista_state(grid, runtime)
    if cfg.iterations == 0:
        return init_carry, jnp.asarray(L, jnp.float32)
    carry_final, _ = jax.lax.scan(step, init_carry, jnp.arange(int(cfg.iterations)))
    return carry_final, jnp.asarray(L, jnp.float32)


def _emit_fista_callback(callback: LossCallback | None, result: _FistaResult) -> None:
    if result.losses.size == 0:
        return
    final_step = max(result.effective_iterations - 1, 0)
    emit_loss_callback_endpoints(
        callback,
        (
            (0, float(result.losses[0])),
            (final_step, float(result.losses[final_step])),
        ),
    )


def fista_tv(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    init_x: jnp.ndarray | None = None,
    config: FistaConfig | None = None,
    callback: LossCallback | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, dict[str, object]]:
    """Run FISTA with TV regularization using an explicit solver configuration.

    If ``callback`` is provided, it fires on the first recorded loss sample and on
    the final recorded loss sample. The callback arguments are ``(step, loss)``,
    where ``step`` is the zero-based iteration index that produced ``loss``. When
    early stopping truncates active iterations, the final callback reports the
    last active iteration rather than the repeated padded tail entry.

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
    init_x : array-like or None, optional
        Initial volume of shape ``(nx, ny, nz)``; None starts from zero.
    config : FistaConfig or None, optional
        Solver settings. None uses FistaConfig defaults.
    callback : callable or None, optional
        Callback receiving ``(step, loss)`` at the first and last logged samples.
    det_grid : tuple of array-like or None, optional
        Explicit detector-coordinate arrays ``(u, v)``, each of shape ``(nv, nu)``.
        None uses the canonical detector grid.

    Returns
    -------
    volume : jax.Array
        Float32 reconstruction of shape ``(nx, ny, nz)``.
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
    >>> from tomojax.recon import fista_tv, FistaConfig, project
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = project(geometry, np.ones((4, 4, 1), np.float32))
    >>> volume, info = fista_tv(
    ...     geometry,
    ...     geometry.grid,
    ...     geometry.detector,
    ...     data,
    ...     config=FistaConfig(iterations=1),
    ... )
    >>> volume.shape
    (4, 4, 1)
    """
    runtime = _prepare_fista_runtime(
        geometry,
        grid,
        detector,
        projections,
        init_x=init_x,
        config=config,
        det_grid=det_grid,
    )
    if runtime.stream:
        with host_source(np.asarray(projections)) as key:
            final, lipschitz = _run_fista_scan(grid, detector, key, runtime)
            final.x.block_until_ready()
    elif runtime.split is not None:  # each device's views on that device, once
        data = runtime.split.place(projections)
        final, lipschitz = _run_fista_scan(grid, detector, data, runtime.split.everywhere(runtime))
        final = final._replace(x=runtime.split.gather(final.x))
    else:
        # Store residuals rather than another volume when that is the smaller object.
        run = run_owned_fista if use_owned_fista(grid, projections, runtime) else _run_fista_scan
        final, lipschitz = run(grid, detector, projections, runtime)
    result = _FistaResult(
        volume=final.x,
        losses=final.loss,
        lipschitz=float(lipschitz),
        effective_iterations=int(final.iters_done),
        early_stop=bool(final.done),
        regulariser=runtime.regulariser,
        huber_delta=runtime.huber_delta,
    )
    _emit_fista_callback(callback, result)
    info = result.info()
    if runtime.config.initial_epochs:
        info["initial_epochs"] = runtime.config.initial_epochs
        info["initial_updates"] = runtime.config.initial_epochs * runtime.config.initial_subsets
    return result.volume, info
