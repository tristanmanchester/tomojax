"""Public FISTA/TV reconstruction adapter."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import progress
from tomojax.backends.host_stream import host_source, should_stream
from tomojax.core.devices import ViewSplit, as_devices, refuse_streaming, view_split
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import (
    validate_grid,
    validate_optional_broadcastable_shape,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)

from ._chunks import Chunked
from ._fista_initialization import first_gradient as estimate_first_gradient
from ._fista_owned import FistaResult, FistaScanState, run_owned_fista, use_owned_fista
from ._noise import estimate_projection_noise
from ._noise_probe import probe_gradient_rms
from ._projection import (
    ConeModel,
    ProjectorBackend,
    least_squares_operators,
    resolve_geometry_projector,
)
from ._projection_batches import ViewBatches
from ._solver_validation import validate_fista_config
from ._tv_metric import TVMetric, raise_tv_weight, resolve_tv_metric
from ._tv_ops import (
    huber_tv_value,
    isotropic_tv_value,
    project_box,
    tv_proximal,
    validate_regularizer,
)
from ._volume_storage import Model, pack_volume, resident_model, volume_shape

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tomojax._typed_arrays import Device
    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from .types import DataTerm, Regularizer


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["lower_bound", "upper_bound"],
)
@dataclass(frozen=True)
class _FistaConstraints:
    lower_bound: float | None
    upper_bound: float | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FistaRuntime:
    """Resolved solver state shared with the initialization module."""

    config: FISTAConfig
    regularizer: Regularizer = field(metadata={"static": True})
    metric: TVMetric | None
    noise_scale: jax.Array | float | None
    huber_delta: jax.Array | float
    constraints: _FistaConstraints
    # None starts from zeros created inside the compiled solve, so the start
    # is not a separate argument buffer alongside the iterates.
    x0: jnp.ndarray | np.ndarray | None
    poses: jnp.ndarray
    # None: estimated inside the compiled solve (batched operators), see _run_fista_scan.
    lipschitz: jax.Array | float | None
    volume_mask: jnp.ndarray | None
    detector_grid: tuple[jnp.ndarray, jnp.ndarray] | None
    # (model, backend, views per batch) for the batched operators.
    projector: tuple[Model, str, int] = field(metadata={"static": True})
    # The data argument is then a host_source key, read one batch at a time.
    stream: bool = field(default=False, metadata={"static": True})
    # The views shared among devices, the data placed accordingly; None for one.
    split: ViewSplit | None = field(default=None, metadata={"static": True})

    def penalty(self, x: jax.Array) -> jax.Array | float:
        """The regularization term in the physical voxel metric."""
        if self.metric is None:
            return 0.0
        scale = self.metric.gradient_scale_xyz
        if self.regularizer == "huber_tv":
            return self.metric.weight * huber_tv_value(
                x, self.huber_delta, gradient_scale_xyz=scale
            )
        return self.metric.weight * isotropic_tv_value(x, gradient_scale_xyz=scale)

    def automatic_metric(self, grid: Grid, detector: Detector) -> TVMetric:
        """Resolve automatic strength through the same configured data transpose."""
        assert self.metric is not None
        assert self.noise_scale is not None
        if detector.nv * detector.nu == 1:
            return self.metric
        model, backend, batch = self.projector
        ops = ViewBatches(
            self.poses,
            grid,
            detector,
            self.detector_grid,
            backend,
            batch,
            model,
            "linear",
            absolute_weights=False,
        )
        backproject = ops.backproject
        rms = probe_gradient_rms(
            self.poses,
            grid,
            detector,
            backproject,
            batch_size=batch,
            shape=volume_shape(model, grid),
            support=self.volume_mask,
        )
        return raise_tv_weight(self.metric, self.noise_scale * rms)

    def initial_state(self, grid: Grid) -> FistaScanState:
        """Build the feasible start and empty diagnostics inside the compiled solve."""
        x0 = self.x0
        if x0 is None:
            x0 = jnp.zeros(volume_shape(self.projector[0], grid), dtype=jnp.float32)
        x0 = _feasible_volume(jnp.asarray(x0, jnp.float32), self)
        return FistaScanState(
            x=x0,
            z=x0,
            t=jnp.float32(1.0),
            loss=jnp.zeros((self.config.iterations,), dtype=jnp.float32),
            prev_obj=jnp.float32(0.0),
            streak=jnp.int32(0),
            done=jnp.bool_(False),
            iters_done=jnp.int32(0),
        )


@dataclass(kw_only=True)
class FISTAConfig:
    """Configuration for public FISTA/TV reconstruction.

    ``backend`` follows :class:`CGLSConfig`: ``"auto"`` projects with CUDA on
    CUDA. ``views_per_batch=None`` batches 64 views.

    ``stream`` reads NumPy or memmap projections from host memory
    one view batch at a time, so they never occupy the device whole. ``None``
    streams when they would take more than 40% of free device memory; ``True``
    always streams host arrays. Streaming needs the batched operators and can
    reduce device memory at the cost of slower execution than host-volume slabs.

    ``lipschitz`` is the data term's Lipschitz constant; ``None`` estimates it
    with ``power_iterations`` power iterations, started from the backprojected data.
    TV and Huber-TV use the proximal step without changing this data-only bound.

    TV uses physical voxel differences and a global data-contrast scale. Thus
    ``tv_weight`` does not change when switching length units or multiplying all
    line integrals by a constant. It weights an integral, not a mean over the ROI;
    padding the volume does not dilute this fixed minimum. Automatic noise
    scaling uses RMS over all volume voxels and can change with volume padding.
    Effective dimensional and selected dimensionless weights are returned in
    ``info`` as ``effective_tv_weight`` and ``resolved_tv_weight``.

    Against synthetic truth FISTA-TV is more accurate than FBP. On phase-contrast
    data it renders propagation fringes as smooth bands where FBP keeps them.
    Apply alignment poses before comparing methods on a real scan.

    Unmasked resident nonnegative Joseph volumes larger than the data reuse two
    owned buffers; this changes FP32 summation order.

    ``devices`` (two or more JAX devices) shares the views among them, each
    holding the whole volume, and sums their backprojections; one device, or
    ``None``, runs on the default device. It needs the batched operators and
    projections held in device memory (no streaming).

    Parameters
    ----------
    iterations : int, optional
        Maximum number of solver updates. For SIRT and MLEM, an update is a complete
        subset sweep. Default is 50.
    data_term : DataTerm, optional
        "absolute" (default) fits line integrals. Use "per_view_contrast" when
        material lies outside the field of view: it profiles an unknown constant
        per view, so absolute background density is not recoverable. This costs
        accuracy on contained objects; do not select it for contained scans.
    tv_weight : float | None, optional
        Dimensionless physical-TV weight; zero disables TV. Default None raises
        an empirical minimum of 0.00005 using a projection-only Haar-MAD noise
        estimate and one centered Gaussian gradient probe (seed 0). The minimum
        was selected on plate validation and image-checked on K11 and the corpus;
        it is not derived. Noise only raises it. Positive explicit values bypass
        estimation. Correlated detector noise and object texture are not separated
        by this iid noise scale. The selected strength is fixed for differentiation;
        physical intensity/unit normalization retains its derivatives.
    regularizer : Regularizer, optional
        Regularization model: "tv" or "huber_tv". Both use a constrained proximal
        solve after each data-gradient step. Default is 'tv'.
    huber_delta : float, optional
        Positive dimensionless Huber transition scale when regulariser is
        "huber_tv". Default is 0.01.
    lipschitz : float | None, optional
        Positive data-gradient Lipschitz constant, or None to estimate it by power
        iteration.
    views_per_batch : int | None, optional
        View batch size. None uses 64.
    tv_prox_iterations : int, optional
        Iterations of the TV or Huber-TV proximal solve per FISTA update. Default is 10.
    tolerance : float | None, optional
        Relative objective-change tolerance for stopping reconstruction early; None disables
        it.
    patience : int, optional
        Number of successive small-change iterations required for early stopping.
        Default is 0.
    power_iterations : int, optional
        Number of power iterations used to estimate the data-gradient Lipschitz
        constant. Default is 3.
    support : array-like | None, optional
        Boolean volume support mask, broadcastable to the grid. False (zero)
        entries fix the reconstructed voxel to zero, including at initialization
        and in every proximal update; True (nonzero) entries are included.
        Bounds apply only at included entries; exclusion wins over a positive
        lower bound. TV differences across the support boundary remain active.
        None uses the entire grid.
    lower_bound : float | None, optional
        Lower voxel-value bound (set ``0.0`` for nonnegativity), or None for no
        lower bound.
    upper_bound : float | None, optional
        Upper voxel-value bound, or None for no upper bound.
    backend : ProjectorBackend, optional
        Execution backend: "auto", "jax" or "cuda". An explicit accelerator
        request never silently falls back. Default is 'auto'.
    stream : bool | None, optional
        Read host projections one view batch at a time. None selects streaming from
        available device memory; False keeps them resident.
    devices : jax.Device or sequence of jax.Device or None, optional
        JAX device or sequence of devices. None uses the current default; multiple
        devices divide the views and each hold the full volume.
    """

    iterations: int = 50
    data_term: DataTerm = "absolute"
    tv_weight: float | None = None
    regularizer: Regularizer = "tv"
    huber_delta: float = 1e-2
    lipschitz: float | None = None
    views_per_batch: int | None = None
    tv_prox_iterations: int = 10
    tolerance: float | None = None
    patience: int = 0
    power_iterations: int = 3
    support: jnp.ndarray | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    backend: ProjectorBackend = "auto"
    stream: bool | None = None
    devices: Device | Sequence[Device] | None = None  # kept as a tuple

    def __post_init__(self) -> None:
        self.devices = as_devices(self.devices)


jax.tree_util.register_dataclass(
    FISTAConfig,
    data_fields=["support"],
    meta_fields=[field.name for field in fields(FISTAConfig) if field.name != "support"],
)


def _prepare_fista_runtime(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    init_x: jnp.ndarray | None,
    config: FISTAConfig | None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> FistaRuntime:
    cfg = FISTAConfig() if config is None else config
    lower_bound, upper_bound = validate_fista_config(cfg)
    volume_mask = (
        None if cfg.support is None else jnp.asarray(cfg.support, bool).astype(jnp.float32)
    )
    regularizer = validate_regularizer(
        cfg.regularizer,
        cfg.huber_delta,
        context="fista config",
    )
    constraints = _FistaConstraints(lower_bound, upper_bound)
    validate_grid(grid, "fista_tv grid")
    n_views, _, _ = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="fista_tv projections",
    )
    validate_optional_broadcastable_shape(
        volume_mask,
        (grid.nz, grid.ny, grid.nx),
        context="fista_tv support",
        name="support",
        fix="use a support mask broadcastable to shape (grid.nz, grid.ny, grid.nx).",
    )
    if init_x is not None:
        validate_volume(init_x, grid, context="fista_tv init_x", name="init_x")

    x0 = (
        None
        if init_x is None
        else (np if isinstance(init_x, np.ndarray) else jnp).asarray(init_x, dtype=np.float32)
    )
    poses = stack_view_poses(geometry, n_views)
    validate_pose_stack(poses, n_views, context="fista_tv geometry")
    projector = _batched_projector(cfg, n_views, det_grid, geometry, detector)
    split = view_split(cfg.devices, n_views)
    refuse_streaming(split, cfg.stream, "fista_tv")
    stream = (
        split is None
        and not isinstance(projections, jax.Array)
        and (
            bool(cfg.stream)
            if cfg.stream is not None  # ~8 volumes of iterates, TV duals, scratch
            else should_stream(projections, reserve_bytes=32 * grid.nx * grid.ny * grid.nz)
        )
    )

    metric = (
        resolve_tv_metric(
            geometry,
            grid,
            detector,
            projections,
            tv_weight=5e-5 if cfg.tv_weight is None else cfg.tv_weight,
            huber_delta=cfg.huber_delta if regularizer == "huber_tv" else 1.0,
        )
        if cfg.tv_weight != 0
        else None
    )
    return FistaRuntime(
        config=cfg,
        regularizer=regularizer,
        metric=metric,
        noise_scale=(
            estimate_projection_noise(projections) if detector.nv * detector.nu > 1 else 0.0
        )
        if cfg.tv_weight is None
        else None,
        huber_delta=1.0 if metric is None else jnp.where(metric.delta > 0, metric.delta, 1),
        constraints=constraints,
        x0=x0,
        poses=poses,
        lipschitz=cfg.lipschitz,
        volume_mask=volume_mask,
        detector_grid=det_grid,
        projector=projector,
        stream=stream,
        split=split,
    )


def _batched_projector(
    cfg: FISTAConfig,
    n_views: int,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    geometry: object,
    detector: Detector,
) -> tuple[str | ConeModel, str, int]:
    """The batched operators' model, backend and views per batch."""
    requested = 64 if cfg.views_per_batch is None else int(cfg.views_per_batch)
    model, backend = resolve_geometry_projector(
        geometry, cfg.backend, detector=detector, det_grid=det_grid, context="fista_tv"
    )
    return model, backend, max(1, min(requested, n_views))


def _data_term(
    grid: Grid, detector: Detector, projections: jnp.ndarray, runtime: FistaRuntime
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Return the data term's value-and-gradient function."""
    mask = runtime.volume_mask
    model, backend, batch = runtime.projector
    least_squares = least_squares_operators(
        runtime.poses, grid, detector, backend, batch, model, stream=runtime.stream,
        split=runtime.split, data_term=runtime.config.data_term, det_grid=runtime.detector_grid,
    )  # fmt: skip

    def value_and_grad(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        v, g = least_squares(z if mask is None else z * mask, projections)
        return v, g if mask is None else g * mask

    return value_and_grad


def _feasible_volume(x: jax.Array, runtime: FistaRuntime) -> jax.Array:
    """Project starts and momentum onto the same constraints as the primal prox.

    Support is a zero constraint, not an amplitude taper on the iterate.
    Excluded voxels stay zero even when the interior has a positive lower bound.
    """
    constraints = runtime.constraints
    return project_box(
        x,
        nonnegative=False,
        lower_bound=constraints.lower_bound,
        upper_bound=constraints.upper_bound,
        support=runtime.volume_mask,
    )


def _fista_proximal(
    runtime: FistaRuntime, lipschitz: jax.Array | float
) -> Callable[[jax.Array], jax.Array]:
    """The joint regularizer/feasibility prox for the selected FISTA objective."""
    cfg, constraints = runtime.config, runtime.constraints
    if cfg.tv_weight == 0.0:
        return functools.partial(_feasible_volume, runtime=runtime)
    assert runtime.metric is not None
    return functools.partial(
        tv_proximal,
        lam_over_L=runtime.metric.weight / lipschitz,
        huber_delta=runtime.huber_delta if runtime.regularizer == "huber_tv" else None,
        # Support must constrain every inner primal update, not only the output:
        # projecting an unconstrained TV minimizer does not solve the joint prox.
        support=runtime.volume_mask,
        gradient_scale_xyz=runtime.metric.gradient_scale_xyz,
        iterations=int(cfg.tv_prox_iterations),
        nonnegative=False,
        lower_bound=constraints.lower_bound,
        upper_bound=constraints.upper_bound,
    )


def _resident_runtime(runtime: FistaRuntime) -> FistaRuntime:
    # An existing input volume owns public storage. A solve that creates its
    # own volumes can choose the layout consumed by every projection batch.
    if runtime.x0 is not None or runtime.volume_mask is not None or runtime.split is not None:
        return runtime
    model, backend, batch = runtime.projector
    storage = resident_model(model, backend)
    metric = runtime.metric
    if storage != model and metric is not None:
        metric = replace(metric, gradient_scale_xyz=metric.gradient_scale_xyz[::-1])
    return replace(runtime, projector=(storage, backend, batch), metric=metric)


def _fista_step(
    grid: Grid, detector: Detector, projections: jax.Array, runtime: FistaRuntime, L: jax.Array
) -> Callable[..., FistaScanState]:
    cfg = runtime.config
    val_and_grad = jax.jit(_data_term(grid, detector, projections, runtime))
    proximal = _fista_proximal(runtime, L)

    tolerance = cfg.tolerance or 0.0

    def step(
        state: FistaScanState,
        initial_data: tuple[jax.Array, jax.Array] | None = None,
    ) -> FistaScanState:
        k = state.iters_done
        # The objective is tracked at z, where the gradient's residual already
        # gives the data term: evaluating it at x would cost a projection.
        z = state.z
        data_loss_val, g = val_and_grad(z) if initial_data is None else initial_data
        y = z - (1.0 / L) * g
        x_new = proximal(y)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * state.t * state.t))
        z_new = x_new + ((state.t - 1.0) / t_new) * (x_new - state.x)
        z_new = _feasible_volume(z_new, runtime)
        obj = data_loss_val + runtime.penalty(z)
        streak, done = state.streak, state.done
        if tolerance > 0 and cfg.patience > 0:
            previous = state.prev_obj
            relative = jnp.abs(obj - previous) / jnp.maximum(jnp.abs(previous), 1e-6)
            streak = jnp.where((k > 0) & (relative <= tolerance), streak + 1, 0)
            done = streak >= cfg.patience
        return state._replace(
            x=x_new,
            z=z_new,
            t=t_new,
            loss=state.loss.at[k].set(obj.astype(jnp.float32)),
            prev_obj=obj,
            streak=streak,
            done=done,
            iters_done=state.iters_done + jnp.int32(1),
        )

    return step


@functools.partial(jax.jit, static_argnames=("grid", "detector"))
def _initialize_fista(
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    runtime: FistaRuntime,
) -> tuple[FistaScanState, jax.Array, TVMetric | None]:
    cfg = runtime.config
    runtime = replace(
        runtime,
        metric=runtime.automatic_metric(grid, detector)
        if cfg.tv_weight is None
        else runtime.metric,
    )
    L, first_value, first_gradient = runtime.lipschitz, None, None
    if L is None:
        L, first_value, first_gradient = estimate_first_gradient(
            grid, detector, projections, runtime
        )
    # A zero data operator imposes no step-size restriction.
    L = jnp.where(L == 0, 1e-6, L)

    initial = runtime.initial_state(grid)
    if cfg.iterations and first_gradient is not None:
        assert first_value is not None
        # Consume the cached gradient before the loop so its buffer can be reused.
        initial = _fista_step(grid, detector, projections, runtime, L)(
            initial, (first_value, first_gradient)
        )

    return initial, L, runtime.metric


@functools.partial(jax.jit, static_argnames=("grid", "detector"), donate_argnames=("state",))
def _advance_fista(
    grid: Grid,
    detector: Detector,
    projections: jax.Array,
    runtime: FistaRuntime,
    L: jax.Array,
    state: FistaScanState,
    end: int,
) -> FistaScanState:
    step = _fista_step(grid, detector, projections, runtime, L)
    cfg = runtime.config
    tolerance = cfg.tolerance or 0.0

    def advance(_: int, state: FistaScanState) -> FistaScanState:
        if tolerance > 0 and cfg.patience > 0:
            return jax.lax.cond(state.done, lambda s: s, step, state)
        return step(state)

    return jax.lax.fori_loop(state.iters_done, end, advance, state)


def _fista_program(
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    runtime: FistaRuntime,
    *,
    drive: Callable[..., FistaScanState],
) -> tuple[FistaResult, jax.Array, TVMetric | None]:
    """Return the final state, Lipschitz constant and resolved regularization metric."""
    runtime = _resident_runtime(runtime)
    initial, L, metric = _initialize_fista(grid, detector, projections, runtime)
    runtime = replace(runtime, metric=metric)

    def advance(state: FistaScanState, end: int) -> FistaScanState:
        return _advance_fista(grid, detector, projections, runtime, L, state, end)

    def completed(state: FistaScanState) -> tuple[int, bool, dict[str, float]]:
        count, stopped, loss = jax.device_get((state.iters_done, state.done, state.prev_obj))
        return int(count), bool(stopped), {"loss": float(loss)}

    iterations = runtime.config.iterations
    final = drive(initial, iterations, advance, completed) if iterations else initial
    loss = final.padded_loss()
    return (
        FistaResult(
            pack_volume(final.x, runtime.projector[0]), final.z, loss, final.iters_done, final.done
        ),
        L,
        metric,
    )


_run_fista_scan = Chunked(_fista_program, static_argnames=("grid", "detector"))


@progress.reports_iterations(FISTAConfig.iterations)
def fista(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    init_x: jnp.ndarray | None = None,
    config: FISTAConfig | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    out: np.ndarray | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Run FISTA with TV regularization using an explicit solver configuration.

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
        Initial volume of shape ``(nz, ny, nx)``; None starts from zero.
    config : FISTAConfig or None, optional
        Solver settings. None uses FISTAConfig defaults.
    det_grid : tuple of array-like or None, optional
        Explicit detector-coordinate arrays ``(u, v)``, each of shape ``(nv, nu)``.
        None uses the canonical detector grid.

    out : numpy.ndarray or None, optional
        Writable float32 host output. CUDA Joseph solves use host slabs when
        supplied, or automatically when their resident state exceeds device memory.
        The host route supports ordinary TV without support masks. Large host
        state uses temporary files in TMPDIR.

    Returns
    -------
    volume : jax.Array or numpy.ndarray
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
    >>> from tomojax.recon import FISTAConfig, fista
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = tj.project(geometry, np.ones((1, 4, 4), np.float32))
    >>> volume, info = fista(
    ...     geometry,
    ...     geometry.grid,
    ...     geometry.detector,
    ...     data,
    ...     config=FISTAConfig(iterations=1),
    ... )
    >>> volume.shape
    (1, 4, 4)
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
    from ._fista_host import solve_on_host

    hosted = solve_on_host(geometry, grid, detector, projections, runtime, out=out)
    if hosted is None:
        if runtime.stream:
            with host_source(np.asarray(projections)) as key:
                final, lipschitz, metric = _run_fista_scan(grid, detector, key, runtime)
                final.x.block_until_ready()
        elif runtime.split is not None:
            data = runtime.split.place(projections)
            final, lipschitz, metric = _run_fista_scan(
                grid, detector, data, runtime.split.everywhere(runtime)
            )
            final = final._replace(x=runtime.split.gather(final.x))
        else:
            # Store residuals rather than another volume when that is the smaller object.
            run = (
                run_owned_fista if use_owned_fista(grid, projections, runtime) else _run_fista_scan
            )
            final, lipschitz, metric = run(grid, detector, projections, runtime)
        volume = final.x
        losses, count = np.asarray(final.loss).tolist(), int(final.iters_done)
        info = {
            "loss": losses,
            "lipschitz": float(lipschitz),
            "effective_iterations": count,
            "early_stop": bool(final.done),
            "regularizer": runtime.regularizer,
            "huber_delta": runtime.huber_delta,
        }
    else:
        volume, info, metric = hosted
    resolved_weight = runtime.config.tv_weight
    if resolved_weight is None:
        assert metric is not None
        assert runtime.metric is not None
        resolved_weight = (
            5e-5
            * float(metric.weight_over_contrast)
            / (float(runtime.metric.weight_over_contrast) or 1)
        )
    info.update(
        {
            "data_term": runtime.config.data_term,
            "background_profiled": runtime.config.data_term == "per_view_contrast",
            "tv_weight": runtime.config.tv_weight,
            "effective_tv_weight": 0.0 if metric is None else float(metric.weight),
            "resolved_tv_weight": resolved_weight,
            "huber_delta": (
                float(runtime.metric.delta)
                if runtime.metric is not None and runtime.regularizer == "huber_tv"
                else None
            ),
            "contrast_scale": None
            if runtime.metric is None
            else float(runtime.metric.contrast_scale),
            "length_scale": None if runtime.metric is None else runtime.metric.length_scale,
        }
    )
    return volume, info
