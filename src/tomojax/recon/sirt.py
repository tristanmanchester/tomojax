"""Row/column-normalized simultaneous and ordered-subset algebraic reconstruction."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
import math
import operator
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends.host_stream import host_source, should_stream
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import (
    validate_detector_grid,
    validate_grid,
    validate_projection_stack,
    validate_volume,
)
from tomojax.recon._axial_slabs import axial_slabs, independent_planes
from tomojax.recon._cone_slabs import RowReader, host_volumes
from tomojax.recon._projection import (
    ConeModel,
    ProjectorBackend,
    ProjectorModel,
    resolve_geometry_projector,
    subset_operators,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry


@dataclass(frozen=True, kw_only=True)
class SIRTConfig:
    """SIRT, or OS-SART with ``subsets > 1`` (one view per subset is SART).

    Each iteration visits all interleaved subsets once, in input view order.
    A subset updates ``x += relaxation * C * A.T * R * (data - A*x)``, where
    ``R = 1 / (A*support)`` and ``C = 1 / (A.T*ones)``; zero denominators give
    zero weights. The forward operator and transpose are exactly matched.
    ``relaxation`` must be in (0, 2); ordered subsets can cycle on inconsistent
    data, so use fewer subsets or lower relaxation for the final refinement.

    ``acceleration=True`` applies fixed FISTA momentum to full-data SIRT
    (``subsets=1``, ``relaxation <= 1``), reducing the iterations needed for
    the same weighted least-squares objective without changing the projector.

    ``lower_bound`` and ``upper_bound`` clip each update. ``support`` is a volume
    mask: zero excludes a voxel from reconstruction and fixes it to zero,
    including in the initial volume; nonzero includes it. Bounds apply only
    inside that support. Only nonnegative projector weights are used.

    Parameters
    ----------
    iterations : int, optional
        Maximum number of solver updates. For SIRT and MLEM, an update is a complete
        subset sweep. Default is 50.
    subsets : int, optional
        Number of interleaved view subsets visited per sweep; one uses the
        simultaneous method. Default is 1.
    relaxation : float, optional
        Update scale strictly between 0 and 2. Default is 1.0.
    acceleration : bool, optional
        Apply FISTA momentum to full-data SIRT; requires subsets=1 and relaxation <=
        1. Default is False.
    lower_bound : float | None, optional
        Lower voxel-value bound, or None for no explicit lower bound.
    upper_bound : float | None, optional
        Upper voxel-value bound, or None for no upper bound.
    support : array-like | None, optional
        Volume support mask. Excluded voxels remain zero; None uses the entire grid.
    views_per_batch : int, optional
        Number of views processed together to bound temporary storage. Default is
        64.
    backend : ProjectorBackend, optional
        Execution backend: "auto", "jax" or "cuda". An explicit accelerator
        request never silently falls back. Default is 'auto'.
    projector_model : ProjectorModel, optional
        Projection model: "auto", "ray" or "joseph". Auto selects Joseph unless an
        explicit detector grid requires rays. Default is 'auto'.
    """

    iterations: int = 50
    subsets: int = 1
    relaxation: float = 1.0
    acceleration: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    support: jax.Array | np.ndarray | None = None
    views_per_batch: int = 64
    backend: ProjectorBackend = "auto"
    projector_model: ProjectorModel = "auto"


def _inverse(value: jax.Array) -> jax.Array:
    positive = value > 0
    return jnp.where(positive, 1.0 / jnp.where(positive, value, 1.0), 0.0)


def _accelerated_steps(
    update: Callable[[jax.Array, jax.Array], jax.Array],
    initial: jax.Array,
    iterations: jax.Array,
) -> jax.Array:
    # Row/column normalization bounds the spectrum by one in the C^-1 norm.
    # Box/support projection is coordinatewise in that norm, so the usual
    # FISTA recurrence applies with a relaxation of at most one.
    def step(
        index: jax.Array, state: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        previous, extrapolated, time = state
        current = update(index, extrapolated)
        next_time = (1.0 + jnp.sqrt(1.0 + 4.0 * time * time)) / 2.0
        momentum = (time - 1.0) / next_time
        return current, current + momentum * (current - previous), next_time

    return jax.lax.fori_loop(0, iterations, step, (initial, initial, jnp.float32(1.0)))[0]


@partial(
    jax.jit,
    static_argnames=(
        "grid",
        "detector",
        "backend",
        "batch",
        "model",
        "subsets",
        "cache_columns",
        "axial",
        "acceleration",
        "stream",
        "column_device",
    ),
)
def _solve(
    poses: jax.Array,
    data: jax.Array,
    initial: jax.Array | None,
    support: jax.Array | None,
    det_grid: tuple[jax.Array, jax.Array] | None,
    iterations: jax.Array,
    relaxation: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch: int,
    model: str | ConeModel,
    subsets: int,
    cache_columns: bool,
    axial: bool,
    acceleration: bool,
    stream: bool,
    host_columns: jax.Array | None = None,
    column_device: jax.sharding.Sharding | None = None,
) -> jax.Array:
    select = subset_operators(
        poses, grid, detector, det_grid, backend, batch, model, subsets, stream=stream
    )
    shape = (grid.nz, grid.ny, grid.nx)
    normalization = select
    normal_rows = detector.nv
    normal_shape = shape
    if axial:
        from tomojax.geometry import grid_volume_origin

        normal_grid = replace(grid, nz=1, vol_origin=(*grid_volume_origin(grid)[:2], 0.0))
        normal_detector = replace(detector, nv=1, center=(detector.center[0], 0.0))
        normalization = subset_operators(
            poses, normal_grid, normal_detector, None, backend, batch, model, subsets
        )
        normal_rows, normal_shape = 1, (1, grid.ny, grid.nx)
    mask = jnp.ones(shape, jnp.float32) if support is None else (support != 0).astype(jnp.float32)

    def constrain(x: jax.Array) -> jax.Array:
        return jnp.where(mask != 0, jnp.clip(x, lower, upper), 0.0)

    initial = constrain(jnp.zeros(shape, jnp.float32) if initial is None else initial)

    def row_weights(index: jax.Array) -> jax.Array:
        forward, _, _ = (normalization if support is None else select)(index)
        return _inverse(forward(jnp.ones(normal_shape, jnp.float32) if support is None else mask))

    def column_weights(index: jax.Array) -> jax.Array:
        _, adjoint, _ = normalization(index)
        count = (poses.shape[0] + subsets - 1) // subsets
        return _inverse(adjoint(jnp.ones((count, normal_rows, detector.nu), jnp.float32)))

    rows = None if stream else jax.lax.map(row_weights, jnp.arange(subsets))
    # Avoid storing a volume for every subset on large scans. The exact subset
    # normalization is then recomputed at each update, not approximated by a
    # rescaled full-scan normalization. Cone updates share its ray geometry
    # with the residual transpose without sharing their accumulators.
    columns = (
        jax.lax.map(column_weights, jnp.arange(subsets))
        if cache_columns and host_columns is None
        else None
    )

    def update(step: jax.Array, x: jax.Array) -> jax.Array:
        index = step % subsets
        _, _, residual_adjoint = select(index)
        row_weights = None if rows is None else rows[index]
        row_support = None if support is None else mask
        if columns is None and not axial:
            gradient, norm = residual_adjoint(
                x, data, row_weights, row_support=row_support, return_column=True
            )
            column = _inverse(norm)
        else:
            column = (
                jax.device_put(host_columns[index], column_device)[None, :, :]
                if host_columns is not None
                else column_weights(index)
                if columns is None
                else columns[index]
            )
            gradient = residual_adjoint(x, data, row_weights, row_support=row_support)
        return constrain(x + relaxation * column * gradient)

    if acceleration:
        return _accelerated_steps(update, initial, iterations)
    return jax.lax.fori_loop(0, iterations * subsets, update, initial)


def _initial_volume(value: jax.Array | np.ndarray | None, *, stream: bool) -> jax.Array | None:
    """Release a streamed solve's pending initialization workspace before dispatch."""
    if value is None:
        return None
    initial = jnp.asarray(value, jnp.float32)
    # A warm FBP can still own large temporary buffers when XLA allocates the
    # iterative executable's workspace. Its output is a dependency anyway;
    # finish it first rather than requiring both workspaces simultaneously.
    return jax.block_until_ready(initial) if stream else initial


def _parameters(cfg: SIRTConfig, n: int) -> tuple[int, int, int, float, float]:
    """Validate iteration controls and bounds before choosing memory placement."""
    iterations = operator.index(cfg.iterations)
    subsets, batch = operator.index(cfg.subsets), operator.index(cfg.views_per_batch)
    if iterations < 0 or batch < 1 or not 1 <= subsets <= n:
        raise ValueError(
            "sirt: iterations >= 0, views_per_batch >= 1 and 1 <= subsets <= views required"
        )
    if not math.isfinite(cfg.relaxation) or not 0 < cfg.relaxation < 2:
        raise ValueError("sirt: relaxation must be finite and in (0, 2)")
    if cfg.acceleration and (subsets != 1 or cfg.relaxation > 1):
        raise ValueError("sirt: acceleration requires subsets=1 and relaxation <= 1")
    lower = -np.inf if cfg.lower_bound is None else float(cfg.lower_bound)
    upper = np.inf if cfg.upper_bound is None else float(cfg.upper_bound)
    if lower > upper:
        raise ValueError("sirt: lower_bound must not exceed upper_bound")
    return iterations, subsets, batch, lower, upper


def _storage(
    projections: jax.Array | np.ndarray,
    init_x: jax.Array | np.ndarray | None,
    *,
    grid: Grid,
    detector: Detector,
    cfg: SIRTConfig,
    views: int,
    backend: str,
    axial: bool,
) -> tuple[jax.Array | None, bool, bool]:
    """Place data and normalizers together, reserving the solve's working buffers."""
    from tomojax.backends import device_free_memory_bytes

    stream = should_stream(projections, fraction=0.2)
    initial = _initial_volume(init_x, stream=stream)
    free = device_free_memory_bytes()
    volume_bytes = grid.nx * grid.ny * grid.nz * 4
    column_bytes = cfg.subsets * grid.nx * grid.ny * (1 if axial else grid.nz) * 4
    budget = 256 * 1024**2 if free is None else free // 4
    cache = column_bytes <= budget
    if (
        not cache
        and free is not None
        and backend == "cuda"
        and cfg.projector_model != "ray"
        and not axial
        and cfg.support is None
        and cfg.iterations > 1
        and cfg.subsets > 1
        and not isinstance(projections, jax.Array)
    ):
        # Finish an asynchronous FBP before deciding whether its transient
        # workspace can instead hold exact, reusable subset columns.
        initial = _initial_volume(initial, stream=True)
        free = device_free_memory_bytes()
        count = (views + cfg.subsets - 1) // cfg.subsets
        image_bytes = min(cfg.views_per_batch, count) * detector.nu * detector.nv * 4
        # Six volumes cover state/update, gradient, column slice, normalization
        # work and optional projector packing; two images cover batch scratch.
        reserve = 6 * volume_bytes + 2 * image_bytes
        if free is not None and column_bytes + reserve <= free:
            stream, cache = True, True
    return initial, stream, cache


def sirt(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    init_x: jax.Array | np.ndarray | None = None,
    config: SIRTConfig | None = None,
    det_grid: tuple[jax.Array, jax.Array] | None = None,
    out: np.ndarray | None = None,
) -> tuple[jax.Array | np.ndarray, dict[str, object]]:
    """Matched FP32 SIRT/OS-SART; each iteration visits every subset once.

    Supports parallel, tilted and cone-beam scans on the existing matched
    projectors. Inputs and updates are FP32. An iteration is a complete sweep,
    not one subset update. No residual-based stopping or quality claim is made.
    Large independent axial planes use host slabs and invocation-owned host
    normalizers. Large host projection stacks
    stream with exact row recomputation, also allowing reusable column weights
    to fit in device memory. Cone-beam solves whose working set exceeds the
    device, or given ``out``, update the volume a z slab at a time; it stays
    on the device when it fits beside that sweep, and otherwise in host memory
    as a C-ordered NumPy volume (``out`` if given).

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
    config : SIRTConfig or None, optional
        Solver settings. None uses SIRTConfig defaults.
    det_grid : tuple of array-like or None, optional
        Explicit detector-coordinate arrays ``(u, v)``, each of shape ``(nv, nu)``.
        None uses the canonical detector grid.
    out : numpy.ndarray or None, optional
        Writable float32 ``(nz, ny, nx)`` host storage for a cone-beam result,
        including a memmap; it must not overlap the projections.

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
    >>> from tomojax.recon import sirt, SIRTConfig, project
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = project(geometry, np.ones((1, 4, 4), np.float32))
    >>> volume, info = sirt(
    ...     geometry,
    ...     geometry.grid,
    ...     geometry.detector,
    ...     data,
    ...     config=SIRTConfig(iterations=1),
    ... )
    >>> volume.shape
    (1, 4, 4)
    """
    return SIRTCall().solve(
        geometry, grid, detector, projections, init_x=init_x, config=config, det_grid=det_grid,
        out=out,
    )  # fmt: skip


_HOST_COLUMN_MIN_BYTES = 128 * 1024**2


@partial(jax.jit, static_argnames=("grid", "detector", "backend", "batch", "model", "subsets"))
def _axial_column(
    poses: jax.Array,
    index: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch: int,
    model: str | ConeModel,
    subsets: int,
) -> jax.Array:
    """Build one exact single-plane column map without a device-sized stack."""
    from tomojax.geometry import grid_volume_origin

    grid = replace(grid, nz=1, vol_origin=(*grid_volume_origin(grid)[:2], 0.0))
    detector = replace(detector, nv=1, center=(detector.center[0], 0.0))
    select = subset_operators(poses, grid, detector, None, backend, batch, model, subsets)
    _, adjoint, _ = select(index)
    count = (poses.shape[0] + subsets - 1) // subsets
    return _inverse(adjoint(jnp.ones((count, 1, detector.nu), jnp.float32)))[0]


class SIRTCall:
    """Own exact axial normalizers across one invocation's translated slabs.

    Each caller owns this object; no geometry- or shape-keyed global cache can
    leak between reconstructions. Slab depth/support may vary, but angles and
    transverse sampling stay those of the original independent-plane problem.
    """

    def __init__(self) -> None:
        self.columns: jax.Array | None = None

    def solve(
        self,
        geometry: Geometry,
        grid: Grid,
        detector: Detector,
        projections: jax.Array | np.ndarray,
        *,
        init_x: jax.Array | np.ndarray | None = None,
        config: SIRTConfig | None = None,
        det_grid: tuple[jax.Array, jax.Array] | None = None,
        out: np.ndarray | None = None,
    ) -> tuple[jax.Array | np.ndarray, dict[str, object]]:
        """Solve this invocation, retaining normalizers across its axial slabs."""
        cfg = SIRTConfig() if config is None else config
        validate_grid(grid, "sirt grid")
        n, _, _ = validate_projection_stack(
            projections, detector, geometry=geometry, context="sirt"
        )
        validate_detector_grid(det_grid, detector, context="sirt")
        iterations, subsets, batch, lower, upper = _parameters(cfg, n)
        for name, value in (("init_x", init_x), ("support", cfg.support)):
            if value is not None:
                validate_volume(value, grid, context="sirt", name=name)
        model, backend = resolve_geometry_projector(
            geometry,
            cfg.projector_model,
            cfg.backend,
            detector=detector,
            det_grid=det_grid,
            context="sirt",
        )
        axial = (
            model == "joseph" and det_grid is None and independent_planes(geometry, grid, detector)
        )
        cone, rows = isinstance(model, ConeModel), isinstance(projections, RowReader)
        if (out is not None or rows) and not cone:
            raise ValueError("sirt: out and row readers need a cone-beam geometry")
        hosted = out is not None or rows or host_volumes(grid, 5 + 2 * cfg.acceleration, backend)
        if axial:
            result = self._slabs(geometry, grid, detector, projections, init_x, cfg)
            if result is not None:
                return result
        elif isinstance(model, ConeModel) and hosted:
            from ._sirt_host import solve as solve_host

            return solve_host(
                geometry, grid, detector, projections,
                None if init_x is None else np.asarray(init_x), cfg,
                model=model, backend=backend, poses=stack_view_poses(geometry, n),
                bounds=(lower, upper), out=out,
            )  # fmt: skip
        initial, stream, cache_columns = _storage(
            projections,
            init_x,
            grid=grid,
            detector=detector,
            cfg=cfg,
            views=n,
            backend=backend,
            axial=axial,
        )
        poses = stack_view_poses(geometry, n)
        host_columns, column_device = self._host_columns(
            poses,
            grid=grid,
            detector=detector,
            backend=backend,
            batch=batch,
            model=model,
            subsets=subsets,
            enabled=axial and cache_columns and iterations > 0,
        )
        source = (
            host_source(np.asarray(projections))
            if stream
            else nullcontext(jnp.asarray(projections, jnp.float32))
        )
        with source as data:
            volume = _solve(
                poses, data, initial,
                None if cfg.support is None else jnp.asarray(cfg.support),
                det_grid, jnp.int32(iterations), jnp.float32(cfg.relaxation),
                jnp.float32(lower), jnp.float32(upper), grid=grid, detector=detector,
                backend=backend, batch=batch, model=model, subsets=subsets,
                cache_columns=cache_columns,
                axial=axial, acceleration=cfg.acceleration, stream=stream,
                host_columns=host_columns, column_device=column_device,
            )  # fmt: skip
            if stream:
                volume.block_until_ready()  # callbacks must finish before unregistering the source
        return volume, {
            "effective_iterations": iterations,
            "subset_updates": iterations * subsets,
            "termination": "iteration_limit",
            "stream_projections": stream,
            "projector_backend": backend,
            "projector_model": model,
        }

    def _host_columns(
        self,
        poses: jax.Array,
        *,
        grid: Grid,
        detector: Detector,
        backend: str,
        batch: int,
        model: str | ConeModel,
        subsets: int,
        enabled: bool,
    ) -> tuple[jax.Array | None, jax.sharding.Sharding | None]:
        """Retain large exact maps on host, leaving small device caches unchanged."""
        host_columns = None
        column_device = None
        column_bytes = subsets * grid.nx * grid.ny * 4
        if enabled and column_bytes >= _HOST_COLUMN_MIN_BYTES:
            device = next(iter(poses.devices()))
            column_device = jax.sharding.SingleDeviceSharding(device)
            if self.columns is None:
                maps = np.empty((subsets, grid.ny, grid.nx), np.float32)
                for index in range(subsets):
                    maps[index] = np.asarray(
                        _axial_column(
                            poses,
                            jnp.int32(index),
                            grid=grid,
                            detector=detector,
                            backend=backend,
                            batch=batch,
                            model=model,
                            subsets=subsets,
                        )
                    )
                host = (
                    jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
                    if device.platform == "gpu"
                    else column_device
                )
                self.columns = jax.block_until_ready(jax.device_put(maps, host))
            host_columns = self.columns
        return host_columns, column_device

    def _slabs(
        self,
        geometry: Geometry,
        grid: Grid,
        detector: Detector,
        projections: jax.Array | np.ndarray,
        init_x: jax.Array | np.ndarray | None,
        cfg: SIRTConfig,
    ) -> tuple[np.ndarray, dict[str, object]] | None:
        """Plan translated independent planes while retaining this call's cache."""

        def solve_slab(
            geometry: ParallelGeometry,
            data: jax.Array | np.ndarray,
            initial: jax.Array | np.ndarray | None,
            support: jax.Array | np.ndarray | None,
        ) -> tuple[jax.Array | np.ndarray, dict[str, object]]:
            return self.solve(
                geometry,
                geometry.grid,
                geometry.detector,
                data,
                init_x=initial,
                config=replace(cfg, support=support),
            )

        return axial_slabs(
            geometry,
            grid,
            detector,
            projections,
            initial=init_x,
            support=cfg.support,
            volume_buffers=3 + min(cfg.subsets, 16) + 2 * cfg.acceleration,
            projection_buffers=2 + 2 / cfg.subsets,
            solve=solve_slab,
        )
