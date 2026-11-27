"""Row/column-normalized simultaneous and ordered-subset algebraic reconstruction."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
import math
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import progress
from tomojax._parameters import ProjectorBackend, check_backend
from tomojax.backends.host_stream import host_source, should_stream
from tomojax.core.devices import announce
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import (
    validate_detector_grid,
    validate_projection_stack,
    validate_volume,
)
from tomojax.errors import InvalidParameter
from tomojax.recon._axial_slabs import axial_slabs, independent_planes
from tomojax.recon._chunks import Chunked, fixed_updates, partial_sweep
from tomojax.recon._cone_slabs import RowReader, host_volumes
from tomojax.recon._discrepancy import sirt_slab_report, sirt_stopping
from tomojax.recon._projection import (
    ConeModel,
    resolve_geometry_projector,
    subset_operators,
)
from tomojax.recon._solver_validation import check_count, check_subset_count
from tomojax.recon._volume_storage import (
    Model,
    pack_volume,
    preview_planes,
    resident_model,
    restore_volume,
    volume_shape,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry

    from ._projection import SubsetOperators

    type Select = Callable[[jax.Array | int], SubsetOperators]


@dataclass(frozen=True, kw_only=True)
class SIRTConfig:
    """SIRT, or OS-SART with ``subsets > 1`` (one view per subset is SART).

    Each iteration visits all interleaved subsets once, in input view order.
    A subset updates ``x += relaxation * C * A.T * R * (data - A*x)``, where
    ``R = 1 / (A*support)`` and ``C = 1 / (A.T*ones)``; zero denominators give
    zero weights. The forward operator and transpose are exactly matched.
    ``relaxation`` must be in (0, 2); ordered subsets can cycle on inconsistent
    data, so use fewer subsets or lower relaxation for the final refinement.

    With ``iterations=None`` and ``subsets=1``, full-data SIRT stops when the
    projection residual RMS at the sweep input reaches an estimate of detector
    noise plus sampling error, capped at 50 sweeps (a one-pixel detector runs to
    the 50-sweep cap). With ``acceleration=None`` (the default), full-data SIRT
    also enables fixed FISTA momentum whenever ``iterations=None`` and
    ``relaxation <= 1``. Pass ``acceleration=False`` for plain SIRT with the
    same discrepancy stop. Passing an explicit ``iterations`` disables
    discrepancy stopping and runs that many plain sweeps unless
    ``acceleration=True`` is also set (requires ``subsets=1`` and
    ``relaxation <= 1``). Ordered subsets (``subsets > 1``) run a fixed sweep
    budget (default 50, no momentum). Momentum reduces the sweeps needed for
    the same weighted least-squares objective without changing the projector.

    ``lower_bound`` and ``upper_bound`` clip each update. ``support`` is a volume
    mask: zero excludes a voxel from reconstruction and fixes it to zero,
    including in the initial volume; nonzero includes it. Bounds apply only
    inside that support. Only nonnegative projector weights are used.

    Parameters
    ----------
    iterations : int, optional
        Number of complete sweeps. None stops a full-data solve at the estimated
        projection discrepancy or 50 sweeps. Ordered subsets use 50 sweeps.
    subsets : int, optional
        Number of interleaved view subsets visited per sweep; one uses the
        simultaneous method. Default is 1.
    relaxation : float, optional
        Update scale strictly between 0 and 2. Default is 1.0.
    acceleration : bool, optional
        Apply FISTA momentum to full-data SIRT; requires subsets=1 and relaxation <=
        1. None enables momentum for an eligible solve with an omitted budget.
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
    """

    method: ClassVar[Literal["sirt"]] = "sirt"

    iterations: int | None = None
    subsets: int = 1
    relaxation: float = 1.0
    acceleration: bool | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    support: jax.Array | np.ndarray | None = None
    views_per_batch: int = 64
    backend: ProjectorBackend = "auto"

    @property
    def accelerated(self) -> bool:
        """Whether this solve uses full-data momentum.

        Returns
        -------
        bool
            Effective acceleration for the requested budget, relaxation and subsets.
        """
        if self.acceleration is not None:
            return self.acceleration
        return self.iterations is None and self.subsets == 1 and self.relaxation <= 1

    def __post_init__(self) -> None:
        check_backend(self.backend, "sirt")
        if self.iterations is not None:
            check_count(self.iterations, "iterations", minimum=0, context="sirt")
        for name, minimum in (("views_per_batch", 1), ("subsets", 1)):
            check_count(getattr(self, name), name, minimum=minimum, context="sirt")
        if not math.isfinite(self.relaxation) or not 0 < self.relaxation < 2:
            raise InvalidParameter("relaxation", "sirt: relaxation must be finite and in (0, 2)")
        if self.acceleration and (self.subsets != 1 or self.relaxation > 1):
            raise InvalidParameter(
                "acceleration", "sirt: acceleration requires subsets=1 and relaxation <= 1"
            )
        lower = -np.inf if self.lower_bound is None else float(self.lower_bound)
        upper = np.inf if self.upper_bound is None else float(self.upper_bound)
        if lower > upper:
            raise InvalidParameter("lower_bound", "sirt: lower_bound must not exceed upper_bound")


def _inverse(value: jax.Array) -> jax.Array:
    positive = value > 0
    return jnp.where(positive, 1.0 / jnp.where(positive, value, 1.0), 0.0)


def _normalization(
    select: Select,
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch: int,
    model: Model,
    subsets: int,
    *,
    axial: bool,
) -> tuple[Select, int, tuple[int, int, int]]:
    """Use shared one-plane normalizers only for exactly independent axial rows."""
    if not axial:
        return select, detector.nv, volume_shape(model, grid)
    from tomojax.geometry import grid_volume_origin

    normal_grid = replace(grid, nz=1, vol_origin=(*grid_volume_origin(grid)[:2], 0.0))
    normal_detector = replace(detector, nv=1, center=(detector.center[0], 0.0))
    normalization = subset_operators(
        poses, normal_grid, normal_detector, None, backend, batch, model, subsets
    )
    return normalization, 1, volume_shape(model, normal_grid)


class _Momentum(NamedTuple):
    previous: jax.Array
    current: jax.Array
    time: jax.Array
    momentum: jax.Array


type _Point = jax.Array | _Momentum
type _Sweep = tuple[jax.Array, jax.Array, _Point]


_SIRT_STATIC = (
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
    "normalizer_device",
)


@partial(jax.jit, static_argnames=_SIRT_STATIC, donate_argnames=("state",))
def _segment(
    poses: jax.Array,
    data: jax.Array,
    initial: jax.Array | None,
    support: jax.Array | None,
    det_grid: tuple[jax.Array, jax.Array] | None,
    relaxation: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    state: jax.Array | _Momentum | None,
    first: int,
    end: int,
    norms: tuple[jax.Array | None, jax.Array | None] | None,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch: int,
    model: Model,
    subsets: int,
    cache_columns: bool,
    axial: bool,
    acceleration: bool,
    stream: bool,
    host_columns: jax.Array | None = None,
    column_device: jax.sharding.Sharding | None = None,
    normalizer_device: jax.sharding.Sharding | None = None,
    target: jax.Array | None = None,
) -> Any:
    initial, support = pack_volume(initial, model), pack_volume(support, model)
    select = subset_operators(
        poses, grid, detector, det_grid, backend, batch, model, subsets,
        stream=stream, measure=target is not None,
    )  # fmt: skip
    shape = volume_shape(model, grid)
    normalization, normal_rows, normal_shape = _normalization(
        select, poses, grid, detector, backend, batch, model, subsets, axial=axial
    )
    mask = jnp.ones(shape, np.float32) if support is None else (support != 0).astype(jnp.float32)

    def constrain(x: jax.Array) -> jax.Array:
        return jnp.where(mask != 0, jnp.clip(x, lower, upper), 0.0)

    def row_weights(index: jax.Array) -> jax.Array:
        forward, _, _ = (normalization if support is None else select)(index)
        return _inverse(forward(jnp.ones(normal_shape, np.float32) if support is None else mask))

    def column_weights(index: jax.Array) -> jax.Array:
        _, adjoint, _ = normalization(index)
        count = (poses.shape[0] + subsets - 1) // subsets
        return _inverse(adjoint(jnp.ones((count, normal_rows, detector.nu), np.float32)))

    if state is None:
        initial = constrain(jnp.zeros(shape, np.float32) if initial is None else initial)

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

        state = (
            _Momentum(initial, initial, jnp.float32(1), jnp.float32(0)) if acceleration else initial
        )
        return state, (rows, columns)

    assert norms is not None
    rows, columns = norms if normalizer_device is None else jax.device_put(norms, normalizer_device)

    def update(step: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        index = step % subsets
        _, _, residual_adjoint = select(index)
        row_weights = None if rows is None else rows[index]
        row_support = None if support is None else mask
        if columns is None and not axial:
            gradient, norm, energy = residual_adjoint(
                x, data, row_weights, row_support=row_support, return_column=True
            )
            assert norm is not None
            column = _inverse(norm)
        else:
            column = (
                jax.device_put(host_columns[index], column_device)
                if host_columns is not None
                else column_weights(index)
                if columns is None
                else columns[index]
            )
            gradient, _, energy = residual_adjoint(x, data, row_weights, row_support=row_support)
        return constrain(x + relaxation * column * gradient), energy

    return _sweeps(update, state, first, end, subsets, target)


def _sweeps(
    update: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    state: _Point,
    first: int,
    end: int,
    subsets: int,
    target: jax.Array | None,
) -> _Point | _Sweep:
    """Advance the point to the budget or first measured discrepancy crossing."""

    def sweep(step: jax.Array, point: _Point) -> tuple[_Point, jax.Array]:
        if not isinstance(point, _Momentum):
            return update(step, point)
        previous, current, time, momentum = point
        extrapolated = current + momentum * (current - previous)
        new, energy = update(step, extrapolated)
        next_time = (1.0 + jnp.sqrt(1.0 + 4.0 * time * time)) / 2.0
        return _Momentum(current, new, next_time, (time - 1.0) / next_time), energy

    if target is None:

        def fixed(index: jax.Array, point: _Point) -> _Point:
            return sweep(index, point)[0]

        return jax.lax.fori_loop(first * subsets, end * subsets, fixed, state)

    def step(carry: _Sweep) -> _Sweep:
        k, _, point = carry
        point, energy = sweep(k, point)
        return k + 1, energy, point

    return jax.lax.while_loop(
        lambda s: (s[0] < end) & (s[1] > target),
        step,
        (jnp.int32(first), jnp.float32(jnp.inf), state),
    )


def _sirt_program(
    poses: jax.Array,
    data: jax.Array,
    initial: jax.Array | None,
    support: jax.Array | None,
    det_grid: tuple[jax.Array, jax.Array] | None,
    iterations: int,
    relaxation: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    *,
    drive: Callable,
    target: jax.Array | None = None,
    **options: Any,
) -> tuple[jax.Array, int | jax.Array, jax.Array | None]:
    segment = partial(
        _segment,
        poses,
        data,
        initial,
        support,
        det_grid,
        relaxation,
        lower,
        upper,
        target=target,
        **options,
    )
    state, norms = segment(None, 0, 0, None)
    if (device := options.get("normalizer_device")) is not None:
        # Keep these maps with the host partitions between chunks. Their device
        # copies then share the compiled workspace with the projector scratch.
        norms = jax.block_until_ready(jax.device_put(norms, device.with_memory_kind("pinned_host")))

    def sample(s: jax.Array | _Momentum) -> dict[str, progress.Sample]:
        return preview_planes(
            s.current if isinstance(s, _Momentum) else s, options["model"], options["grid"]
        )

    count, energy = iterations, None
    if target is None:
        result = fixed_updates(state, iterations, partial(segment, norms=norms), drive, sample)
    else:

        def advance(carry: _Sweep, end: int) -> _Sweep:
            first, _, point = carry
            return segment(point, first, end, norms=norms)

        def completed(carry: _Sweep) -> tuple[int, bool, dict[str, float]]:
            return int(carry[0]), bool(carry[1] <= target), {}

        def preview(carry: _Sweep) -> dict[str, progress.Sample]:
            return sample(carry[2])

        count, energy, result = drive(
            (jnp.int32(0), jnp.float32(jnp.inf), state),
            iterations,
            advance,
            completed,
            preview,
        )
    return result.current if isinstance(result, _Momentum) else result, count, energy


_solve = Chunked(_sirt_program, static_argnames=_SIRT_STATIC)


def _initial_volume(value: jax.Array | np.ndarray | None, *, stream: bool) -> jax.Array | None:
    """Release a streamed solve's pending initialization workspace before dispatch."""
    if value is None:
        return None
    initial = jnp.asarray(value, np.float32)
    # A warm FBP can still own large temporary buffers when XLA allocates the
    # iterative executable's workspace. Its output is a dependency anyway;
    # finish it first rather than requiring both workspaces simultaneously.
    return jax.block_until_ready(initial) if stream else initial


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
        and not axial
        and cfg.support is None
        and (cfg.iterations is None or cfg.iterations > 1)
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


@progress.reports_iterations(50)
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

    This solver fits the supplied data as given, and the workflows ``tj.reconstruct``
    and ``tj.reconstruct_slabs`` handle scans extending beyond the detector.

    Supports parallel, tilted and cone-beam scans on the existing matched
    projectors. Inputs and updates are FP32. An iteration is a complete sweep,
    not one subset update. With an omitted budget, a full-data sweep measures the
    projection residual at its input point and stops after the first sweep that
    reaches the estimated measurement and sampling error.
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
        Solver diagnostics and effective work: ``effective_iterations``,
        ``subset_updates``, ``termination`` (``"projection_discrepancy"`` or
        ``"iteration_limit"``), ``early_stop``, resolved ``acceleration``,
        ``measurement_noise_rms``, ``projection_error_rms``, and
        ``pre_update_residual_rms`` (measured at the point projected for the
        last sweep, including momentum when accelerated; the returned volume
        completes the sweep and respects bounds and support). Independent axial
        slab solves also record per-slab counts and residuals in
        ``slab_stopping``. Replay an automatic run by passing both
        ``iterations=info["effective_iterations"]`` and
        ``acceleration=info["acceleration"]``. A small residual or iteration
        count does not establish reconstruction accuracy.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.

    Examples
    --------
    >>> import numpy as np
    >>> import tomojax as tj
    >>> from tomojax.recon import SIRTConfig, sirt
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = tj.project(geometry, np.ones((1, 4, 4), np.float32))
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
    model: Model,
    subsets: int,
) -> jax.Array:
    """Build one exact single-plane column map, in volume storage, without a device stack."""
    from tomojax.geometry import grid_volume_origin

    grid = replace(grid, nz=1, vol_origin=(*grid_volume_origin(grid)[:2], 0.0))
    detector = replace(detector, nv=1, center=(detector.center[0], 0.0))
    select = subset_operators(poses, grid, detector, None, backend, batch, model, subsets)
    _, adjoint, _ = select(index)
    count = (poses.shape[0] + subsets - 1) // subsets
    return _inverse(adjoint(jnp.ones((count, 1, detector.nu), np.float32)))


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
        announce(array=projections)
        cfg = SIRTConfig() if config is None else config
        n, _, _ = validate_projection_stack(
            projections, detector, geometry=geometry, context="sirt"
        )
        validate_detector_grid(det_grid, detector, context="sirt")
        automatic = cfg.iterations is None and cfg.subsets == 1
        acceleration = cfg.accelerated
        iterations = 50 if cfg.iterations is None else cfg.iterations
        check_subset_count("sirt", cfg.subsets, n)
        lower = -np.inf if cfg.lower_bound is None else cfg.lower_bound
        upper = np.inf if cfg.upper_bound is None else cfg.upper_bound
        for name, value in (("init_x", init_x), ("support", cfg.support)):
            if value is not None:
                _ = validate_volume(value, grid, context="sirt", name=name)
        model, backend = resolve_geometry_projector(
            geometry,
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
        hosted = out is not None or rows or host_volumes(grid, cfg, backend)
        if axial:
            result = self._slabs(geometry, grid, detector, projections, init_x, cfg)
            if result is not None:
                return result
        from ._discrepancy import estimate

        discrepancy = (
            estimate(projections, geometry, grid, detector, linear=det_grid is None)
            if automatic
            else None
        )
        target = None if discrepancy is None else discrepancy.target(math.prod(projections.shape))
        cfg = replace(cfg, iterations=iterations, acceleration=acceleration)
        if isinstance(model, ConeModel) and hosted:
            from ._sirt_host import solve as solve_host

            volume, info = solve_host(
                geometry, grid, detector, projections,
                None if init_x is None else np.asarray(init_x), cfg,
                model=model, backend=backend, poses=stack_view_poses(geometry, n),
                bounds=(lower, upper), out=out, target=target,
            )  # fmt: skip
        else:
            volume, info = self._resident(
                grid,
                detector,
                projections,
                init_x,
                det_grid,
                cfg,
                stack_view_poses(geometry, n),
                (lower, upper),
                backend=backend,
                model=model,
                axial=axial,
                target=target,
            )
        info["measurement_noise_rms"] = None if discrepancy is None else discrepancy.noise
        return volume, info

    def _resident(
        self,
        grid: Grid,
        detector: Detector,
        projections: jax.Array | np.ndarray,
        init_x: jax.Array | np.ndarray | None,
        det_grid: tuple[jax.Array, jax.Array] | None,
        cfg: SIRTConfig,
        poses: jax.Array,
        bounds: tuple[float, float],
        *,
        backend: str,
        model: Model,
        axial: bool,
        target: float | None,
    ) -> tuple[jax.Array, dict[str, object]]:
        """Place the normalizers and solve while the iterate fits on the device."""
        lower, upper = bounds
        model = resident_model(model, backend)
        iterations, subsets, batch = cfg.iterations, cfg.subsets, cfg.views_per_batch
        assert iterations is not None
        initial, stream, cache_columns = _storage(
            projections,
            init_x,
            grid=grid,
            detector=detector,
            cfg=cfg,
            views=poses.shape[0],
            backend=backend,
            axial=axial,
        )
        host_columns = None
        column_device = None
        column_bytes = subsets * grid.nx * grid.ny * 4
        if axial and cache_columns and iterations and column_bytes >= _HOST_COLUMN_MIN_BYTES:
            device = next(iter(poses.devices()))
            column_device = jax.sharding.SingleDeviceSharding(device)
            if self.columns is None:
                plane = replace(grid, nz=1)
                maps = np.empty((subsets, *volume_shape(model, plane)), np.float32)
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
        normalizer_device = None
        if (
            partial_sweep.get() is not None
            and progress.for_unit("iterations") is not None
            and next(iter(poses.devices())).platform == "gpu"
        ):
            normalizer_device = poses.sharding
        source = (
            host_source(np.asarray(projections))
            if stream
            else nullcontext(jnp.asarray(projections, np.float32))
        )
        with source as data:
            volume, count, energy = _solve(
                poses, data, initial,
                None if cfg.support is None else jnp.asarray(cfg.support),
                det_grid, iterations, jnp.float32(cfg.relaxation),
                jnp.float32(lower), jnp.float32(upper), grid=grid, detector=detector,
                backend=backend, batch=batch, model=model, subsets=subsets,
                cache_columns=cache_columns,
                axial=axial, acceleration=cfg.acceleration, stream=stream,
                host_columns=host_columns, column_device=column_device,
                normalizer_device=normalizer_device, target=target,
            )  # fmt: skip
            if stream:
                volume = volume.block_until_ready()  # Finish callbacks before unregistering.
        return restore_volume(volume, model), {
            **sirt_stopping(int(count), energy, target, math.prod(projections.shape)),
            "acceleration": cfg.acceleration,
            "subset_updates": int(count) * subsets,
            "stream": stream,
            "projector_backend": backend,
        }

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
            _section: slice,
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
            config=cfg,
            solve=solve_slab,
            summarize=sirt_slab_report if cfg.iterations is None and cfg.subsets == 1 else None,
        )
