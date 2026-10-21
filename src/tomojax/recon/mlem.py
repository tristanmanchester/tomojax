"""Poisson maximum-likelihood reconstruction with matched nonnegative projectors."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
import operator
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends import device_free_memory_bytes
from tomojax.backends.host_stream import host_source, should_stream
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import (
    validate_detector_grid,
    validate_grid,
    validate_projection_stack,
    validate_volume,
)
from tomojax.geometry import grid_volume_origin
from tomojax.recon._axial_slabs import independent_planes
from tomojax.recon._cone_slabs import RowReader, host_volumes
from tomojax.recon._host_arrays import validate_host_views
from tomojax.recon._projection import (
    ConeModel,
    ProjectorBackend,
    ProjectorModel,
    poisson_operators,
    resolve_geometry_projector,
)

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Geometry, Grid

_SENSITIVITY_CACHE_BYTES = 256 * 2**20


def _cache_sensitivities(
    nbytes: int,
    subsets: int,
    iterations: int,
    *,
    workspace_bytes: int,
    initial: jax.Array | None = None,
) -> bool:
    if subsets == 1 or nbytes <= _SENSITIVITY_CACHE_BYTES:
        return True
    if iterations <= 1:
        return False  # A single sweep cannot reuse the extra cached volumes.
    # An asynchronous initializer can still retain its inputs and temporaries.
    # Finish it before measuring reusable memory for a large sensitivity cache.
    if initial is not None:
        initial.block_until_ready()
    free = device_free_memory_bytes()
    # A cache is only part of the compiled solve's live memory. Its separate
    # allocation can use the remaining pool after reserving the working buffers;
    # a fixed fraction would needlessly recompute norms when that reserve fits.
    return free is not None and nbytes <= free - workspace_bytes


@dataclass(frozen=True, kw_only=True)
class MLEMConfig:
    """Reconstruct nonnegative linear Poisson measurements, ``y ~ Poisson(A x)``.

    ``subsets=1`` is MLEM; larger values use ordered-subsets EM (OSEM).
    Each iteration visits every view once, in angle-interleaved groups. OSEM
    accelerates early progress but can cycle; use one subset for convergence.
    ``support`` is a boolean mask broadcastable to the volume shape: voxels
    outside it stay zero. The default initial volume is one inside the observed support.
    A supplied zero voxel stays zero under multiplicative updates.
    ``tj.reconstruct(..., warm_start=True)`` instead starts from nonnegative
    FBP/FDK, including its zero voxels. This can improve early image quality,
    but fixes the initial zero set for all subsequent EM updates.

    These are linear emission-style measurements, not transmission counts
    with mean ``I0 * exp(-A x)``. Applying MLEM to nonnegative log-transformed
    attenuation data is possible, but does not give it Poisson statistics.
    The projectors are the same matched pair as :class:`CGLSConfig`, with
    nonnegative interpolation weights. A single sensitivity is retained; multiple
    subset sensitivities are cached within 256 MiB, or in available memory
    after reserving working buffers when reused across multiple sweeps;
    otherwise they are recomputed to bound memory use. Exactly independent
    axial planes share their single-plane sensitivities.

    Parameters
    ----------
    iterations : int, optional
        Maximum number of solver updates. For SIRT and MLEM, an update is a complete
        subset sweep. Default is 50.
    subsets : int, optional
        Number of interleaved view subsets visited per sweep; one uses the
        simultaneous method. Default is 1.
    support : array-like | None, optional
        Boolean support mask broadcastable to the volume shape; excluded voxels
        remain zero. None uses the entire grid.
    views_per_batch : int, optional
        Number of views processed together to bound temporary storage. Default is
        32.
    backend : ProjectorBackend, optional
        Execution backend: "auto", "jax" or "cuda". An explicit accelerator
        request never silently falls back. Default is 'auto'.
    projector_model : ProjectorModel, optional
        Projection model: "auto", "ray" or "joseph". Auto selects Joseph unless an
        explicit detector grid requires rays. Default is 'auto'.
    """

    iterations: int = 50
    subsets: int = 1
    support: jax.Array | np.ndarray | None = None
    views_per_batch: int = 32
    backend: ProjectorBackend = "auto"
    projector_model: ProjectorModel = "auto"


@partial(jax.jit, static_argnames=("grid", "detector", "backend", "batch_size", "model", "axial"))
def _normalizer(
    poses: jax.Array,
    indices: jax.Array,
    valid: jax.Array,
    det_grid: tuple[jax.Array, jax.Array] | None,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str | ConeModel,
    axial: bool,
) -> jax.Array:
    """Own a large norm stack separately from the iterative solver's workspace."""
    if axial:
        grid = replace(grid, nz=1, vol_origin=(*grid_volume_origin(grid)[:2], 0.0))
        detector = replace(detector, nv=1, center=(detector.center[0], 0.0))
    sensitivity, _ = poisson_operators(
        poses, grid, detector, det_grid, backend, batch_size, model, indices, valid
    )
    return jax.lax.map(sensitivity, jnp.arange(indices.shape[0]))


@partial(
    jax.jit,
    static_argnames=(
        "grid",
        "detector",
        "backend",
        "batch_size",
        "model",
        "cache",
        "stream",
        "axial",
    ),
)
def _solve(
    poses: jax.Array,
    data: jax.Array,
    initial: jax.Array | None,
    support: jax.Array | None,
    indices: jax.Array,
    valid: jax.Array,
    iterations: jax.Array,
    det_grid: tuple[jax.Array, jax.Array] | None,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str | ConeModel,
    cache: bool,
    stream: bool,
    axial: bool,
    sensitivities: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    sensitivity, correction = poisson_operators(
        poses, grid, detector, det_grid, backend, batch_size, model, indices, valid, stream=stream
    )
    if axial:
        # Equal axial rows share A.T 1, even when activity/support varies by plane.
        normal_grid = replace(grid, nz=1, vol_origin=(*grid_volume_origin(grid)[:2], 0.0))
        normal_detector = replace(detector, nv=1, center=(detector.center[0], 0.0))
        sensitivity, _ = poisson_operators(
            poses, normal_grid, normal_detector, None, backend, batch_size, model, indices, valid
        )
    shape = (grid.nx, grid.ny, grid.nz)
    x = jnp.ones(shape, jnp.float32) if initial is None else initial
    if support is not None:
        x = jnp.where(support, x, 0.0)
    inputs_valid = jnp.bool_(True) if stream else jnp.all(jnp.isfinite(data) & (data >= 0))
    inputs_valid &= jnp.all(jnp.isfinite(x) & (x >= 0)) & jnp.all(jnp.isfinite(poses))
    subsets = indices.shape[0]
    # A full subset stack on a large volume can exceed device memory. Retain
    # only one current sensitivity in that case, independent of subset count.
    if sensitivities is None and cache:
        sensitivities = jax.lax.map(sensitivity, jnp.arange(subsets))
    # A wholly unobserved voxel has no likelihood information: choose zero,
    # not an arbitrary unit activity. Visibility must include every subset;
    # a voxel missed by one subset can still be measured by another.
    observed = jnp.zeros(shape, bool)
    if sensitivities is not None:
        # A stack-shaped > 0 comparison can retain a second full zero stack
        # alongside the map's writable cache. Compare one volume at a time.
        observed = jax.lax.fori_loop(
            0, subsets, lambda i, seen: seen | (sensitivities[i] > 0), observed
        )

    def step(k: jax.Array, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        volume, seen = state
        i = k % subsets
        norm = sensitivity(i) if sensitivities is None else sensitivities[i]
        update = correction(i, volume, data)
        volume = volume * jnp.where(norm > 0, update / jnp.where(norm > 0, norm, 1), 1)
        return volume, seen if cache else seen | (norm > 0)

    # Unseen voxels cannot affect any ray, so zeroing them can wait until the
    # end. Collecting visibility from each uncached norm avoids an extra full
    # backprojection pass, without dropping voxels missed by only one subset.
    x, observed = jax.lax.fori_loop(0, iterations * subsets, step, (x, observed))
    if not cache:
        observed = jax.lax.cond(
            iterations == 0,
            lambda: jax.lax.fori_loop(
                0, subsets, lambda i, seen: seen | (sensitivity(i) > 0), jnp.zeros(shape, bool)
            ),
            lambda: observed,
        )
    return jnp.where(observed, x, 0.0), inputs_valid


def _device(array: object) -> jax.Array:
    if isinstance(array, jax.Array):
        return jnp.asarray(array, jnp.float32)
    return jax.device_put(np.asarray(array, np.float32))


def _run_device(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    config: MLEMConfig,
    model: str | ConeModel,
    backend: str,
    initial: jax.Array | np.ndarray | None,
    det_grid: tuple[jax.Array, jax.Array] | None,
) -> jax.Array:
    """The existing resident-volume update, retaining view-streaming and norm placement."""
    iterations, subsets, batch = map(
        operator.index, (config.iterations, config.subsets, config.views_per_batch)
    )
    shape = (grid.nx, grid.ny, grid.nz)
    n = projections.shape[0]
    support = (
        None
        if config.support is None
        else jnp.broadcast_to(jnp.asarray(config.support, bool), shape)
    )
    axial = model == "joseph" and det_grid is None and independent_planes(geometry, grid, detector)
    order = np.argsort(np.asarray(getattr(geometry, "angles", np.arange(n))), kind="stable")
    count = (n + subsets - 1) // subsets
    indices = np.arange(subsets)[:, None] + subsets * np.arange(count)[None, :]
    valid = indices < n
    indices = order[np.minimum(indices, n - 1)].astype(np.int32)
    initial = None if initial is None else _device(initial)
    stream = should_stream(projections)
    if stream:
        validate_host_views(np.asarray(projections))
        if np.min(projections) < 0:
            raise ValueError("mlem: projections must be nonnegative")
    context = host_source(np.asarray(projections)) if stream else nullcontext(_device(projections))
    with context as data:
        # Query after these persistent inputs occupy their device storage.
        volume_bytes = int(np.prod(shape)) * 4
        norm_bytes = grid.nx * grid.ny * 4 if axial else volume_bytes
        image_bytes = detector.nv * detector.nu * 4
        # Four volume buffers cover the iterate, update, current norm and
        # visibility/output workspace. Both resident and streamed data need
        # only the current predicted/measured view batches.
        workspace = 4 * volume_bytes + 2 * min(batch, count) * image_bytes
        cache = _cache_sensitivities(
            subsets * norm_bytes, subsets, iterations, workspace_bytes=workspace, initial=initial
        )
        poses = stack_view_poses(geometry, n)
        device_indices, device_valid = jax.device_put((indices, valid))
        norms = None
        if cache and subsets * norm_bytes > _SENSITIVITY_CACHE_BYTES:
            norms = _normalizer(
                poses, device_indices, device_valid, det_grid, grid=grid, detector=detector,
                backend=backend, batch_size=batch, model=model, axial=axial,
            )  # fmt: skip
            # Release builder scratch before allocating the iteration workspace.
            norms.block_until_ready()
        volume, good = _solve(
            poses, data, initial, support,
            device_indices, device_valid, jnp.int32(iterations), det_grid,
            grid=grid, detector=detector, backend=backend, batch_size=batch, model=model,
            cache=cache, stream=stream, axial=axial, sensitivities=norms,
        )  # fmt: skip
        if stream:
            volume.block_until_ready()  # Host registrations must outlive every callback.
    if not bool(good):
        raise ValueError("mlem: projections and initial volume must be finite and nonnegative")
    return volume


def mlem(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    init_x: jax.Array | np.ndarray | None = None,
    config: MLEMConfig | None = None,
    det_grid: tuple[jax.Array, jax.Array] | None = None,
    out: np.ndarray | None = None,
) -> tuple[jax.Array | np.ndarray, dict[str, object]]:
    """MLEM or OSEM with the exact transpose; see :class:`MLEMConfig`.

    Returns the FP32 volume and an iteration record, without a convergence
    claim. Wholly unobserved voxels are zero; zero measurements drive
    observed voxels to zero. Large host measurement stacks stream in bounded
    view batches. Cone solves whose workspace exceeds the device, or given
    ``out``, update the volume a z slab at a time, as for cone SIRT; every
    subset projection sums all slabs before its coupled update. The
    resident-volume path remains the default whenever its workspace fits.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry, including view angles and any per-view poses.
    grid : Grid
        Reconstruction grid in ``(x, y, z)`` order, with physical voxel spacing.
    detector : Detector
        Detector dimensions and placement in the same length unit as the grid.
    projections : array-like
        Nonnegative linear Poisson measurements (not raw transmission counts),
        with shape ``(views, detector.nv, detector.nu)``.
    init_x : array-like or None, optional
        Initial volume of shape ``(nx, ny, nz)``; None starts from one in observed support.
    config : MLEMConfig or None, optional
        Solver settings. None uses MLEMConfig defaults.
    det_grid : tuple of array-like or None, optional
        Explicit detector-coordinate arrays ``(u, v)``, each of shape ``(nv, nu)``.
        None uses the canonical detector grid.
    out : numpy.ndarray or None, optional
        Writable float32 ``(nx, ny, nz)`` host storage for a cone-beam result,
        including a memmap; it must not overlap the projections.

    Returns
    -------
    volume : jax.Array or numpy.ndarray
        Float32 reconstruction of shape ``(nx, ny, nz)``, host-owned when the
        cone solver workspace does not fit the device.
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
    >>> from tomojax.recon import mlem, MLEMConfig, project
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = project(geometry, np.ones((4, 4, 1), np.float32))
    >>> volume, info = mlem(
    ...     geometry,
    ...     geometry.grid,
    ...     geometry.detector,
    ...     data,
    ...     config=MLEMConfig(iterations=1),
    ... )
    >>> volume.shape
    (4, 4, 1)
    """
    return run_mlem(
        geometry, projections, grid=grid, detector=detector, init_x=init_x,
        config=MLEMConfig() if config is None else config, det_grid=det_grid, out=out,
    )  # fmt: skip


def run_mlem(
    geometry: Geometry,
    projections: jax.Array | np.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    init_x: jax.Array | np.ndarray | None = None,
    config: MLEMConfig,
    det_grid: tuple[jax.Array, jax.Array] | None = None,
    warm_start: bool = False,
    out: np.ndarray | None = None,
) -> tuple[jax.Array | np.ndarray, dict[str, object]]:
    """Own workflow initialization inside the solver, never copying a private host start."""
    cfg = config
    iterations, subsets, batch = map(
        operator.index, (cfg.iterations, cfg.subsets, cfg.views_per_batch)
    )
    if iterations < 0 or subsets < 1 or batch < 1:
        raise ValueError(
            "mlem: iterations must be nonnegative; subsets and views_per_batch positive"
        )
    validate_grid(grid, "mlem grid")
    n, _, _ = validate_projection_stack(projections, detector, geometry=geometry, context="mlem")
    validate_detector_grid(det_grid, detector, context="mlem")
    if subsets > n:
        raise ValueError("mlem: subsets must not exceed the number of views")
    if init_x is not None:
        validate_volume(init_x, grid, context="mlem", name="init_x")
    model, backend = resolve_geometry_projector(
        geometry, cfg.projector_model, cfg.backend, detector=detector,
        det_grid=det_grid, context="mlem",
    )  # fmt: skip
    storage, rows = {}, isinstance(projections, RowReader)
    if isinstance(model, ConeModel) and (out is not None or rows or host_volumes(grid, 5, backend)):
        from ._mlem_host import solve as solve_host

        volume, storage = solve_host(
            geometry, grid, detector, projections, model=model, backend=backend,
            poses=stack_view_poses(geometry, n),
            order=np.argsort(np.asarray(getattr(geometry, "angles", np.arange(n))), kind="stable"),
            subsets=subsets, iterations=iterations, batch=batch, initial=init_x,
            support=cfg.support, warm_start=warm_start, out=out,
        )  # fmt: skip
    elif out is not None or rows:
        raise ValueError("mlem: out and row readers need a cone-beam geometry")
    else:
        if warm_start:
            from ._warm_start import fbp_start
            from .fbp import FBPConfig

            init_x = jnp.maximum(
                fbp_start(
                    geometry, grid, detector, projections, config=FBPConfig(), det_grid=det_grid
                ),
                0,
            )
        volume = _run_device(
            geometry,
            grid,
            detector,
            projections,
            config=cfg,
            model=model,
            backend=backend,
            initial=init_x,
            det_grid=det_grid,
        )
    return volume, {
        "effective_iterations": iterations,
        "subset_updates": iterations * subsets,
        "termination": "iteration_limit",
        "projector_backend": backend,
        "projector_model": model,
        **storage,
    }
