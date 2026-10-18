"""Joint free-voxel/pose Gauss-Newton with bounded pose-column storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
import logging

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.alignment._model.dofs import POSE_WIDTH
from tomojax.alignment._objectives.loss_specs import L2LossSpec
from tomojax.core.devices import ViewSplit, shared_devices, view_split
from tomojax.core.projector import get_detector_grid_device

from ._coupled_linear import CoupledLinearResult
from ._coupled_program import CoupledArrays, CoupledSpec, JointRefinement, run_loss, run_update
from ._pose_context import _PoseObjectiveContext
from ._pose_jacobian import PoseJacobianOptions

# Recomputing a view's active Jacobian columns costs a derivative
# projection, and the joint solve applies them in every conjugate-gradient
# iteration, so cache them whenever that many sinograms fit in this share of
# free device memory. The complete working set is still checked below, and a
# cached update that cannot fit retries without columns before staging the solve.
_POSE_CACHE_FRACTION = 0.5
# Fallback cap when free device memory cannot be queried.
_POSE_CACHE_BYTES = 512 * 1024**2
LOG = logging.getLogger(__name__)


def _pose_cache_limit() -> int:
    from tomojax.backends import device_free_memory_bytes

    if _POSE_CACHE_BYTES <= 0:
        return 0
    free = device_free_memory_bytes() if jax.default_backend() == "gpu" else None
    return _POSE_CACHE_BYTES if free is None else int(_POSE_CACHE_FRACTION * free)


class AlignmentMemoryError(MemoryError):
    """A level's joint pose and volume update needs more device memory than is free."""

    def __init__(self, needed: int, available: int) -> None:
        self.needed, self.available = needed, available
        super().__init__(
            f"the joint pose and volume update needs {needed / 2**30:.1f} GiB of device "
            f"memory, {available / 2**30:.1f} GiB is free: align at coarser levels, on a "
            "binned scan, or with more device memory"
        )


# The cheap estimate below must leave this factor of room before the
# compiled update's own memory figure is skipped.
_MEMORY_MARGIN = 2


def _free_device_memory(array: jax.Array) -> int | None:
    """Free allocator capacity on the GPU holding ``array``, when queryable.

    The compiled update reports total live storage, not one allocation. A pool's
    largest free block is not a bound on that total; it can even be zero while
    a non-preallocating allocator still has capacity to grow. Fragmentation can
    still cause an actual OOM, handled by the alignment level's normal fallback.
    """
    device = next(iter(array.devices()))
    if device.platform != "gpu":
        return None
    from tomojax.backends import device_free_memory_bytes

    return device_free_memory_bytes(device=device)


def _check_device_memory(arrays: CoupledArrays, spec: CoupledSpec) -> Callable | None:
    """Raise :class:`AlignmentMemoryError` if the update cannot fit on its GPU.

    The update keeps about a dozen volume-sized and a few projection-sized
    arrays, plus the cached pose columns and per-pixel weights. Only when that
    estimate comes within a factor of two of free memory is the update compiled
    ahead to read XLA's own figure; the compiled update is then returned for
    use, so it is compiled once. Recheck current free memory at dispatch: an
    intervening reconstruction can retain another volume after this admission.
    """
    available = _free_device_memory(arrays.poses)
    if available is None:
        return None
    grid, detector = spec.grid, spec.detector
    n_views = int(arrays.poses.shape[0])
    volume = 4 * grid.nx * grid.ny * grid.nz
    if spec.split is not None:  # each device holds its share of the views
        n_views //= spec.split.mesh.size
    sinogram = 4 * n_views * detector.nu * detector.nv
    sinograms = (
        3 + (len(spec.active_columns) if spec.cache_columns else 0) + (arrays.weights.ndim > 0)
    )
    if _MEMORY_MARGIN * (13 * volume + sinograms * sinogram) < available:
        return None
    p = jax.ShapeDtypeStruct((int(arrays.poses.shape[0]), POSE_WIDTH), jnp.float32)
    x = jax.ShapeDtypeStruct((grid.nx, grid.ny, grid.nz), jnp.float32)
    compiled = run_update.lower(arrays, p, x, spec=spec).compile()
    analysis = compiled.memory_analysis()
    if analysis is None:
        return compiled
    needed = int(analysis.temp_size_in_bytes) + int(analysis.output_size_in_bytes)
    # A host scan has not occupied device memory yet; include its input
    # transfer as well as XLA's temporaries in the admission decision.
    needed += sum(a.nbytes for a in jax.tree.leaves(arrays) if isinstance(a, np.ndarray))
    if needed > available:
        raise AlignmentMemoryError(needed, available)

    def checked_update(arrays: CoupledArrays, p: jax.Array, x: jax.Array) -> CoupledLinearResult:
        available = _free_device_memory(x)
        if available is not None and needed > available:
            raise AlignmentMemoryError(needed, available)
        return compiled(arrays, p, x)

    return checked_update


def _shared(arrays: CoupledArrays, split: ViewSplit) -> CoupledArrays:
    """``arrays`` laid out on ``split``'s devices: each device's views, the rest everywhere."""
    weights = arrays.weights
    return CoupledArrays(
        poses=split.place(arrays.poses, repeat=True),
        projections=split.place(arrays.projections),
        weights=split.everywhere(weights) if weights.ndim == 0 else split.place(weights),
        mask=split.everywhere(arrays.mask),
        active=split.everywhere(arrays.active),
        smoothness=split.everywhere(arrays.smoothness),
        det_grid=(split.everywhere(arrays.det_grid[0]), split.everywhere(arrays.det_grid[1])),
        frames=None if arrays.frames is None else split.place(arrays.frames, repeat=True),
        counts=split.place(split.counted().astype(jnp.float32)),
    )


def _on_devices(
    update: Callable[[jax.Array, jax.Array], CoupledLinearResult],
    loss: Callable[[jax.Array, jax.Array], jax.Array],
    split: ViewSplit,
) -> tuple[
    Callable[[jax.Array, jax.Array], CoupledLinearResult],
    Callable[[jax.Array, jax.Array], jax.Array],
]:
    """``update`` and ``loss`` of poses and a volume on the first device, run on ``split``."""

    def shared_update(p: jax.Array, x: jax.Array) -> CoupledLinearResult:
        result = update(split.place(p, repeat=True), split.everywhere(x))
        dx, dp = result.increment
        return result._replace(
            increment=(split.gather(dx), split.take(dp)),
            iterations=split.gather(result.iterations),
            relative_residual=split.gather(result.relative_residual),
            finite=split.gather(result.finite),
        )

    def shared_loss(p: jax.Array, x: jax.Array) -> jax.Array:
        return split.gather(loss(split.place(p, repeat=True), split.everywhere(x)))

    return shared_update, shared_loss


@dataclass(frozen=True)
class CoupledObjective:
    update: Callable[[jax.Array, jax.Array], CoupledLinearResult]
    loss: Callable[[jax.Array, jax.Array], jax.Array]
    projector_backend: str
    pose_columns_cached: bool


def build_coupled_objective(
    ctx: _PoseObjectiveContext, *, refinement: JointRefinement | None = None
) -> CoupledObjective:
    cfg = ctx.cfg
    if not ctx.loss_adapter.supports_gauss_newton:
        raise ValueError("joint GN requires a least-squares alignment loss")
    if cfg.gather_dtype not in {"fp32", "float32"}:
        raise ValueError("joint GN requires gather_dtype='fp32' for matched linear operators")
    # Pallas exact and Joseph integration accept dynamic rigid poses. The
    # sampled reference keeps its matched explicit transpose and bounded view loop.
    backend = (
        "pallas" if cfg.projector_backend == "pallas" and jax.default_backend() == "gpu" else "jax"
    )
    if cfg.ray_integrator not in {"exact", "joseph", "joseph_cubic"}:
        backend = "jax"
    if ctx.cone is not None:
        # Cone beams: the CUDA kernels where available ("pallas" names the CUDA path).
        from tomojax.core.cone import use_cuda_cone

        backend = "pallas" if cfg.projector_backend != "jax" and use_cuda_cone() else "jax"
    canonical = get_detector_grid_device(ctx.detector)
    if (
        backend == "pallas"
        and cfg.ray_integrator == "exact"
        and not all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(ctx.det_grid, canonical, strict=True)
        )
    ):
        backend = "jax"
    arrays = CoupledArrays(
        poses=ctx.pose_stack,
        projections=ctx.projections,
        # Plain least squares weighs every pixel alike: a scalar, not a stack of ones.
        weights=jnp.float32(1)
        if isinstance(ctx.loss_adapter.spec, L2LossSpec) and not ctx.has_loss_mask
        else ctx.loss_adapter.gauss_newton_weights(
            ctx.projections, ctx.loss_mask if ctx.has_loss_mask else None
        ),
        mask=ctx.volume_mask if ctx.volume_mask is not None else jnp.float32(1),
        active=ctx.active_mask.astype(jnp.float32),
        smoothness=ctx.smoothness_weights,
        det_grid=ctx.det_grid,
        frames=ctx.frames,
    )
    split = view_split(shared_devices(), ctx.n_views)
    if split is not None:
        arrays = _shared(arrays, split)
    # Each device caches its own views' columns.
    views = ctx.n_views if split is None else split.total // split.mesh.size
    active_columns = tuple(int(i) for i in np.flatnonzero(np.asarray(ctx.active_mask)))
    cache_columns = views * ctx.nv * ctx.nu * len(active_columns) * 4 <= _pose_cache_limit()
    spec = CoupledSpec(
        grid=ctx.grid,
        detector=ctx.detector,
        backend=backend,
        jacobian=replace(PoseJacobianOptions.from_config(cfg), cone_beam=ctx.cone is not None),
        active_columns=active_columns,
        cache_columns=cache_columns,
        regulariser=cfg.regulariser,
        huber_delta=float(cfg.huber_delta),
        tv_weight=float(cfg.tv_weight),
        nonnegative=bool(cfg.nonnegative),
        gn_volume_damping=float(cfg.gn_volume_damping),
        gn_damping=float(cfg.gn_damping),
        gn_joint_solver=cfg.gn_joint_solver,
        gn_joint_rtol=float(cfg.gn_joint_rtol),
        gn_joint_iterations=int(cfg.gn_joint_iterations),
        has_smoothness=bool(cfg.w_rot or cfg.w_trans),
        split=split,
        refinement=refinement,
    )
    return _prepare_execution(arrays, spec)


def _prepare_execution(arrays: CoupledArrays, spec: CoupledSpec) -> CoupledObjective:
    """Admit a device program, or stage the same system without its large temporaries."""
    for cached in (True, False) if spec.cache_columns else (False,):
        spec = replace(spec, cache_columns=cached)
        try:
            compiled = _check_device_memory(arrays, spec)
            break
        except AlignmentMemoryError as error:
            if cached:
                LOG.info("Cached pose update does not fit (%s); recomputing columns instead", error)
                continue
            if spec.split is not None or spec.refinement is not None:
                raise
            from ._coupled_host import build_host_objective

            LOG.info("Device joint update does not fit (%s); staging Krylov vectors on host", error)
            update, loss = build_host_objective(arrays, spec)
            return CoupledObjective(update, loss, spec.backend, False)
    update = (
        partial(run_update, arrays, spec=spec) if compiled is None else partial(compiled, arrays)
    )
    loss = partial(run_loss, arrays, spec=spec)
    if spec.split is not None:
        update, loss = _on_devices(update, loss, spec.split)
    return CoupledObjective(update, loss, spec.backend, spec.cache_columns)
