"""Cone OSEM with the volume in host memory, updated one z slab at a time.

Each subset update is one sweep (see :mod:`._cone_slabs`): the slabs' exact
projections are summed in the ring of detector rows before each row's Poisson
ratio is formed, and each slab is updated from the transposes of the ratios
and of unit images, so this is one coupled reconstruction, not independent
slices. Caller inputs are never modified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax import progress
from tomojax.core import checkpoint
from tomojax.core.checkpoint import Completed, Loop, Saved
from tomojax.core.host_volume import fill, release_planes

from ._cone_slabs import algebraic_sweep, cone_problem, host_projections
from ._host_arrays import validate_host_arrays

if TYPE_CHECKING:
    import jax

    from tomojax.core.cone import ConeModel
    from tomojax.geometry import Detector, Geometry, Grid

    from ._cone_slabs import Batch, ConeProblem
    from .mlem import MLEMConfig


def solve(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    model: ConeModel,
    backend: str,
    poses: jax.Array,
    order: np.ndarray,
    cfg: MLEMConfig,
    initial: jax.Array | np.ndarray | None,
    warm_start: bool = False,
    background: np.ndarray | None = None,
    out: np.ndarray | None = None,
) -> tuple[Any, dict[str, object]]:
    """Run validated EM settings; return the volume (``out`` if given) and its storage record.

    Voxels no subset sees are zero; zero initial voxels stay zero.
    """
    subsets, iterations, support = cfg.subsets, cfg.iterations, cfg.support
    data = host_projections(projections, "mlem", nonnegative=True)
    if not np.isfinite(np.asarray(poses)).all():
        raise ValueError("mlem: poses must be finite")
    x = None
    if warm_start:
        from ._warm_start import fbp_start
        from .fbp import FBPConfig

        # Before planning the sweeps: XLA keeps the device memory FDK has used.
        x = validate_host_arrays(data, out, (grid.nz, grid.ny, grid.nx), "mlem")
        _ = fbp_start(geometry, grid, detector, data, config=FBPConfig(), det_grid=None, out=x)
    problem = cone_problem(
        geometry, grid, detector, data, model=model, backend=backend, poses=poses,
        **algebraic_sweep(cfg, data.shape[0], host_output=out is not None, warm_start=warm_start),
    )  # fmt: skip
    if x is None:
        x = problem.iterate(initial, out, "mlem")
        if initial is None:
            fill(x, 1.0)
    xp = np if isinstance(x, np.ndarray) else problem.engine.xp
    _constrain(x, support, problem.plan.depth, nonnegative=warm_start, xp=xp)
    with checkpoint.loop() as saves:
        em = _HostEM(problem, x, background=background, updating=iterations > 0, saved=saves.saved)
        with problem.using(out, em.volume):
            em.run(order, subsets, iterations, saves)
    for z0 in range(0, grid.nz, problem.plan.depth):
        z1 = min(grid.nz, z0 + problem.plan.depth)
        x[z0:z1, :, :] *= xp.asarray(np.unpackbits(em.seen[z0:z1, :, :], axis=2, count=grid.nx))
        release_planes(x, z0, z1)
    record = {"host_volume": not problem.on_device, "slices_per_batch": problem.plan.depth}
    return problem.result(x), record


def _constrain(x: Any, support: Any, depth: int, *, nonnegative: bool, xp: Any) -> None:
    """Clip a warm start at zero and zero voxels outside the support; check the start."""
    mask = None if support is None else np.broadcast_to(np.asarray(support, bool), x.shape)
    for z0 in range(0, x.shape[0], depth):
        planes = x[z0 : z0 + depth, :, :]
        if nonnegative:
            xp.maximum(planes, 0, out=planes)
        if mask is not None:
            planes[~xp.asarray(mask[z0 : z0 + depth, :, :])] = 0
        if not bool(xp.isfinite(planes).all()) or float(xp.min(planes)) < 0:
            raise ValueError("mlem: initial volume must be finite and nonnegative")
        release_planes(x, z0, z0 + depth)


class _HostEM:
    """One host EM volume's subset sweeps, recording which voxels any subset sees."""

    def __init__(
        self,
        problem: ConeProblem,
        x: np.ndarray,
        *,
        background: np.ndarray | None,
        updating: bool,
        saved: Saved | None,
    ) -> None:
        self.background = background
        self.problem, self.plan, self.xp = problem, problem.plan, problem.engine.xp
        self.volume, self.updating = problem.volume(x), updating
        grid = self.plan.grid
        self.seen = np.zeros((grid.nz, grid.ny, (grid.nx + 7) // 8), np.uint8, order="C")
        self.steps = 0
        if saved is not None:  # a checkpoint's iterate, seen voxels and steps
            self.volume.restore(saved.arrays[0])
            np.copyto(self.seen, saved.arrays[1])
            self.steps = saved.scalars(Completed).steps

    def run(self, order: np.ndarray, subsets: int, iterations: int, saves: Loop) -> None:
        """Every subset's update in each iteration after the completed ones; at least one sweep."""
        problem = self.problem
        for step in range(self.steps + 1, max(1, iterations) + 1):
            progress.check_canceled()
            for group in range(subsets):
                problem.sweep.run(
                    order[group::subsets], source=self.source, finish=self.finish, sink=self.sink,
                    transposed=int(self.updating), ones=True, keep=problem.keep,
                    prefetch=lambda k: self.volume.prefetch(*problem.plan.planes(k)),
                )  # fmt: skip
            if iterations and (task := progress.for_unit("iterations")) is not None:
                problem.completed(task.advance, step, {})
            if saves.due():
                saves.save((self.volume.contents(), self.seen), Completed(step))

    def source(self, k: int) -> tuple[Any, ...]:
        retain = not self.problem.keep
        return (self.volume.get(*self.plan.planes(k), retain=retain),) if self.updating else ()

    def finish(self, batch: Batch, first: int, stop: int, rows: tuple[Any, ...]) -> None:
        """Rows become measured / predicted ratios, zero where nothing is predicted."""
        if self.updating:
            xp, predicted = self.xp, rows[0]
            measured = self.problem.measured(batch, first, stop)
            if self.background is not None:
                additive = self.problem.engine.upload(self.background[batch.views, first:stop])
                predicted += xp.maximum(additive, 0)
            positive = predicted > 0
            predicted[...] = xp.where(positive, measured / xp.where(positive, predicted, 1), 0)

    def sink(self, k: int, accumulated: tuple[Any, ...], sources: tuple[Any, ...]) -> None:
        """Multiply slab ``k`` by its correction; unseen voxels keep their values until masked."""
        xp, (z0, z1) = self.xp, self.plan.planes(k)
        norm = accumulated[-1][: z1 - z0, :, :]
        visible = norm > 0
        visible = visible.get() if hasattr(visible, "get") else visible
        self.seen[z0:z1, :, :] |= np.packbits(visible, axis=2)
        if self.updating:
            updated = accumulated[0][: z1 - z0, :, :]
            norm[norm <= 0] = xp.inf
            updated /= norm
            updated[xp.isinf(norm)] = 1
            updated *= (sources or (self.volume.view(z0, z1),))[0][: z1 - z0, :, :]
            self.volume.put(z0, z1, updated)
