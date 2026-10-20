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

from tomojax.core.host_volume import fill, release_planes

from ._cone_slabs import cone_problem, host_projections
from ._host_arrays import validate_host_arrays

if TYPE_CHECKING:
    import jax

    from tomojax.core.cone import ConeModel
    from tomojax.geometry import Detector, Geometry, Grid

    from ._cone_slabs import Batch


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
    subsets: int,
    iterations: int,
    batch: int,
    initial: jax.Array | np.ndarray | None,
    support: jax.Array | np.ndarray | None,
    warm_start: bool = False,
    out: np.ndarray | None = None,
) -> tuple[Any, dict[str, object]]:
    """Run validated EM settings; return the volume (``out`` if given) and its storage record.

    Voxels no subset sees are zero; zero initial voxels stay zero.
    """
    data = host_projections(projections, "mlem", nonnegative=True)
    if not np.isfinite(np.asarray(poses)).all():
        raise ValueError("mlem: poses must be finite")
    problem = cone_problem(
        geometry, grid, detector, data, model=model, backend=backend, poses=poses,
        batch=batch, channels=1, views=-(-data.shape[0] // subsets),
        resident=0 if out is not None or warm_start else 1,
    )  # fmt: skip
    if warm_start:
        from ._warm_start import fbp_start
        from .fbp import FBPConfig

        x = validate_host_arrays(data, out, (grid.nx, grid.ny, grid.nz), "mlem", order="F")
        fbp_start(geometry, grid, detector, data, config=FBPConfig(), det_grid=None, out=x)
    else:
        x = problem.iterate(initial, out, "mlem")
        if initial is None:
            fill(x, 1.0)
    xp = np if isinstance(x, np.ndarray) else problem.engine.xp
    _constrain(x, support, problem.plan.depth, nonnegative=warm_start, xp=xp)
    em = _HostEM(problem, x, updating=iterations > 0)
    for _ in range(max(1, iterations)):
        for group in range(subsets):
            problem.sweep.run(
                order[group::subsets], source=em.source, finish=em.finish, sink=em.sink,
                transposed=int(em.updating), ones=True, keep=problem.keep,
                prefetch=lambda k: em.volume.prefetch(*problem.plan.planes(k)),
            )  # fmt: skip
    em.volume.close()
    for z0 in range(0, grid.nz, problem.plan.depth):
        z1 = min(grid.nz, z0 + problem.plan.depth)
        x[:, :, z0:z1] *= xp.asarray(np.unpackbits(em.seen[:, :, z0:z1], axis=0, count=grid.nx))
        release_planes(x, z0, z1)
    record = {"host_volume": not problem.on_device, "slices_per_batch": problem.plan.depth}
    return problem.result(x), record


def _constrain(x: Any, support: Any, depth: int, *, nonnegative: bool, xp: Any) -> None:
    """Clip a warm start at zero and zero voxels outside the support; check the start."""
    mask = None if support is None else np.broadcast_to(np.asarray(support, bool), x.shape)
    for z0 in range(0, x.shape[2], depth):
        planes = x[:, :, z0 : z0 + depth]
        if nonnegative:
            xp.maximum(planes, 0, out=planes)
        if mask is not None:
            planes[~xp.asarray(mask[:, :, z0 : z0 + depth])] = 0
        if not bool(xp.isfinite(planes).all()) or float(xp.min(planes)) < 0:
            raise ValueError("mlem: initial volume must be finite and nonnegative")
        release_planes(x, z0, z0 + depth)


class _HostEM:
    """One host EM volume's subset sweeps, recording which voxels any subset sees."""

    def __init__(self, problem: Any, x: np.ndarray, *, updating: bool) -> None:
        self.problem, self.plan, self.xp = problem, problem.plan, problem.engine.xp
        self.volume, self.updating = problem.volume(x), updating
        grid = self.plan.grid
        self.seen = np.zeros(((grid.nx + 7) // 8, grid.ny, grid.nz), np.uint8, order="F")

    def source(self, k: int) -> tuple[Any, ...]:
        retain = not self.problem.keep
        return (self.volume.get(*self.plan.planes(k), retain=retain),) if self.updating else ()

    def finish(self, batch: Batch, first: int, stop: int, rows: tuple[Any, ...]) -> None:
        """Rows become measured / predicted ratios, zero where nothing is predicted."""
        if self.updating:
            xp, predicted = self.xp, rows[0]
            measured = self.problem.measured(batch, first, stop)
            positive = predicted > 0
            predicted[...] = xp.where(positive, measured / xp.where(positive, predicted, 1), 0)

    def sink(self, k: int, accumulated: tuple[Any, ...], sources: tuple[Any, ...]) -> None:
        """Multiply slab ``k`` by its correction; unseen voxels keep their values until masked."""
        xp, (z0, z1) = self.xp, self.plan.planes(k)
        norm = accumulated[-1][:, :, : z1 - z0]
        visible = norm > 0
        visible = visible.get() if hasattr(visible, "get") else visible
        self.seen[:, :, z0:z1] |= np.packbits(visible, axis=0)
        if self.updating:
            updated = accumulated[0][:, :, : z1 - z0]
            norm[norm <= 0] = xp.inf
            updated /= norm
            updated[xp.isinf(norm)] = 1
            updated *= (sources or (self.volume.view(z0, z1),))[0][:, :, : z1 - z0]
            self.volume.put(z0, z1, updated, scratch=accumulated[1])
