"""SIRT and OS-SART with the volume in host memory, for cone scans larger than the device.

Each subset update is one sweep: the volume's slabs are projected into the
ring of detector rows, finished rows become row-normalized residuals, and
each slab is updated from its exact transpose and column sums as soon as its
window is complete. The volume is read from and written to its storage once
per subset; a zero start is neither stored nor projected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax.core.host_volume import fill, release_planes

from ._cone_slabs import Batch, cone_problem, host_projections

if TYPE_CHECKING:
    import jax

    from tomojax.core.cone import ConeModel
    from tomojax.geometry import Detector, Geometry, Grid

    from .sirt import SIRTConfig


def solve(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: Any,
    initial: np.ndarray | None,
    cfg: SIRTConfig,
    *,
    model: ConeModel,
    backend: str,
    poses: jax.Array,
    bounds: tuple[float, float],
    out: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run validated SIRT settings with a host volume, in ``out`` or a C-ordered volume."""
    projections = host_projections(projections, "sirt")
    problem = cone_problem(
        geometry, grid, detector, projections, model=model, backend=backend, poses=poses,
        batch=cfg.views_per_batch, channels=1 if cfg.support is None else 2,
        views=-(-projections.shape[0] // cfg.subsets),
        resident=0 if out is not None else 1 + cfg.acceleration,
    )  # fmt: skip
    solver = _HostSIRT(problem, cfg, bounds, initial, out)
    for momentum in _momenta(cfg):
        solver.momentum = momentum
        for subset in range(cfg.subsets):
            solver.update(np.arange(subset, projections.shape[0], cfg.subsets))
    solver.volume.close()
    solver.point.close()
    return problem.result(solver.x), {
        "effective_iterations": cfg.iterations,
        "subset_updates": cfg.iterations * cfg.subsets,
        "termination": "iteration_limit",
        "projector_backend": backend,
        "projector_model": model,
        "host_slabs": not problem.on_device,
        "slices_per_batch": problem.plan.depth,
        "bytes_to_device": problem.engine.traffic["to_device"],
        "bytes_to_host": problem.engine.traffic["to_host"],
    }


def _momenta(cfg: SIRTConfig) -> list[np.float32]:
    """Each iteration's FISTA momentum (zero without acceleration), as the resident solver's."""
    momenta, time = [], np.float32(1)
    for _ in range(cfg.iterations):
        following = (1 + np.sqrt(1 + 4 * time * time, dtype=np.float32)) / 2
        momenta.append((time - 1) / following if cfg.acceleration else np.float32(0))
        time = following
    return momenta


class _HostSIRT:
    """The host iterate, its extrapolated point and support, and one subset's sweep.

    With FISTA momentum (full-data SIRT only) updates are taken at the
    extrapolated point, which then moves past the new iterate; ``x`` keeps the
    iterates. Voxels outside the support stay zero.
    """

    def __init__(
        self,
        problem: Any,
        cfg: SIRTConfig,
        bounds: tuple[float, float],
        initial: np.ndarray | None,
        out: np.ndarray | None,
    ) -> None:
        self.problem, self.plan, self.xp = problem, problem.plan, problem.engine.xp
        grid = self.plan.grid
        shape = (grid.nz, grid.ny, grid.nx)
        self.lower, self.upper = (np.float32(b) for b in bounds)
        # A zero start is neither stored nor projected: the first sweep writes every voxel.
        self.zero = initial is None and self.lower <= 0 <= self.upper
        self.x = problem.iterate(initial, out, "sirt", written=self.zero)
        self.support = (
            None if cfg.support is None else np.broadcast_to(np.asarray(cfg.support) != 0, shape)
        )
        xp = np if isinstance(self.x, np.ndarray) else self.xp
        bounded = np.isfinite(self.lower) or np.isfinite(self.upper)
        # Unconstrained starts are left untouched: a file-backed start is not rewritten.
        constrained = (bounded or self.support is not None) and not self.zero
        for k in range(self.plan.count if constrained else 0):
            z0, z1 = self.plan.planes(k)
            planes = self.x[z0:z1, :, :]
            if bounded:
                xp.clip(planes, self.lower, self.upper, out=planes)
            if self.support is not None:
                planes[~xp.asarray(self.support[z0:z1, :, :])] = 0
            release_planes(self.x, z0, z1)
        self.volume = problem.volume(self.x)
        self.point = self.volume
        if cfg.acceleration:
            work = problem.work(out)
            self.point = problem.volume(work if self.zero else fill(work, self.x))
        self.relaxation, self.momentum = np.float32(cfg.relaxation), np.float32(0)
        self.accelerated = cfg.acceleration

    def update(self, views: np.ndarray) -> None:
        """One subset's update of every slab."""
        self.problem.sweep.run(
            views, source=self.source, finish=self.finish, sink=self.sink,
            channels=1 if self.support is None else 2, transposed=1, ones=True,
            keep=self.problem.keep,
            prefetch=None if self.zero else lambda k: self.point.prefetch(*self.plan.planes(k)),
        )  # fmt: skip
        self.zero = False

    def source(self, k: int) -> tuple[Any, ...]:
        """Slab ``k`` of the point (and support) on the device; see :attr:`ConeProblem.keep`."""
        if self.zero:
            return (None, *self._support(k))
        point = self.point.get(*self.plan.planes(k), retain=not self.problem.keep)
        return (point, *self._support(k))

    def _support(self, k: int) -> tuple[Any, ...]:
        if self.support is None:
            return ()
        z0, z1 = self.plan.planes(k)
        mask = np.ascontiguousarray(self.support[z0:z1, :, :], np.float32)
        return (self.point.kernel(self.problem.engine.upload(mask)),)

    def finish(self, batch: Batch, first: int, stop: int, rows: tuple[Any, ...]) -> None:
        """Rows become ``R (y - A x)`` with ``R = 1 / (A support)``; zero sums weigh zero."""
        xp, problem = self.xp, self.problem
        if self.support is None:
            coeff = problem.sweep.row_coefficients(batch, first)
            sums = problem.engine.project_ones(coeff, stop - first)
        else:
            sums = rows[1]
        positive = sums > 0
        weights = xp.where(positive, 1 / xp.where(positive, sums, 1), 0).astype(xp.float32)
        rows[0][...] = weights * (problem.measured(batch, first, stop) - rows[0])

    def sink(self, k: int, accumulated: tuple[Any, ...], sources: tuple[Any, ...]) -> None:
        """Update slab ``k`` in place, starting from the slab projected in this sweep."""
        xp, (z0, z1) = self.xp, self.plan.planes(k)
        if not sources:  # unkept slabs are fetched again
            sources = (None if self.zero else self.point.view(z0, z1), *self._support(k))
        updated, norm = (a[: z1 - z0, :, :] for a in accumulated)
        xp.copyto(norm, xp.inf, where=norm <= 0)  # zero column sums give zero weight
        updated /= norm
        updated *= self.relaxation
        if sources[0] is not None:
            updated += sources[0][: z1 - z0, :, :]
        xp.clip(updated, self.lower, self.upper, out=updated)
        if self.support is not None:
            updated *= sources[1][: z1 - z0, :, :]
        previous = None
        if self.accelerated:
            previous = xp.zeros_like(updated) if self.zero else self.volume.view(z0, z1)
        self.volume.put(z0, z1, updated)
        if previous is not None:
            # updated + momentum (updated - previous), in the uploaded buffer
            previous -= updated
            previous *= -self.momentum
            previous += updated
            self.point.put(z0, z1, previous)
