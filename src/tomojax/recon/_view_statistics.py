"""Whole-detector statistics for objectives evaluated through slab row windows."""

from __future__ import annotations

import numpy as np

from . import _cone_slabs


class ViewStatistics:
    """Small per-view reductions; zero-ray detector pixels still count as measurements.

    A slab sweep only visits rows reached by some slab. Data reductions visit the
    entire detector once, while projection reductions divide by its full size.
    Neither operation retains a sinogram or a second volume.
    """

    def __init__(self, problem: _cone_slabs.ConeProblem, views: np.ndarray) -> None:
        self.problem, self.views = problem, views
        xp = problem.engine.xp
        count, nv, nu = problem.data.shape
        self.pixels = nv * nu
        sums = xp.zeros((4, count), xp.float64)
        covered = np.zeros(nv, bool)
        for first, stop in problem.plan.windows:
            covered[first:stop] = True
        self.outside_pixels = int(np.count_nonzero(~covered)) * nu
        for first in range(0, nv, _cone_slabs.ROW_BAND):
            stop = min(first + _cone_slabs.ROW_BAND, nv)
            outside = xp.asarray(~covered[first:stop])[None, :, None]
            for start in range(0, len(views), problem.sweep.batch):
                chosen = views[start : start + problem.sweep.batch]
                index = xp.asarray(chosen)
                data = problem.measured(_cone_slabs.Batch(chosen, index), first, stop)
                square = data.astype(xp.float64) ** 2
                sums[0, index] += data.sum((1, 2), dtype=xp.float64)
                sums[1, index] += xp.abs(data).sum((1, 2), dtype=xp.float64)
                sums[2, index] += (data * outside).sum((1, 2), dtype=xp.float64)
                sums[3, index] += (square * outside).sum((1, 2))
        self.mean = (sums[0] / self.pixels).astype(xp.float32)
        self.absolute_mean = (sums[1] / self.pixels).astype(xp.float32)
        self.outside_sum, self.outside_square = sums[2], sums[3]

    def project(self, source, *, absolute: bool = False, prefetch=None):
        """Mean of the complete projection (or its magnitude), in a forward-only sweep.

        ``source(k)`` supplies one volume channel and must be safe to read again
        in the caller's subsequent gradient sweep. No backprojection is performed.
        """
        problem, xp = self.problem, self.problem.engine.xp
        total = xp.zeros(problem.data.shape[0], xp.float64)

        def finish(batch, _first, _stop, rows):
            values = xp.abs(rows[0]) if absolute else rows[0]
            total[batch.index] += values.sum((1, 2), dtype=xp.float64)

        problem.sweep.run(
            self.views, source=source, finish=finish, sink=lambda *_: None,
            transposed=0, keep=False, prefetch=prefetch,
        )  # fmt: skip
        return (total / self.pixels).astype(xp.float32)

    def projection_mean(self, rows, *, absolute: bool = False):
        """Reduce an already resident full-detector projection in ``views`` order.

        Bounded row/view chunks match :meth:`project` without an extra forward
        sweep or a full-size magnitude/FP64 temporary.
        """
        problem, xp = self.problem, self.problem.engine.xp
        total = xp.zeros(problem.data.shape[0], xp.float64)
        for first in range(0, problem.data.shape[1], _cone_slabs.ROW_BAND):
            stop = first + _cone_slabs.ROW_BAND
            for start in range(0, len(self.views), problem.sweep.batch):
                end = min(start + problem.sweep.batch, len(self.views))
                values = rows[start:end, first:stop]
                if absolute:
                    values = xp.abs(values)
                total[xp.asarray(self.views[start:end])] += values.sum((1, 2), dtype=xp.float64)
        return (total / self.pixels).astype(xp.float32)

    def outside_energy(self, offset):
        """Squared norm of ``data-offset`` on rows omitted from every slab window."""
        offset = self.problem.engine.xp.asarray(offset, dtype=self.problem.engine.xp.float64)
        return (
            self.outside_square
            - 2 * offset * self.outside_sum
            + self.outside_pixels * offset * offset
        )[self.views].sum()
