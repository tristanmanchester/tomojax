"""Global-coordinate Joseph slabs for parallel and laminographic host-volume solves.

Every slab contributes before forming a data residual. The transpose visits
those slabs with the same global-coordinate weights, without recentering geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax.core.joseph import validate_plane_geometry
from tomojax.core.joseph_cuda import coefficients, slab_adjoint_into, slab_forward

from ._cone_slabs import ConeProblem, SlabPlan, Sweep, slab_depth
from ._slab_engines import CudaEngine

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Geometry, Grid


@dataclass(frozen=True)
class SlabCoefficients:
    """Global affine maps and the integer first plane of locally addressed storage."""

    values: Any
    z0: int

    def __getitem__(self, index: Any) -> SlabCoefficients:
        return SlabCoefficients(self.values[index], self.z0)


class ParallelEngine(CudaEngine):
    """The matched Joseph pair on slabs, retaining resident coordinate arithmetic."""

    def project(self, volume: Any, coeff: Any, rows: int, *, out: Any, row_start: int) -> None:
        """Add a global-coordinate slab into its full-detector projection window."""
        del row_start, rows
        out += slab_forward(
            volume, coeff.values, grid=self.grid, detector=self.detector, z0=coeff.z0
        )

    def transpose(
        self, images: tuple[Any, ...], coeff: Any, accumulate: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        """Accumulate each RHS using the resident gather's global z coordinate."""
        for image, target in zip(images, accumulate, strict=True):
            slab_adjoint_into(
                image, coeff.values, out=target, grid=self.grid, detector=self.detector, z0=coeff.z0
            )
        return accumulate


class ParallelSweep(Sweep):
    """The common row-ring sweep with global affine maps instead of cone coefficients."""

    def slab_coefficients(self, k: int) -> SlabCoefficients:
        return SlabCoefficients(self._device_coefficients, self.plan.planes(k)[0])


def parallel_problem(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data: np.ndarray,
    *,
    poses: Any,
    batch: int,
    channels: int = 1,
) -> ConeProblem:
    """Plan a CUDA Joseph sweep without rounding a new origin for each slab."""
    del geometry
    validate_plane_geometry(poses, grid, detector)
    engine = ParallelEngine(grid, detector)
    count, batch = len(poses), min(batch, len(poses))

    def plans(depth: int) -> SlabPlan:
        slabs = -(-grid.nz // depth)
        return SlabPlan(
            grid,
            detector,
            depth,
            ((0, detector.nv),) * slabs,
            tuple(range(slabs)),
            detector.nv,
            slabs,
        )

    depth = slab_depth(plans, count, channels, batch, engine)
    coeff = coefficients(np.asarray(poses), grid, detector, xp=np)
    sweep = ParallelSweep(plans(depth), engine, coeff, batch)
    return ConeProblem(sweep, data, keep=False, retain=False)
