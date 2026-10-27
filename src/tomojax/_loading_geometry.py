"""Record-to-geometry resolution, before any detector pixels are read.

Explicit geometry and grids remain authoritative. A complete recorded pitch
may prepare an internal axial centre seed; the loading boundary must estimate
that centre from calibrated pixels before returning it to the caller.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from tomojax._scan import geometry_of, rebuild
from tomojax.geometry import Detector, Grid, ParallelGeometry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.geometry import ScanGeometry
    from tomojax.io import ProjectionDataset


def frame_output_grid(
    geometry: ScanGeometry, source: ProjectionDataset | None, *, binning: int = 1
) -> ScanGeometry:
    """Keep explicit grids; size detector-derived grids at the corrected resolution.

    Only an unposed axial parallel model identifies detector rows with object z.
    Other models retain their physical output box, resampling it when binned.
    This is a numerical grid policy, not a change to acquired ray geometry.
    """
    if source is None or source.grid is not None:
        return geometry
    if isinstance(geometry, ParallelGeometry):
        d = geometry.detector
        grid = Grid(d.nu, d.nu, d.nv, d.du, d.du, d.dv, vol_center=(0, 0, d.center[1]))
    elif binning > 1:
        g = geometry.grid
        grid = g.resampled(
            (max(1, g.nx // binning), max(1, g.ny // binning), max(1, g.nz // binning))
        )
    else:
        return geometry
    return rebuild(geometry, grid=grid)


def frame_geometry(
    record: ProjectionDataset,
    *,
    angles: Sequence[float] | np.ndarray | None = None,
    detector: Detector | None = None,
    grid: Grid | None = None,
    geometry: ScanGeometry | None = None,
    poses: bool = True,
) -> tuple[ProjectionDataset, ScanGeometry]:
    """Resolve caller metadata before constructing any geometry from a file.

    Readers may return incomplete records. An explicit geometry must not first
    require constructing (and guessing) the missing source geometry.
    """
    meta = dict(record.geometry_metadata)
    if geometry is not None or detector is not None:
        meta["detector_center_status"] = "provided"
    if geometry is not None:
        meta["axis_status"] = "provided"
    if geometry is not None and "acquisition" in meta:
        meta["acquisition"] = {
            **meta["acquisition"],
            "geometry_status": "provided",
            "geometry_note": "Geometry supplied explicitly by caller.",
        }
    record = replace(
        record,
        angles=record.angles if angles is None else np.asarray(angles, np.float32),
        detector=detector or record.detector,
        grid=geometry.grid if geometry is not None else grid or record.grid,
        geometry_metadata=meta,
    )
    record = record.prepare_center_seed() if geometry is None else record
    resolved = geometry if geometry is not None else geometry_of(record, poses=poses)
    return record, frame_output_grid(resolved, record)
