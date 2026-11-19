"""Record-to-geometry resolution, before any detector pixels are read.

Explicit geometry and grids remain authoritative. A complete recorded pitch
may prepare an internal axial centre seed; the loading boundary must estimate
that centre from calibrated pixels before returning it to the caller.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast
import warnings

import numpy as np

from tomojax._scan import frame_output_grid, geometry_of
from tomojax.geometry import Detector, Grid
from tomojax.io.api import dropped_offsets_message, with_view_offsets, without_view_offsets

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.geometry import ScanGeometry
    from tomojax.io import ProjectionDataset


def _vendor_corrections(record: ProjectionDataset, *, applied: bool) -> ProjectionDataset:
    """A scanner's own per-view shifts and centre shift, applied or left out, said either way."""
    vendor = record.geometry_metadata.get("vendor_corrections")
    if vendor is None:
        return record
    if not applied:
        record = without_view_offsets(record)
    meta = {**record.geometry_metadata, "vendor_corrections": {**vendor, "applied": applied}}
    if not applied and meta.get("detector_center_status") == "recorded":
        meta["detector_center_status"] = "nominal"
    return replace(record, geometry_metadata=meta)


def frame_geometry(
    record: ProjectionDataset,
    *,
    angles: Sequence[float] | np.ndarray | None = None,
    detector: Detector | None = None,
    grid: Grid | None = None,
    geometry: ScanGeometry | None = None,
    apply_poses: bool = True,
    pixel_size: float | tuple[float, float] | None = None,
) -> tuple[ProjectionDataset, ScanGeometry]:
    """Resolve caller metadata before constructing any geometry from a file.

    Readers may return incomplete records. An explicit geometry must not first
    require constructing (and guessing) the missing source geometry. Recorded
    per-view detector offsets move from the file's detector to a caller's
    ``detector``; an explicit ``geometry`` replaces them, with a warning.
    """
    if pixel_size is not None and (detector is not None or geometry is not None):
        raise ValueError(
            "pixel_size= and detector= (or geometry=) both describe the detector: pass one"
        )
    if geometry is not None and (message := dropped_offsets_message(record)):
        warnings.warn(message, stacklevel=4)
    record = _vendor_corrections(record, applied=apply_poses and geometry is None)
    meta = dict(record.geometry_metadata)
    if pixel_size is not None:  # the shape from the frames, the centre from the file or the middle
        du, dv = np.broadcast_to(np.asarray(pixel_size, np.float64), (2,)).tolist()
        _, nv, nu = cast("np.ndarray", record.projections).shape  # frames: always present
        center = (0.0, 0.0) if record.detector is None else record.detector.center
        detector = Detector(nu, nv, du, dv, center)
        meta["detector_pixel_size_status"] = "provided"
        meta.pop("length_unit", None)
    if geometry is not None or (detector is not None and pixel_size is None):
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
    if geometry is None and detector is not None:
        record = with_view_offsets(record, detector)
    record = record.with_parallel_detector() if geometry is None else record
    resolved = geometry if geometry is not None else geometry_of(record, apply_poses=apply_poses)
    return record, frame_output_grid(resolved, record)
