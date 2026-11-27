"""Construct lazy detector frames from saved scans, recognized acquisitions or TIFF stacks."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax._loading.geometry import frame_geometry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tomojax._loading.geometry import DeclaredGeometry
    from tomojax._loading.loader import Frames
    from tomojax._scan.model import Scan
    from tomojax.geometry import Detector, Grid, ScanGeometry


def line_integral_frames(scan: Scan, stack: Any) -> Frames:
    """Pair lazy sample storage with the saved scan's geometry and acquisition positions."""
    from tomojax._loading.loader import Frames

    return Frames(
        stack,
        scan.geometry,
        name=scan.name,
        source=scan.source,
        view_positions=scan.view_positions,
        corrections=scan.corrections,
    )


def tiff_frames(
    file: Path,
    *,
    angles: Sequence[float] | np.ndarray | None,
    detector: Detector | None = None,
    pixel_size: float | tuple[float, float] | None = None,
    grid: Grid | None = None,
    geometry: ScanGeometry | None = None,
    declared: DeclaredGeometry | None = None,
) -> Frames:
    from tomojax._loading.loader import Frames
    from tomojax.io import ProjectionDataset
    from tomojax.io.api import read_tiff_frames

    if geometry is not None and angles is None:
        angles = np.asarray(cast("Any", geometry).angles)
    if angles is None:
        raise ValueError(
            f"{file} is a TIFF stack, which holds no angles: pass angles=... (degrees), "
            "or use tomojax correct --angles"
        )
    counts = read_tiff_frames(file, lazy=True)
    views, rows, cols = counts.shape
    angles = np.asarray(angles, np.float32)
    if angles.shape != (views,):
        raise ValueError(f"{file} has {views} frames but {angles.size} angles")
    record = ProjectionDataset(
        projections=np.broadcast_to(np.float32(0), counts.shape),
        angles=angles,
        detector=detector,
        source_path=str(file),
        source_format="tiff_stack",
    )
    record, resolved = frame_geometry(
        record, pixel_size=pixel_size, grid=grid, geometry=geometry, declared=declared
    )
    return Frames(counts, resolved, source=record)


def selected_frames(frames: Frames, kept: np.ndarray, counts: Any) -> Frames:
    """Attach selected sample storage while preserving calibration coordinates."""
    from tomojax._data import views_of
    from tomojax._scan.model import cut

    at = np.arange(frames.views) + 0.5 if frames.view_positions is None else frames.view_positions
    return replace(
        frames,
        counts=counts,
        geometry=views_of(frames.geometry, kept),
        view_positions=np.asarray(at, np.float64)[kept],
        exposure_times=(
            None
            if frames.exposure_times is None
            else np.broadcast_to(frames.exposure_times, (frames.views,))[kept]
        ),
        corrections=(*frames.corrections, cut("selected", views=kept.tolist())),
    )
