"""Construct lazy detector frames from saved scans, recognized acquisitions or TIFF stacks."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax._loading_geometry import frame_geometry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tomojax._loading import Frames
    from tomojax._loading_geometry import DeclaredGeometry
    from tomojax._scan import Scan
    from tomojax.geometry import Detector, Grid, ScanGeometry


def line_integral_frames(scan: Scan, stack: Any, key: np.ndarray) -> Frames:
    """Pair lazy sample storage with the saved scan's geometry and acquisition positions."""
    from tomojax._loading import Frames

    frames = Frames(
        stack,
        scan.geometry,
        name=scan.name,
        source=scan.source,
        view_positions=scan.view_positions,
        corrections=scan.corrections,
    )
    if np.any(key != 0):
        frames = replace(frames.selected(key == 0), corrections=scan.corrections)
    return frames


def acquisition_frames(
    file: Path,
    *,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
    **overrides: Any,
) -> Frames | None:
    from tomojax._loading import Frames
    from tomojax.io.api import read_acquisition

    found = read_acquisition(
        file, data_path=data_path, image_key_path=image_key_path, angles_path=angles_path
    )
    if found is None:
        return None
    record, resolved = frame_geometry(found.record, **overrides)
    return Frames(
        counts=found.counts,
        geometry=resolved,
        flats=found.flats,
        darks=found.darks,
        flat_positions=found.flat_positions,
        white_level=found.white_level,
        view_positions=found.view_positions,
        exposure_times=found.exposure_times,
        flat_exposure_times=found.flat_exposure_times,
        dark_source=found.dark_source,
        linear_limit=found.linear_limit,
        name=record.sample_name or "sample",
        source=record,
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
    from tomojax._loading import Frames
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
    from tomojax._scan import cut

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
