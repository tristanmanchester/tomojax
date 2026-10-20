"""The grid ``--roi`` and ``--grid`` choose, for ``tomojax recon`` and ``tomojax align``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from tomojax.geometry import (
    Detector,
    Grid,
    compute_roi,
    cylindrical_mask_xy,
    grid_from_detector_fov,
    grid_from_detector_fov_cube,
    grid_from_detector_fov_slices,
)

if TYPE_CHECKING:
    import jax

    from tomojax import Scan

ROI_CHOICES = ("auto", "off", "cube", "bbox", "cyl")
ROI_HELP = (
    "Crop the grid to the detector's field of view: auto (default), cube, bbox, "
    "cyl (auto, zeroing outside the cylinder every view sees), or off"
)


def region_grid(scan: Scan, *, roi: str, grid: tuple[int, int, int] | None) -> tuple[Grid, bool]:
    """The grid ``--roi`` and ``--grid`` choose, and whether to zero outside the cylinder.

    ``--grid`` resamples the selected ROI to new voxel counts, preserving its
    physical extent. Cylindrical support is retained at the new resolution.
    """
    geometry_type = "parallel" if scan.source is None else scan.source.geometry_type
    is_parallel = str(geometry_type).lower() == "parallel"
    chosen = _resolve_roi_grid(scan.grid, scan.detector, is_parallel=is_parallel, roi_mode=roi)
    if grid is not None:
        return chosen.resampled(grid), roi == "cyl"
    return chosen, roi == "cyl"


def cylinder_support(grid: Grid, detector: Detector) -> np.ndarray:
    """``(nx, ny, nz)``: 1 inside the cylinder every view sees, 0 outside."""
    inside = np.asarray(cylindrical_mask_xy(grid, detector), np.float32)[:, :, None]
    return np.broadcast_to(inside, (grid.nx, grid.ny, grid.nz)).copy()


def zero_outside_cylinder(
    volume: np.ndarray | jax.Array, grid: Grid, detector: Detector
) -> np.ndarray:
    """``volume`` with the voxels outside the cylinder every view sees set to zero."""
    return np.asarray(volume) * cylinder_support(grid, detector)


def _resolve_roi_grid(
    grid: Grid,
    detector: Detector,
    *,
    is_parallel: bool,
    roi_mode: str,
) -> Grid:
    if roi_mode == "off":
        return grid
    try:
        return _roi_grid(grid, detector, is_parallel=is_parallel, roi_mode=roi_mode)
    except Exception as exc:
        if roi_mode != "auto":
            raise ValueError(f"Failed to apply requested --roi={roi_mode!r}") from exc
        logging.warning(
            "--roi=auto could not be applied; continuing without ROI crop: %s",
            exc,
            exc_info=True,
        )
        return grid


def _roi_grid(grid: Grid, detector: Detector, *, is_parallel: bool, roi_mode: str) -> Grid:
    """The grid ``roi_mode`` asks for; ``auto`` crops only when the detector sees less."""
    if roi_mode == "auto":
        roi = compute_roi(grid, detector, crop_y_to_u=is_parallel)
        if (roi.nx_roi, roi.ny_roi, roi.nz_roi) == (grid.nx, grid.ny, grid.nz):
            return grid
        if is_parallel:
            return grid_from_detector_fov_slices(grid, detector, crop_y_to_u=True)
        return grid_from_detector_fov(grid, detector, crop_y_to_u=False)
    crop = {
        "cube": grid_from_detector_fov_cube,
        "cyl": grid_from_detector_fov_slices,
        "bbox": grid_from_detector_fov,
    }.get(roi_mode)
    return grid if crop is None else crop(grid, detector, crop_y_to_u=is_parallel)


__all__ = ["ROI_CHOICES", "ROI_HELP", "cylinder_support", "region_grid", "zero_outside_cylinder"]
