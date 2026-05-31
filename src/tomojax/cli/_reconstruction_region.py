"""Shared reconstruction region resolution for reconstruction and alignment CLIs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging

import jax.numpy as jnp

from tomojax._typed_arrays import jax_float32_array
from tomojax.geometry import (
    Detector,
    Grid,
    compute_roi,
    cylindrical_mask_xy,
    grid_from_detector_fov,
    grid_from_detector_fov_cube,
    grid_from_detector_fov_slices,
)


@dataclass(frozen=True, slots=True)
class ReconstructionRegion:
    """Resolved reconstruction grid and mask policy from CLI region options."""

    recon_grid: Grid
    apply_output_mask: bool
    solver_volume_mask_mode: str
    roi_metadata: dict[str, object]


def resolve_reconstruction_region(
    grid: Grid,
    detector: Detector,
    *,
    geometry_type: str,
    roi_mode: str,
    grid_override: tuple[int, int, int] | list[int] | None,
    mask_mode: str = "off",
) -> ReconstructionRegion:
    """Resolve the reconstruction grid and mask policy shared by CLI workflows."""
    roi_requested = str(roi_mode).lower()
    is_parallel = str(geometry_type).lower() == "parallel"
    recon_grid = _resolve_roi_grid(
        grid,
        detector,
        is_parallel=is_parallel,
        roi_mode=roi_requested,
    )
    apply_output_mask = roi_requested == "cyl"
    if grid_override is not None:
        nx, ny, nz = map(int, grid_override)
        recon_grid = replace(recon_grid, nx=nx, ny=ny, nz=nz)
        apply_output_mask = False

    return ReconstructionRegion(
        recon_grid=recon_grid,
        apply_output_mask=apply_output_mask,
        solver_volume_mask_mode=_normalize_mask_mode(mask_mode),
        roi_metadata={
            "requested": roi_requested,
            "is_parallel": bool(is_parallel),
            "grid_changed": recon_grid != grid,
        },
    )


def solver_volume_mask(region: ReconstructionRegion, detector: Detector) -> jnp.ndarray | None:
    """Return the solver mask implied by a resolved reconstruction region."""
    if region.solver_volume_mask_mode != "cyl":
        return None
    try:
        m_xy = cylindrical_mask_xy(region.recon_grid, detector)
        return jax_float32_array(m_xy)[:, :, None]
    except Exception as exc:
        raise ValueError(
            f"Failed to apply requested --mask-vol={region.solver_volume_mask_mode!r}"
        ) from exc


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
        roi = compute_roi(grid, detector, crop_y_to_u=is_parallel)
        full_half_x = ((grid.nx / 2.0) - 0.5) * float(grid.vx)
        full_half_y = ((grid.ny / 2.0) - 0.5) * float(grid.vy)
        full_half_z = ((grid.nz / 2.0) - 0.5) * float(grid.vz)
        det_smaller = (
            (roi.r_u + 1e-6) < full_half_x
            or (is_parallel and (roi.r_u + 1e-6) < full_half_y)
            or (roi.r_v + 1e-6) < full_half_z
        )
        if roi_mode == "auto" and det_smaller:
            if is_parallel:
                return grid_from_detector_fov_slices(grid, detector, crop_y_to_u=True)
            return grid_from_detector_fov(grid, detector, crop_y_to_u=False)
        if roi_mode == "cube":
            return grid_from_detector_fov_cube(grid, detector, crop_y_to_u=is_parallel)
        if roi_mode == "cyl":
            return grid_from_detector_fov_slices(grid, detector, crop_y_to_u=is_parallel)
        if roi_mode == "bbox":
            return grid_from_detector_fov(grid, detector, crop_y_to_u=is_parallel)
        return grid
    except Exception as exc:
        if roi_mode == "auto":
            logging.warning(
                "--roi=auto could not be applied; continuing without ROI crop: %s",
                exc,
                exc_info=True,
            )
            return grid
        raise ValueError(f"Failed to apply requested --roi={roi_mode!r}") from exc


def _normalize_mask_mode(mask_mode: str) -> str:
    mode = str(mask_mode).lower()
    if mode in {"off", "none"}:
        return "off"
    if mode in {"cyl", "cylindrical"}:
        return "cyl"
    raise ValueError(f"unsupported volume mask mode {mask_mode!r}")


__all__ = ["ReconstructionRegion", "resolve_reconstruction_region", "solver_volume_mask"]
