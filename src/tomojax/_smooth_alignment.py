"""One joint alignment workflow for declared acquisition geometries."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax._scan import Reconstruction, describe, record_of, with_grid
from tomojax.alignment._integration import cone_frames, fit_poses
from tomojax.geometry import stack_view_poses
from tomojax.recon import FistaConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax._scan import Scan
    from tomojax.geometry import Grid


def align_poses(
    scan: Scan, *, fit: str | tuple[str, ...] = "translations", grid: Grid | None = None
) -> Reconstruction:
    """Jointly fit one image and the selected global and per-view parameters."""
    from tomojax.alignment._calibration import publish_calibration

    _require_declared_geometry(scan)
    if grid is not None:
        scan = with_grid(scan, grid)
    views = len(scan.angles)
    current = np.asarray(stack_view_poses(scan.geometry, views), np.float64)
    native = (scan.detector.du, scan.detector.dv)
    _, metadata, _, _ = describe(scan.geometry)
    volume, working, fitted, record = fit_poses(
        scan.projections,
        current,
        scan.grid,
        scan.detector,
        fit=fit,
        keep_grid=grid is not None,
        frames=cone_frames(scan.geometry),
        detector_roll_deg=float(cast("float", metadata.get("detector_roll_deg", 0.0))),
    )
    corrected, calibration = publish_calibration(
        scan, fitted, record["corrections"], native_pixel_size=native
    )
    corrected = with_grid(corrected, working)
    info = {**record, "method": "joint_volume_pose", "calibration": calibration}
    source = record_of(corrected)
    source.geometry_metadata["alignment_info"] = info
    corrected = replace(corrected, source=source)
    return Reconstruction(volume, corrected, "joint_volume_pose", info)


def _require_declared_geometry(scan: Scan) -> None:
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    if acquisition.get("geometry_status") == "not_recorded":
        raise ValueError("alignment requires declared acquisition geometry")


def reconstruct_with_motion(scan: Scan, *, reconstruct: Callable, grid: Grid | None = None):
    """Reconstruct a parallel or laminography scan, fitting its poses unless it has them.

    A scan with saved poses is reconstructed with them by FISTA at its grid; an
    unposed scan is aligned with :func:`align_poses`.
    """
    _require_declared_geometry(scan)
    if scan.poses is None:
        return align_poses(scan, grid=grid)
    if grid is not None:
        scan = with_grid(scan, grid)
    result = reconstruct(
        scan,
        "fista",
        config=FistaConfig(
            initial_epochs=10
            if scan.grid.vx > 1.5 * max(scan.detector.du, scan.detector.dv)
            else 8,
            initial_subsets=16,
            tv_weight=0,
            iterations=1,
            stream_projections=False,
        ),
    )
    automatic = {
        "method": "fista",
        "geometry": "preserved_pose_corrections",
        "status": "ok",
        "reasons": [],
    }
    return replace(result, info={**result.info, "automatic": automatic})
