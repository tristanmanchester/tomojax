"""Automatic detector-motion fitting for declared parallel and laminography geometries."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np

from tomojax._scan import Reconstruction, describe, record_of, with_grid
from tomojax.alignment._integration import fit_translations
from tomojax.geometry import stack_view_poses
from tomojax.recon import FistaConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax._scan import Scan
    from tomojax.geometry import Grid


def uses_laminography_workflow(scan: Scan) -> bool:
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    kind = describe(scan.geometry)[0]
    return kind == "lamino" or (
        kind == "parallel" and acquisition.get("geometry_status") == "not_recorded"
    )


def reconstruct_with_motion(scan: Scan, *, reconstruct: Callable, grid: Grid | None = None):
    """Fit per-view detector translations jointly with a bounded-scale image.

    The declared geometry is kept and the physical support is never inferred
    or cropped. Motion is smooth with free steps: a spline with knots every ten
    views plus view-to-view jumps under a Huber prior, in native detector
    pixels. The returned scan keeps native projections and detector
    coordinates; its image is the fit's working-scale volume. An explicit grid
    is the working grid itself, never coarsened. Missing axis metadata
    requires explicit setup or caller-supplied geometry.
    """
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    if acquisition.get("geometry_status") == "not_recorded":
        raise ValueError(
            "automatic alignment requires declared geometry; "
            "supply geometry or use an explicit setup mode"
        )
    if grid is not None:
        scan = with_grid(scan, grid)
    if scan.poses is not None:
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
        return replace(
            result,
            info={
                **result.info,
                "automatic": {
                    "method": "fista",
                    "geometry": "preserved_pose_corrections",
                    "status": "ok",
                    "reasons": [],
                },
            },
        )
    views = len(scan.angles)
    nominal = np.asarray(stack_view_poses(scan.geometry, views), np.float64)
    volume, working, shifts, record = fit_translations(
        scan.projections, nominal, scan.grid, scan.detector, keep_grid=grid is not None
    )
    poses = np.zeros((views, 6))
    poses[:, 3] = scan.detector.du * shifts[:, 0]
    poses[:, 4] = scan.detector.dv * shifts[:, 1]
    corrected = with_grid(scan, working).with_poses(poses)
    automatic = {
        "method": "joint_volume_pose",
        "status": "ok",
        "reasons": [],
        "geometry": "declared",
        "grid": asdict(working),
        "detector_binning": record["detector_binning"],
        "pose_model": "spline_with_jumps",
        "pose_translation_frame": "detector",
        "fit": record,
        "warning": "Fitted motion is not physical truth; inspect images and held-out projections.",
    }
    source = record_of(corrected)
    source.geometry_metadata["alignment_info"] = {"automatic": automatic}
    corrected = replace(corrected, source=source)
    return Reconstruction(volume, corrected, "joint_volume_pose", {"automatic": automatic})
