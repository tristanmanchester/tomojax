"""One bounded smooth-pose workflow for declared laminography geometry."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import time
from typing import TYPE_CHECKING

import numpy as np

from tomojax._scan import Reconstruction, describe, record_of, with_grid
from tomojax.alignment.api import coupled_pose_config
from tomojax.core.multires import scale_grid
from tomojax.recon import FistaConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax._scan import Scan
    from tomojax.geometry import Grid

# An alignment image, not the final reconstruction sampling: 64 MiB leaves
# room for Krylov state and pose derivatives on a small CUDA device.
_ALIGNMENT_VOXELS = 16 * 1024**2


def uses_laminography_workflow(scan: Scan) -> bool:
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    kind = describe(scan.geometry)[0]
    return kind == "lamino" or (
        kind == "parallel" and acquisition.get("geometry_status") == "not_recorded"
    )


def reconstruct_laminography(
    scan: Scan, *, align: Callable, reconstruct: Callable, grid: Grid | None = None
):
    """Fit smooth detector motion on a bounded image, keeping declared geometry.

    The physical support is never inferred or cropped. The returned scan keeps
    native projections and detector coordinates; its image may be coarse.
    An explicit grid fixes output sampling. Missing axis metadata requires
    explicit setup or caller-supplied geometry, never an automatic axis guess.
    """
    started = time.perf_counter()
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    if acquisition.get("geometry_status") == "not_recorded":
        raise ValueError(
            "automatic alignment requires declared geometry; "
            "supply geometry or use an explicit setup mode"
        )
    if grid is None and scan.poses is None:
        g = scan.grid
        factor = max(1, math.ceil((g.nx * g.ny * g.nz / _ALIGNMENT_VOXELS) ** (1 / 3)))
        grid = scale_grid(g, factor)
        while grid.nx * grid.ny * grid.nz > _ALIGNMENT_VOXELS:
            factor += 1
            grid = scale_grid(g, factor)
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
    # Square binning must retain at least one pixel on a thin detector.
    factor = min(
        scan.detector.nu,
        scan.detector.nv,
        max(1, math.ceil(max(scan.detector.nu, scan.detector.nv) / 320)),
    )
    work = scan.binned(factor) if factor > 1 else scan
    config = coupled_pose_config(
        pose_model="spline",
        knot_spacing=max(1, math.ceil(len(scan.angles) / 12)),
        degree=3,
        pose_translation_frame="detector",
        optimise_dofs=("dx", "dz"),
        seed_translations=False,
        outer_iterations=4,
        iterations=5,
        gn_joint_iterations=20,
    )
    result = align(work, mode="pose", levels=(1,), config=config)
    poses = np.asarray(result.scan.poses)
    corrected = scan.with_poses(poses)
    automatic = {
        "method": "joint_volume_pose",
        "status": "ok",
        "reasons": [],
        "geometry": "declared",
        "grid": asdict(scan.grid),
        "detector_binning": factor,
        "pose_model": "spline",
        "pose_translation_frame": "detector",
        "preparation_seconds": time.perf_counter() - started,
        "warning": "Smooth motion is not physical truth; inspect images and held-out projections.",
    }
    source = record_of(corrected)
    source.geometry_metadata["alignment_info"] = {"automatic": automatic}
    corrected = replace(corrected, source=source)
    return Reconstruction(
        result.volume, corrected, "joint_volume_pose", {**result.info, "automatic": automatic}
    )
