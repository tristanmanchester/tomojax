"""One joint alignment workflow for declared acquisition geometries."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax._scan import Reconstruction, describe, record_of, with_grid
from tomojax.alignment._integration import cone_frames, fit_poses
from tomojax.backends.api import is_oom_error
from tomojax.geometry import stack_view_poses
from tomojax.progress import warn

if TYPE_CHECKING:
    from tomojax._scan import Scan
    from tomojax.alignment import AlignConfig
    from tomojax.geometry import Grid


def align_poses(
    scan: Scan,
    *,
    fit: str | tuple[str, ...] = "translations",
    grid: Grid | None = None,
    config: AlignConfig | None = None,
) -> Reconstruction:
    """Jointly fit one image and the selected global and per-view parameters."""
    from tomojax.alignment._calibration import publish_calibration

    _require_declared_geometry(scan)
    if grid is not None:
        scan = with_grid(scan, grid)
    views = len(scan.angles)
    native = (scan.detector.du, scan.detector.dv)
    _, metadata, _, _ = describe(scan.geometry)
    try:
        volume, working, fitted, record = fit_poses(
            scan.projections,
            np.asarray(stack_view_poses(scan.geometry, views), np.float64),
            scan.grid,
            scan.detector,
            fit=fit,
            keep_grid=grid is not None,
            frames=cone_frames(scan.geometry),
            detector_roll=float(cast("float", metadata.get("detector_roll", 0.0))),
            view_positions=scan.view_positions,
            declared_poses=np.asarray(stack_view_poses(scan.with_poses(None).geometry, views)),
            config=config,
        )
    except Exception as exc:
        if grid is None or not is_oom_error(exc):
            raise
        raise MemoryError(
            f"tj.align ran out of device memory on the {grid.nx} x {grid.ny} x {grid.nz} grid it "
            "was given: an explicit grid= is used as given, at its full resolution. Omit grid= "
            "to let the fit bound its own working scale."
        ) from exc
    changes = record["calibration"]
    corrected, values = publish_calibration(scan, fitted, changes, native_pixel_size=native)
    corrected = with_grid(corrected, working)
    calibration = {name: {"value": values[name], "change": changes[name]} for name in changes}
    info = {**record, "method": "joint_volume_pose", "calibration": calibration}
    _warn_about_views(record["report"])
    source = record_of(corrected)
    source.geometry_metadata["alignment_info"] = info
    corrected = replace(corrected, source=source)
    return Reconstruction(volume, corrected.geometry, "joint_volume_pose", info, corrected)


def _warn_about_views(report: dict) -> None:
    """Name the views a user should look at before trusting the corrections."""
    isolated, unusual = report["isolated_views"], report["unusual_views"]
    parts = list(report["advisories"])
    if isolated:
        parts.append(f"{len(isolated)} views jump away from both neighbours ({_first(isolated)})")
    if unusual:
        parts.append(f"{len(unusual)} views fit unusually badly ({_first(unusual)})")
    if not parts:
        return
    warn(
        f"tj.align: {'; '.join(parts)}. The report is advisory: see info['report'] and "
        "'Check the result' in docs/align-a-scan.md.",
        stacklevel=3,
    )


def _first(views: list[int], count: int = 5) -> str:
    shown = ", ".join(map(str, views[:count]))
    return f"views {shown}, ..." if len(views) > count else f"views {shown}"


def _require_declared_geometry(scan: Scan) -> None:
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    if acquisition.get("geometry_status") == "not_recorded":
        raise ValueError(
            "alignment requires declared acquisition geometry: this file records no rotation "
            "axis; pass geometry=... to tomojax.load"
        )
