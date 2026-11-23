"""Resolve workflow settings before allocating solver state or reading image samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from tomojax._parameters import Method
from tomojax._scan import Scan, with_grid
from tomojax.errors import InvalidParameter
from tomojax.recon.api import MethodConfig, method_config

if TYPE_CHECKING:
    from tomojax._field_of_view import Cylinder
    from tomojax._parameters import RegionName
    from tomojax._region import RegionEstimate
    from tomojax.geometry import Grid


@dataclass(frozen=True)
class ResolvedReconstruction:
    """Acquisition, support and numerical configuration consumed by the solver."""

    scan: Scan
    method: Method
    config: MethodConfig
    cylinder: Cylinder | None
    reasons: tuple[str, ...]
    info: dict[str, Any]


def resolve_grid(
    scan: Scan,
    grid: Grid | None,
    region: RegionName | RegionEstimate | None,
) -> tuple[Grid | None, Cylinder | None, dict[str, Any]]:
    """Resolve the support and keep its physical mask and provenance together."""
    if region is None:
        return grid, None, {}
    from tomojax._field_of_view import region_grid

    if grid is not None:
        raise InvalidParameter("grid", "pass grid= or region=, not both")
    chosen, cylinder = region_grid(scan, region)
    info: dict[str, Any] = {"region": region if isinstance(region, str) else "estimate"}
    if not isinstance(region, str):
        info["region_estimate"] = dict(region.info)
    return chosen, cylinder, info


def resolve_reconstruction(
    scan: Scan,
    method: Method = "auto",
    *,
    grid: Grid | None = None,
    region: RegionName | RegionEstimate | None = None,
    config: MethodConfig | None = None,
    warm_start: bool | None = None,
    devices: Any = None,
    **settings: Any,
) -> ResolvedReconstruction:
    """Select the inverse, its grid and all solver settings once for plan and run."""
    grid, cylinder, info = resolve_grid(scan, grid, region)
    if cylinder is not None:
        info["cylindrical_output_mask"] = True
    settings = {k: v for k, v in settings.items() if v is not None}
    if method == "auto":
        method, reasons = _select_automatic_method(scan)
        options = {
            k: v
            for k, v in {**settings, "config": config, "warm_start": warm_start}.items()
            if v is not None
        }
        if options:
            taking = _methods_taking(options)
            suggestion = f"pass an explicit method that takes them: {', '.join(taking)}"
            raise InvalidParameter(
                sorted(options)[0],
                f"method='auto' would select {method!r} for this scan and does not take "
                f"method-specific options ({', '.join(sorted(options))}); "
                + (suggestion if taking else "no one method takes all of them"),
            )
        center_status = _automatic_axial_setup(scan, method=method, reasons=reasons)
        info["automatic"] = {
            "method": method,
            "status": "limited" if reasons else "ok",
            "reasons": list(reasons),
            "center_status": center_status,
            "geometry": "preserved",
            "warning": (
                "Method selection does not validate geometry, support or detector artefacts."
            ),
        }
        if not reasons:
            reasons.append(
                "Automatic selection from the declared acquisition and angular coverage."
            )
    else:
        reasons = ["Explicit method selection."]
    if devices is not None:
        settings["devices"] = devices
    cfg = method_config(method, config=config, warm_start=warm_start, **settings)
    return ResolvedReconstruction(
        scan if grid is None else with_grid(scan, grid), method, cfg, cylinder, tuple(reasons), info
    )


def _methods_taking(options: dict[str, Any]) -> list[str]:
    """The explicit methods whose settings take every one of ``options``."""
    taking = []
    for name in get_args(Method.__value__)[1:]:  # all but "auto"
        try:
            _ = method_config(name, **options)
        except ValueError:
            continue
        taking.append(name)
    return taking


def _select_automatic_method(scan: Scan) -> tuple[Method, list[str]]:
    """Choose the default reconstruction method and any advisory reasons for ``scan``."""
    from tomojax.geometry import (
        ConeSegments,
        LaminographyGeometry,
        ParallelGeometry,
        RotationAxisGeometry,
    )

    base: Any = scan.geometry
    while hasattr(base, "base"):
        base = base.base
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    unknown_axis = (
        isinstance(acquisition, dict) and acquisition.get("geometry_status") == "not_recorded"
    )
    if unknown_axis:
        raise ValueError(
            "reconstruction requires declared acquisition geometry: this file records no "
            "rotation axis; pass geometry=... to tomojax.load"
        )
    reasons: list[str] = []
    method: Method = "fbp"
    if isinstance(base, ConeSegments):
        # Per-view detector offsets (a Zeiss scan's shifts) make segments of one orbit, which
        # FDK weights and backprojects view by view, exactly. Several trajectories, or
        # arrangements that differ (a source at several heights), are not one circular orbit.
        beams = [getattr(segment, "beam", None) for segment in base.segments]
        if len(base.trajectory_lengths) > 1 or any(b != beams[0] for b in beams):
            method = "cgls"
            reasons.append(
                "Segmented cone geometry uses matched CGLS because FDK requires one arrangement; "
                "iteration count and image quality are not automatically validated."
            )
    elif type(scan.geometry) is ParallelGeometry:
        angles = np.unique(np.mod(np.asarray(scan.angles, np.float64), 180.0))
        gaps = np.diff(np.r_[angles, angles[0] + 180])
        if len(angles) >= 16 and np.max(gaps) <= max(22.5, 2 * float(np.median(gaps))):
            method = "fourier"
        else:
            reasons.append(
                "Sparse or incomplete half-turn coverage: Fourier inversion was not selected."
            )
    elif isinstance(base, LaminographyGeometry | RotationAxisGeometry):
        reasons.append(
            "FBP leaves the frequencies no view measures (the missing cone of a tilted "
            "axis) empty; method='fista' estimates them from nonnegativity and its "
            "regulariser, at the cost of its iterations."
        )
    elif getattr(base, "beam", None) is None:
        reasons.append(
            "Posed or custom parallel geometry uses FBP; "
            "automatic iterative validation is unavailable."
        )
    return method, reasons


def _automatic_axial_setup(scan: Scan, *, method: Method, reasons: list[str]) -> str | None:
    """The centre the scan's file recorded, and unresolved raw-workflow limitations."""
    status = (
        None if scan.source is None else scan.source.geometry_metadata.get("detector_center_status")
    )
    if method == "fourier" and any(step.name == "log" for step in scan.corrections):
        reasons.append(
            "Corrected projections were already materialized; use reconstruct_slabs on raw "
            "Frames to avoid that host round trip. Raw-workflow speed is not guaranteed here."
        )
        if not any(step.name in {"stripes", "rings"} for step in scan.corrections):
            reasons.append(
                "No stripe correction is recorded; detector-ring suppression is unassessed."
            )
    return status
