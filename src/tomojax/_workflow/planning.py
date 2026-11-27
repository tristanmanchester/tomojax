"""Resolve workflow settings and measurement constraints before allocating solver state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from tomojax._parameters import Method, ReconMethod
from tomojax._scan.model import Scan, with_grid
from tomojax.errors import InvalidParameter
from tomojax.recon.api import MethodConfig, method_config, validate_reconstruction

if TYPE_CHECKING:
    from tomojax._parameters import RegionName
    from tomojax._workflow.field_of_view import Cylinder
    from tomojax._workflow.region import RegionEstimate
    from tomojax.geometry import Grid


UNDECLARED_GEOMETRY = (
    "requires declared acquisition geometry: this file records no rotation axis or pixel "
    "scale. Declare it when loading: pass geometry=... to tomojax.load in Python, "
    '--geometry parallel to tomojax correct, or geometry {"kind": "parallel"} to the '
    "worker's or MCP server's open"
)


@dataclass(frozen=True)
class ResolvedReconstruction:
    """Acquisition, support and numerical configuration consumed by the solver."""

    scan: Scan
    config: MethodConfig
    cylinder: Cylinder | None
    reasons: tuple[str, ...]
    info: dict[str, Any]
    warm_start: bool

    @property
    def iteration_budget(self) -> int | None:
        """Maximum updates in the resolved plan, including an automatic solver's cap."""
        iterations = getattr(self.config, "iterations", None)
        if self.config.method in ("cgls", "sirt") and iterations is None:
            return 50
        return iterations

    @property
    def method(self) -> ReconMethod:
        """Reconstruction method, named by the configuration's class."""
        return self.config.method


def plane_dimension(grid: Grid, axis: str, index: int) -> int:
    """Resolve an index against the original reconstruction grid."""
    if axis not in ("x", "y", "z"):
        raise InvalidParameter("axis", "plane axis must be x, y or z")
    dimension = "zyx".index(axis)
    if not 0 <= index < grid.shape[dimension]:
        raise InvalidParameter("index", "plane index outside volume")
    return dimension


def resolve_grid(
    scan: Scan,
    grid: Grid | None,
    region: RegionName | RegionEstimate | None,
    *,
    shape: tuple[int, int, int] | None = None,
) -> tuple[Grid | None, Cylinder | None, dict[str, Any]]:
    """Resolve the support and keep its physical mask and provenance together.

    ``shape`` (``(nz, ny, nx)``) resamples the chosen region, or the scan's grid, to
    those voxel counts over the same physical extent; it replaces ``grid``.
    """
    from tomojax._workflow.field_of_view import region_grid

    if grid is not None and (region is not None or shape is not None):
        raise InvalidParameter("grid", "pass grid= or region= and shape=, not both")
    if region is None:
        return (grid if shape is None else scan.grid.resampled(shape=shape)), None, {}
    chosen, cylinder = region_grid(scan, region, shape=shape)
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
    plane: tuple[str, int] | None = None,
    metadata_only: bool = False,
    shape: tuple[int, int, int] | None = None,
    **settings: Any,
) -> ResolvedReconstruction:
    """Select the inverse, its grid and all solver settings once for plan and run."""
    grid, cylinder, info = resolve_grid(scan, grid, region, shape=shape)
    if cylinder is not None:
        info["cylindrical_output_mask"] = True
    settings = {k: v for k, v in settings.items() if v is not None}
    if method == "auto":
        method, reason, recommendation = _select_automatic_method(scan)
        reasons = [] if recommendation is None else [recommendation]
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
            "reason": reason,
            "status": "limited" if reasons else "ok",
            "reasons": list(reasons),
            "center_status": center_status,
            "geometry": "preserved",
            "warning": (
                "Method selection does not validate geometry, support or detector artifacts."
            ),
        }
        reasons.insert(0, reason)
    else:
        reasons = ["Explicit method selection."]
    if devices is not None:
        settings["devices"] = devices
    cfg = method_config(method, config=config, warm_start=warm_start, **settings)
    scan = scan if grid is None else with_grid(scan, grid)
    axial_index = (
        plane[1] if plane is not None and plane_dimension(scan.grid, *plane) == 0 else None
    )
    validate_reconstruction(
        scan.geometry,
        scan.grid,
        cfg,
        warm_start=bool(warm_start),
        projections=scan.projections,
        check_samples=not metadata_only,
        axial_index=axial_index,
    )
    return ResolvedReconstruction(scan, cfg, cylinder, tuple(reasons), info, bool(warm_start))


def _methods_taking(options: dict[str, Any]) -> list[str]:
    """The explicit methods whose settings take every one of ``options``."""
    taking = []
    for name in get_args(ReconMethod.__value__):
        try:
            _ = method_config(name, **options)
        except ValueError:
            continue
        taking.append(name)
    return taking


# Ray directions closer than this, in degrees, are one direction.
_SAME_DIRECTION = 1e-3


def _select_automatic_method(scan: Scan) -> tuple[Method, str, str | None]:
    """Choose a method with a reason and an optional recommendation."""
    from tomojax.geometry import (
        ConeSegments,
        LaminographyGeometry,
        RotationAxisGeometry,
        beam_of,
    )
    from tomojax.recon.fdk import check_angular_coverage

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
        raise ValueError(f"reconstruction {UNDECLARED_GEOMETRY}")
    method, reason = "fbp", "Filtered backprojection supports the declared acquisition geometry."
    if isinstance(base, ConeSegments):
        # Segments with detector offsets can still belong to one circular orbit.
        orbits = {
            (beam.source_to_axis, beam.source_height, beam.axis_offset)
            for segment in base.segments
            if (beam := beam_of(segment)) is not None
        }
        if len(base.trajectory_lengths) > 1 or len(orbits) > 1:
            return "cgls", "CGLS fits the cone-beam segments jointly.", None
    elif isinstance(base, LaminographyGeometry | RotationAxisGeometry):
        return (
            "fbp",
            "FBP follows the tilted rotation axis.",
            "Use method='fista' with nonnegative=True to estimate the frequencies no view "
            "measures from nonnegativity and regularization.",
        )
    elif getattr(base, "beam", None) is None:
        return _parallel_method(scan, _coverage_advice(scan))
    try:  # One orbit: FDK weights a full turn or 180 degrees plus the fan angle; CGLS any arc.
        check_angular_coverage(scan.geometry, scan.detector, len(scan.angles))
    except ValueError as err:
        method, reason = "cgls", f"CGLS, because {err}."
    return method, reason, None


def _parallel_method(scan: Scan, recommendation: str | None) -> tuple[Method, str, str | None]:
    """Choose the axial inverse from the measured directions and any out-of-plane motion."""
    from tomojax.geometry import ParallelGeometry

    if type(scan.geometry) is not ParallelGeometry:
        # Pure axial angle corrections describe the same Fourier experiment
        # as an encoder list. Keep the general FBP route for moving samples.
        from tomojax.recon.fourier import axial_trajectory

        try:
            _, shifts, _ = axial_trajectory(scan.geometry)
        except ValueError:
            return "fbp", "FBP uses the scan's per-view geometry.", recommendation
        if shifts is not None:
            return "fbp", "FBP uses the scan's per-view geometry.", recommendation
    if recommendation is None:
        return "fourier", "Fourier inversion uses the scan's half-turn coverage.", None
    reason = (
        "FBP, because automatic selection takes Fourier inversion only for 16 or more "
        "directions spread over the half turn."
    )
    return "fbp", reason, recommendation


def _coverage_advice(scan: Scan) -> str | None:
    """What a parallel-beam scan's ray directions leave unmeasured, or None."""
    # Parallel rays, as declared or with per-view poses, run along the lab's +y
    # axis; in the object frame that is row 1 of each view's rotation, so saved
    # rotation corrections count. Opposite views measure the same direction.
    poses = np.asarray([scan.geometry.pose_matrix(i) for i in range(len(scan.angles))])
    directions = np.sort(np.mod(np.degrees(np.arctan2(poses[:, 1, 1], poses[:, 1, 0])), 180.0))
    angles = directions[np.r_[True, np.diff(directions) > _SAME_DIRECTION]]
    if len(angles) > 1 and angles[0] + 180.0 - angles[-1] <= _SAME_DIRECTION:
        angles = angles[:-1]
    gaps = np.diff(np.r_[angles, angles[0] + 180])
    if np.max(gaps) > max(22.5, 2 * float(np.median(gaps))):
        return (
            "The views leave gaps in the half turn: expect streaks along the unmeasured "
            "directions. method='fista' with nonnegative=True fills them from "
            "nonnegativity and regularization."
        )
    if len(angles) < 16:
        return (
            f"The views measure {len(angles)} directions, fewer than 16: expect streaks. "
            "method='fista' with nonnegative=True reduces them through regularization."
        )
    return None


def _automatic_axial_setup(scan: Scan, *, method: Method, reasons: list[str]) -> str | None:
    """The center the scan's file recorded, and unresolved raw-workflow limitations."""
    status = (
        None if scan.source is None else scan.source.geometry_metadata.get("detector_center_status")
    )
    if method == "fourier" and any(step.name == "log" for step in scan.corrections):
        reasons.append(
            "Use reconstruct_slabs on raw Frames to avoid storing all corrected projections."
        )
        if not any(step.name == "rings" for step in scan.corrections):
            reasons.append("Apply Rings() if detector rings appear in the reconstruction.")
    return status
