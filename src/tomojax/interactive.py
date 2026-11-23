"""Registered slices, centre sweeps, detector bands and region overlays for interactive clients."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from tomojax import progress
from tomojax._loading import corrected_data
from tomojax._planning import ResolvedReconstruction, resolve_grid, resolve_reconstruction
from tomojax._region import RegionEstimate, estimate_region
from tomojax._scan import Scan, binned_geometry
from tomojax._workflow import reconstruct, reconstruct_resolved
from tomojax.core.registration import planes, recorded_length_unit
from tomojax.errors import InvalidParameter
from tomojax.geometry import ConeSegments, grid_volume_origin
from tomojax.recon import FBPConfig
from tomojax.recon.api import AngularCoverageError, axial_fbp_plane

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax._loading import Frames
    from tomojax._parameters import Method, RegionName
    from tomojax.corrections import Step
    from tomojax.geometry import Detector, Grid, ScanGeometry


__all__ = [
    "CorrectionBand",
    "RegionOverlay",
    "centre_sweep",
    "correction_preview",
    "reconstruct_slice",
    "region_preview",
]


type Axis = Literal["z", "y", "x"]


def plane_dimension(grid: Grid, axis: Axis, index: int) -> int:
    """Resolve an index against the original reconstruction grid."""
    if axis not in ("x", "y", "z"):
        raise InvalidParameter("axis", "plane axis must be x, y or z")
    dimension = "zyx".index(axis)
    if not 0 <= index < grid.shape[dimension]:
        raise InvalidParameter("index", "plane index outside volume")
    return dimension


def resolved_slice(
    plan: ResolvedReconstruction, axis: Axis, index: int, *, warm_start: bool
) -> Any:
    """Select the plane after resolving options and any volume support on the full grid."""
    dim = plane_dimension(plan.scan.grid, axis, index)
    if axis == "z" and plan.method == "fbp" and isinstance(plan.config, FBPConfig):
        scan = plan.scan
        value = axial_fbp_plane(
            scan.geometry, scan.grid, scan.detector, scan.projections, plan.config, index
        )
        if value is not None:
            if plan.cylinder is not None:
                from tomojax._field_of_view import cylinder_support

                mask = cylinder_support(replace(scan.grid, nz=1), plan.cylinder)[0]
                value = np.where(mask, value, 0)
            return value
    result = reconstruct_resolved(plan, warm_start=warm_start)
    key: list[slice | int] = [slice(None)] * 3
    key[dim] = index
    return result.volume[tuple(key)].copy()


def reconstruct_slice(
    scan: Scan, axis: Axis, index: int, *, method: Method = "auto", **options: Any
) -> Any:
    """Return the indexed plane of the requested public reconstruction on its original grid.

    Parameters
    ----------
    scan : Scan
        Input projections and calibrated geometry.
    axis : {"z", "y", "x"}
        Dimension held fixed in the returned plane.
    index : int
        Zero-based plane index on the resolved grid.
    method : str, optional
        Public reconstruction method.
    **options : dict
        Keyword arguments to the reconstruction resolver, including grid, region and solver options.

    Returns
    -------
    numpy.ndarray or jax.Array
        Float32 attenuation in the remaining public array axes.
    """
    plan = resolve_reconstruction(scan, method, **options)
    return resolved_slice(plan, axis, index, warm_start=bool(options.get("warm_start")))


def _shift_center(geometry: ScanGeometry, shift: float) -> ScanGeometry:
    """Translate each calibrated detector horizontally, keeping poses and panel orientations."""
    if isinstance(geometry, ConeSegments):
        return replace(
            geometry, segments=tuple(_shift_center(part, shift) for part in geometry.segments)
        )
    if not is_dataclass(geometry) or isinstance(geometry, type):
        raise TypeError(f"cannot shift a {type(geometry).__name__} geometry")
    if base := getattr(geometry, "base", None):
        return replace(geometry, base=_shift_center(base, shift))
    detector = geometry.detector
    return replace(
        geometry,
        detector=replace(detector, center=(detector.center[0] + shift, detector.center[1])),
    )


def centre_sweep(
    scan: Scan,
    axis: Axis,
    index: int,
    centres: Sequence[float],
    *,
    method: Method = "auto",
    **options: Any,
) -> Any:
    """Reconstruct ordered planes at signed horizontal offsets in input detector pixels.

    Resolve the requested region once on the input scan. Every offset uses that fixed
    grid, support and plane index. A full reconstruction oracle shifts the detector and
    passes this resolved grid explicitly at each centre. Repeated offsets retain their order.

    Parameters
    ----------
    scan : Scan
        Input projections and calibrated geometry.
    axis : {"z", "y", "x"}
        Dimension held fixed in the returned plane.
    index : int
        Zero-based plane index on the resolved grid.
    centres : sequence of float
        Signed detector offsets relative to the central column, in input pixels.
    method : str, optional
        Public reconstruction method.
    **options : dict
        Keyword arguments to the reconstruction resolver, including grid, region and solver options.

    Returns
    -------
    numpy.ndarray or jax.Array
        Float32 attenuation with a leading centre coordinate, in the supplied order.
    """
    plans = resolve_sweep(scan, axis, index, centres, method=method, **options)
    return resolved_sweep(plans, axis, index, warm_start=bool(options.get("warm_start")))


def resolve_sweep(
    scan: Scan,
    axis: Axis,
    index: int,
    centres: Sequence[float],
    *,
    method: Method = "auto",
    grid: Grid | None = None,
    region: RegionName | RegionEstimate | None = None,
    **options: Any,
) -> tuple[ResolvedReconstruction, ...]:
    """Resolve support on the input and validate the actual acquisition for each requested plane."""
    if len(centres) == 0 or not np.isfinite(centres).all():
        raise InvalidParameter(
            "centres", "centres must be a nonempty sequence of finite detector offsets"
        )
    fixed_grid, cylinder, info = resolve_grid(scan, grid, region)
    plane_dimension(fixed_grid or scan.grid, axis, index)
    plans = []
    for i, centre in enumerate(centres):
        progress.check_cancelled()
        shift = float(centre) * scan.detector.du - scan.detector.center[0]
        shifted = replace(scan, geometry=_shift_center(scan.geometry, shift))
        try:
            plan = resolve_reconstruction(shifted, method, grid=fixed_grid, **options)
        except AngularCoverageError as err:
            raise InvalidParameter(f"centres[{i}]", str(err)) from err
        plans.append(replace(plan, cylinder=cylinder, info={**plan.info, **info}))
    return tuple(plans)


def resolved_sweep(
    plans: Sequence[ResolvedReconstruction],
    axis: Axis,
    index: int,
    *,
    warm_start: bool,
) -> Any:
    """Execute the validated acquisitions in order, keeping only their requested planes."""
    import jax
    import jax.numpy as jnp

    images = []
    for plan in plans:
        progress.check_cancelled()
        images.append(resolved_slice(plan, axis, index, warm_start=warm_start))
    return jnp.stack(images) if isinstance(images[0], jax.Array) else np.stack(images)


@dataclass(frozen=True)
class RegionOverlay:
    """Estimated reconstruction grid, its voxel-edge bounds and registered central images.

    Parameters
    ----------
    estimate : RegionEstimate
        The estimator's result, usable in a subsequent reconstruction.
    bounds : tuple
        Lower and upper voxel edges in physical x, y and z order.
    images : dict
        Registered central planes indexed by their fixed axis.
    """

    estimate: RegionEstimate
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    images: dict[str, progress.Sample]


def region_preview(scan: Scan) -> RegionOverlay:
    """Return the estimator's grid and its central FBP overlays in physical space.

    Parameters
    ----------
    scan : Scan
        Calibrated laminography projections supplied to the region estimator.

    Returns
    -------
    RegionOverlay
        Estimate, physical XYZ bounds and registered central planes.
    """
    estimate = estimate_region(scan)
    grid = estimate.scan.grid
    origin = grid_volume_origin(grid)
    x, y, z = (
        (o - d / 2, o + (n - 0.5) * d)
        for o, d, n in zip(
            origin, (grid.vx, grid.vy, grid.vz), (grid.nx, grid.ny, grid.nz), strict=True
        )
    )
    unit = recorded_length_unit(scan.source.geometry_metadata) if scan.source else None
    images = planes(reconstruct(estimate.scan, "fbp").volume, grid, unit=unit or "scan_length")
    return RegionOverlay(estimate, (x, y, z), images)


@dataclass(frozen=True)
class CorrectionBand:
    """Native input counts, binned output line integrals and retained source-view indices.

    Parameters
    ----------
    before : numpy.ndarray
        Native counts across the requested complete blocks.
    after : numpy.ndarray
        Line integrals on the selected binned rows.
    views : numpy.ndarray
        Indices of retained views in the input acquisition.
    detector : Detector
        Full binned detector used to register the output rows.
    """

    before: np.ndarray
    after: np.ndarray
    views: np.ndarray
    detector: Detector


def correction_preview(
    frames: Frames,
    rows: tuple[int, int],
    *steps: Step,
    binning: int = 1,
    epsilon: float = 1e-6,
    views_per_batch: int | None = None,
) -> CorrectionBand:
    """Correct complete native blocks for a half-open interval in the binned output.

    Parameters
    ----------
    frames : Frames
        Raw acquisition with calibration and exposure metadata.
    rows : tuple of int
        First binned row and exclusive stop.
    *steps : Step
        Ordered corrections, as for the full acquisition.
    binning : int, optional
        Width and height of a native detector block.
    epsilon : float, optional
        Lower bound for flat and transmission values.
    views_per_batch : int or None, optional
        Maximum number of views in a correction batch.

    Returns
    -------
    CorrectionBand
        Native counts, binned line integrals and original input view indices.
    """
    low, high = rows
    done = corrected_data(frames, steps, epsilon, views_per_batch, binning, output_rows=rows)
    native = slice(low * binning, high * binning)
    counts = frames.counts
    before = (
        counts.window(native, slice(None))[:] if hasattr(counts, "window") else counts[:, native, :]
    )
    result_detector = binned_geometry(frames.geometry, binning).detector
    return CorrectionBand(np.array(before, copy=True), done.projections, done.kept, result_detector)
