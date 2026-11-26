"""Registered slices, center sweeps, detector bands and region overlays for interactive clients."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from tomojax import progress
from tomojax._loading import corrected_data
from tomojax._planning import (
    ResolvedReconstruction,
    plane_dimension,
    resolve_grid,
    resolve_reconstruction,
)
from tomojax._region import RegionEstimate, estimate_region
from tomojax._scan import Scan, bind_step_geometry, binned_geometry, shifted_center
from tomojax._workflow import reconstruct, reconstruct_resolved
from tomojax.core.registration import planes, recorded_length_unit
from tomojax.errors import InvalidParameter
from tomojax.geometry import grid_volume_origin
from tomojax.recon import FBPConfig
from tomojax.recon.api import AngularCoverageError, axial_fbp_plane

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tomojax._loading import Frames
    from tomojax._parameters import Method, RegionName
    from tomojax.corrections import Step
    from tomojax.geometry import Detector, Grid


__all__ = [
    "CorrectionBand",
    "RegionOverlay",
    "center_sweep",
    "correction_preview",
    "reconstruct_slice",
    "reconstruct_slices",
    "region_preview",
]


type Axis = Literal["z", "y", "x"]


def _axial_plane(plan: ResolvedReconstruction, axis: Axis, index: int) -> Any:
    """The plane on its own, where axial FBP can make it without the rest of the volume."""
    if axis != "z" or not isinstance(plan.config, FBPConfig):
        return None
    scan = plan.scan
    value = axial_fbp_plane(
        scan.geometry, scan.grid, scan.detector, scan.projections, plan.config, index
    )
    if value is not None and plan.cylinder is not None:
        from tomojax._field_of_view import cylinder_support

        mask = cylinder_support(replace(scan.grid, nz=1), plan.cylinder)[0]
        value = np.where(mask, value, 0)
    return value


def _cut(volume: Any, axis: Axis, index: int) -> Any:
    key: list[slice | int] = [slice(None)] * 3
    key["zyx".index(axis)] = index
    return volume[tuple(key)].copy()


def resolved_slice(plan: ResolvedReconstruction, axis: Axis, index: int) -> Any:
    """Select the plane after resolving options and any volume support on the full grid."""
    _ = plane_dimension(plan.scan.grid, axis, index)
    value = _axial_plane(plan, axis, index)
    if value is not None:
        return value
    return _cut(reconstruct_resolved(plan).volume, axis, index)


def reconstruct_slices(
    scan: Scan, requests: Sequence[tuple[Axis, int]], *, method: Method = "auto", **options: Any
) -> list[Any]:
    """Reconstruct several planes, sharing one full solve when any of them needs one.

    Every index is checked before anything runs. Axial FBP planes that take one
    detector row each are made on their own; otherwise one full reconstruction is
    cut into every requested plane.

    Parameters
    ----------
    scan : Scan
        Input projections and calibrated geometry.
    requests : sequence of (axis, index)
        Planes to make.
    method : str, optional
        Public reconstruction method.
    **options : dict
        Keyword arguments to the reconstruction resolver.

    Returns
    -------
    list of array
        Float32 attenuation planes, one per request, in order.
    """
    if not requests:
        raise InvalidParameter("requests", "request at least one plane")
    plans = [resolve_reconstruction(scan, method, plane=plane, **options) for plane in requests]
    if all(axis == "z" for axis, _ in requests):
        images = []
        for plan, (axis, index) in zip(plans, requests, strict=True):
            value = _axial_plane(plan, axis, index)
            if value is None:
                break
            images.append(value)
        else:
            return images
    volume = reconstruct_resolved(resolve_reconstruction(scan, method, **options)).volume
    return [np.asarray(_cut(volume, axis, index)) for axis, index in requests]


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
    plan = resolve_reconstruction(scan, method, plane=(axis, index), **options)
    return resolved_slice(plan, axis, index)


def center_sweep(
    scan: Scan,
    axis: Axis,
    index: int,
    centers: Sequence[float],
    *,
    method: Method = "auto",
    **options: Any,
) -> Any:
    """Reconstruct ordered planes at signed horizontal offsets in input detector pixels.

    Resolve the requested region once on the input scan. Every offset uses that fixed
    grid, support and plane index. A full reconstruction oracle shifts the detector and
    passes this resolved grid explicitly at each center. Repeated offsets retain their order.

    Parameters
    ----------
    scan : Scan
        Input projections and calibrated geometry.
    axis : {"z", "y", "x"}
        Dimension held fixed in the returned plane.
    index : int
        Zero-based plane index on the resolved grid.
    centers : sequence of float
        Signed detector offsets relative to the central column, in input pixels.
    method : str, optional
        Public reconstruction method.
    **options : dict
        Keyword arguments to the reconstruction resolver, including grid, region and solver options.

    Returns
    -------
    numpy.ndarray or jax.Array
        Float32 attenuation with a leading center coordinate, in the supplied order.
    """
    plans = resolve_sweep(scan, axis, index, centers, method=method, **options)
    return resolved_sweep(plans, axis, index)


def resolve_sweep(
    scan: Scan,
    axis: Axis,
    index: int,
    centers: Sequence[float],
    *,
    method: Method = "auto",
    grid: Grid | None = None,
    region: RegionName | RegionEstimate | None = None,
    **options: Any,
) -> tuple[ResolvedReconstruction, ...]:
    """Resolve support on the input and validate the actual acquisition for each candidate."""
    if len(centers) == 0 or not np.isfinite(centers).all():
        raise InvalidParameter(
            "centers", "centers must be a nonempty sequence of finite detector offsets"
        )
    fixed_grid, cylinder, info = resolve_grid(scan, grid, region)
    plans = []
    for i, center in enumerate(centers):
        progress.check_canceled()
        shift = float(center) * scan.detector.du - scan.detector.center[0]
        shifted = replace(scan, geometry=shifted_center(scan.geometry, shift))
        try:
            plan = resolve_reconstruction(
                shifted, method, grid=fixed_grid, plane=(axis, index), **options
            )
        except AngularCoverageError as err:
            raise InvalidParameter(f"centers[{i}]", str(err)) from err
        plans.append(replace(plan, cylinder=cylinder, info={**plan.info, **info}))
    return tuple(plans)


def sweep_planes(plans: Sequence[ResolvedReconstruction], axis: Axis, index: int) -> Iterator[Any]:
    """Each candidate's plane in order, one candidate in memory at a time."""
    for plan in plans:
        progress.check_canceled()
        yield resolved_slice(plan, axis, index)


def resolved_sweep(plans: Sequence[ResolvedReconstruction], axis: Axis, index: int) -> Any:
    """Execute the validated candidates in order and stack their planes."""
    import jax
    import jax.numpy as jnp

    images = list(sweep_planes(plans, axis, index))
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
    """The input rows, the corrected binned rows and the views kept, for a row band.

    Parameters
    ----------
    before : numpy.ndarray
        The input across the requested complete blocks: counts of raw frames, or the
        line integrals of a scan.
    after : numpy.ndarray
        Line integrals on the selected binned rows.
    views : numpy.ndarray
        Indices of retained views in the input acquisition.
    detector : Detector
        Full binned detector used to register the output rows.
    domain : {"counts", "line_integrals"}, optional
        What ``before`` holds.
    """

    before: np.ndarray
    after: np.ndarray
    views: np.ndarray
    detector: Detector
    domain: Literal["counts", "line_integrals"] = "counts"


def correction_preview(
    source: Frames | Scan,
    rows: tuple[int, int],
    *steps: Step,
    binning: int = 1,
    epsilon: float = 1e-6,
    views_per_batch: int | None = None,
) -> CorrectionBand:
    """Correct complete native blocks for a half-open interval in the binned output.

    The rows equal the same rows of a full correction followed by binning: rows
    the band does not hold are read when a step needs them (Paganin's propagation,
    view rejection's statistics).

    Parameters
    ----------
    source : Frames or Scan
        Raw acquisition with calibration and exposure metadata, or a scan of line
        integrals for the steps that act on them.
    rows : tuple of int
        First binned row and exclusive stop.
    *steps : Step
        Ordered corrections, as for the full acquisition.
    binning : int, optional
        Width and height of a native detector block.
    epsilon : float, optional
        Lower bound for flat and transmission values (raw frames).
    views_per_batch : int or None, optional
        Maximum number of views in a correction batch.

    Returns
    -------
    CorrectionBand
        The input rows, binned line integrals and original input view indices.
    """
    low, high = rows
    native = slice(low * binning, high * binning)
    result_detector = binned_geometry(source.geometry, binning).detector
    if isinstance(source, Scan):
        from tomojax.core.array_ops import check_binning, mean_detector_pixels
        from tomojax.corrections import correct_projections
        from tomojax.corrections.api import check_rows, check_scan_steps, preview_window

        projections = np.asarray(source.projections)
        _, height, width = projections.shape
        check_binning(binning, height, width)
        check_rows(rows, height, binning)
        check_scan_steps(steps)
        bound = bind_step_geometry(steps, source.geometry)  # as Scan.corrected binds them
        band, local = preview_window(bound, (height, width), low, high, binning)
        done = correct_projections(projections[:, band], local, views_per_batch=views_per_batch)
        first = band.start // binning
        binned = mean_detector_pixels(done.projections, binning)[:, low - first : high - first]
        before = np.array(projections[:, native], copy=True)
        return CorrectionBand(before, binned, done.kept, result_detector, "line_integrals")
    done = corrected_data(source, steps, epsilon, views_per_batch, binning, output_rows=rows)
    counts = source.counts
    before = (
        counts.window(native, slice(None))[:] if hasattr(counts, "window") else counts[:, native, :]
    )
    return CorrectionBand(np.array(before, copy=True), done.projections, done.kept, result_detector)
