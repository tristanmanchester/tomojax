"""Registered slices, centre sweeps, detector bands and region overlays for interactive clients."""

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
from tomojax._scan import Scan, binned_geometry, shifted_center
from tomojax._workflow import reconstruct, reconstruct_resolved
from tomojax.core.registration import planes, recorded_length_unit
from tomojax.errors import InvalidParameter
from tomojax.geometry import grid_volume_origin
from tomojax.recon import FBPConfig
from tomojax.recon.api import AngularCoverageError, AxialRoute, axial_fbp_plane

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tomojax._loading import Frames
    from tomojax._parameters import Method, RegionName
    from tomojax.corrections import Step
    from tomojax.geometry import Detector, Grid


__all__ = [
    "CorrectionBand",
    "PlanePlan",
    "RegionOverlay",
    "SweepPlan",
    "centre_sweep",
    "correction_preview",
    "plan_planes",
    "planes_of",
    "reconstruct_slice",
    "region_preview",
    "resolve_sweep",
    "sweep_planes",
]


type Axis = Literal["z", "y", "x"]


def resolved_slice(plan: ResolvedReconstruction, axis: Axis, index: int) -> Any:
    """Select the plane after resolving options and any volume support on the full grid."""
    dim = plane_dimension(plan.scan.grid, axis, index)
    if axis == "z" and plan.route is not None and plan.route.kind != "full":
        scan = plan.scan
        config = plan.config
        assert isinstance(config, FBPConfig)
        value = axial_fbp_plane(
            scan.geometry, scan.grid, scan.detector, scan.projections, config, index, plan.route
        )
        if value is not None:
            if plan.cylinder is not None:
                from tomojax._field_of_view import cylinder_support

                mask = cylinder_support(replace(scan.grid, nz=1), plan.cylinder)[0]
                value = np.where(mask, value, 0)
            return value
    result = reconstruct_resolved(plan)
    key: list[slice | int] = [slice(None)] * 3
    key[dim] = index
    return result.volume[tuple(key)].copy()


@dataclass(frozen=True)
class PlanePlan:
    """How requested planes are made: each on its own, or all from one full solve.

    Parameters
    ----------
    requests : tuple
        The distinct ``(axis, index)`` planes, in the order first requested.
    plans : tuple
        One resolved plane each, when every plane has its own route; else empty.
    full : ResolvedReconstruction or None
        The one full reconstruction every plane is cut from, when any plane needs it.
    """

    requests: tuple[tuple[Axis, int], ...]
    plans: tuple[ResolvedReconstruction, ...]
    full: ResolvedReconstruction | None


def plan_planes(
    scan: Scan, requests: Sequence[tuple[Axis, int]], *, method: Method = "auto", **options: Any
) -> PlanePlan:
    """Resolve several planes, sharing one full solve when any of them needs one.

    Every index is checked before anything runs. A plane only an axial FBP route can
    make on its own; any other plane makes the whole set a single full reconstruction,
    validated on the whole volume.

    Parameters
    ----------
    scan : Scan
        Input projections and calibrated geometry.
    requests : sequence of (axis, index)
        Planes to make; repeats are made once.
    method : str, optional
        Public reconstruction method.
    **options : dict
        Keyword arguments to the reconstruction resolver.

    Returns
    -------
    PlanePlan
        The planes' routes, or the single full reconstruction.
    """
    distinct: tuple[tuple[Axis, int], ...] = tuple(
        dict.fromkeys((axis, int(index)) for axis, index in requests)
    )
    if not distinct:
        raise InvalidParameter("requests", "request at least one plane")
    plans = []
    for i, (axis, index) in enumerate(distinct):  # each index is checked before any solve
        try:
            plans.append(resolve_reconstruction(scan, method, plane=(axis, index), **options))
        except InvalidParameter as err:
            raise InvalidParameter(f"requests[{i}].{err.path}", str(err)) from err
    if all(plan.route is not None and plan.route.kind != "full" for plan in plans):
        return PlanePlan(distinct, tuple(plans), None)
    # A full solve makes every plane, so its validation covers the whole volume.
    return PlanePlan(distinct, (), resolve_reconstruction(scan, method, **options))


def planes_of(plan: PlanePlan) -> list[Any]:
    """The planes of ``plan`` in its request order, solving the full volume at most once.

    Parameters
    ----------
    plan : PlanePlan
        Planes resolved by :func:`plan_planes`.

    Returns
    -------
    list of array
        Float32 attenuation planes, one per distinct request.
    """
    if plan.full is None:
        return [
            resolved_slice(resolved, axis, index)
            for resolved, (axis, index) in zip(plan.plans, plan.requests, strict=True)
        ]
    volume = reconstruct_resolved(plan.full).volume
    images = []
    for axis, index in plan.requests:
        key: list[slice | int] = [slice(None)] * 3
        key["zyx".index(axis)] = index
        images.append(np.asarray(volume[tuple(key)]).copy())
    return images


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
    return resolved_sweep(resolve_sweep(scan, axis, index, centres, method=method, **options))


@dataclass(frozen=True)
class SweepPlan:
    """A validated centre sweep: one template plan and each candidate's route.

    A candidate's shifted geometry is rebuilt only when it runs, so a long sweep over a
    segmented scan holds one candidate's geometry at a time.

    Parameters
    ----------
    template : ResolvedReconstruction
        The first candidate's plan; the others differ only in detector centre and route.
    axis, index : str, int
        The swept plane.
    centres : tuple of float
        Detector centre positions in pixels from the detector middle, in order.
    routes : tuple
        Each candidate's axial route, or None.
    """

    template: ResolvedReconstruction
    axis: Axis
    index: int
    centres: tuple[float, ...]
    routes: tuple[AxialRoute | None, ...]

    @property
    def full_solves(self) -> int:
        """How many candidates need a full reconstruction rather than one plane.

        Returns
        -------
        int
            Candidates without their own axial route.
        """
        return sum(route is None or route.kind == "full" for route in self.routes)

    def candidate(self, i: int) -> ResolvedReconstruction:
        """Candidate ``i``'s plan, its geometry shifted from the template's.

        Parameters
        ----------
        i : int
            Candidate ordinal.

        Returns
        -------
        ResolvedReconstruction
            The candidate's plan with its own detector centre and route.
        """
        template = self.template
        du = (self.centres[i] - self.centres[0]) * template.scan.detector.du
        scan = replace(template.scan, geometry=shifted_center(template.scan.geometry, du))
        return replace(template, scan=scan, route=self.routes[i])


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
) -> SweepPlan:
    """Resolve support on the input and validate the actual acquisition for each candidate.

    Only each candidate's route is kept; its shifted geometry is rebuilt when it runs.

    Parameters
    ----------
    scan : Scan
        Input projections and calibrated geometry.
    axis : {"z", "y", "x"}
        Dimension held fixed in the swept plane.
    index : int
        Zero-based plane index on the resolved grid.
    centres : sequence of float
        Detector centre positions in pixels from the detector middle.
    method : str, optional
        Public reconstruction method.
    grid : Grid or None, optional
        Output grid shared by every candidate.
    region : str, RegionEstimate or None, optional
        Region resolved once on the input scan.
    **options : dict
        Further keyword arguments to the reconstruction resolver.

    Returns
    -------
    SweepPlan
        The validated sweep.

    Raises
    ------
    InvalidParameter
        If the centres are empty or not finite, or a candidate fails validation.
    """
    if len(centres) == 0 or not np.isfinite(centres).all():
        raise InvalidParameter(
            "centres", "centres must be a nonempty sequence of finite detector offsets"
        )
    fixed_grid, cylinder, info = resolve_grid(scan, grid, region)
    template = None
    routes = []
    for i, centre in enumerate(centres):
        progress.check_cancelled()
        shift = float(centre) * scan.detector.du - scan.detector.center[0]
        shifted = replace(scan, geometry=shifted_center(scan.geometry, shift))
        try:
            plan = resolve_reconstruction(
                shifted, method, grid=fixed_grid, plane=(axis, index), **options
            )
        except AngularCoverageError as err:
            raise InvalidParameter(f"centres[{i}]", str(err)) from err
        routes.append(plan.route)
        if template is None:
            template = replace(plan, cylinder=cylinder, info={**plan.info, **info})
    assert template is not None
    return SweepPlan(template, axis, index, tuple(float(c) for c in centres), tuple(routes))


def sweep_planes(plan: SweepPlan) -> Iterator[Any]:
    """Each candidate's plane in order, one candidate in memory at a time.

    Parameters
    ----------
    plan : SweepPlan
        A sweep validated by :func:`resolve_sweep`.

    Yields
    ------
    array
        Float32 attenuation of the swept plane at each candidate centre.
    """
    for i in range(len(plan.centres)):
        progress.check_cancelled()
        yield resolved_slice(plan.candidate(i), plan.axis, plan.index)


def resolved_sweep(plan: SweepPlan) -> Any:
    """Execute the validated candidates in order and stack their planes."""
    import jax
    import jax.numpy as jnp

    images = list(sweep_planes(plan))
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
