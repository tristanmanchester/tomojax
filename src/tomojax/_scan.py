"""Scans: line-integral projections with the geometry that produced them.

:class:`Scan` and what changes one (views kept, a detector block, line-integral
corrections, binning), and its record in a dataset file. Files become scans in
:mod:`tomojax._loading`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax.core.array_ops import mean_detector_pixels
from tomojax.core.validation import view_indices
from tomojax.geometry import (
    ConeGeometry,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import jax

    from tomojax.corrections import Correction, Step
    from tomojax.geometry import ConeBeam, ConeSegments, Detector, Geometry, Grid, ScanGeometry
    from tomojax.io import ProjectionDataset

LOG = logging.getLogger(__name__)

# Geometry metadata keys a geometry object owns; other keys (provenance) carry over.
GEOMETRY_KEYS = frozenset(
    {
        "cone_beam",
        "cone_segments",
        "trajectory_lengths",
        "tilt_deg",
        "tilt_about",
        "axis_unit_lab",
        "detector_roll_deg",
    }
)


@dataclass(frozen=True)
class Scan:
    """Projections and the geometry that produced them.

    ``projections`` are ``(views, rows, columns)`` line integrals (absorption,
    not intensities); ``geometry`` describes every view, and carries the
    reconstruction ``grid`` and the ``detector``. A scan loaded from a file or
    returned by :func:`align` may carry per-view pose corrections, which every
    operation applies (see :attr:`poses`). ``corrections`` records how its
    projections were made from detector frames (see :mod:`tomojax.corrections`),
    and is saved with it.

    Parameters
    ----------
    projections : numpy.ndarray or jax.Array
        Line integrals, not intensities, shaped (views, rows, columns).
    geometry : ScanGeometry
        Per-view geometry carrying the reconstruction grid and detector.
    name : str, optional
        Sample name.
    corrections : tuple of Correction, optional
        Record of corrections already applied to the projections.
    source : ProjectionDataset or None, optional
        Source metadata retained for saving and provenance.

    Raises
    ------
    ValueError
        If projection dimensions disagree with the detector or view count.

    Examples
    --------
    >>> import numpy as np
    >>> import tomojax as tj
    >>> grid = tj.Grid(8, 8, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(8, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, [0.0, 90.0])
    >>> scan = tj.Scan(np.zeros((2, 1, 8), dtype=np.float32), geometry)
    """

    projections: np.ndarray | jax.Array
    geometry: ScanGeometry
    name: str = "sample"
    corrections: tuple[Correction, ...] = ()
    source: ProjectionDataset | None = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        views, rows, cols = self.projections.shape
        text = (
            f"Scan({self.name!r}: {views} views of {rows} x {cols}, {type(self.geometry).__name__}"
        )
        if self.corrections:
            text += ", corrections: " + ", ".join(str(c) for c in self.corrections)
        return text + ")"

    def corrected(self, *steps: Step, batch_views: int | None = None) -> Scan:
        """The scan with line-integral ``steps`` applied (see :mod:`tomojax.corrections`).

        Steps on counts or transmission need the frames: use :func:`load_frames`
        and :meth:`Frames.corrected`.

        Parameters
        ----------
        *steps : Step
            Line-integral correction steps, applied in order.
        batch_views : int or None, optional
            Positive views per batch; None chooses automatically.

        Returns
        -------
        Scan
            Corrected projections and geometry, with appended correction records.

        Raises
        ------
        ValueError
            If a step needs raw counts or transmission, or settings are invalid.
        """
        from tomojax.corrections import correct_projections

        done = correct_projections(self.projections, steps, batch_views=batch_views)
        geometry = (
            self.geometry
            if len(done.kept) == self.projections.shape[0]
            else views_of(self.geometry, done.kept)
        )
        return replace(
            self,
            projections=done.projections,
            geometry=geometry,
            corrections=(*self.corrections, *done.records),
        )

    def cropped(self, rows: slice, cols: slice) -> Scan:
        """Crop a contiguous detector window and move the detector to match.

        Parameters
        ----------
        rows : slice
            Nonempty contiguous detector row window.
        cols : slice
            Nonempty contiguous detector column window.

        Returns
        -------
        Scan
            Cropped projections with the detector centre moved to match.

        Raises
        ------
        ValueError
            If a window is empty or has a step other than one.
        """
        window = detector_window(rows, cols, self.detector)
        return replace(
            self,
            projections=np.asarray(self.projections)[:, rows, cols],
            geometry=rebuild(self.geometry, detector=window),
        )

    def selected(self, views: slice | Sequence[int] | np.ndarray) -> Scan:
        """Select views and their geometry in acquisition order.

        Parameters
        ----------
        views : slice, sequence of int or numpy.ndarray
            Forward slice, increasing indices or nonempty boolean view mask.

        Returns
        -------
        Scan
            Selected projections and their geometry, in acquisition order.

        Raises
        ------
        ValueError
            If indices are empty, unordered, repeated or out of bounds.
        """
        kept = view_indices(views, self.projections.shape[0])
        return replace(
            self,
            projections=np.asarray(self.projections)[kept],
            geometry=views_of(self.geometry, kept),
        )

    def __post_init__(self) -> None:
        check_shape("Scan: projections", self.projections.shape, self.geometry)

    @property
    def grid(self) -> Grid:
        """The reconstruction grid.

        Returns
        -------
        Grid
            Reconstruction grid carried by the geometry.
        """
        return self.geometry.grid

    @property
    def detector(self) -> Detector:
        """The detector.

        Returns
        -------
        Detector
            Detector carried by the geometry.
        """
        return self.geometry.detector

    @property
    def angles(self) -> np.ndarray:
        """Rotation angle of each view, in degrees.

        Returns
        -------
        numpy.ndarray
            One rotation angle per view, in degrees.
        """
        return np.asarray(self.geometry.angles, dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def poses(self) -> np.ndarray | None:
        """Per-view pose corrections, ``(views, 6)`` as alpha, beta, phi, dx, dz, dy.

        Rotations in radians, translations in the geometry's length unit, in
        the detector frame. None when the scan carries no corrections.

        Returns
        -------
        numpy.ndarray or None
            Pose corrections with shape (views, 6), or None when absent.
        """
        from tomojax._data.geometry_meta import detector_poses

        return detector_poses(self.geometry)

    def with_poses(self, poses: np.ndarray | None) -> Scan:
        """The scan with its per-view corrections replaced by ``poses``.

        The table has one row per view and six columns, in :attr:`poses`'s
        order, units and detector translation frame. It is relative to the
        calibrated geometry, not to the scan's current corrections: applying
        the same table twice does not double the motion. ``None`` removes
        corrections while retaining the calibrated axis, detector and grid.

        Use this to attach externally estimated or interpolated poses to a
        full scan after fitting selected views. Interpolate small corrections
        in acquisition order, not wrapped rotation angle; no interpolation or
        additional alignment is performed here.

        Parameters
        ----------
        poses : numpy.ndarray or None
            Replacement (views, 6) correction table; None removes all corrections.

        Returns
        -------
        Scan
            Scan with replacement poses and unchanged calibrated geometry.

        Raises
        ------
        ValueError
            If the pose table does not have shape (views, 6).
        """
        from tomojax._data.geometry_meta import AugmentedGeometry, nominal_geometry

        geometry = nominal_geometry(self.geometry)
        if poses is not None:
            table = np.asarray(poses, dtype=np.float32)
            shape = (self.projections.shape[0], 6)
            if table.shape != shape:
                raise ValueError(f"Scan.with_poses needs a {shape} table, not {table.shape}")
            geometry = AugmentedGeometry(geometry, table, translation_frame="detector")
        return replace(self, geometry=geometry)

    @classmethod
    def from_astra(
        cls,
        projections: np.ndarray,
        proj_geom: Mapping[str, Any],
        vol_geom: Mapping[str, Any],
        *,
        name: str = "sample",
    ) -> Scan:
        """A scan from ASTRA Toolbox data and geometries.

        ``projections`` are in ASTRA's ``(rows, views, columns)`` layout;
        ``proj_geom`` is a ``cone`` or ``cone_vec`` geometry and ``vol_geom`` a
        3-D volume geometry, as ``astra.create_proj_geom`` and
        ``astra.create_vol_geom`` make them. The scan's grid is the ASTRA
        volume, and its volumes are ASTRA's ``(z, y, x)`` arrays transposed to
        ``(x, y, z)``. The fitted circular orbit is the scan's geometry; any
        departure from it (tilts, wobble, per-view corrections) becomes
        :attr:`poses`. ASTRA turns the source and detector about the object
        and TomoJAX the object, so :attr:`angles` run the other way.

        Parameters
        ----------
        projections : numpy.ndarray
            ASTRA line integrals in (rows, views, columns) order.
        proj_geom : mapping
            ASTRA cone or cone_vec projection geometry.
        vol_geom : mapping
            ASTRA three-dimensional volume geometry.
        name : str, optional
            Sample name.

        Returns
        -------
        Scan
            Imported cone-beam scan, including departures from its fitted orbit as poses.

        Raises
        ------
        ValueError
            If ASTRA geometry or data shape is unsupported or inconsistent.
        """
        from tomojax._astra import from_astra
        from tomojax.geometry import ConeSegments

        data, segments = from_astra(projections, proj_geom, vol_geom)
        posed = [_with_corrections(geometry, poses) for geometry, poses in segments]
        if len(posed) == 1:
            return cls(data, posed[0], name=name)
        return cls(data, ConeSegments(tuple(posed)), name=name)

    @classmethod
    def combine(cls, scans: Sequence[Scan], *, name: str | None = None) -> Scan:
        """One scan of the same object from several cone-beam scans, view after view.

        For scans that differ in their source and detector arrangement (a
        multi-orbit scan, a tall sample scanned in stacked sections): each
        becomes a :class:`~tomojax.geometry.ConeSegments` segment. They share
        ``scans[0]``'s grid and need equal detector pixel counts.

        Parameters
        ----------
        scans : sequence of Scan
            Nonempty cone-beam scans with equal detector pixel counts.
        name : str or None, optional
            Combined sample name; None keeps the first scan's name.

        Returns
        -------
        Scan
            Concatenated views on the first scan's reconstruction grid.

        Raises
        ------
        ValueError
            If scans are empty or their geometry and detector counts are incompatible.
        """
        from tomojax.geometry import ConeSegments

        if not scans:
            raise ValueError("Scan.combine needs at least one scan")
        grid = scans[0].grid
        parts = []
        trajectories = []
        for scan in scans:
            geometry = rebuild(scan.geometry, grid=grid)
            parts.extend(_calibration_parts(geometry))
            layout = geometry
            while (base := getattr(layout, "base", None)) is not None:
                layout = base
            trajectories.extend(
                layout.trajectory_lengths
                if isinstance(layout, ConeSegments)
                else (len(layout.angles),)
            )
        projections = np.concatenate([np.asarray(s.projections) for s in scans])
        geometry = ConeSegments(tuple(parts), trajectory_lengths=tuple(trajectories))
        return cls(projections, geometry, name=name or scans[0].name)

    def binned(self, factor: int) -> Scan:
        """The scan with ``factor x factor`` detector pixels averaged into one.

        Use it when the detector samples finer than the reconstruction grid (a
        pixel's footprint at the rotation axis smaller than a voxel): projection
        cost falls by ``factor**2`` for little change in the volume. Pixels that
        do not fill a block at the detector's edges are dropped, and the
        detector centre moves to match.

        Parameters
        ----------
        factor : int
            Positive detector-block side length, in pixels.

        Returns
        -------
        Scan
            Block-averaged projections and detector geometry; the grid is unchanged.

        Raises
        ------
        ValueError
            If factor is less than one or leaves no complete detector block.
        """
        if factor < 1:
            raise ValueError(f"binned needs a factor of at least 1, not {factor}")
        if factor == 1:
            return self
        binned = mean_detector_pixels(np.asarray(self.projections, np.float32), factor)
        return replace(self, projections=binned, geometry=binned_geometry(self.geometry, factor))

    def to_astra(self) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        """Export projections and cone-beam geometry for ASTRA.

        The inverse of :meth:`from_astra`, for cone-beam scans; transpose a
        volume ``(x, y, z) -> (z, y, x)`` to use it with ASTRA.

        Returns
        -------
        projections : numpy.ndarray
            Line integrals in ASTRA (rows, views, columns) order.
        proj_geom : dict
            ASTRA cone_vec projection geometry.
        vol_geom : dict
            ASTRA three-dimensional volume geometry.

        Raises
        ------
        ValueError
            If the scan is not a supported cone-beam geometry.
        """
        from tomojax._astra import to_astra

        return to_astra(np.asarray(self.projections), self.geometry, self.grid, self.detector)


def detector_window(rows: slice, cols: slice, detector: Detector) -> Callable[[Detector], Detector]:
    """The map from a detector to its block of ``rows`` and ``cols``, its centre moved to match."""
    r0, r1, rstep = rows.indices(detector.nv)
    c0, c1, cstep = cols.indices(detector.nu)
    if rstep != 1 or cstep != 1 or r1 <= r0 or c1 <= c0:
        raise ValueError(
            f"a crop is a non-empty block of the {detector.nv} x {detector.nu} detector's "
            "rows and columns"
        )

    def crop(d: Detector) -> Detector:
        u, v = d.center
        moved = (u + (c0 + c1 - d.nu) / 2 * d.du, v + (r0 + r1 - d.nv) / 2 * d.dv)
        return replace(d, nu=c1 - c0, nv=r1 - r0, center=moved)

    return crop


def _calibration_parts(geometry: ScanGeometry) -> tuple[ScanGeometry, ...]:
    """Flatten calibration blocks once, slicing enclosing pose tables in acquisition order."""
    from tomojax._data.geometry_meta import AugmentedGeometry
    from tomojax.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        return geometry.segments
    base = getattr(geometry, "base", None)
    if base is None:
        return (geometry,)
    parts = _calibration_parts(base)
    if len(parts) == 1 and parts[0] is base:
        return (geometry,)
    if not is_dataclass(geometry) or isinstance(geometry, type):
        raise TypeError(f"cannot split a {type(geometry).__name__} geometry")
    wrapped, start = [], 0
    for part in parts:
        stop = start + len(part.angles)
        fields: dict[str, Any] = {"base": part}
        if isinstance(geometry, AugmentedGeometry):
            fields["align_params"] = np.asarray(geometry.align_params)[start:stop]
        wrapped.append(replace(geometry, **fields))
        start = stop
    return tuple(wrapped)


def views_of(geometry: ScanGeometry, kept: np.ndarray) -> ScanGeometry:
    """``geometry`` of the views ``kept`` (increasing indices), wrappers and segments kept."""
    from tomojax._data.geometry_meta import AugmentedGeometry
    from tomojax.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        parts, start = [], 0
        for segment in geometry.segments:
            n = _view_count(segment)
            mine = kept[(kept >= start) & (kept < start + n)] - start
            if len(mine):
                parts.append(views_of(segment, mine))
            start += n
        stops = np.searchsorted(kept, np.cumsum(geometry.trajectory_lengths))
        counts = np.diff(np.r_[0, stops])
        trajectories = tuple(int(n) for n in counts if n)
        if len(parts) == 1 and len(trajectories) == 1:
            return parts[0]
        return ConeSegments(tuple(parts), trajectory_lengths=trajectories)
    if isinstance(geometry, AugmentedGeometry):
        return replace(
            geometry,
            base=views_of(geometry.base, kept),
            align_params=np.asarray(geometry.align_params)[kept],
        )
    if not is_dataclass(geometry) or isinstance(geometry, type):
        raise TypeError(f"cannot select views of a {type(geometry).__name__} geometry")
    fields: dict[str, Any] = vars(geometry)  # a wrapper's base, or a geometry's own angles
    if "base" in fields:
        return replace(geometry, base=views_of(fields["base"], kept))
    return replace(geometry, angles=np.asarray(fields["angles"])[kept])


def _view_count(geometry: ScanGeometry) -> int:
    return len(geometry.angles)  # pyright: ignore[reportAttributeAccessIssue]


def check_shape(what: str, shape: tuple[int, ...], geometry: ScanGeometry) -> None:
    detector = geometry.detector
    views = _view_count(geometry)
    if tuple(int(s) for s in shape) != (views, detector.nv, detector.nu):
        raise ValueError(
            f"{what} are {tuple(shape)} but the geometry has {views} views of "
            f"{detector.nv} rows x {detector.nu} columns"
        )


def _with_corrections(geometry: ConeGeometry, poses: np.ndarray) -> ScanGeometry:
    """``geometry`` with per-view pose corrections, unless they move nothing.

    Corrections moving no voxel a hundredth of a detector pixel are rounding noise.
    """
    grid, detector, beam = geometry.grid, geometry.detector, geometry.beam
    reach = float(np.linalg.norm([grid.nx * grid.vx, grid.ny * grid.vy, grid.nz * grid.vz]))
    largest = np.max(np.abs(poses[:, :3])) * reach / 2 + np.max(np.abs(poses[:, 3:]))
    magnification = beam.source_to_detector / beam.source_to_axis
    if largest * magnification <= 0.01 * min(detector.du, detector.dv):
        return geometry
    from tomojax._data.geometry_meta import AugmentedGeometry

    return AugmentedGeometry(geometry, poses, translation_frame="detector")


def describe(geometry: Geometry) -> tuple[str, dict[str, object], Any, Geometry]:
    """Geometry type, metadata and per-view poses of ``geometry``, and its base geometry."""
    from tomojax._data.geometry_meta import (
        AugmentedGeometry,
        DetectorRollGeometry,
        detector_poses,
    )
    from tomojax.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        return _describe_segments(geometry)
    if isinstance(geometry, AugmentedGeometry):
        kind, meta, inner_poses, base = describe(geometry.base)
        # One update keeps its saved frame; nested updates need one table
        # relative to the truly nominal geometry, not an already-posed base.
        poses = (
            (np.asarray(geometry.align_params), str(geometry.translation_frame))
            if inner_poses is None
            else (detector_poses(geometry), "detector")
        )
        return kind, meta, poses, base
    if isinstance(geometry, DetectorRollGeometry):
        kind, meta, poses, base = describe(geometry.base)
        return kind, {**meta, "detector_roll_deg": float(geometry.detector_roll_deg)}, poses, base
    meta: dict[str, object] = {}
    if isinstance(geometry, ConeGeometry):
        kind, meta = "cone", geometry.geometry_metadata()
    elif isinstance(geometry, LaminographyGeometry):
        kind = "lamino"
        meta = {"tilt_deg": float(geometry.tilt_deg), "tilt_about": str(geometry.tilt_about)}
    elif isinstance(geometry, RotationAxisGeometry):
        kind = "lamino"
        meta["axis_unit_lab"] = [float(x) for x in geometry.axis_unit_lab]
    elif isinstance(geometry, ParallelGeometry):
        kind = "parallel"
    else:
        raise TypeError(f"cannot describe a {type(geometry).__name__} geometry")
    return kind, meta, None, geometry


def _describe_segments(
    geometry: ConeSegments,
) -> tuple[str, dict[str, object], Any, Geometry]:
    """:func:`describe` for segments: each segment's arrangement under ``cone_segments``."""
    entries = []
    for segment in geometry.segments:
        _, meta, _, _ = describe(segment)
        detector = segment.detector.to_dict()
        entries.append({"views": len(segment.angles), "detector": detector, **meta})
    meta: dict[str, object] = {
        "cone_beam": entries[0]["cone_beam"],
        "cone_segments": entries,
        "trajectory_lengths": list(geometry.trajectory_lengths),
    }
    from tomojax._data.geometry_meta import detector_poses

    table = detector_poses(geometry)  # one frame for every segment's poses
    return "cone", meta, None if table is None else (table, "detector"), geometry


def record_of(scan: Scan, *, grid: Grid | None = None) -> ProjectionDataset:
    """A dataset record for ``scan``: its geometry, keeping the source's provenance."""
    from tomojax.io import ProjectionDataset

    geometry_type, meta, poses, _ = describe(scan.geometry)
    source = scan.source
    kept = (
        {}
        if source is None
        else {
            k: v
            for k, v in source.geometry_metadata.items()
            if k not in GEOMETRY_KEYS and k != "corrections"
        }
    )
    fields: dict[str, Any] = {
        "projections": np.asarray(scan.projections),
        "angles": scan.angles.astype(np.float32),
        "volume": None,
        "detector": scan.detector,
        "grid": scan.grid if grid is None else grid,
        "geometry_type": geometry_type,
        "geometry_metadata": {**kept, **meta},
        # Angle offsets are already in the geometry's angles.
        "angle_offset_deg": None,
        "align_params": None if poses is None else poses[0],
        "align_gauge": None if poses is None else {"pose_translation_frame": poses[1]},
        "sample_name": scan.name,
    }
    if scan.corrections:
        fields["geometry_metadata"]["corrections"] = [c.to_dict() for c in scan.corrections]
    if source is None:
        return ProjectionDataset(**fields)
    # A Scan holds only sample-view line integrals, not the acquisition's raw
    # frame labels. Keep the other provenance without reusing a stale view axis.
    record = replace(source, **fields)
    metadata = record.copy_metadata()
    metadata.image_key = None
    return replace(record, _metadata=metadata)


def scan_from_record(record: ProjectionDataset, *, poses: bool) -> Scan:
    from tomojax.corrections import Correction

    saved = record.geometry_metadata.get("corrections") or ()
    return Scan(
        projections=record.projections,
        geometry=geometry_of(record, poses=poses),
        name=record.sample_name or "sample",
        corrections=tuple(Correction.from_dict(c) for c in saved),
        source=record,
    )


def geometry_of(record: ProjectionDataset, *, poses: bool) -> ScanGeometry:
    from tomojax.io import build_geometry_from_dataset_metadata

    _, _, geometry = build_geometry_from_dataset_metadata(record.geometry_inputs(), poses=poses)
    return geometry


def rebuild(
    geometry: ScanGeometry,
    *,
    grid: Grid | None = None,
    detector: Callable[[Detector], Detector] | None = None,
) -> ScanGeometry:
    """``geometry`` with a new grid and/or each detector mapped by ``detector``, wrappers kept."""
    from tomojax.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        segments = (rebuild(s, grid=grid, detector=detector) for s in geometry.segments)
        return replace(geometry, segments=tuple(segments))
    if not is_dataclass(geometry) or isinstance(geometry, type):
        raise TypeError(f"cannot rebuild a {type(geometry).__name__} geometry")
    inner = getattr(geometry, "base", None)
    if inner is not None:
        return replace(geometry, base=rebuild(inner, grid=grid, detector=detector))
    changes: dict[str, object] = {}
    if grid is not None:
        changes["grid"] = grid
    if detector is not None:
        mapped = detector(geometry.detector)
        if isinstance(geometry, ConeGeometry):
            mapped, beam = _windowed_cone_detector(geometry, mapped)
            changes["beam"] = beam
        changes["detector"] = mapped
    return replace(geometry, **changes)


def _windowed_cone_detector(geometry: ConeGeometry, mapped: Detector) -> tuple[Detector, ConeBeam]:
    """Move a cropped/binned cone detector along its own pixel axes, not lab x/z."""
    detector, beam = geometry.detector, geometry.beam
    if mapped.center == detector.center or not any(
        (beam.detector_roll_deg, beam.detector_pitch_deg, beam.detector_yaw_deg)
    ):
        return mapped, beam
    du = mapped.center[0] - detector.center[0]
    dv = mapped.center[1] - detector.center[1]
    _, u_dir, v_dir = beam.detector_frame(detector)
    shift = du * u_dir + dv * v_dir
    # Detector.center stores lab x/z; the distance stores lab y. Moving the
    # detector centre in its tilted plane leaves the source and object poses fixed.
    return (
        replace(
            mapped,
            center=(float(detector.center[0] + shift[0]), float(detector.center[1] + shift[2])),
        ),
        replace(beam, source_to_detector=float(beam.source_to_detector + shift[1])),
    )


def binned_geometry(geometry: ScanGeometry, factor: int) -> ScanGeometry:
    """The geometry of detector block means, retaining physical pixel positions."""
    return rebuild(geometry, detector=lambda d: _binned_detector(d, factor))


def _binned_detector(detector: Detector, factor: int) -> Detector:
    """``detector`` with ``factor x factor`` pixels averaged; partial edge blocks dropped."""
    from tomojax.geometry import Detector

    nu, nv = detector.nu // factor, detector.nv // factor
    # Dropped edge pixels move the binned detector's centre (see Scan.binned).
    cu = detector.center[0] + detector.du * (factor * nu - detector.nu) / 2
    cv = detector.center[1] + detector.dv * (factor * nv - detector.nv) / 2
    return Detector(nu, nv, detector.du * factor, detector.dv * factor, (cu, cv))


def with_grid(scan: Scan, grid: Grid) -> Scan:
    """``scan`` reconstructed on ``grid`` instead of its own."""
    if grid == scan.grid:
        return scan
    return replace(scan, geometry=rebuild(scan.geometry, grid=grid))


@dataclass(frozen=True)
class Reconstruction:
    """A reconstructed volume, the scan it came from and how it was made.

    ``volume`` is ``(nx, ny, nz)`` on :attr:`grid`, in the reciprocal of the
    geometry's length unit (a NumPy or JAX array; ``np.asarray`` gives NumPy).
    ``info`` records the method's settings.

    Parameters
    ----------
    volume : numpy.ndarray or jax.Array
        Volume in (x, y, z) order and reciprocal geometry-length units.
    scan : Scan
        Input scan, including the output grid and any inferred geometry.
    method : str
        Reconstruction method used.
    info : mapping
        Resolved method settings and solver records.
    """

    volume: np.ndarray | jax.Array
    scan: Scan
    method: str
    info: Mapping[str, object]

    @property
    def grid(self) -> Grid:
        """The grid the volume is on, the scan's.

        Returns
        -------
        Grid
            Reconstruction grid carried by the scan.
        """
        return self.scan.grid


@dataclass(frozen=True)
class Alignment:
    """The result of :func:`align`.

    ``volume`` is the reconstruction the alignment converged with. ``scan`` is
    the input scan with the estimated geometry and per-view poses applied:
    reconstruct it, align it again or save it. ``poses`` are its ``(views, 6)``
    poses (see :attr:`Scan.poses`), always a table, zero where nothing moved;
    ``info`` is the solver's record.

    Parameters
    ----------
    volume : numpy.ndarray or jax.Array
        Reconstruction used by the alignment, in (x, y, z) order.
    scan : Scan
        Scan with estimated geometry and per-view corrections applied.
    poses : numpy.ndarray
        Complete (views, 6) correction table, zero where nothing moved.
    info : mapping
        Solver record and resolved alignment settings.
    """

    volume: np.ndarray | jax.Array
    scan: Scan
    poses: np.ndarray
    info: Mapping[str, object]


__all__ = ["Alignment", "Reconstruction", "Scan"]
