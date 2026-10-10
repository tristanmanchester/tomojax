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
    from tomojax.geometry import ConeSegments, Detector, Geometry, Grid, ScanGeometry
    from tomojax.io import ProjectionDataset

LOG = logging.getLogger(__name__)

# Geometry metadata keys a geometry object owns; other keys (provenance) carry over.
GEOMETRY_KEYS = frozenset(
    {
        "cone_beam",
        "cone_segments",
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
        """The scan of this block of detector ``rows`` and ``cols``, the detector moved to match."""
        window = detector_window(rows, cols, self.detector)
        return replace(
            self,
            projections=np.asarray(self.projections)[:, rows, cols],
            geometry=rebuild(self.geometry, detector=window),
        )

    def selected(self, views: slice | Sequence[int] | np.ndarray) -> Scan:
        """These ``views`` and geometry: a forward slice, increasing indices or a nonempty mask."""
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
        """The reconstruction grid."""
        return self.geometry.grid

    @property
    def detector(self) -> Detector:
        """The detector."""
        return self.geometry.detector

    @property
    def angles(self) -> np.ndarray:
        """Rotation angle of each view, in degrees."""
        return np.asarray(self.geometry.angles, dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def poses(self) -> np.ndarray | None:
        """Per-view pose corrections, ``(views, 6)`` as alpha, beta, phi, dx, dz, dy.

        Rotations in radians, translations in the geometry's length unit, in
        the detector frame. None when the scan carries no corrections.
        """
        from tomojax._data.geometry_meta import detector_poses

        return detector_poses(self.geometry)

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
        """
        from tomojax.geometry import ConeSegments

        if not scans:
            raise ValueError("Scan.combine needs at least one scan")
        grid = scans[0].grid
        parts = []
        for scan in scans:
            geometry = rebuild(scan.geometry, grid=grid)
            parts.extend(geometry.segments if isinstance(geometry, ConeSegments) else (geometry,))
        projections = np.concatenate([np.asarray(s.projections) for s in scans])
        return cls(projections, ConeSegments(tuple(parts)), name=name or scans[0].name)

    def binned(self, factor: int) -> Scan:
        """The scan with ``factor x factor`` detector pixels averaged into one.

        Use it when the detector samples finer than the reconstruction grid (a
        pixel's footprint at the rotation axis smaller than a voxel): projection
        cost falls by ``factor**2`` for little change in the volume. Pixels that
        do not fill a block at the detector's edges are dropped, and the
        detector centre moves to match.
        """
        if factor < 1:
            raise ValueError(f"binned needs a factor of at least 1, not {factor}")
        if factor == 1:
            return self
        data = np.asarray(self.projections, np.float32)
        views, nv, nu = data.shape
        rows, cols = nv // factor, nu // factor
        blocks = data[:, : rows * factor, : cols * factor].reshape(
            views, rows, factor, cols, factor
        )
        geometry = rebuild(self.geometry, detector=lambda d: _binned_detector(d, factor))
        return replace(self, projections=blocks.mean(axis=(2, 4)), geometry=geometry)

    def to_astra(self) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        """ASTRA ``(rows, views, columns)`` projections, ``cone_vec`` and volume geometries.

        The inverse of :meth:`from_astra`, for cone-beam scans; transpose a
        volume ``(x, y, z) -> (z, y, x)`` to use it with ASTRA.
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


def view_indices(views: slice | Sequence[int] | np.ndarray, count: int) -> np.ndarray:
    """``views`` of ``count`` as nonempty, increasing integer indices."""
    if isinstance(views, slice):
        if views.step is not None and views.step <= 0:
            raise ValueError("views must follow a forward slice with a positive step")
        chosen = np.arange(count)[views]
    else:
        chosen = np.asarray(views)
    if chosen.dtype == bool:
        if chosen.shape != (count,):
            raise ValueError(f"a mask of views needs {count} entries, not {chosen.shape}")
        chosen = np.flatnonzero(chosen)
    if not chosen.size:
        raise ValueError("views must select at least one view")
    if chosen.ndim != 1 or chosen.dtype.kind not in "iu":
        raise ValueError("views must be a one-dimensional array of integer indices")
    if chosen.min() < 0 or chosen.max() >= count:
        raise ValueError(f"views must be in 0..{count - 1}")
    if np.any(chosen[1:] <= chosen[:-1]):
        raise ValueError("views must be increasing indices, each once")
    return chosen.astype(np.int64)


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
        return parts[0] if len(parts) == 1 else ConeSegments(tuple(parts))
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


def _with_corrections(geometry: ScanGeometry, poses: np.ndarray) -> ScanGeometry:
    """``geometry`` with per-view pose corrections, unless they move nothing.

    Corrections moving no voxel a hundredth of a pixel are rounding noise.
    """
    grid, detector = geometry.grid, geometry.detector
    reach = float(np.linalg.norm([grid.nx * grid.vx, grid.ny * grid.vy, grid.nz * grid.vz]))
    largest = np.max(np.abs(poses[:, :3])) * reach / 2 + np.max(np.abs(poses[:, 3:]))
    if largest <= 0.01 * detector.du:
        return geometry
    from tomojax._data.geometry_meta import AugmentedGeometry

    return AugmentedGeometry(geometry, poses, translation_frame="detector")


def describe(geometry: Geometry) -> tuple[str, dict[str, object], Any, Geometry]:
    """Geometry type, metadata and per-view poses of ``geometry``, and its base geometry."""
    from tomojax.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        return _describe_segments(geometry)
    poses = None
    meta: dict[str, object] = {}
    base = geometry
    if hasattr(base, "align_params") and hasattr(base, "base"):
        poses = (np.asarray(base.align_params), str(base.translation_frame))  # pyright: ignore[reportAttributeAccessIssue]
        base = base.base  # pyright: ignore[reportAttributeAccessIssue]
    if hasattr(base, "detector_roll_deg") and hasattr(base, "base"):
        meta["detector_roll_deg"] = float(base.detector_roll_deg)  # pyright: ignore[reportAttributeAccessIssue]
        base = base.base  # pyright: ignore[reportAttributeAccessIssue]
    if isinstance(base, ConeGeometry):
        return "cone", {**meta, **base.geometry_metadata()}, poses, base
    if isinstance(base, LaminographyGeometry):
        meta |= {"tilt_deg": float(base.tilt_deg), "tilt_about": str(base.tilt_about)}
        return "lamino", meta, poses, base
    if isinstance(base, RotationAxisGeometry):
        meta["axis_unit_lab"] = [float(x) for x in base.axis_unit_lab]
        return "lamino", meta, poses, base
    if isinstance(base, ParallelGeometry):
        return "parallel", meta, poses, base
    raise TypeError(f"cannot describe a {type(base).__name__} geometry")


def _describe_segments(
    geometry: ConeSegments,
) -> tuple[str, dict[str, object], Any, Geometry]:
    """:func:`describe` for segments: each segment's arrangement under ``cone_segments``."""
    entries = []
    for segment in geometry.segments:
        _, meta, _, _ = describe(segment)
        detector = segment.detector.to_dict()
        entries.append({"views": len(segment.angles), "detector": detector, **meta})
    meta = {"cone_beam": entries[0]["cone_beam"], "cone_segments": entries}
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
    return replace(source, **fields)


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
        return ConeSegments(tuple(segments))
    if not is_dataclass(geometry) or isinstance(geometry, type):
        raise TypeError(f"cannot rebuild a {type(geometry).__name__} geometry")
    inner = getattr(geometry, "base", None)
    if inner is not None:
        return replace(geometry, base=rebuild(inner, grid=grid, detector=detector))
    changes: dict[str, object] = {}
    if grid is not None:
        changes["grid"] = grid
    if detector is not None:
        changes["detector"] = detector(geometry.detector)
    return replace(geometry, **changes)


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
    from tomojax.geometry import ConeSegments

    if grid == scan.grid:
        return scan
    if isinstance(scan.geometry, ConeSegments):
        return replace(scan, geometry=rebuild(scan.geometry, grid=grid))
    record = record_of(scan, grid=grid)
    return replace(scan_from_record(record, poses=True), source=scan.source)


__all__ = ["Scan"]
