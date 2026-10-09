"""Scans and raw frames: what TomoJAX reconstructs and aligns, and how files become them.

A :class:`Scan` is line-integral projections with the geometry that produced
them. :func:`load` reads one from a file, correcting raw detector frames on the
way (the correction is recorded in :attr:`Scan.corrections`); :func:`load_frames`
reads the frames themselves, as :class:`Frames`, for other corrections.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from tomojax.geometry import (
    ConeGeometry,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from os import PathLike

    import jax

    from tomojax.geometry import ConeSegments, Detector, Geometry, Grid, ScanGeometry
    from tomojax.io import ProjectionDataset

type Method = Literal["fbp", "cgls", "fista", "spdhg"]

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
    operation applies (see :attr:`poses`).
    """

    projections: np.ndarray | jax.Array
    geometry: ScanGeometry
    name: str = "sample"
    source: ProjectionDataset | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        shape = tuple(self.projections.shape)
        detector = self.geometry.detector
        views = len(self.geometry.angles)  # pyright: ignore[reportAttributeAccessIssue]
        if shape != (views, detector.nv, detector.nu):
            raise ValueError(
                f"Scan: projections are {shape} but the geometry has {views} views of "
                f"{detector.nv} rows x {detector.nu} columns"
            )

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
            geometry = _rebuild(scan.geometry, grid=grid)
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
        geometry = _rebuild(self.geometry, detector=lambda d: _binned_detector(d, factor))
        return replace(self, projections=blocks.mean(axis=(2, 4)), geometry=geometry)

    def to_astra(self) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        """ASTRA ``(rows, views, columns)`` projections, ``cone_vec`` and volume geometries.

        The inverse of :meth:`from_astra`, for cone-beam scans; transpose a
        volume ``(x, y, z) -> (z, y, x)`` to use it with ASTRA.
        """
        from tomojax._astra import to_astra

        return to_astra(np.asarray(self.projections), self.geometry, self.grid, self.detector)


def load(path: str | PathLike[str], *, poses: bool = True) -> Scan:
    """Load a scan from a TomoJAX dataset (``.nxs``, ``.h5``, ``.npz``) or a Nikon ``.xtekct``.

    The scan carries the per-view corrections saved with it (by :func:`align`
    or ``tomojax align``) unless ``poses`` is False. TIFF stacks need their geometry
    stated: import them with ``tomojax import`` or :func:`tomojax.io.load_tiff_stack`.
    """
    from tomojax.io import load_dataset, load_nikon_xtekct

    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no such file: {file}")
    record = load_nikon_xtekct(file) if file.suffix.lower() == ".xtekct" else load_dataset(file)
    return scan_from_record(record, poses=poses)


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
        else {k: v for k, v in source.geometry_metadata.items() if k not in GEOMETRY_KEYS}
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
    if source is None:
        return ProjectionDataset(**fields)
    return replace(source, **fields)


def scan_from_record(record: ProjectionDataset, *, poses: bool) -> Scan:
    from tomojax.io import build_geometry_from_dataset_metadata

    _, _, geometry = build_geometry_from_dataset_metadata(record.geometry_inputs(), poses=poses)
    return Scan(
        projections=record.projections,
        geometry=geometry,
        name=record.sample_name or "sample",
        source=record,
    )


def _rebuild(
    geometry: ScanGeometry,
    *,
    grid: Grid | None = None,
    detector: Callable[[Detector], Detector] | None = None,
) -> ScanGeometry:
    """``geometry`` with a new grid and/or each detector mapped by ``detector``, wrappers kept."""
    from tomojax.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        segments = (_rebuild(s, grid=grid, detector=detector) for s in geometry.segments)
        return ConeSegments(tuple(segments))
    if not is_dataclass(geometry) or isinstance(geometry, type):
        raise TypeError(f"cannot rebuild a {type(geometry).__name__} geometry")
    inner = getattr(geometry, "base", None)
    if inner is not None:
        return replace(geometry, base=_rebuild(inner, grid=grid, detector=detector))
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
        return replace(scan, geometry=_rebuild(scan.geometry, grid=grid))
    record = record_of(scan, grid=grid)
    return replace(scan_from_record(record, poses=True), source=scan.source)


__all__ = ["Scan", "load"]
