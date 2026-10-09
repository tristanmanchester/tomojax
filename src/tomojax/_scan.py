"""Scans and raw frames: what TomoJAX reconstructs and aligns, and how files become them.

A :class:`Scan` is line-integral projections with the geometry that produced
them. :func:`load` reads one from a file, correcting raw detector frames on the
way (the correction is recorded in :attr:`Scan.corrections`); :func:`load_frames`
reads the frames themselves, as :class:`Frames`, for other corrections.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
import logging
import os
from pathlib import Path
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
    from os import PathLike

    import jax

    from tomojax.corrections import Correction, Step
    from tomojax.geometry import ConeSegments, Detector, Geometry, Grid, ScanGeometry
    from tomojax.io import ProjectionDataset
    from tomojax.io.api import LoadedNXTomo

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
            else _views_of(self.geometry, done.kept)
        )
        return replace(
            self,
            projections=done.projections,
            geometry=geometry,
            corrections=(*self.corrections, *done.records),
        )

    def selected(self, views: slice | Sequence[int] | np.ndarray) -> Scan:
        """The scan of these ``views`` (a slice, increasing indices or a mask) and geometry."""
        kept = _view_indices(views, self.projections.shape[0])
        return replace(
            self,
            projections=np.asarray(self.projections)[kept],
            geometry=_views_of(self.geometry, kept),
        )

    def __post_init__(self) -> None:
        _check_shape("Scan: projections", self.projections.shape, self.geometry)

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
    """Load a scan of line integrals from a dataset (``.nxs``, ``.h5``, ``.npz``) or ``.xtekct``.

    A file of raw detector frames (an NXtomo ``image_key`` marking flats or
    darks, or a Nikon scan) is corrected with the standard chain, as
    ``load_frames(path).corrected()``: dark subtraction, flat division and
    ``-log``; :attr:`Scan.corrections` records it. For other corrections, load
    the frames with :func:`load_frames`. The scan carries the per-view
    corrections saved with it (by :func:`align` or ``tomojax align``) unless
    ``poses`` is False.
    """
    from tomojax.io import load_dataset
    from tomojax.io.api import holds_flats_or_darks

    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no such file: {file}")
    suffix = file.suffix.lower()
    if file.is_dir() or suffix in {".tif", ".tiff"}:
        raise ValueError(
            f"{file} is a TIFF stack, which holds no geometry: load it with "
            "tomojax.load_frames(path, angles=..., detector=...)"
        )
    if suffix == ".xtekct":
        return _corrected_on_load(load_frames(file), file)
    if suffix in {".nxs", ".h5", ".hdf5"} and holds_flats_or_darks(str(file)):
        return _corrected_on_load(load_frames(file), file)
    record = load_dataset(file)
    if np.issubdtype(np.asarray(record.projections).dtype, np.integer):
        raise ValueError(
            f"{file} holds integer detector counts and no flat frames (image_key 1): "
            "load them with tomojax.load_frames(path, flats=... or white_level=...)"
        )
    return scan_from_record(record, poses=poses)


def _corrected_on_load(frames: Frames, file: Path) -> Scan:
    scan = frames.corrected()
    LOG.info("%s: corrected detector frames to line integrals: %s", file.name,
             ", ".join(str(c) for c in scan.corrections))  # fmt: skip
    return scan


@dataclass(frozen=True, repr=False)
class Frames:
    """Detector frames as recorded, with the geometry of each sample view.

    ``counts`` are the sample frames, ``(views, rows, columns)``, in memory, a
    memmap or read lazily from the file; ``flats`` the frames of the beam
    alone, with ``flat_positions`` the number of views recorded before each
    (flats taken before and after a scan are interpolated between); ``darks``
    the frames with the beam off. Scanners that calibrate their flat field
    record a ``white_level`` instead. ``view_positions`` places each view
    among the flats, in the units of ``flat_positions`` (by default view ``i``
    at ``i + 1/2``; :meth:`selected` keeps it). :meth:`corrected` makes the
    :class:`Scan` of line integrals. Made by :func:`load_frames`.
    """

    counts: Any
    geometry: ScanGeometry
    flats: np.ndarray | None = None
    darks: np.ndarray | None = None
    flat_positions: np.ndarray | None = None
    white_level: float | None = None
    view_positions: np.ndarray | None = None
    name: str = "sample"
    source: ProjectionDataset | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _check_shape("Frames: counts", self.counts.shape, self.geometry)
        if self.view_positions is not None and np.shape(self.view_positions) != (self.views,):
            raise ValueError(f"view_positions needs one entry for each of the {self.views} views")

    def __repr__(self) -> str:
        views, rows, cols = self.counts.shape
        parts = [f"{views} views of {rows} x {cols} {np.dtype(self.counts.dtype).name}"]
        if self.flats is not None:
            sets = 1 if self.flat_positions is None else len(np.unique(self.flat_positions))
            parts.append(f"{_count(len(self.flats), 'flat')} in {_count(sets, 'set')}")
        if self.white_level is not None:
            parts.append(f"white level {self.white_level:g}")
        parts.append(_count(0 if self.darks is None else len(self.darks), "dark"))
        return f"Frames({self.name!r}: {', '.join(parts)}, {type(self.geometry).__name__})"

    @property
    def views(self) -> int:
        """The number of sample views."""
        return int(self.counts.shape[0])

    def corrected(
        self, *steps: Step, epsilon: float = 1e-6, batch_views: int | None = None
    ) -> Scan:
        """The scan of line integrals ``-log((I - D) / (F - D))``, with ``steps`` applied.

        Steps run by the data they act on: counts steps first, then the flat
        and dark fields, transmission steps, the log, and line-integral steps,
        each in the order given (see :mod:`tomojax.corrections`). ``epsilon``
        bounds the flat field and transmission below. The views are corrected
        on the device in batches of ``batch_views`` (by default, 256 MiB).
        """
        from tomojax.corrections import correct_frames

        done = correct_frames(
            self.counts,
            flats=self.flats,
            flat_positions=self.flat_positions,
            view_positions=self.view_positions,
            darks=self.darks,
            white_level=self.white_level,
            steps=steps,
            epsilon=epsilon,
            batch_views=batch_views,
        )
        geometry = (
            self.geometry if len(done.kept) == self.views else _views_of(self.geometry, done.kept)
        )
        return Scan(done.projections, geometry, self.name, done.records, self.source)

    def selected(self, views: slice | Sequence[int] | np.ndarray) -> Frames:
        """These sample ``views`` only (a slice, increasing indices or a mask), with their geometry.

        Each view keeps its place among the flats (:attr:`view_positions`), so
        its flat is interpolated as before.
        """
        kept = _view_indices(views, self.views)
        at = np.arange(self.views) + 0.5 if self.view_positions is None else self.view_positions
        counts = self.counts
        counts = (
            counts.frames_at(kept) if hasattr(counts, "frames_at") else np.asarray(counts)[kept]
        )
        return replace(
            self,
            counts=counts,
            geometry=_views_of(self.geometry, kept),
            view_positions=np.asarray(at, np.float64)[kept],
        )


def load_frames(
    path: str | PathLike[str],
    *,
    flats: np.ndarray | str | PathLike[str] | None = None,
    darks: np.ndarray | str | PathLike[str] | None = None,
    angles: Sequence[float] | np.ndarray | None = None,
    detector: Detector | None = None,
    grid: Grid | None = None,
    white_level: float | None = None,
) -> Frames:
    """Load detector frames from an NXtomo (or ``.npz``) file, a Nikon ``.xtekct`` or TIFFs.

    An NXtomo file's ``image_key`` marks its flats (1) and darks (2); its
    sample frames are read only as they are corrected. A Nikon scan carries
    its white level. A TIFF file or folder of sample frames needs its
    ``angles`` (degrees), and its ``detector`` unless pixels are unit length.
    ``flats`` and ``darks`` (arrays, or TIFF files or folders), ``angles``,
    ``detector``, ``grid`` and ``white_level`` replace what the file holds.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no such file: {file}")
    suffix = file.suffix.lower()
    if suffix == ".xtekct":
        frames = _nikon_frames(file)
    elif file.is_dir() or suffix in {".tif", ".tiff"}:
        frames = _tiff_frames(file, angles=angles, detector=detector)
    elif suffix in {".nxs", ".h5", ".hdf5"}:
        from tomojax.io.api import load_nxtomo

        frames = _frames_from_payload(load_nxtomo(str(file), lazy=True), file)
    else:
        from tomojax.io import load_dataset

        record = load_dataset(file)
        frames = Frames(
            record.projections,
            _geometry_of(record, poses=True),
            name=record.sample_name or "sample",
            source=record,
        )
    changes: dict[str, Any] = {}
    if flats is not None:
        changes |= {"flats": _frames_of(flats, "flats"), "flat_positions": None}
    if darks is not None:
        changes["darks"] = _frames_of(darks, "darks")
    if white_level is not None:
        changes["white_level"] = float(white_level)
    if (
        angles is not None or detector is not None or grid is not None
    ) and frames.source is not None:
        record = replace(
            frames.source,
            angles=frames.source.angles if angles is None else np.asarray(angles, np.float32),
            detector=detector or frames.source.detector,
            grid=grid or frames.source.grid,
        )
        changes |= {"geometry": _geometry_of(record, poses=True), "source": record}
    return replace(frames, **changes)


def _view_indices(views: slice | Sequence[int] | np.ndarray, count: int) -> np.ndarray:
    """``views`` of ``count`` as increasing indices."""
    if isinstance(views, slice):
        return np.arange(count)[views]
    chosen = np.asarray(views)
    if chosen.dtype == bool:
        if chosen.shape != (count,):
            raise ValueError(f"a mask of views needs {count} entries, not {chosen.shape}")
        return np.flatnonzero(chosen)
    chosen = chosen.astype(np.int64).reshape(-1)
    if chosen.size and (chosen.min() < 0 or chosen.max() >= count):
        raise ValueError(f"views must be in 0..{count - 1}")
    if np.any(np.diff(chosen) <= 0):
        raise ValueError("views must be increasing indices, each once")
    return chosen


def _views_of(geometry: ScanGeometry, kept: np.ndarray) -> ScanGeometry:
    """``geometry`` of the views ``kept`` (increasing indices), wrappers and segments kept."""
    from tomojax._data.geometry_meta import AugmentedGeometry
    from tomojax.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        parts, start = [], 0
        for segment in geometry.segments:
            n = _view_count(segment)
            mine = kept[(kept >= start) & (kept < start + n)] - start
            if len(mine):
                parts.append(_views_of(segment, mine))
            start += n
        return parts[0] if len(parts) == 1 else ConeSegments(tuple(parts))
    if isinstance(geometry, AugmentedGeometry):
        return replace(
            geometry,
            base=_views_of(geometry.base, kept),
            align_params=np.asarray(geometry.align_params)[kept],
        )
    if not is_dataclass(geometry) or isinstance(geometry, type):
        raise TypeError(f"cannot select views of a {type(geometry).__name__} geometry")
    fields: dict[str, Any] = vars(geometry)  # a wrapper's base, or a geometry's own angles
    if "base" in fields:
        return replace(geometry, base=_views_of(fields["base"], kept))
    return replace(geometry, angles=np.asarray(fields["angles"])[kept])


def _view_count(geometry: ScanGeometry) -> int:
    return len(geometry.angles)  # pyright: ignore[reportAttributeAccessIssue]


def _check_shape(what: str, shape: tuple[int, ...], geometry: ScanGeometry) -> None:
    detector = geometry.detector
    views = _view_count(geometry)
    if tuple(int(s) for s in shape) != (views, detector.nv, detector.nu):
        raise ValueError(
            f"{what} are {tuple(shape)} but the geometry has {views} views of "
            f"{detector.nv} rows x {detector.nu} columns"
        )


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' * (n != 1)}"


def _frames_of(value: np.ndarray | str | PathLike[str], name: str) -> np.ndarray:
    if isinstance(value, str | os.PathLike):
        from tomojax.io.api import read_tiff_frames

        return read_tiff_frames(Path(value))
    frames = np.asarray(value)
    if frames.ndim == 2:
        frames = frames[None]
    if frames.ndim != 3:
        raise ValueError(f"{name} must be frames (frames, rows, columns), not {frames.shape}")
    return frames


def _frames_from_payload(payload: LoadedNXTomo, file: Path) -> Frames:
    """The sample frames, flats and darks of an NXtomo payload, split by ``image_key``."""
    from tomojax.io import ProjectionDataset

    stack = payload.projections
    views_total = int(stack.shape[0])
    key = payload.metadata.image_key
    key = np.zeros(views_total, np.int32) if key is None else np.asarray(key)
    sample = key == 0
    if not sample.any():
        raise ValueError(f"{file} holds no sample frames (image_key 0)")
    counts = stack[sample] if isinstance(stack, np.ndarray) else stack.frames_at(sample)
    flats = np.asarray(stack[np.flatnonzero(key == 1)]) if (key == 1).any() else None
    darks = np.asarray(stack[np.flatnonzero(key == 2)]) if (key == 2).any() else None
    positions = np.cumsum(sample)[key == 1] if flats is not None else None
    metadata = replace(
        payload.metadata,
        angles=None
        if payload.metadata.angles is None
        else np.asarray(payload.metadata.angles)[sample],
        image_key=None,
    )
    # The geometry needs the views' shape, not their values: read none of them.
    shape = (int(sample.sum()), *(int(s) for s in stack.shape[1:]))
    placeholder = replace(
        payload, projections=np.broadcast_to(np.float32(0), shape), metadata=metadata
    )
    record = ProjectionDataset.from_nxtomo(placeholder, source_path=file)
    return Frames(
        counts=counts,
        geometry=_geometry_of(record, poses=True),
        flats=flats,
        darks=darks,
        flat_positions=positions,
        name=record.sample_name or "sample",
        source=record,
    )


def _nikon_frames(file: Path) -> Frames:
    from tomojax.io import load_nikon_xtekct

    record = load_nikon_xtekct(file, absorption=False)
    white = float(record.geometry_metadata["nikon_xtekct"]["white_level"])
    return Frames(
        counts=record.projections,
        geometry=_geometry_of(record, poses=True),
        white_level=white,
        name=record.sample_name or "sample",
        source=record,
    )


def _tiff_frames(
    file: Path, *, angles: Sequence[float] | np.ndarray | None, detector: Detector | None
) -> Frames:
    from tomojax.geometry import Detector
    from tomojax.io import ProjectionDataset
    from tomojax.io.api import read_tiff_frames

    if angles is None:
        raise ValueError(
            f"{file} is a TIFF stack, which holds no angles: pass angles=... (degrees)"
        )
    counts = read_tiff_frames(file)
    views, rows, cols = counts.shape
    angles = np.asarray(angles, np.float32)
    if angles.shape != (views,):
        raise ValueError(f"{file} has {views} frames but {angles.size} angles")
    record = ProjectionDataset(
        projections=counts,
        angles=angles,
        detector=detector or Detector(nu=cols, nv=rows, du=1.0, dv=1.0),
        source_path=str(file),
        source_format="tiff_stack",
    )
    return Frames(counts, _geometry_of(record, poses=True), source=record)


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
        geometry=_geometry_of(record, poses=poses),
        name=record.sample_name or "sample",
        corrections=tuple(Correction.from_dict(c) for c in saved),
        source=record,
    )


def _geometry_of(record: ProjectionDataset, *, poses: bool) -> ScanGeometry:
    from tomojax.io import build_geometry_from_dataset_metadata

    _, _, geometry = build_geometry_from_dataset_metadata(record.geometry_inputs(), poses=poses)
    return geometry


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
