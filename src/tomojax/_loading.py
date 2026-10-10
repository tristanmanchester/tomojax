"""How files become scans: :func:`load`, and raw detector frames as :class:`Frames`.

:func:`load` reads a scan of line integrals, correcting raw detector frames on
the way (the correction is recorded in :attr:`Scan.corrections`);
:func:`load_frames` reads the frames themselves, as :class:`Frames`, for other
corrections: from HDF5 (NXtomo or another layout), Nikon ``.xtekct`` scans and
TIFFs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax._scan import (
    Scan,
    check_shape,
    detector_window,
    geometry_of,
    rebuild,
    scan_from_record,
    view_indices,
    views_of,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    from tomojax.corrections import Step
    from tomojax.geometry import Detector, Grid, ScanGeometry
    from tomojax.io import ProjectionDataset
    from tomojax.io.api import LocatedFrames

LOG = logging.getLogger(__name__)


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
    located = _located(file) if suffix in _HDF5 else None
    if located is not None and located.raw:
        return _corrected_on_load(_hdf5_frames(file, located, angles=None), file)
    try:
        record = load_dataset(file)
    except KeyError as exc:  # HDF5 frames not laid out as TomoJAX's
        if located is None:
            raise
        raise ValueError(
            f"{file} is not a TomoJAX dataset and holds no flat frames: load its frames with "
            "tomojax.load_frames(path, flats=..., angles=...)"
        ) from exc
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
        check_shape("Frames: counts", self.counts.shape, self.geometry)
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
        must be finite and positive; it bounds the flat field and transmission
        below. Views are corrected on the device in batches of ``batch_views``,
        a positive integer (by default, 256 MiB).
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
            self.geometry if len(done.kept) == self.views else views_of(self.geometry, done.kept)
        )
        return Scan(done.projections, geometry, self.name, done.records, self.source)

    def cropped(self, rows: slice, cols: slice) -> Frames:
        """These frames' block of detector ``rows`` and ``cols``, the detector moved to match.

        A lazily read file reads only that block.
        """
        window = detector_window(rows, cols, self.geometry.detector)
        counts = self.counts
        if hasattr(counts, "window"):
            counts = counts.window(rows, cols)
        else:
            counts = np.asarray(counts)[:, rows, cols]
        flats, darks = (
            None if f is None else np.asarray(f)[:, rows, cols] for f in (self.flats, self.darks)
        )
        return replace(
            self,
            counts=counts,
            flats=flats,
            darks=darks,
            geometry=rebuild(self.geometry, detector=window),
        )

    def selected(self, views: slice | Sequence[int] | np.ndarray) -> Frames:
        """These sample ``views`` only, with their geometry.

        Take a forward slice, increasing integer indices or a nonempty mask.
        Each view keeps its place among the flats (:attr:`view_positions`), so
        its flat is interpolated as before.
        """
        kept = view_indices(views, self.views)
        at = np.arange(self.views) + 0.5 if self.view_positions is None else self.view_positions
        counts = self.counts
        counts = (
            counts.frames_at(kept) if hasattr(counts, "frames_at") else np.asarray(counts)[kept]
        )
        return replace(
            self,
            counts=counts,
            geometry=views_of(self.geometry, kept),
            view_positions=np.asarray(at, np.float64)[kept],
        )


def load_frames(
    path: str | PathLike[str],
    *,
    flats: np.ndarray | float | str | PathLike[str] | None = None,
    darks: np.ndarray | float | str | PathLike[str] | None = None,
    angles: Sequence[float] | np.ndarray | None = None,
    detector: Detector | None = None,
    grid: Grid | None = None,
    geometry: ScanGeometry | None = None,
    white_level: float | None = None,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
) -> Frames:
    """Load detector frames from an HDF5 (NXtomo) or ``.npz`` file, a Nikon ``.xtekct`` or TIFFs.

    An HDF5 file's ``image_key`` marks its flats (1) and darks (2), and its
    sample frames are read only as they are corrected. The frames, key and
    angles are found at their NXtomo paths or as the file's only datasets of
    that name (``data``, ``image_key``, ``rotation_angle``); ``data_path``,
    ``image_key_path`` and ``angles_path`` name them otherwise. A Nikon scan
    carries its white level. TIFF frames need their ``angles`` (degrees), and
    their ``detector`` unless pixels are unit length.

    ``flats`` and ``darks`` (arrays, TIFF files or folders, or one level for
    every pixel), ``angles``, ``detector``, ``grid``, ``white_level`` and a
    whole ``geometry`` (of every sample view; then TIFFs need no ``angles``)
    replace what the file holds.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no such file: {file}")
    suffix = file.suffix.lower()
    if suffix == ".xtekct":
        frames = _nikon_frames(file)
    elif file.is_dir() or suffix in {".tif", ".tiff"}:
        if geometry is not None and angles is None:
            angles = np.asarray(cast("Any", geometry).angles)
        frames = _tiff_frames(file, angles=angles, detector=detector)
    elif suffix in _HDF5:
        from tomojax.io.api import locate_frames

        located = locate_frames(
            str(file), data_path=data_path, image_key_path=image_key_path, angles_path=angles_path
        )
        frames = _hdf5_frames(file, located, angles=angles)
    else:
        from tomojax.io import load_dataset

        record = load_dataset(file)
        frames = Frames(
            record.projections,
            geometry_of(record, poses=True),
            name=record.sample_name or "sample",
            source=record,
        )
    frames = _with_fields(frames, flats=flats, darks=darks, white_level=white_level)
    if (
        angles is not None or detector is not None or grid is not None
    ) and frames.source is not None:
        record = replace(
            frames.source,
            angles=frames.source.angles if angles is None else np.asarray(angles, np.float32),
            detector=detector or frames.source.detector,
            grid=grid or frames.source.grid,
        )
        frames = replace(frames, geometry=geometry_of(record, poses=True), source=record)
    return frames if geometry is None else replace(frames, geometry=geometry)


def _with_fields(
    frames: Frames,
    *,
    flats: np.ndarray | float | str | PathLike[str] | None,
    darks: np.ndarray | float | str | PathLike[str] | None,
    white_level: float | None,
) -> Frames:
    """``frames`` with the flat and dark fields given in place of the file's."""
    changes: dict[str, Any] = {}
    shape = tuple(int(n) for n in frames.counts.shape[1:])
    if isinstance(flats, int | float):  # one level for every pixel: a white level
        white_level = float(flats)
    elif flats is not None:
        changes |= {
            "flats": _frames_of(flats, "flats"),
            "flat_positions": None,
            "white_level": None,
        }
    if white_level is not None:
        changes |= {"flats": None, "flat_positions": None, "white_level": float(white_level)}
    if isinstance(darks, int | float):
        changes["darks"] = np.full((1, *shape), float(darks), np.float32)
    elif darks is not None:
        changes["darks"] = _frames_of(darks, "darks")
    return replace(frames, **changes)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' * (n != 1)}"


def _frames_of(value: np.ndarray | float | str | PathLike[str], name: str) -> np.ndarray:
    if isinstance(value, str | os.PathLike):
        from tomojax.io.api import read_tiff_frames

        return read_tiff_frames(Path(value))
    frames = np.asarray(value)
    if frames.ndim == 2:
        frames = frames[None]
    if frames.ndim != 3:
        raise ValueError(f"{name} must be frames (frames, rows, columns), not {frames.shape}")
    return frames


_HDF5 = frozenset({".nxs", ".h5", ".hdf5"})


def _located(file: Path) -> LocatedFrames | None:
    """The file's frames, key and angles at their usual places, or None if not found."""
    from tomojax.io.api import locate_frames

    try:
        return locate_frames(str(file))
    except KeyError:
        return None


def _hdf5_frames(
    file: Path, located: LocatedFrames, *, angles: Sequence[float] | np.ndarray | None
) -> Frames:
    """The sample frames, flats and darks of an HDF5 file, split by ``image_key``.

    The geometry is the NXtomo file's (``tomojax`` metadata) when it has one,
    else parallel-beam with unit pixels.
    """
    from tomojax.geometry import Detector
    from tomojax.io import ProjectionDataset
    from tomojax.io.api import LoadedNXTomo, NXTomoMetadata, load_nxtomo

    stack = located.stack
    key = located.image_key
    key = np.zeros(len(stack), np.int32) if key is None else key
    sample = key == 0
    if not sample.any():
        raise ValueError(f"{file} holds no sample frames (image_key 0)")
    views, rows, cols = int(sample.sum()), stack.rows, stack.cols
    if angles is not None:
        view_angles = np.asarray(angles, np.float64)
    elif located.angles is not None:
        view_angles = located.angles[sample]
    else:
        raise ValueError(f"{file} records no rotation angles: pass angles=... (degrees)")
    if view_angles.shape != (views,):
        raise ValueError(f"{file} has {views} sample frames but {view_angles.size} angles")
    try:
        metadata = load_nxtomo(str(file), lazy=True).metadata
    except KeyError:  # not laid out as NXtomo: the frames alone
        metadata = NXTomoMetadata(detector=Detector(nu=cols, nv=rows, du=1.0, dv=1.0))
    metadata = replace(metadata, angles=view_angles.astype(np.float32), image_key=None)
    # The geometry needs the views' shape, not their values: read none of them.
    placeholder = LoadedNXTomo(np.broadcast_to(np.float32(0), (views, rows, cols)), metadata)
    record = ProjectionDataset.from_nxtomo(placeholder, source_path=file)
    flats = stack[np.flatnonzero(key == 1)] if (key == 1).any() else None
    darks = stack[np.flatnonzero(key == 2)] if (key == 2).any() else None
    if located.flats is not None:  # kept apart from the frames: one set
        flats = located.flats[0 : len(located.flats)]
    if located.darks is not None:
        darks = located.darks[0 : len(located.darks)]
    return Frames(
        counts=stack.frames_at(sample),
        geometry=geometry_of(record, poses=True),
        flats=flats,
        darks=darks,
        flat_positions=None
        if flats is None or located.flats is not None
        else np.cumsum(sample)[key == 1],
        name=record.sample_name or "sample",
        source=record,
    )


def _nikon_frames(file: Path) -> Frames:
    from tomojax.io import load_nikon_xtekct

    record = load_nikon_xtekct(file, absorption=False)
    white = float(record.geometry_metadata["nikon_xtekct"]["white_level"])
    return Frames(
        counts=record.projections,
        geometry=geometry_of(record, poses=True),
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
    return Frames(counts, geometry_of(record, poses=True), source=record)


__all__ = ["Frames", "load", "load_frames"]
