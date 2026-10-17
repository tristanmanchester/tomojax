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
import numbers
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax._scan import (
    Scan,
    binned_geometry,
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
    """Load line integrals from a dataset or Nikon scan.

    A file of raw detector frames (an NXtomo ``image_key`` marking flats or
    darks, or a Nikon scan) is corrected with the standard chain, as
    ``load_frames(path).corrected()``: dark subtraction, flat division and
    ``-log``; :attr:`Scan.corrections` records it. For other corrections, load
    the frames with :func:`load_frames`. The scan carries the per-view
    corrections saved with it (by :func:`align` or ``tomojax align``) unless
    ``poses`` is False. Frames that ``image_key`` marks as other than sample
    views are left out.

    Parameters
    ----------
    path : str or path-like
        Dataset or Nikon scan to read.
    poses : bool, optional
        Apply the saved per-view pose corrections; default is True.

    Returns
    -------
    Scan
        Line-integral projections with geometry and correction records.

    Raises
    ------
    FileNotFoundError
        If the input does not exist.
    ValueError
        If TIFFs need geometry, counts lack calibration, or metadata are invalid.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    """
    from tomojax.io import load_dataset

    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no such file: {file}")
    discovered = _acquisition_frames(file)
    if discovered is not None:
        return _corrected_on_load(discovered, file)
    suffix = file.suffix.lower()
    if file.is_dir() or suffix in {".tif", ".tiff"}:
        raise ValueError(
            f"{file} is a TIFF stack, which holds no geometry: load it with "
            "tomojax.load_frames(path, angles=..., detector=...)"
        )
    located = _located(file) if suffix in _HDF5 else None
    if located is not None and located.raw:
        return _corrected_on_load(_hdf5_frames(file, located, angles=None, poses=poses), file)
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
    scan = scan_from_record(record, poses=poses)
    if located is not None and located.image_key is not None and np.any(located.image_key != 0):
        scan = scan.selected(located.image_key == 0)
    return scan


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
    the frames with the beam off, both in memory or lazy like ``counts``.
    Scanners that calibrate their flat field record a ``white_level`` instead.
    ``view_positions`` places each view among the flats, in the units of
    ``flat_positions`` (by default view ``i`` at ``i + 1/2``; :meth:`selected`
    keeps it). :meth:`corrected` makes the :class:`Scan` of line integrals.
    Made by :func:`load_frames`.

    Parameters
    ----------
    counts : array-like
        Sample detector frames, shape (views, rows, columns); may be lazy.
    geometry : ScanGeometry
        Geometry of every sample view, including the grid and detector.
    flats : array-like or None, optional
        Open-beam calibration frames; may be lazy.
    darks : array-like or None, optional
        Beam-off calibration frames; may be lazy.
    flat_positions : numpy.ndarray or None, optional
        Number of sample views preceding each flat frame.
    white_level : float or None, optional
        Calibrated open-beam count level, instead of flat frames.
    view_positions : numpy.ndarray or None, optional
        Sample positions in flat-position units; default is i + 1/2.
    name : str, optional
        Sample name.
    source : ProjectionDataset or None, optional
        Original dataset metadata retained when loading from a file.

    Raises
    ------
    ValueError
        If frame shape or view-position count does not match the geometry.
    """

    counts: Any
    geometry: ScanGeometry
    flats: Any = None
    darks: Any = None
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
        """The number of sample views.

        Returns
        -------
        int
            Number of sample frames.
        """
        return int(self.counts.shape[0])

    def corrected(
        self, *steps: Step, epsilon: float = 1e-6, batch_views: int | None = None, binning: int = 1
    ) -> Scan:
        """Correct detector counts to line integrals with the requested steps.

        Steps run by the data they act on: counts steps first, then the flat
        and dark fields, transmission steps, the log, and line-integral steps,
        each in the order given (see :mod:`tomojax.corrections`). ``epsilon``
        must be finite and positive; it bounds the flat field and transmission
        below. Views are corrected on the device in batches of ``batch_views``,
        a positive integer. Automatic view batches target 32 MiB for simple
        pipelines, or 256 MiB including additional workspace such as the padded
        FFT arrays needed for Paganin phase retrieval.

        ``binning`` averages blocks of ``binning`` x ``binning`` detector pixels of
        the line integrals, as :meth:`Scan.binned` does, without holding the
        unbinned stack on the host unless a step such as ``RejectViews`` or a
        non-row step needs it. Pixels where the flat does not exceed the dark by
        three standard errors of their means take no absorption; the flat-field
        record counts them.

        Parameters
        ----------
        *steps : Step
            Corrections applied in domain order, then in the order supplied.
        epsilon : float, optional
            Positive finite lower bound for flat field and transmission.
        batch_views : int or None, optional
            Positive views per device batch; None selects a workspace-based size.
        binning : int, optional
            Positive factor for averaging detector blocks after correction.

        Returns
        -------
        Scan
            Corrected line integrals with matching geometry and correction records.

        Raises
        ------
        ValueError
            If calibration, correction settings or batch dimensions are invalid.
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
            binning=binning,
        )
        geometry = (
            self.geometry if len(done.kept) == self.views else views_of(self.geometry, done.kept)
        )
        geometry = binned_geometry(geometry, binning) if binning > 1 else geometry
        return Scan(done.projections, geometry, self.name, done.records, self.source)

    def cropped(self, rows: slice, cols: slice) -> Frames:
        """These frames' block of detector ``rows`` and ``cols``, the detector moved to match.

        A lazily read file reads only that block, of the flats and darks too.

        Parameters
        ----------
        rows : slice
            Nonempty contiguous detector row window.
        cols : slice
            Nonempty contiguous detector column window.

        Returns
        -------
        Frames
            Selected frames with matching geometry and calibration.

        Raises
        ------
        ValueError
            If the selection is empty, out of bounds or not ordered as required.
        """
        window = detector_window(rows, cols, self.geometry.detector)

        def crop(frames: Any) -> Any:
            if frames is None or hasattr(frames, "window"):
                return None if frames is None else frames.window(rows, cols)
            return np.asarray(frames)[:, rows, cols]

        counts, flats, darks = crop(self.counts), crop(self.flats), crop(self.darks)
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

        Parameters
        ----------
        views : slice, sequence of int or numpy.ndarray
            Forward slice, increasing indices or nonempty boolean sample-view mask.

        Returns
        -------
        Frames
            Selected frames with matching geometry and calibration.

        Raises
        ------
        ValueError
            If the selection is empty, out of bounds or not ordered as required.
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
    """Load detector frames and acquisition metadata from supported scan formats.

    An HDF5 file's ``image_key`` marks its flats (1) and darks (2), and its
    sample, flat and dark frames are read only as they are corrected. The
    frames, key and angles are found at their NXtomo paths or as the file's only datasets of
    that name (``data``, ``image_key``, ``rotation_angle``), or their paired GDA imaging
    paths; ``data_path``, ``image_key_path`` and ``angles_path`` name them otherwise.
    Nikon, SkyScan, Zeiss and ESRF acquisitions are recognized by content and
    supply their own geometry and calibration. SkyScan logs require geometrically
    and flat-field corrected TIFF exports; their bit depth supplies the white level.
    A GDA command mislabeled as mm is read in degrees with a warning; discovered
    paths and this correction stay in source metadata.
    Generic TIFF stacks without acquisition metadata need ``angles`` (degrees),
    ``flats`` or ``white_level``, and
    their ``detector`` unless pixels are unit length.

    ``flats`` and ``darks`` (arrays, HDF5 stacks, TIFF files or folders, or one level for
    every pixel), ``angles``, ``detector``, ``grid``, ``white_level`` and a
    whole ``geometry`` (of every sample view; then TIFFs need no ``angles``)
    replace what the file holds. HDF5 calibration paths use the same automatic
    dataset discovery as the input and stay lazy; every frame in their stack
    contributes to the calibration mean.

    Parameters
    ----------
    path : str or path-like
        HDF5, NPZ, Nikon, SkyScan, Zeiss or ESRF acquisition, or generic TIFF stack.
    flats : array-like, float, str, path-like or None, optional
        Replacement flat frames, calibration stack path or constant open-beam level.
    darks : array-like, float, str, path-like or None, optional
        Replacement dark frames, calibration stack path or constant dark level.
    angles : array-like or None, optional
        Sample-view angles in degrees, replacing file metadata.
    detector : Detector or None, optional
        Replacement detector dimensions, pixel sizes and physical centre.
    grid : Grid or None, optional
        Replacement reconstruction grid.
    geometry : ScanGeometry or None, optional
        Complete sample-view geometry, overriding individual metadata.
    white_level : float or None, optional
        Replacement calibrated open-beam level, used instead of flats.
    data_path : str or None, optional
        Explicit HDF5 frame dataset path; None discovers it.
    image_key_path : str or None, optional
        Explicit HDF5 sample/flat/dark key path; None discovers it.
    angles_path : str or None, optional
        Explicit HDF5 rotation-angle path; None discovers it.

    Returns
    -------
    Frames
        Raw sample and calibration frames with geometry; HDF5/TIFF reads stay lazy.

    Raises
    ------
    FileNotFoundError
        If the input does not exist.
    ValueError
        If required angles are absent or metadata and frame shapes disagree.
    KeyError
        If a required dataset cannot be found in the file.

    Examples
    --------
    >>> import tomojax as tj
    >>> frames = tj.load_frames("raw_scan.nxs")  # doctest: +SKIP
    >>> scan = frames.corrected(binning=4)  # doctest: +SKIP
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no such file: {file}")
    suffix = file.suffix.lower()
    frames = _acquisition_frames(file, data_path=data_path)
    if frames is not None:
        pass
    elif file.is_dir() or suffix in {".tif", ".tiff"}:
        if geometry is not None and angles is None:
            angles = np.asarray(cast("Any", geometry).angles)
        frames = _tiff_frames(file, angles=angles, detector=detector)
    elif suffix in _HDF5:
        from tomojax.io.api import locate_frames

        located = locate_frames(
            str(file), data_path=data_path, image_key_path=image_key_path, angles_path=angles_path
        )
        frames = _hdf5_frames(file, located, angles=angles, poses=True)
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
    if geometry is not None:
        source = frames.source
        if source is not None and "acquisition" in source.geometry_metadata:
            source = replace(
                source,
                geometry_metadata={
                    **source.geometry_metadata,
                    "acquisition": {
                        **source.geometry_metadata["acquisition"],
                        "geometry_status": "provided",
                        "geometry_note": "Geometry supplied explicitly by caller.",
                    },
                },
            )
        frames = replace(frames, geometry=geometry, source=source)
    if (geometry is not None or detector is not None) and frames.source is not None:
        frames = replace(
            frames,
            source=replace(
                frames.source,
                geometry_metadata={
                    **frames.source.geometry_metadata,
                    "detector_center_status": "provided",
                },
            ),
        )
    return frames


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
    if isinstance(flats, numbers.Real):  # one level for every pixel: a white level
        white_level = float(flats)
    elif flats is not None:
        changes |= {
            "flats": _frames_of(flats, "flats"),
            "flat_positions": None,
            "white_level": None,
        }
    if white_level is not None:
        changes |= {"flats": None, "flat_positions": None, "white_level": float(white_level)}
    if isinstance(darks, numbers.Real):
        changes["darks"] = np.full((1, *shape), float(darks), np.float32)
    elif darks is not None:
        changes["darks"] = _frames_of(darks, "darks")
    return replace(frames, **changes)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' * (n != 1)}"


def _frames_of(value: np.ndarray | float | str | PathLike[str], name: str) -> Any:
    if isinstance(value, str | os.PathLike):
        from tomojax.io.api import locate_frames, read_tiff_frames

        if Path(value).suffix.lower() in _HDF5:
            return locate_frames(str(value)).stack
        return read_tiff_frames(Path(value), lazy=True)
    frames = np.asarray(value)
    if frames.ndim == 2:
        frames = frames[None]
    if frames.ndim != 3:
        raise ValueError(f"{name} must be frames (frames, rows, columns), not {frames.shape}")
    return frames


_HDF5 = frozenset({".nxs", ".h5", ".hdf", ".hdf5"})


def _located(file: Path) -> LocatedFrames | None:
    """The file's frames, key and angles at their usual places, or None if not found."""
    from tomojax.io.api import locate_frames

    try:
        return locate_frames(str(file))
    except KeyError:
        return None


def _hdf5_frames(
    file: Path,
    located: LocatedFrames,
    *,
    angles: Sequence[float] | np.ndarray | None,
    poses: bool,
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
        metadata = NXTomoMetadata(
            detector=Detector(nu=cols, nv=rows, du=1.0, dv=1.0),
            geometry_meta={"detector_center_status": "not_recorded"},
        )
        if located.acquisition is not None:
            metadata.geometry_meta = {
                "detector_center_status": "not_recorded",
                "acquisition": {
                    **located.acquisition,
                    "geometry_status": "not_recorded",
                    "geometry_note": "Source has no calibrated rotation axis or pixel scale.",
                },
            }
    metadata = replace(metadata, angles=view_angles.astype(np.float32), image_key=None)
    # The geometry needs the views' shape, not their values: read none of them.
    placeholder = LoadedNXTomo(np.broadcast_to(np.float32(0), (views, rows, cols)), metadata)
    record = ProjectionDataset.from_nxtomo(placeholder, source_path=file)
    flats = stack.frames_at(key == 1) if (key == 1).any() else None
    darks = stack.frames_at(key == 2) if (key == 2).any() else None
    if located.flats is not None:  # kept apart from the frames: one set
        flats = located.flats
    if located.darks is not None:
        darks = located.darks
    return Frames(
        counts=stack.frames_at(sample),
        geometry=geometry_of(record, poses=poses),
        flats=flats,
        darks=darks,
        flat_positions=None
        if flats is None or located.flats is not None
        else np.cumsum(sample)[key == 1],
        name=record.sample_name or "sample",
        source=record,
    )


def _acquisition_frames(file: Path, *, data_path: str | None = None) -> Frames | None:
    from tomojax.io.api import read_acquisition

    found = read_acquisition(file, data_path=data_path)
    if found is None:
        return None
    record = found.record
    return Frames(
        counts=found.counts,
        geometry=geometry_of(record, poses=True),
        flats=found.flats,
        darks=found.darks,
        flat_positions=found.flat_positions,
        white_level=found.white_level,
        view_positions=found.view_positions,
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
    counts = read_tiff_frames(file, lazy=True)
    views, rows, cols = counts.shape
    angles = np.asarray(angles, np.float32)
    if angles.shape != (views,):
        raise ValueError(f"{file} has {views} frames but {angles.size} angles")
    record = ProjectionDataset(
        projections=np.broadcast_to(np.float32(0), counts.shape),
        angles=angles,
        detector=detector or Detector(nu=cols, nv=rows, du=1.0, dv=1.0),
        source_path=str(file),
        source_format="tiff_stack",
    )
    return Frames(counts, geometry_of(record, poses=True), source=record)


__all__ = ["Frames", "load", "load_frames"]
