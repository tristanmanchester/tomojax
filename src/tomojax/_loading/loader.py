"""How files become scans: :func:`load`, and raw detector frames as :class:`Frames`.

:func:`load` reads a scan of line integrals, correcting raw detector frames on
the way (the correction is recorded in :attr:`Scan.corrections`);
:func:`load_frames` reads the frames themselves, as :class:`Frames`, for other
corrections: from HDF5 (NXtomo or another layout), Nikon ``.xtekct`` and Zeiss
Xradia acquisitions, or TIFFs.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import KW_ONLY, dataclass, field, replace
import logging
import numbers
import os
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, Unpack, cast

import numpy as np

from tomojax import progress
from tomojax._data import calibration_issues, read_nxtomo, views_of
from tomojax._loading.frame_sources import (
    acquisition_frames,
    line_integral_frames,
    selected_frames,
    tiff_frames,
)
from tomojax._loading.geometry import center_of, centered, frame_geometry
from tomojax._parameters import AnnotationModule
from tomojax._scan.model import (
    Scan,
    bind_step_geometry,
    binned_geometry,
    check_shape,
    cropped_record,
    cut,
    detector_window,
    frame_output_grid,
    rebuild,
    scan_from_record,
)
from tomojax.core.host_volume import discard_failed_outputs
from tomojax.core.registration import registration
from tomojax.core.validation import view_indices
from tomojax.geometry import Detector, Grid, ScanGeometry
from tomojax.io.projection_layout import saved_projection_layout

if TYPE_CHECKING:
    from tomojax import corrections
    from tomojax._data import FrameStack
    from tomojax._loading.geometry import DeclaredGeometry
    from tomojax.corrections import Correction
    from tomojax.io import ProjectionDataset
    from tomojax.io.api import LocatedFrames
else:
    corrections = AnnotationModule("tomojax.corrections")

if TYPE_CHECKING:

    class _FrameOverrides(TypedDict, total=False):
        detector: Detector | None
        pixel_size: float | tuple[float, float] | None
        grid: Grid | None
        geometry: ScanGeometry | None
        declared: DeclaredGeometry | None

    class _StoredOverrides(_FrameOverrides):
        angles: Sequence[float] | np.ndarray | None

    class _StoredFrameSettings(_StoredOverrides):
        apply_poses: bool


LOG = logging.getLogger(__name__)


def is_raw_input(
    path: str | PathLike[str],
    *,
    flats: np.ndarray | float | str | PathLike[str] | None = None,
    darks: np.ndarray | float | str | PathLike[str] | None = None,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
) -> bool:
    """Whether ``path`` holds raw detector frames rather than saved line integrals."""
    from tomojax.io.api import read_acquisition

    file = Path(path)
    if flats is not None or darks is not None:
        return True
    paths = {"data_path": data_path, "image_key_path": image_key_path, "angles_path": angles_path}
    if read_acquisition(file, **paths) is not None:
        return True
    located = _located(file, **paths) if file.suffix.lower() in _HDF5 else None
    return _line_integrals(file, located) is None


def load(
    path: str | PathLike[str],
    *,
    flats: np.ndarray | float | str | PathLike[str] | None = None,
    darks: np.ndarray | float | str | PathLike[str] | None = None,
    angles: Annotated[Sequence[float] | np.ndarray | None, "degree"] = None,
    detector: Detector | None = None,
    pixel_size: Annotated[float | tuple[float, float] | None, "geometry_length"] = None,
    grid: Grid | None = None,
    geometry: ScanGeometry | None = None,
    exposure_times: Annotated[np.ndarray | float | None, "s"] = None,
    flat_exposure_times: Annotated[np.ndarray | float | None, "s"] = None,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
    apply_poses: bool = True,
) -> Scan:
    """Load line integrals from a dataset or recognized raw acquisition.

    A file of raw detector frames (an NXtomo ``image_key`` marking flats or
    darks, or a Nikon, SkyScan, Zeiss or ESRF acquisition, or a TIFF stack) is
    corrected with the standard chain, as ``load_frames(path, ...).corrected()``:
    dark subtraction, flat division and ``-log``; :attr:`Scan.corrections`
    records it. For other corrections, load the frames with :func:`load_frames`.
    Every other argument has :func:`load_frames`'s meaning and supplies what the
    file does not record. The scan carries the per-view corrections saved with
    it (by :func:`align` or ``tomojax align``) unless ``apply_poses`` is False. Frames
    that ``image_key`` marks as other than sample views are left out.

    Parameters
    ----------
    path : str or path-like
        Dataset or recognized acquisition file or directory.
    flats, darks, angles, detector, pixel_size, grid, geometry : optional
        As :func:`load_frames`.
    exposure_times, flat_exposure_times : optional
        As :func:`load_frames`.
    data_path, image_key_path, angles_path : str or None, optional
        As :func:`load_frames`.
    apply_poses : bool, optional
        Apply the saved per-view pose corrections, and a scanner's own recorded
        per-view shifts and center shift (Zeiss); default is True. False gives the
        nominal geometry; the record names the scanner's corrections either way.

    Returns
    -------
    Scan
        Line-integral projections with geometry and correction records.

    Raises
    ------
    FileNotFoundError
        If the input does not exist.
    ValueError
        If metadata the file does not record are needed, counts lack
        calibration, or metadata are invalid; the message names the argument.
    KeyError
        If no stack of frames is found; ``data_path`` names it.

    Examples
    --------
    >>> import tomojax as tj
    >>> scan = tj.load("scan.nxs")  # doctest: +SKIP
    >>> scan = tj.load("tomo.h5", detector=tj.Detector(960, 600, 0.65, 0.65))  # doctest: +SKIP
    """
    return load_scan_with_declaration(
        path,
        flats=flats,
        darks=darks,
        angles=angles,
        detector=detector,
        pixel_size=pixel_size,
        grid=grid,
        geometry=geometry,
        exposure_times=exposure_times,
        flat_exposure_times=flat_exposure_times,
        data_path=data_path,
        image_key_path=image_key_path,
        angles_path=angles_path,
        apply_poses=apply_poses,
    )


def load_scan_with_declaration(
    path: str | PathLike[str],
    *,
    flats: np.ndarray | float | str | PathLike[str] | None = None,
    darks: np.ndarray | float | str | PathLike[str] | None = None,
    exposure_times: np.ndarray | float | None = None,
    flat_exposure_times: np.ndarray | float | None = None,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
    apply_poses: bool = True,
    **overrides: Any,
) -> Scan:
    center, overrides["declared"] = center_of(overrides.get("declared"))
    with discard_failed_outputs() as outputs:
        progress.check_canceled()
        file = Path(path)
        if not file.exists():
            with progress.task("load", file.name, None, "views"):
                raise FileNotFoundError(f"no such file: {file}")
        paths = {
            "data_path": data_path,
            "image_key_path": image_key_path,
            "angles_path": angles_path,
        }
        overrides.setdefault("angles", None)
        discovered, located = discover_frames(file, apply_poses=apply_poses, **paths, **overrides)
        if discovered is None:
            found = _line_integrals(file, located)
            if found is not None:
                outputs.own(found[0].projections)
                scan = _record_scan(file, *found, apply_poses=apply_poses, **overrides)
                return centered(scan, center)
        frames = _frames(file, discovered, apply_poses=apply_poses, **paths, **overrides)
        fields = _with_fields(
            frames,
            flats=flats,
            darks=darks,
            exposure_times=exposure_times,
            flat_exposure_times=flat_exposure_times,
        )
        scan = _corrected_on_load(fields, file)
        outputs.own(scan.projections)
        return centered(scan, center)


def discover_frames(
    file: Path,
    *,
    apply_poses: bool,
    layout_only: bool = False,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
    **overrides: Any,
) -> tuple[Frames | None, LocatedFrames | None]:
    """Discover vendor frames first, then labeled HDF5 frames, before reading pixels."""
    paths = {"data_path": data_path, "image_key_path": image_key_path, "angles_path": angles_path}
    frames = acquisition_frames(file, apply_poses=apply_poses, **paths, **overrides)
    if frames is None and (file.is_dir() or file.suffix.lower() in {".tif", ".tiff"}):
        frames = tiff_frames(file, **overrides)
    if frames is not None or file.suffix.lower() not in _HDF5:
        return frames, None
    located = _located(file, **paths)
    if located is not None and located.raw:
        frames = _hdf5_frames(
            file, located, apply_poses=apply_poses, layout_only=layout_only, **overrides
        )
    return frames, located


def projection_frames(
    file: Path,
    *,
    require_raw: bool = False,
    apply_poses: bool = True,
    flats: np.ndarray | float | str | PathLike[str] | None = None,
    darks: np.ndarray | float | str | PathLike[str] | None = None,
    exposure_times: np.ndarray | float | None = None,
    flat_exposure_times: np.ndarray | float | None = None,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
    angles: Sequence[float] | np.ndarray | None = None,
    **overrides: Unpack[_FrameOverrides],
) -> tuple[Frames, bool]:
    """Resolve geometry and calibration for a scan without reading its detector pixels."""
    if not file.exists():
        raise FileNotFoundError(f"no such file: {file}")
    center, overrides["declared"] = center_of(overrides.get("declared"))
    if TYPE_CHECKING:
        settings = _StoredFrameSettings(apply_poses=apply_poses, angles=angles, **overrides)
    else:
        settings = dict(apply_poses=apply_poses, angles=angles, **overrides)
    frames, located = discover_frames(
        file,
        layout_only=True,
        data_path=data_path,
        image_key_path=image_key_path,
        angles_path=angles_path,
        **settings,
    )
    if frames is None:
        frames, raw = stored_frames(file, located, require_raw=require_raw, **settings)
        if not raw:
            return centered(frames, center), False
    fields = _with_fields(
        frames,
        flats=flats,
        darks=darks,
        exposure_times=exposure_times,
        flat_exposure_times=flat_exposure_times,
    )
    from tomojax.corrections.calibration import check_calibration_layout

    _require_calibration(fields)
    check_calibration_layout(fields)
    return centered(fields, center), True


def stored_frames(
    file: Path,
    located: LocatedFrames | None,
    *,
    require_raw: bool,
    apply_poses: bool,
    **overrides: Unpack[_StoredOverrides],
) -> tuple[Frames, bool]:
    if TYPE_CHECKING:
        settings = _StoredFrameSettings(apply_poses=apply_poses, **overrides)
    else:
        settings = dict(apply_poses=apply_poses, **overrides)
    if located is not None and require_raw:
        return _hdf5_frames(file, located, layout_only=True, **settings), True
    try:
        record, stack = saved_projection_layout(file)
    except KeyError:
        if located is None:
            raise
        return _hdf5_frames(file, located, layout_only=True, **settings), True
    _frames_in(record, file)
    record.volume = None
    key, raw = sample_key(record, located)
    if raw and located is not None:
        return _hdf5_frames(file, located, layout_only=True, **settings), True
    if raw or require_raw or any(value is not None for value in overrides.values()):
        return _record_frames(file, record, stack=stack, **settings), raw or require_raw
    scan = scan_from_record(record, apply_poses=apply_poses)
    return line_integral_frames(scan, stack, key), False


def _line_integrals(
    file: Path, located: LocatedFrames | None
) -> tuple[ProjectionDataset, np.ndarray] | None:
    """A TomoJAX dataset of line integrals and its discovered key, or None for frames."""
    from tomojax.io import load_dataset

    suffix = file.suffix.lower()
    if file.is_dir() or suffix in {".tif", ".tiff"}:
        return None
    if located is not None and located.raw:
        return None
    try:
        record = load_dataset(file)
    except KeyError as exc:  # try the raw frame layout
        if progress.handler_failed(exc):
            raise
        return None
    key, raw = sample_key(record, located)
    return None if raw else (record, key)


def sample_key(record: ProjectionDataset, located: LocatedFrames | None) -> tuple[np.ndarray, bool]:
    discovered = None if located is None else located.image_key
    key = _saved_image_key(record) if discovered is None else discovered
    raw = np.isin(key, (1, 2)).any() or np.issubdtype(
        np.asarray(record.projections).dtype, np.integer
    )
    return key, bool(raw)


def _frames_in(record: ProjectionDataset, file: Path) -> np.ndarray:
    """``record``'s frames; a reconstruction saved without its scan has none to load."""
    if record.projections is None:
        raise ValueError(
            f"{file} holds a reconstruction saved without its scan: tj.load_reconstruction reads it"
        )
    return record.projections


def _record_scan(
    file: Path,
    record: ProjectionDataset,
    key: np.ndarray,
    *,
    apply_poses: bool,
    **overrides: Any,
) -> Scan:
    """A TomoJAX record's sample views of line integrals, with any caller metadata."""
    _frames_in(record, file)
    if all(value is None for value in overrides.values()):
        scan = scan_from_record(record, apply_poses=apply_poses)
        if not np.any(key != 0):
            return scan
        # The file's sample frames, not a selection made by the caller: no record.
        return replace(scan.selected(key == 0), corrections=scan.corrections)
    frames = _record_frames(file, record, apply_poses=apply_poses, **overrides)
    meta = frames.source.geometry_metadata if frames.source is not None else {}
    saved = meta.get("corrections") or ()
    from tomojax.corrections import Correction

    return Scan(
        projections=np.asarray(frames.counts),
        geometry=frames.geometry,
        name=frames.name,
        corrections=tuple(Correction.from_dict(c) for c in saved),
        source=frames.source,
    )


def _corrected_on_load(frames: Frames, file: Path) -> Scan:
    from tomojax.core.host_volume import scratch_directory

    scan = _require_calibration(frames).corrected()
    LOG.info("%s: corrected detector frames to line integrals: %s", file.name,
             ", ".join(str(c) for c in scan.corrections))  # fmt: skip
    directory = scratch_directory(scan.projections)
    if directory is not None:
        LOG.info("%s: the corrected stack (%.1f GiB) is held in a temporary file in %s, "
                 "as more than half of host memory", file.name,
                 scan.projections.nbytes / 2**30, directory)  # fmt: skip
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
    exposure_times : numpy.ndarray or float or None, optional
        Sample exposure durations in seconds, one per view or a common duration.
    flat_exposure_times : numpy.ndarray or float or None, optional
        Flat exposure durations in seconds, one per flat or a common duration.
    dark_source : str or None, optional
        Where the file states that the instrument removed the dark before writing the
        counts (``darks`` is then a zero frame, and the correction record names it).
    linear_limit : tuple of (float, str) or None, optional
        The camera's stated upper count of its linear range and where the file states
        it. Counts above it are reported in the correction record, not changed.
    name : str, optional
        Sample name.
    corrections : tuple of Correction, optional
        How these frames were cut down (``selected``, ``cropped``), carried into the
        corrected scan's records.
    source : ProjectionDataset or None, optional
        Original dataset metadata retained when loading from a file.

    Raises
    ------
    ValueError
        If frame shape or view-position count does not match the geometry.
    """

    counts: Any
    geometry: ScanGeometry
    _: KW_ONLY
    flats: Any = None
    darks: Any = None
    flat_positions: np.ndarray | None = None
    white_level: float | None = None
    view_positions: np.ndarray | None = None
    exposure_times: np.ndarray | float | None = None
    flat_exposure_times: np.ndarray | float | None = None
    dark_source: str | None = None
    linear_limit: tuple[float, str] | None = None
    name: str = "sample"
    corrections: tuple[Correction, ...] = ()
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
        if self.dark_source is not None:
            parts.append("dark removed by the instrument")
        else:
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
        self,
        *steps: corrections.Step,
        epsilon: float = 1e-6,
        views_per_batch: int | None = None,
        binning: int = 1,
    ) -> Scan:
        """Correct detector counts to line integrals with the requested steps.

        Steps run by the data they act on: counts steps first, then the flat
        and dark fields, transmission steps, the log, and line-integral steps,
        each in the order given (see :mod:`tomojax.corrections`). ``epsilon``
        must be finite and positive; it bounds the flat field and transmission
        below. Views are corrected on the device in batches of ``views_per_batch``,
        a positive integer. Automatic view batches target 32 MiB for simple
        pipelines, or 256 MiB including additional workspace such as the padded
        FFT arrays needed for Paganin phase retrieval.

        ``binning`` averages blocks of ``binning`` x ``binning`` detector pixels of
        the line integrals, as :meth:`Scan.binned` does, without holding the
        unbinned stack on the host unless a step such as ``RejectViews`` or a
        non-row step needs it. Pixels where the flat does not exceed the dark by
        three standard errors of their means take no absorption; the flat-field
        record counts them. Detector-derived output grids follow this resolution;
        grids supplied by the file or caller remain unchanged.

        Parameters
        ----------
        *steps : Step
            Corrections applied in domain order, then in the order supplied.
        epsilon : float, optional
            Positive finite lower bound for flat field and transmission.
        views_per_batch : int or None, optional
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
        done = corrected_data(self, steps, epsilon, views_per_batch, binning)
        everything = len(done.kept) == self.views
        geometry = self.geometry if everything else views_of(self.geometry, done.kept)
        geometry = binned_geometry(geometry, binning) if binning > 1 else geometry
        geometry = frame_output_grid(geometry, self.source, binning=binning)
        at = np.arange(self.views) + 0.5 if self.view_positions is None else self.view_positions
        positions = self.view_positions if everything else np.asarray(at, np.float64)[done.kept]
        binned = (cut("binned", binning=binning),) if binning > 1 else ()
        return Scan(
            done.projections,
            geometry,
            self.name,
            (*self.corrections, *done.records, *binned),
            self.source,
            view_positions=positions,
        )

    def cropped(self, rows: slice, cols: slice) -> Frames:
        """These frames' block of detector ``rows`` and ``cols``, the detector moved to match.

        A lazily read file reads only that block, of the flats and darks too.
        An automatically chosen axial-parallel output grid follows the window;
        an explicit grid, or a non-axial output box, is retained.

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
            geometry=frame_output_grid(rebuild(self.geometry, detector=window), self.source),
            corrections=(*self.corrections, cropped_record(rows, cols, self.geometry.detector)),
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
        counts = self.counts
        counts = (
            counts.frames_at(kept) if hasattr(counts, "frames_at") else np.asarray(counts)[kept]
        )
        return selected_frames(self, kept, counts)


def corrected_data(
    frames: Frames,
    steps: Sequence[corrections.Step],
    epsilon: float,
    views_per_batch: int | None,
    binning: int,
    *,
    output_rows: tuple[int, int] | None = None,
) -> corrections.Corrected:
    """Run the shared correction pipeline and retain its original acquisition indices."""
    from tomojax.corrections import correct_frames

    progress.check_canceled()
    bound = bind_step_geometry(steps, frames.geometry)
    source = frames.source.source_path if frames.source is not None else None
    reading = not isinstance(frames.counts, np.ndarray)
    with (
        registration(frames),
        progress.task("load", Path(source).name, frames.views, "views")
        if reading and source is not None
        else nullcontext(),
    ):
        return correct_frames(
            frames.counts,
            flats=frames.flats,
            flat_positions=frames.flat_positions,
            view_positions=frames.view_positions,
            exposure_times=frames.exposure_times,
            flat_exposure_times=frames.flat_exposure_times,
            darks=frames.darks,
            dark_source=frames.dark_source,
            linear_limit=frames.linear_limit,
            white_level=frames.white_level,
            steps=bound,
            epsilon=epsilon,
            views_per_batch=views_per_batch,
            binning=binning,
            output_rows=output_rows,
        )


def load_frames(
    path: str | PathLike[str],
    *,
    flats: np.ndarray | float | str | PathLike[str] | None = None,
    darks: np.ndarray | float | str | PathLike[str] | None = None,
    angles: Annotated[Sequence[float] | np.ndarray | None, "degree"] = None,
    detector: Detector | None = None,
    pixel_size: Annotated[float | tuple[float, float] | None, "geometry_length"] = None,
    grid: Grid | None = None,
    geometry: ScanGeometry | None = None,
    exposure_times: Annotated[np.ndarray | float | None, "s"] = None,
    flat_exposure_times: Annotated[np.ndarray | float | None, "s"] = None,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
    apply_poses: bool = True,
) -> Frames:
    """Load detector frames and acquisition metadata from supported scan formats.

    An HDF5 file's ``image_key`` marks its flats (1) and darks (2), and its
    sample, flat and dark frames are read only as they are corrected. The
    frames, key and angles are found at their NXtomo paths or as the file's only datasets of
    that name (``data``, ``image_key``, ``rotation_angle``), or their paired GDA imaging
    paths; ``data_path``, ``image_key_path`` and ``angles_path`` name them otherwise.
    Nikon, SkyScan, Zeiss and ESRF acquisitions are recognized by content and
    supply their own geometry and calibration. SkyScan logs require geometrically
    and flat-field corrected TIFF exports; their storage bit depth is not a
    measured white level, so supply ``flats=`` (frames or one level) explicitly.
    Zeiss Xradia projection files keep their embedded reference images,
    physical calibration and per-view image shifts in the geometry.
    A GDA command mislabeled as mm is read in degrees with a warning; discovered
    paths and this correction stay in source metadata.
    Generic TIFF stacks without acquisition metadata need ``angles`` (degrees) and
    ``flats`` (frames or one level). A parallel-beam file that records no pixel size
    loads in pixel units (a pitch of one; its record says ``length_unit`` "pixel"), and
    ``pixel_size`` gives the physical one; a cone-beam file needs one, since its
    distances fix the unit. A file that records the pitch but not the rotation center
    gives the middle of the detector, and its record says the center was not recorded.
    Missing dark fields require ``darks=``; use ``darks=0`` only for already
    dark-corrected counts.

    ``flats`` and ``darks`` (arrays, HDF5 stacks, TIFF files or folders, Zeiss XRM
    images, or one level for every pixel), ``angles``, ``detector``, ``grid`` and a
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
        A TIFF path is read in the same panel orientation as TIFF projections.
    darks : array-like, float, str, path-like or None, optional
        Replacement dark frames, calibration stack path or constant dark level.
    angles : array-like or None, optional
        Sample-view angles in degrees, replacing file metadata.
    detector : Detector or None, optional
        Replacement detector dimensions, pixel sizes and physical center.
    pixel_size : float, (float, float) or None, optional
        The detector's pixel size (u, v) in the geometry's length unit, when the file
        records none or another; the shape comes from the frames and the center from the
        file or the middle. Not with ``detector`` or ``geometry``.
    grid : Grid or None, optional
        Replacement reconstruction grid.
    geometry : ScanGeometry or None, optional
        Complete sample-view geometry, overriding individual metadata.
    exposure_times : numpy.ndarray or float or None, optional
        Sample exposure durations in seconds, replacing file metadata.
    flat_exposure_times : numpy.ndarray or float or None, optional
        Flat exposure durations in seconds, replacing file metadata.
    data_path : str or None, optional
        Explicit HDF5 frame dataset path; None discovers it.
    image_key_path : str or None, optional
        Explicit HDF5 sample/flat/dark key path; None discovers it.
    angles_path : str or None, optional
        Explicit HDF5 rotation-angle path; None discovers it.
    apply_poses : bool, optional
        Apply saved per-view pose corrections, and a scanner's own recorded
        per-view shifts and center shift (Zeiss); default is True. False gives
        the nominal geometry while keeping vendor shift provenance in metadata.

    Returns
    -------
    Frames
        Raw sample and calibration frames with geometry; HDF5/Zeiss/TIFF reads stay lazy.

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
    return load_raw_frames_with_declaration(
        path,
        flats=flats,
        darks=darks,
        angles=angles,
        detector=detector,
        pixel_size=pixel_size,
        grid=grid,
        geometry=geometry,
        exposure_times=exposure_times,
        flat_exposure_times=flat_exposure_times,
        data_path=data_path,
        image_key_path=image_key_path,
        angles_path=angles_path,
        apply_poses=apply_poses,
    )


def load_raw_frames_with_declaration(
    path: str | PathLike[str],
    *,
    flats: np.ndarray | float | str | PathLike[str] | None = None,
    darks: np.ndarray | float | str | PathLike[str] | None = None,
    exposure_times: np.ndarray | float | None = None,
    flat_exposure_times: np.ndarray | float | None = None,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
    apply_poses: bool = True,
    **overrides: Any,
) -> Frames:
    center, overrides["declared"] = center_of(overrides.get("declared"))
    with discard_failed_outputs() as outputs, progress.task("metadata", Path(path).name, None, ""):
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"no such file: {file}")
        paths = {
            "data_path": data_path,
            "image_key_path": image_key_path,
            "angles_path": angles_path,
        }
        overrides.setdefault("angles", None)
        frames = _frames(
            file,
            acquisition_frames(file, apply_poses=apply_poses, **paths, **overrides),
            apply_poses=apply_poses,
            **paths,
            **overrides,
        )
        outputs.own(frames.counts)
        fields = _with_fields(
            frames,
            flats=flats,
            darks=darks,
            exposure_times=exposure_times,
            flat_exposure_times=flat_exposure_times,
        )
        return centered(_require_calibration(fields), center)


def _frames(
    file: Path,
    discovered: Frames | None,
    *,
    apply_poses: bool,
    data_path: str | None,
    image_key_path: str | None,
    angles_path: str | None,
    **overrides: Any,
) -> Frames:
    """The file's detector frames: a recognized acquisition's, a TIFF stack's, or HDF5/NPZ."""
    if discovered is not None:
        return discovered
    suffix = file.suffix.lower()
    if file.is_dir() or suffix in {".tif", ".tiff"}:
        return tiff_frames(file, **overrides)
    if suffix in _HDF5:
        from tomojax.io.api import locate_frames

        located = locate_frames(
            str(file), data_path=data_path, image_key_path=image_key_path, angles_path=angles_path
        )
        return _hdf5_frames(file, located, apply_poses=apply_poses, **overrides)
    from tomojax.io import load_dataset

    return _record_frames(file, load_dataset(file), apply_poses=apply_poses, **overrides)


def _require_calibration(frames: Frames) -> Frames:
    """Missing file calibration requires a caller choice, not a numerical guess."""
    issues = calibration_issues(frames.flats, frames.darks, frames.white_level)
    if issues:
        raise ValueError(issues[0])
    return frames


def _with_fields(
    frames: Frames,
    *,
    flats: np.ndarray | float | str | PathLike[str] | None,
    darks: np.ndarray | float | str | PathLike[str] | None,
    exposure_times: np.ndarray | float | None = None,
    flat_exposure_times: np.ndarray | float | None = None,
) -> Frames:
    """``frames`` with the flat and dark fields given in place of the file's."""
    changes: dict[str, Any] = {}
    shape = tuple(int(n) for n in frames.counts.shape[1:])
    if isinstance(flats, numbers.Real):  # one level for every pixel: a white level
        changes |= {"flats": None, "flat_positions": None, "white_level": float(flats)}
    elif flats is not None:
        flat_frames = _frames_of(flats, "flats", frames.counts)
        changes |= {
            "flats": flat_frames,
            "flat_positions": None,
            "white_level": None,
            "flat_exposure_times": getattr(flat_frames, "exposure_times", None),
        }
    if isinstance(darks, numbers.Real):
        changes |= {"darks": np.broadcast_to(np.float32(darks), (1, *shape)), "dark_source": None}
    elif darks is not None:
        changes |= {"darks": _frames_of(darks, "darks", frames.counts), "dark_source": None}
    if exposure_times is not None:
        changes["exposure_times"] = exposure_times
    if flat_exposure_times is not None:
        changes["flat_exposure_times"] = flat_exposure_times
    return replace(frames, **changes)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' * (n != 1)}"


def _frames_of(value: np.ndarray | float | str | PathLike[str], name: str, counts: Any) -> Any:
    if isinstance(value, str | os.PathLike):
        from tomojax.io.api import read_calibration_frames

        # A vendor's TIFF panel origin (Nikon, SkyScan) applies to its flats too.
        return read_calibration_frames(Path(value), like=counts)
    frames = np.asarray(value)
    if frames.ndim == 2:
        frames = frames[None]
    if frames.ndim != 3:
        raise ValueError(f"{name} must be frames (frames, rows, columns), not {frames.shape}")
    return frames


_HDF5 = frozenset({".nx", ".nxs", ".h5", ".hdf", ".hdf5"})


def _located(file: Path, **paths: str | None) -> LocatedFrames | None:
    """The file's frames, key and angles (at the given or usual places), or None if not found."""
    from tomojax.io.api import locate_frames

    try:
        return locate_frames(str(file), **paths)
    except KeyError:
        return None


def _hdf5_frames(
    file: Path,
    located: LocatedFrames,
    *,
    angles: Sequence[float] | np.ndarray | None,
    apply_poses: bool,
    layout_only: bool = False,
    detector: Detector | None = None,
    pixel_size: float | tuple[float, float] | None = None,
    grid: Grid | None = None,
    geometry: ScanGeometry | None = None,
    declared: DeclaredGeometry | None = None,
) -> Frames:
    """The sample frames, flats and darks of an HDF5 file, split by ``image_key``.

    The geometry is the NXtomo file's (``tomojax`` metadata) when it has one,
    else incomplete until the caller supplies detector/geometry metadata.
    """
    from tomojax.io import ProjectionDataset
    from tomojax.io.api import LoadedNXTomo, NXTomoMetadata

    stack = located.stack
    key = located.image_key
    key = np.zeros(len(stack), np.int32) if key is None else key
    sample = key == 0
    if not sample.any():
        raise ValueError(f"{file} holds no sample frames (image_key 0)")
    views, rows, cols = int(sample.sum()), stack.rows, stack.cols
    if geometry is not None and angles is None:
        angles = np.asarray(cast("Any", geometry).angles)
    if angles is not None:
        view_angles = np.asarray(angles, np.float64)
    elif located.angles is not None:
        view_angles = located.angles[sample]
    else:
        raise ValueError(
            f"{file} records no rotation angles: pass angles=... (degrees), "
            "or use tomojax correct --angles"
        )
    if view_angles.shape != (views,):
        raise ValueError(f"{file} has {views} sample frames but {view_angles.size} angles")
    try:
        metadata = read_nxtomo(str(file), lazy=True, layout_only=layout_only).metadata
    except KeyError:  # not laid out as NXtomo: the frames alone
        metadata = NXTomoMetadata(
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
    record, resolved = frame_geometry(
        ProjectionDataset.from_nxtomo(placeholder, source_path=file),
        detector=detector,
        pixel_size=pixel_size,
        grid=grid,
        geometry=geometry,
        apply_poses=apply_poses,
        declared=declared,
    )
    flats = stack.frames_at(key == 1) if (key == 1).any() else None
    darks = stack.frames_at(key == 2) if (key == 2).any() else None
    if located.flats is not None:  # kept apart from the frames: one set
        flats = located.flats
    if located.darks is not None:
        darks = located.darks
    return Frames(
        counts=stack.frames_at(sample),
        geometry=resolved,
        flats=flats,
        darks=darks,
        flat_positions=None
        if flats is None or located.flats is not None
        else np.cumsum(sample)[key == 1],
        name=record.sample_name or "sample",
        source=record,
    )


def _saved_image_key(record: ProjectionDataset) -> np.ndarray:
    """The record's saved frame labels; every frame is a sample view without any."""
    key = record.image_key
    return np.zeros(len(record.angles), np.int32) if key is None else np.asarray(key)


def _record_frames(
    file: Path,
    record: ProjectionDataset,
    *,
    apply_poses: bool,
    stack: Any = None,
    angles: Sequence[float] | np.ndarray | None = None,
    detector: Detector | None = None,
    pixel_size: float | tuple[float, float] | None = None,
    grid: Grid | None = None,
    geometry: ScanGeometry | None = None,
    declared: DeclaredGeometry | None = None,
) -> Frames:
    """A TomoJAX record's sample frames, flats and darks, split by its saved ``image_key``."""
    key = _saved_image_key(record)
    sample = key == 0
    if not sample.any():
        raise ValueError(f"{file} holds no sample frames (image_key 0)")
    stack = _frames_in(record, file) if stack is None else stack
    samples = stack
    if not sample.all():
        frames = len(key)
        samples = stack.frames_at(sample) if hasattr(stack, "frames_at") else stack[sample]

        def views(values: np.ndarray | None) -> np.ndarray | None:
            if values is None or np.ndim(values) == 0 or len(values) != frames:
                return values
            return values[sample]

        from tomojax.io.api import replace_dataset

        metadata = record.to_nxtomo_metadata()
        metadata.image_key = None
        record = replace_dataset(
            record,
            projections=np.broadcast_to(np.zeros((), stack.dtype), samples.shape)
            if hasattr(stack, "frames_at")
            else samples,
            angles=views(record.angles),
            angle_offset=views(record.angle_offset),
            align_params=views(record.align_params),
            metadata=metadata,
        )
    record, resolved = frame_geometry(
        record,
        angles=angles,
        detector=detector,
        pixel_size=pixel_size,
        grid=grid,
        geometry=geometry,
        apply_poses=apply_poses,
        declared=declared,
    )

    def labeled(label: int) -> np.ndarray | FrameStack | None:
        mask = key == label
        if not mask.any():
            return None
        return stack.frames_at(mask) if hasattr(stack, "frames_at") else stack[mask]

    flats = labeled(1)
    return Frames(
        samples,
        resolved,
        flats=flats,
        darks=labeled(2),
        flat_positions=None if flats is None else np.cumsum(sample)[key == 1],
        name=record.sample_name or "sample",
        source=record,
    )


__all__ = ["Frames", "load", "load_frames"]
