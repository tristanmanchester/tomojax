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
from dataclasses import KW_ONLY, dataclass, field, fields, replace
import logging
import numbers
import os
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, Unpack

import numpy as np

from tomojax import progress
from tomojax._data import Hdf5Frames, calibration_issues, read_projections, views_of
from tomojax._loading.frame_sources import (
    line_integral_frames,
    selected_frames,
    tiff_frames,
)
from tomojax._loading.geometry import center_of, centered, frame_geometry
from tomojax._parameters import FLAT_WEIGHTS, AnnotationModule, FlatWeighting
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
    record_of,
    scan_from_record,
    selected_scan,
)
from tomojax.core.host_volume import discard_failed_outputs
from tomojax.core.registration import registration
from tomojax.core.validation import view_indices
from tomojax.geometry import Detector, Grid, ScanGeometry
from tomojax.io.projection_layout import ProjectionStorage, select_storage

if TYPE_CHECKING:
    from tomojax import corrections
    from tomojax._loading.geometry import DeclaredGeometry
    from tomojax.corrections import Correction
    from tomojax.io import ProjectionDataset
    from tomojax.io.projection_layout import ProjectionPixels, SelectedStorage
else:
    corrections = AnnotationModule("tomojax.corrections")

if TYPE_CHECKING:

    class _FrameOverrides(TypedDict, total=False):
        detector: Detector | None
        pixel_size: float | tuple[float, float] | None
        grid: Grid | None
        geometry: ScanGeometry | None
        declared: DeclaredGeometry | None


LOG = logging.getLogger(__name__)


def is_raw_input(path: Path) -> bool:
    """Whether the selected storage records detector counts, before requiring geometry."""
    selected = select_storage(path, layout_only=True)
    return not isinstance(selected, ProjectionStorage) or _Calibration().counts(
        selected, require_raw=False
    )


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

    A file of raw detector frames (an NXtomo or NeXus scan file with an
    ``image_key`` marking flats or darks, Data Exchange, a Nikon, SkyScan,
    Zeiss or ESRF acquisition, or a TIFF stack) is
    corrected with the standard chain, as ``load_frames(path, ...).corrected()``:
    dark subtraction, flat division and ``-log``; :attr:`Scan.corrections`
    records it. Bracketing multi-frame flat sets are weighted from a sample of
    the views by default while interpolating common row gain by position, and a
    warning is logged when views move the flat by more than 1% over more than
    1% of the detector; pass another scan's file as ``flats=`` to borrow its
    flat frames, or use ``load_frames(path, ...).corrected(flat_weights="position")``
    to blend by position alone. For other corrections, load the frames with
    :func:`load_frames`. Every other argument has :func:`load_frames`'s meaning
    and can replace recorded metadata. Explicit flats or darks identify unlabeled
    projections as counts regardless of storage dtype. A recorded log correction
    or phase measurement rules out further flat/dark correction. The scan carries the per-view
    corrections saved with it (by :func:`align` or ``tomojax align``) unless
    ``apply_poses`` is False. Frames that ``image_key`` marks as other than
    sample views are left out.

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
        Apply per-view poses. On a raw Zeiss acquisition, False also omits the
        scanner's shifts and center correction. A saved scan that incorporated
        those corrections requires the original acquisition for nominal geometry.
        Use ``scan.with_poses(None)`` to clear its rigid poses.

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


def load_scan_with_declaration(path: str | PathLike[str], **options: Any) -> Scan:
    with discard_failed_outputs() as outputs:
        item = load_native(path, **options)
        scan = _corrected_on_load(item, Path(path)) if isinstance(item, Frames) else item
        outputs.own(scan.projections)
        return scan


@dataclass(frozen=True)
class _StoredScan:
    """Complete saved scan state, paired with the selected pixel source."""

    scan: Scan
    pixels: ProjectionPixels

    def materialize(self) -> Scan:
        from tomojax.io.api import replace_dataset

        data = (
            read_projections(self.pixels)
            if isinstance(self.pixels, Hdf5Frames)
            else self.pixels
            if isinstance(self.pixels, np.ndarray)
            else self.pixels[:]
        )
        source = self.scan.source
        if source is not None:
            source = replace_dataset(source, projections=data)
        return replace(self.scan, projections=data, source=source)


def load_native(path: str | PathLike[str], **options: Any) -> Scan | Frames:
    """Open the selected measurements in their resolved domain."""
    file = Path(path)
    progress.check_canceled()
    if not file.exists():
        with progress.task("load", file.name, None, "views"):
            raise FileNotFoundError(f"no such file: {file}")
    with discard_failed_outputs() as outputs:
        item = resolve_input(file, layout_only=False, **options)
        if isinstance(item, Frames):
            return item
        outputs.own(item.pixels)
        return item.materialize()


def projection_frames(
    file: Path, *, require_raw: bool = False, **options: Any
) -> tuple[Frames, bool]:
    """Resolve the same scan as an eager load, keeping its detector storage lazy."""
    item = resolve_input(file, layout_only=True, require_raw=require_raw, **options)
    if isinstance(item, Frames):
        return item, True
    return line_integral_frames(item.scan, item.pixels), False


def check_geometry_options(
    angles_path: str | None,
    angles: Sequence[float] | np.ndarray | None,
    overrides: _FrameOverrides,
) -> None:
    """Check every supplied description of the same part of the geometry for agreement."""
    geometry = overrides.get("geometry")
    if geometry is not None:
        if angles is not None and not np.array_equal(angles, geometry.angles):
            raise ValueError("angles disagree with the supplied geometry")
        for name, value, supplied in (
            ("detector", overrides.get("detector"), geometry.detector),
            ("grid", overrides.get("grid"), geometry.grid),
        ):
            if value is not None and value != supplied:
                raise ValueError(f"{name} disagrees with the supplied geometry")
    if angles_path is not None and (angles is not None or geometry is not None):
        raise ValueError(
            "angles_path and angles or geometry both supply rotation angles; choose one"
        )
    if overrides.get("pixel_size") is not None and (
        geometry is not None or overrides.get("detector") is not None
    ):
        raise ValueError(
            "pixel_size= and detector= (or geometry=) both describe the detector: pass one"
        )


def resolve_input(
    file: Path,
    *,
    layout_only: bool,
    require_raw: bool = False,
    flats: np.ndarray | float | str | PathLike[str] | None = None,
    darks: np.ndarray | float | str | PathLike[str] | None = None,
    exposure_times: np.ndarray | float | None = None,
    flat_exposure_times: np.ndarray | float | None = None,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
    angles: Sequence[float] | np.ndarray | None = None,
    apply_poses: bool = True,
    **overrides: Unpack[_FrameOverrides],
) -> Frames | _StoredScan:
    """Resolve the file selection, geometry and measurement domain for all consumers."""
    check_geometry_options(angles_path, angles, overrides)
    selected = select_storage(
        file,
        layout_only=layout_only,
        data_path=data_path,
        image_key_path=image_key_path,
        angles_path=angles_path,
    )
    return resolve_measurements(
        selected,
        _Calibration(flats, darks, exposure_times, flat_exposure_times),
        require_raw=require_raw,
        angles=angles,
        apply_poses=apply_poses,
        **overrides,
    )


def loading_record(file: Path) -> ProjectionDataset | None:
    """Read saved metadata directly. Raw inputs need their acquisition geometry resolved."""
    selected = select_storage(file, layout_only=True)
    calibration = _Calibration()
    if isinstance(selected, ProjectionStorage) and not calibration.counts(
        selected, require_raw=False
    ):
        return selected.record
    item = resolve_measurements(selected, calibration, check_calibration=False)
    return item.source if isinstance(item, Frames) else item.scan.source


def resolve_measurements(
    selected: SelectedStorage,
    calibration: _Calibration,
    *,
    require_raw: bool = False,
    check_calibration: bool = True,
    angles: Sequence[float] | np.ndarray | None = None,
    apply_poses: bool = True,
    **overrides: Unpack[_FrameOverrides],
) -> Frames | _StoredScan:
    """Bind the requested geometry and calibration to the selected measurements."""
    center, overrides["declared"] = center_of(overrides.get("declared"))
    if isinstance(selected, Path):
        frames = tiff_frames(selected, angles=angles, **overrides)
    elif isinstance(selected, ProjectionStorage):
        raw = calibration.counts(selected, require_raw=require_raw)
        with discard_failed_outputs() as outputs:
            outputs.own(selected.stack)
            scan, samples = _sample_scan(
                selected, raw=raw, angles=angles, apply_poses=apply_poses, **overrides
            )
            if not raw:
                return _StoredScan(centered(scan, center), samples)
            frames = _calibration_frames(selected, scan, samples)
    else:
        record, resolved = frame_geometry(
            selected.nominal_record,
            offsets=selected.offsets,
            angles=angles,
            apply_poses=apply_poses,
            **overrides,
        )
        frames = Frames(
            counts=selected.counts,
            geometry=resolved,
            flats=selected.flats,
            darks=selected.darks,
            flat_positions=selected.flat_positions,
            white_level=selected.white_level,
            view_positions=selected.view_positions,
            exposure_times=selected.exposure_times,
            flat_exposure_times=selected.flat_exposure_times,
            dark_source=selected.dark_source,
            linear_limit=selected.linear_limit,
            name=record.sample_name or "sample",
            source=record,
        )
    frames = calibration.apply(frames)
    if check_calibration:
        from tomojax.corrections.calibration import check_calibration_layout

        issues = calibration_issues(frames.flats, frames.darks, frames.white_level)
        if issues:
            raise ValueError(issues[0])
        check_calibration_layout(frames)
    return centered(frames, center)


def _corrected_on_load(frames: Frames, file: Path) -> Scan:
    from tomojax.core.host_volume import scratch_directory

    scan = frames.corrected()
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
    (sets taken before and after a scan are weighted by the views); ``darks``
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
        flat_weights: FlatWeighting = FLAT_WEIGHTS,
        views_per_batch: int | None = None,
        binning: int = 1,
    ) -> Scan:
        """Correct detector counts to line integrals with the requested steps.

        Steps run by the data they act on: counts steps first, then the flat
        and dark fields, transmission steps, the log, and line-integral steps,
        each in the order given (see :mod:`tomojax.corrections`). ``epsilon``
        must be finite and positive; it bounds the flat field and transmission
        below. Views are corrected on the device in batches of ``views_per_batch``,
        a positive integer; the active bracketing flat sets stay on the device
        across consecutive batches that share them. Automatic view batches target
        32 MiB for simple pipelines, or 256 MiB including additional workspace
        such as the padded FFT arrays needed for Paganin phase retrieval.
        Unfiltered HDF5 stacks are read in parallel positional blocks, and
        calibration means and host output copies run across multiple CPU threads.

        ``binning`` averages blocks of ``binning`` x ``binning`` detector pixels of
        the line integrals, as :meth:`Scan.binned` does, without holding the
        unbinned stack on the host unless a step such as ``RejectViews`` or a
        non-row step needs it. Pixels where the flat does not exceed the dark by
        three standard errors of their means take no absorption; the flat-field
        record counts them. Detector-derived output grids follow this resolution;
        grids supplied by the file or caller remain unchanged.

        With ``flat_weights="views"`` (the default), bracketing multi-frame flat
        sets are weighted for local texture from a sample of views while common
        row gain is interpolated by position and flat level stays continuous
        across one-sided edges; single-frame sets interpolate by position. The
        ``flat_dark`` record stores ``flat_weights`` and, when views set weights,
        ``flat_weights_from_views``, ``later_flat_weight``,
        ``flat_structure_unmatched`` and ``flat_moved_by_views``. When views move
        the flat by more than 1% over more than 1% of the detector, a warning
        names both remedies: borrowing another scan's flats with ``flats=`` on
        :func:`load_frames`, or passing ``flat_weights="position"`` to blend by
        position alone.

        Parameters
        ----------
        *steps : Step
            Corrections applied in domain order, then in the order supplied.
        epsilon : float, optional
            Positive finite lower bound for flat field and transmission.
        flat_weights : {'views', 'position'}, optional
            How a view between two flat sets blends them: by what a sample of the views
            shows of each set (see :mod:`tomojax.corrections`), or by its position alone.
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
        done = corrected_data(self, steps, epsilon, views_per_batch, binning, flat_weights)
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
        its flat is weighted at the same position.

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
    flat_weights: FlatWeighting,
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
            flat_weights=flat_weights,
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
    sample, flat, and dark frames are read only as they are corrected.
    Discovery checks fixed NXtomo, Data Exchange, and paired GDA imaging
    paths, then the NeXus structure (the default plot's frames, the
    ``image_key`` of their ``NXdetector``, and the plotted angular coordinate
    along the frames), and finally unique datasets named ``data``,
    ``image_key``, or ``rotation_angle``; pass ``data_path``,
    ``image_key_path``, or ``angles_path`` to override discovery. When a key
    or angle dataset paired by path or structure has more or fewer entries
    than the frame stack (as in a scan stopped early), the common prefix from
    frame 0 is read with a warning and the counts are stored in
    ``frame_records``; datasets matched only by name must match the stack
    length.
    Nikon, SkyScan, Zeiss and ESRF acquisitions are recognized by content and
    supply their own geometry and calibration. SkyScan logs require geometrically
    and flat-field corrected TIFF exports; their storage bit depth is not a
    measured white level, so supply ``flats=`` (frames or one level) explicitly.
    Zeiss Xradia projection files keep their embedded reference images,
    physical calibration and per-view image shifts in the geometry.
    A GDA command mislabeled as mm is read in degrees with a warning; discovered
    paths and this correction stay in source metadata.
    Generic TIFF stacks without acquisition metadata need ``angles`` (degrees) and
    ``flats`` (frames or one level). A parallel-beam file with no recorded pixel
    size loads in pixel units (pitch 1, ``length_unit`` "pixel"), and
    ``pixel_size`` sets physical units; a cone-beam file requires a physical
    pixel size because its distances fix the unit. When the rotation center is
    not recorded, it defaults to the middle of the detector with
    ``detector_center_status`` "not_recorded". A file that records no rotation
    axis or pixel scale logs a warning (``geometry_status`` "not_recorded") and
    is refused by reconstruction and alignment until ``geometry`` is declared.
    Missing dark fields require ``darks=``; use ``darks=0`` only for already
    dark-corrected counts.

    ``flats`` and ``darks`` (arrays, HDF5 stacks, TIFF files or folders, Zeiss XRM
    images, or one level for every pixel), ``angles``, ``detector``, ``grid`` and a
    whole ``geometry`` (of every sample view; then TIFFs need no ``angles``)
    replace what the file holds. Dataset paths select arrays inside saved HDF5
    files too. Bliss accepts ``data_path`` for its sequence. Fixed-layout vendor
    formats, TIFF, NPZ and MATLAB phase files reject HDF5 dataset paths.
    HDF5 calibration paths use the same automatic
    dataset discovery as the input and stay lazy: another scan's file supplies its
    flat or dark frames, and a stack that labels none supplies every frame.

    Parameters
    ----------
    path : str or path-like
        HDF5, NPZ, Nikon, SkyScan, Zeiss or ESRF acquisition, or generic TIFF stack.
    flats : array-like, float, str, path-like or None, optional
        Replacement flat frames, calibration stack path (another scan's file
        supplies its flat frames) or constant open-beam level. A TIFF path is
        read in the same panel orientation as TIFF projections.
    darks : array-like, float, str, path-like or None, optional
        Replacement dark frames, calibration stack path (another scan's file
        supplies its dark frames) or constant dark level.
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
        Apply per-view poses. On a raw Zeiss acquisition, False also omits the
        scanner's shifts and center correction. A saved scan that incorporated
        those corrections requires the original acquisition for nominal geometry.
        Use ``scan.with_poses(None)`` to clear its rigid poses.

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


def load_raw_frames_with_declaration(path: str | PathLike[str], **options: Any) -> Frames:
    with discard_failed_outputs() as outputs, progress.task("metadata", Path(path).name, None, ""):
        frames, _ = projection_frames(Path(path), require_raw=True, **options)
        outputs.own(frames.counts)
        return frames


@dataclass(frozen=True)
class _Calibration:
    flats: np.ndarray | float | str | PathLike[str] | None = None
    darks: np.ndarray | float | str | PathLike[str] | None = None
    exposure_times: np.ndarray | float | None = None
    flat_exposure_times: np.ndarray | float | None = None

    def counts(self, storage: ProjectionStorage, *, require_raw: bool) -> bool:
        """Resolve count-domain requests against the file's measurement record."""
        explicit = next((f.name for f in fields(self) if getattr(self, f.name) is not None), None)
        meta = storage.record.geometry_metadata
        logged = any(step["name"] == "log" for step in meta.get("corrections", ()))
        if logged or meta.get("quantity") == "phase":
            if explicit or require_raw:
                raise ValueError(
                    f"{explicit or 'load_frames'} requires counts; the file records line integrals"
                )
            return False
        raw = storage.counts or require_raw or self.flats is not None or self.darks is not None
        if explicit and not raw:
            raise ValueError(f"{explicit} requires raw detector counts and flat frames")
        return raw

    def apply(self, frames: Frames) -> Frames:
        """Replace the supplied fields and require the calibration the selected counts need."""
        flats, darks = self.flats, self.darks
        exposure_times, flat_exposure_times = self.exposure_times, self.flat_exposure_times
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
            changes |= {
                "darks": np.broadcast_to(np.float32(darks), (1, *shape)),
                "dark_source": None,
            }
        elif darks is not None:
            changes |= {"darks": _frames_of(darks, "darks", frames.counts), "dark_source": None}
        if exposure_times is not None:
            changes["exposure_times"] = exposure_times
        if flat_exposure_times is not None:
            changes["flat_exposure_times"] = flat_exposure_times
        fields = replace(frames, **changes)
        if fields.flats is None and (exposure_times is not None or flat_exposure_times is not None):
            raise ValueError(
                "exposure_times and flat_exposure_times require flat frames, not a white level"
            )
        return fields


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' * (n != 1)}"


def _frames_of(value: np.ndarray | float | str | PathLike[str], name: str, counts: Any) -> Any:
    if isinstance(value, str | os.PathLike):
        from tomojax.io.api import read_calibration_frames

        # A vendor's TIFF panel origin (Nikon, SkyScan) applies to its flats too.
        return read_calibration_frames(Path(value), like=counts, image_key=1 + (name == "darks"))
    frames = np.asarray(value)
    if frames.ndim == 2:
        frames = frames[None]
    if frames.ndim != 3:
        raise ValueError(f"{name} must be frames (frames, rows, columns), not {frames.shape}")
    return frames


def stored_scan(
    storage: ProjectionStorage,
    samples: ProjectionPixels,
    *,
    angles: Sequence[float] | np.ndarray | None,
    apply_poses: bool,
    **overrides: Unpack[_FrameOverrides],
) -> Scan:
    """Resolve geometry on the saved view axis, then select the complete scan state."""
    record, sample = storage.record, storage.key == 0
    if angles is not None and not sample.all():
        expanded = np.zeros(len(sample), np.float32)
        expanded[sample] = angles
        angles = expanded
    geometry = None
    if angles is not None or any(value is not None for value in overrides.values()):
        record, geometry = frame_geometry(
            record, angles=angles, apply_poses=apply_poses, **overrides
        )
    scan = scan_from_record(record, apply_poses=apply_poses, geometry=geometry)
    if sample.all():
        return scan
    shape = np.broadcast_to(np.zeros((), samples.dtype), samples.shape)
    scan = replace(selected_scan(scan, np.flatnonzero(sample), shape), corrections=scan.corrections)
    return replace(scan, source=record_of(scan))


def _sample_scan(
    selected: ProjectionStorage,
    *,
    raw: bool,
    apply_poses: bool,
    angles: Sequence[float] | np.ndarray | None,
    **overrides: Unpack[_FrameOverrides],
) -> tuple[Scan, ProjectionPixels]:
    """Select per-view records on the same axis as the pixels before resolving geometry."""
    record, stack, key = selected.record, selected.stack, selected.key
    sample = key == 0
    if not sample.any():
        raise ValueError(f"{record.source_path} holds no sample frames (image_key 0)")
    samples = stack if sample.all() else selected.frames_at(sample)
    geometry = overrides.get("geometry")
    if angles is not None and np.shape(angles) != (int(sample.sum()),):
        raise ValueError(f"angles needs one entry per sample view ({int(sample.sum())})")
    if not raw and geometry is None:
        return stored_scan(
            selected, samples, angles=angles, apply_poses=apply_poses, **overrides
        ), samples
    if not sample.all():
        frames = len(key)

        def views(values: np.ndarray | None) -> np.ndarray | None:
            if values is None or np.ndim(values) == 0 or len(values) != frames:
                return values
            return values[sample]

        from tomojax.io.api import replace_dataset

        meta = dict(record.geometry_metadata)
        positions = meta.get("view_positions")
        if positions is None and not selected.counts:
            positions = np.arange(frames) + 0.5
        if positions is not None and len(positions) == frames:
            meta["view_positions"] = np.asarray(positions)[sample].tolist()
        metadata = record.to_nxtomo_metadata()
        metadata.image_key = None
        record = replace_dataset(
            record,
            projections=np.broadcast_to(np.zeros((), stack.dtype), samples.shape)
            if not isinstance(samples, np.ndarray)
            else samples,
            angles=views(record.angles),
            angle_offset=views(record.angle_offset),
            align_params=views(record.align_params),
            geometry_metadata=meta,
            metadata=metadata,
        )
    if geometry is not None:
        angles = geometry.angles
    record, resolved = frame_geometry(record, angles=angles, apply_poses=apply_poses, **overrides)
    return scan_from_record(record, apply_poses=apply_poses, geometry=resolved), samples


def _calibration_frames(
    selected: ProjectionStorage, scan: Scan, samples: ProjectionPixels
) -> Frames:
    key = selected.key

    def labeled(label: int) -> ProjectionPixels | None:
        mask = key == label
        if not mask.any():
            return None
        return selected.frames_at(mask)

    flats = labeled(1)
    darks = labeled(2)
    positions = None if flats is None else np.cumsum(key == 0)[key == 1]
    if selected.located is not None:
        if selected.located.flats is not None:
            flats, positions = selected.located.flats, None
        if selected.located.darks is not None:
            darks = selected.located.darks
    return replace(
        line_integral_frames(scan, samples), flats=flats, darks=darks, flat_positions=positions
    )


__all__ = ["Frames", "load", "load_frames"]
