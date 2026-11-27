"""Public projection dataset loading and saving facade."""
# pyright: reportUnknownMemberType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
from zipfile import BadZipFile

import numpy as np

from tomojax._data import (
    LoadedGeometryMeta,
    LoadedNXTomo,
    LocatedFrames,
    NXTomoMetadata,
    build_geometry_from_meta,
    calibration_issues,
    load_npz,
    load_nxtomo,
    locate_frames,
    save_npz,
    save_nxtomo,
    saved_volume_axes,
    saved_without_frames,
    validate_nxtomo,
)
from tomojax.core.geometry import Detector, Grid, ScanGeometry
from tomojax.errors import InvalidParameter
from tomojax.io._tiff import TIFF_SUFFIXES, read_tiff_frames

__all__ = [
    "LoadedNXTomo",
    "LocatedFrames",
    "NXTomoMetadata",
    "ProjectionDataset",
    "ValidationReport",
    "build_geometry_from_dataset_metadata",
    "convert_dataset",
    "load_dataset",
    "load_nxtomo",
    "load_tiff_stack",
    "locate_frames",
    "save_dataset",
    "save_nxtomo",
    "saved_volume_axes",
    "saved_without_frames",
    "validate_dataset",
    "validate_nxtomo",
]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import jax

    from tomojax.core.json_values import JsonValue

type PathLike = str | os.PathLike[str]

_HDF5_SUFFIXES = {".nx", ".nxs", ".h5", ".hdf", ".hdf5"}
_TIFF_SUFFIXES = TIFF_SUFFIXES


class ValidationReport(TypedDict):
    """Public dataset validation result.

    Attributes
    ----------
    issues : list of str
        Validation problems; an empty list means none were found.
    """

    issues: list[str]


def build_geometry_from_dataset_metadata(
    meta: Mapping[str, Any],
    *,
    grid_override: Grid | tuple[int, int, int] | list[int] | None = None,
    apply_poses: bool = False,
    volume_shape: Sequence[int] | None = None,
) -> tuple[Grid, Detector, ScanGeometry]:
    """Build geometry objects from normalized dataset metadata.

    ``apply_poses`` applies the per-view corrections and angle offsets saved with
    the data. Keeping this wrapper in `tomojax.io` makes the command/data dependency
    explicit and gives the `ProjectionDataset` solver contract one replacement
    point.

    Parameters
    ----------
    meta : mapping
        Normalized geometry metadata, as returned by geometry_inputs.
    grid_override : Grid, tuple of int, list of int or None, optional
        Replacement grid or (nx, ny, nz) dimensions.
    apply_poses : bool, optional
        Apply saved per-view alignment and angle corrections; defaults to False.
    volume_shape : sequence of int or None, optional
        Fallback (nz, ny, nx) array dimensions when no grid is recorded.

    Returns
    -------
    grid : Grid
        Resolved reconstruction grid.
    detector : Detector
        Resolved detector geometry.
    geometry : ScanGeometry
        Per-view geometry, with saved corrections when requested.

    Raises
    ------
    ValueError
        If geometry metadata are incomplete or inconsistent.
    """
    return build_geometry_from_meta(
        cast("LoadedGeometryMeta", dict(meta)),
        grid_override=grid_override,
        apply_poses=apply_poses,
        volume_shape=volume_shape,
    )


@dataclass(slots=True)
class ProjectionDataset:
    """Standard public in-memory projection dataset.

    This is the current IO boundary. It contains measured projection data,
    normalized metadata needed by reconstruction/alignment code, and the
    optional reconstructed volume when a processed container stores one.
    ``angles`` holds each view's rotation angle, in degrees.

    Parameters
    ----------
    projections : numpy.ndarray or None
        Measured projections in (view, v, u) order; None in a reconstruction saved
        without its scan.
    angles : numpy.ndarray
        One rotation angle per view, in degrees.
    volume : numpy.ndarray, jax.Array or None, optional
        Reconstructed volume in (z, y, x) order, when present.
    detector : Detector or None, optional
        Detector pixel counts, pitches and physical center.
    grid : Grid or None, optional
        Reconstruction voxel grid.
    geometry_type : str, optional
        Geometry family, such as parallel, lamino or cone.
    geometry_metadata : dict, optional
        Additional geometry and acquisition metadata.
    angle_offset : numpy.ndarray or None, optional
        Per-view angular corrections in degrees.
    align_params : numpy.ndarray or None, optional
        Saved per-view alignment parameters, with convention in align_gauge.
    align_gauge : dict or None, optional
        Gauge and frame metadata for saved alignment parameters.
    misalign_spec : dict or None, optional
        Simulation misalignment record; angles already include simulated offsets.
    source_path : str or None, optional
        Absolute input path retained as provenance.
    source_format : str or None, optional
        Original container format.
    sample_name : str or None, optional
        Sample name retained on output.
    """

    projections: np.ndarray | None
    angles: np.ndarray
    volume: np.ndarray | jax.Array | None = None
    detector: Detector | None = None
    grid: Grid | None = None
    geometry_type: str = "parallel"
    geometry_metadata: dict[str, Any] = field(default_factory=dict)
    angle_offset: np.ndarray | None = None
    align_params: np.ndarray | None = None
    align_gauge: dict[str, JsonValue] | None = None
    # Simulated misalignment, whose angle offsets ``angles`` already include.
    misalign_spec: dict[str, JsonValue] | None = None
    source_path: str | None = None
    source_format: str | None = None
    sample_name: str | None = None
    _metadata: NXTomoMetadata | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.source_path is not None:
            self.source_path = str(Path(self.source_path).resolve())

    @classmethod
    def from_nxtomo(
        cls,
        payload: LoadedNXTomo,
        *,
        source_path: PathLike | None = None,
    ) -> ProjectionDataset:
        """Build a public dataset from the lower-level NXtomo payload.

        Parameters
        ----------
        payload : LoadedNXTomo
            NXtomo projection payload and persistence metadata.
        source_path : str, pathlib.Path or None, optional
            Input path retained as provenance.

        Returns
        -------
        ProjectionDataset
            Normalized in-memory dataset with arrays and typed grid/detector metadata.

        Raises
        ------
        ValueError
            If grid or detector metadata have invalid coordinate dimensions.
        """
        metadata = payload.metadata
        detector = metadata.detector
        if isinstance(detector, dict) and not {"du", "dv", "det_center"} <= detector.keys():
            # NPZ can carry an incomplete record too. Preserve its absence so
            # load_frames can apply an explicit detector before materialization.
            detector = None
        meta = dict(metadata.geometry_meta or {})
        # Older saves stored raw Zeiss tables alongside the effective geometry.
        # Keep them with the acquisition they describe, including after a re-save.
        if "detector_view_offsets_pixels" in meta or "cone_view_distances" in meta:
            original = {
                key: meta.pop(key)
                for key in ("detector_view_offsets_pixels", "cone_view_distances")
                if key in meta
            }
            vendor = meta.get("vendor_corrections")
            if isinstance(vendor, dict) and vendor.get("applied") is False:
                # The old importer zeroed this table when it omitted the shifts.
                original.pop("detector_view_offsets_pixels", None)
            zeiss = meta.get("zeiss", {})
            if isinstance(zeiss, dict):
                meta["zeiss"] = {**zeiss, **original}
        grid = metadata.grid
        dataset = cls(
            projections=None if payload.projections is None else np.asarray(payload.projections),
            angles=np.asarray(
                np.empty(0, dtype=np.float32) if metadata.angles is None else metadata.angles,
                dtype=np.float32,
            ),
            volume=None if metadata.volume is None else np.asarray(metadata.volume),
            detector=detector
            if isinstance(detector, Detector)
            else (Detector.from_dict(detector) if detector is not None else None),
            grid=grid
            if isinstance(grid, Grid)
            else (Grid.from_dict(grid) if grid is not None else None),
            geometry_type=str(metadata.geometry_type),
            geometry_metadata=meta,
            angle_offset=(
                None
                if metadata.angle_offset is None
                else np.asarray(metadata.angle_offset, dtype=np.float32)
            ),
            align_params=(
                None if metadata.align_params is None else np.asarray(metadata.align_params)
            ),
            align_gauge=None if metadata.align_gauge is None else dict(metadata.align_gauge),
            misalign_spec=None if metadata.misalign_spec is None else dict(metadata.misalign_spec),
            source_path=None if source_path is None else str(source_path),
            source_format="nxtomo",
            sample_name=metadata.sample_name,
        )
        dataset._metadata = metadata
        return dataset

    @property
    def image_key(self) -> np.ndarray | None:
        """Saved frame labels (0 sample, 1 flat, 2 dark, 3 invalid), when recorded.

        Returns
        -------
        numpy.ndarray or None
            One label per stored frame, or None when the record has none.
        """
        return None if self._metadata is None else self._metadata.image_key

    def to_nxtomo_metadata(self) -> NXTomoMetadata:
        """Return writable NXtomo metadata reflecting the dataset's current fields.

        Returns
        -------
        NXTomoMetadata
            Writable persistence metadata with copied arrays and retained
            acquisition information.
        """
        if self._metadata is None:
            metadata = NXTomoMetadata()
        else:
            # Dataset-owned fields below replace the stored ones; copy only the
            # retained acquisition key and the mutable dictionaries kept as is.
            stored = self._metadata
            metadata = replace(
                stored,
                image_key=None if stored.image_key is None else np.array(stored.image_key),
                geometry_calibration=_copied(stored.geometry_calibration),
                simulation_artifacts=_copied(stored.simulation_artifacts),
            )
        metadata.angles = np.array(self.angles, dtype=np.float32, copy=True)
        metadata.volume = None if self.volume is None else np.array(self.volume, copy=True)
        metadata.grid = self.grid
        metadata.detector = self.detector
        metadata.geometry_type = self.geometry_type
        metadata.geometry_meta = _copied(self.geometry_metadata)
        metadata.angle_offset = (
            None
            if self.angle_offset is None
            else np.array(self.angle_offset, dtype=np.float32, copy=True)
        )
        metadata.align_params = (
            None if self.align_params is None else np.array(self.align_params, copy=True)
        )
        metadata.align_gauge = _copied(self.align_gauge)
        metadata.misalign_spec = _copied(self.misalign_spec)
        metadata.sample_name = self.sample_name or metadata.sample_name or "sample"
        return metadata

    def geometry_inputs(self) -> dict[str, Any]:
        """Return normalized geometry metadata for reconstruction/alignment.

        Returns
        -------
        dict
            Normalized geometry, detector, angle and alignment metadata.

        Raises
        ------
        ValueError
            If detector metadata are missing.
        """
        detector = self.detector
        if self.geometry_metadata.get("axis_status") == "not_recorded":
            raise ValueError("Missing calibrated rotation axis: pass geometry=...")
        if detector is None:
            raise ValueError(
                "Missing detector pixel sizes: pass pixel_size=..., detector=... or "
                "geometry=..., or use tomojax correct --pixel-size"
            )
        views = len(self.angles) if self.projections is None else self.projections.shape[0]
        if not views or self.angles.shape != (views,):
            raise ValueError(
                "Missing view angles: pass angles=..., or use tomojax correct --angles"
            )
        payload: dict[str, Any] = {
            "detector": detector.to_dict(),
            "angles": np.asarray(self.angles, dtype=np.float32),
            "geometry_type": self.geometry_type,
        }
        if self.grid is not None:
            payload["grid"] = self.grid.to_dict()
        # The dataset's own fields, not the metadata it was loaded with: callers
        # replace them (an alignment's poses, a rebinned geometry).
        _merge_dataset_solver_metadata(payload, self)
        _merge_geometry_metadata_dict(payload, self.geometry_metadata)
        return payload

    def _recorded_pitches(self) -> tuple[float | None, float | None]:
        """The pixel pitches the file recorded without a whole detector, wherever it put them.

        A raw acquisition reader keeps them under ``acquisition``; a partial NXtomo
        record at the top level; a partial NPZ detector in its own mapping.
        """
        meta = self.geometry_metadata
        for found in (
            meta.get("acquisition", {}).get("detector_pixel_size"),
            meta.get("detector_pixel_size"),
        ):
            if found is not None and all(p is not None for p in found):
                return float(found[0]), float(found[1])
        partial = getattr(self._metadata, "detector", None)
        if isinstance(partial, dict) and {"du", "dv"} <= partial.keys():
            return float(partial["du"]), float(partial["dv"])
        return None, None

    def with_detector(self, detector: Detector) -> ProjectionDataset:
        """Replace the first panel, preserving relative centers across cone segments.

        Parameters
        ----------
        detector : Detector
            First panel center in geometry length units and sampling shared by all views.

        Returns
        -------
        ProjectionDataset
            Dataset with the new detector sampling and each panel displaced equally.
        """
        meta = dict(self.geometry_metadata)
        if entries := meta.get("cone_segments"):
            first = Detector.from_dict(entries[0]["detector"])
            du, dv = detector.center[0] - first.center[0], detector.center[1] - first.center[1]
            segments = []
            for entry in entries:
                old = Detector.from_dict(entry["detector"])
                center = (
                    detector.center
                    if old.center == first.center
                    else (old.center[0] + du, old.center[1] + dv)
                )
                segments.append({**entry, "detector": replace(detector, center=center).to_dict()})
            meta["cone_segments"] = segments
        return replace_dataset(self, detector=detector, geometry_metadata=meta)

    def with_parallel_detector(self) -> ProjectionDataset:
        """A parallel record's detector, from its recorded pitches or in pixel units.

        Some files record the pixel pitches but not the rotation center, and some record
        neither. The detector takes the recorded pitches, or a pitch of one pixel when
        the file records none (the record then says ``detector_pixel_size_status``
        "not_recorded" and ``length_unit`` "pixel": lengths, poses and the center are in
        pixels and attenuation is per pixel). The center is the middle, and the record
        keeps ``detector_center_status`` as the file gave it: nothing is estimated. A cone
        record is left alone, since its pitch and distances must share a unit.

        Returns
        -------
        ProjectionDataset
            Original record, or a copy with that detector.
        """
        if self.detector is not None or self.geometry_type != "parallel":
            return self
        meta = {"detector_center_status": "not_recorded", **self.geometry_metadata}
        du, dv = self._recorded_pitches()
        if du is None or dv is None:
            du = dv = 1.0
            meta = {**meta, "detector_pixel_size_status": "not_recorded", "length_unit": "pixel"}
        _, nv, nu = cast("np.ndarray", self.projections).shape  # a record without a detector
        detector = Detector(nu, nv, du, dv, (0.0, 0.0))
        return replace_dataset(self, detector=detector, geometry_metadata=meta)


def load_dataset(path: PathLike) -> ProjectionDataset:
    """Load a standard TomoJAX projection dataset.

    Supported direct inputs are NXtomo/HDF5 (`.nx`, `.nxs`, `.h5`, `.hdf`, `.hdf5`), TomoJAX
    `.npz` payloads and PSI cSAXS ptychographic tomography `.mat` files (their aligned
    phase projections). TIFF stacks need explicit angle metadata, so use
    :func:`load_tiff_stack`.

    Parameters
    ----------
    path : str or pathlib.Path
        NXtomo/HDF5 or TomoJAX NPZ dataset to read.

    Returns
    -------
    ProjectionDataset
        Measured projections, normalized metadata and any saved volume.

    Raises
    ------
    ValueError
        If the format is unsupported or TIFF angles must be supplied separately.
    KeyError
        If required datasets are missing.
    OSError
        If the input cannot be opened.
    """
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in _HDF5_SUFFIXES:
        return ProjectionDataset.from_nxtomo(load_nxtomo(str(input_path)), source_path=input_path)
    if suffix == ".npz":
        return ProjectionDataset.from_nxtomo(load_npz(str(input_path)), source_path=input_path)
    if suffix == ".mat":
        return psi_record(input_path)
    if input_path.is_dir() or suffix in _TIFF_SUFFIXES:
        raise ValueError(
            "TIFF inputs require angle metadata; use load_tiff_stack(path, angles=...)"
        )
    raise ValueError(f"unsupported dataset format for {input_path}")


def psi_record(path: Path, *, layout_only: bool = False) -> ProjectionDataset:
    """A PSI cSAXS ptychographic tomography file's phase line integrals, as a parallel scan."""
    from tomojax.io._psi import read_psi_mat

    projections, angles, pixel, wavelength, name = read_psi_mat(path, layout_only=layout_only)
    record = ProjectionDataset(
        projections=projections,
        angles=angles,
        geometry_type="parallel",
        geometry_metadata={
            "detector_pixel_size": [pixel, pixel],
            "length_unit": "mm",
            "quantity": "phase",
            "wavelength": wavelength,
        },
        source_path=str(path),
        source_format="psi_mat",
        sample_name=name,
    )
    return record.with_parallel_detector()


def dataset_output_format(path: Path) -> Literal["npz", "hdf5"]:
    """Select the dataset container from its output name.

    Parameters
    ----------
    path : pathlib.Path
        Requested destination.

    Returns
    -------
    str
        Container used by the dataset writer.

    Raises
    ------
    InvalidParameter
        If the destination has an unsupported suffix.
    """
    suffix = path.suffix.lower()
    if suffix in _HDF5_SUFFIXES:
        return "hdf5"
    if suffix == ".npz":
        return "npz"
    known = ", ".join(sorted(_HDF5_SUFFIXES | {".npz"}))
    raise InvalidParameter("path", f"cannot save to {path.name}: give a path ending in {known}")


def save_dataset(path: PathLike, dataset: ProjectionDataset) -> None:
    """Save a public projection dataset as NXtomo/HDF5 or NPZ.

    Parameters
    ----------
    path : str or pathlib.Path
        NXtomo/HDF5 or NPZ output path; existing files are overwritten.
    dataset : ProjectionDataset
        Projections, geometry and optional volume to persist.

    Returns
    -------
    None
        Write the dataset to disk without changing its metadata.

    Raises
    ------
    ValueError
        If the output suffix is not supported.
    OSError
        If the output cannot be written.
    """
    output_path = Path(path)
    writer = save_nxtomo if dataset_output_format(output_path) == "hdf5" else save_npz
    # The metadata copy would duplicate a large volume that the writer only reads.
    metadata = replace_dataset(dataset, volume=None).to_nxtomo_metadata()
    metadata.volume = dataset.volume
    projections = None if dataset.projections is None else np.asarray(dataset.projections)
    writer(str(output_path), projections, metadata=metadata)


def convert_dataset(input_path: PathLike, output_path: PathLike) -> None:
    """Convert between supported TomoJAX dataset container formats.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Input NXtomo/HDF5 or NPZ container.
    output_path : str or pathlib.Path
        Output container of the other format.

    Returns
    -------
    None
        Write the converted dataset with its metadata.

    Raises
    ------
    ValueError
        If the conversion is not NPZ to HDF5 or HDF5 to NPZ.
    OSError
        If a file cannot be read or written.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    input_suffix = in_path.suffix.lower()
    output_suffix = out_path.suffix.lower()
    if input_suffix == ".npz" and output_suffix in _HDF5_SUFFIXES:
        data = load_npz(str(in_path))
        save_nxtomo(
            str(out_path),
            np.asarray(data.projections),
            metadata=data.copy_metadata(),
        )
        return
    if input_suffix in _HDF5_SUFFIXES and output_suffix == ".npz":
        data = load_nxtomo(str(in_path))
        save_npz(str(out_path), np.asarray(data.projections), metadata=data.copy_metadata())
        return
    raise ValueError("Unsupported conversion. Use .npz <-> .nxs/.h5/.hdf/.hdf5")


def load_tiff_stack(
    path: PathLike,
    *,
    angles: Sequence[float] | np.ndarray,
    detector: Detector | None = None,
    grid: Grid | None = None,
    geometry_type: str = "parallel",
    geometry_metadata: Mapping[str, Any] | None = None,
) -> ProjectionDataset:
    """Load grayscale TIFF pages with explicit view ``angles``, in degrees.

    Parameters
    ----------
    path : str or pathlib.Path
        Multipage TIFF or folder of naturally sorted TIFF files.
    angles : array-like
        Finite one-dimensional view angles in degrees, one per grayscale page.
    detector : Detector or None, optional
        Detector geometry to attach to the dataset.
    grid : Grid or None, optional
        Reconstruction grid to attach to the dataset.
    geometry_type : str, optional
        Geometry family; defaults to parallel.
    geometry_metadata : mapping or None, optional
        Additional geometry metadata.

    Returns
    -------
    ProjectionDataset
        Float32 TIFF samples and supplied metadata; no intensity correction is applied.

    Raises
    ------
    ValueError
        If pages are not consistent grayscale frames or angles do not match them.
    FileNotFoundError
        If the input contains no TIFF frames.
    """
    input_path = Path(path)
    angles = np.asarray(angles, dtype=np.float32)
    if angles.ndim != 1:
        raise ValueError("angles must be one-dimensional")
    if not np.isfinite(angles).all():
        raise ValueError("angles must contain only finite values")
    projections = read_tiff_frames(input_path).astype(np.float32, copy=False)
    if angles.shape[0] != projections.shape[0]:
        raise ValueError(
            f"angles length {angles.shape[0]} does not match projection count "
            f"{projections.shape[0]}"
        )

    return ProjectionDataset(
        projections=projections,
        angles=angles,
        detector=detector,
        grid=grid,
        geometry_type=geometry_type,
        geometry_metadata=dict(geometry_metadata or {}),
        source_path=str(input_path),
        source_format="tiff_stack",
    )


def validate_dataset(path: PathLike) -> ValidationReport:
    """Validate a user dataset path using the public IO boundary.

    Parameters
    ----------
    path : str or pathlib.Path
        Dataset path to inspect for structural validity and missing raw calibration.

    Returns
    -------
    ValidationReport
        Issues found in the file or format; an empty issues list means none were found.
    """
    try:
        issues = _dataset_issues(Path(path))
    except (OSError, ValueError, KeyError) as exc:
        issues = [str(exc)]
    return {"issues": issues}


def _dataset_issues(input_path: Path) -> list[str]:
    from ._acquisition import read_acquisition

    raw = read_acquisition(input_path)
    if raw is not None:
        issues = calibration_issues(raw.flats, raw.darks, raw.white_level)
        try:
            raw.record.with_parallel_detector().geometry_inputs()
        except ValueError as exc:
            issues.insert(0, str(exc))
        return issues
    suffix = input_path.suffix.lower()
    whole = suffix in {".npz", ".mat"} or (
        suffix in _HDF5_SUFFIXES and saved_without_frames(input_path)
    )
    if whole:
        try:
            from .projection_layout import saved_projection_layout

            saved_projection_layout(input_path)
        except (BadZipFile, EOFError, OSError, ValueError, KeyError, TypeError) as exc:
            return [str(exc)]
        return []
    if suffix in _HDF5_SUFFIXES:
        located = locate_frames(str(input_path))
        issues = []
        if located.angles is None:
            issues.append("Missing rotation angles; supply angles when loading the frames")
        if located.image_key is not None and not np.any(located.image_key == 0):
            issues.append("No sample frames (image_key == 0)")
        return issues
    if input_path.is_dir() or suffix in _TIFF_SUFFIXES:
        return [
            "TIFF stacks require an angle sidecar or explicit angles; "
            "pass them with tomojax correct --angles, or tj.load(angles=...)"
        ]
    return [f"unsupported dataset format for {input_path}"]


def _merge_dataset_solver_metadata(
    payload: dict[str, Any],
    dataset: ProjectionDataset,
) -> None:
    if dataset.angle_offset is not None:
        payload["angle_offset"] = np.asarray(dataset.angle_offset, dtype=np.float32)
    if dataset.align_params is not None:
        payload["align_params"] = np.asarray(dataset.align_params)
    if dataset.align_gauge is not None:
        payload["align_gauge"] = dict(dataset.align_gauge)
    if dataset.misalign_spec is not None:
        payload["misalign_spec"] = dict(dataset.misalign_spec)


def _merge_geometry_metadata_dict(
    payload: dict[str, Any],
    geometry_metadata: Mapping[str, Any],
) -> None:
    for key in (
        "tilt",
        "tilt_about",
        "axis_unit_lab",
        "detector_roll",
        "cone_beam",
        "cone_segments",
        "trajectory_lengths",
        "voxel_size",
    ):
        value = geometry_metadata.get(key)
        if value is not None:
            payload[key] = value


def replace_dataset(
    dataset: ProjectionDataset,
    *,
    metadata: NXTomoMetadata | None = None,
    keep_metadata: bool = True,
    **changes: Any,
) -> ProjectionDataset:
    """Return a copy of ``dataset`` with updated fields while preserving NXtomo metadata.

    Parameters
    ----------
    dataset : ProjectionDataset
        Source dataset to copy.
    metadata : NXTomoMetadata or None, optional
        Replacement low-level NXtomo metadata record; None keeps the source
        metadata when ``keep_metadata`` is True.
    keep_metadata : bool, optional
        Retain the source dataset's internal NXtomo metadata when ``metadata``
        is None. Default is True.
    **changes : Any
        Field updates forwarded to :func:`dataclasses.replace`.

    Returns
    -------
    ProjectionDataset
        Copied dataset with the requested field updates and preserved metadata.
    """
    updated = replace(dataset, **changes)
    if metadata is not None:
        retained: NXTomoMetadata | None = metadata
    elif keep_metadata:
        retained = cast("NXTomoMetadata | None", object.__getattribute__(dataset, "_metadata"))
    else:
        retained = None
    object.__setattr__(updated, "_metadata", retained)
    return updated


def _copied(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if value is None else copy.deepcopy(value)
