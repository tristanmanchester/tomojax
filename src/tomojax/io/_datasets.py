"""Public projection dataset loading and saving facade."""
# pyright: reportUnknownMemberType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np

from tomojax._data import (
    LoadedGeometryMeta,
    LoadedNXTomo,
    LocatedFrames,
    NXTomoMetadata,
    build_geometry_from_meta,
    load_npz,
    load_nxtomo,
    locate_frames,
    save_npz,
    save_nxtomo,
    validate_nxtomo,
)
from tomojax.core.geometry import Detector, Grid, ScanGeometry
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
    "load_projection_payload",
    "load_tiff_stack",
    "locate_frames",
    "save_dataset",
    "save_nxtomo",
    "save_projection_payload",
    "validate_dataset",
    "validate_nxtomo",
]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import jax

    from tomojax.io._json import JsonValue

type PathLike = str | Path

_HDF5_SUFFIXES = {".nxs", ".h5", ".hdf", ".hdf5"}
_TIFF_SUFFIXES = TIFF_SUFFIXES


class ValidationReport(TypedDict):
    """Public dataset validation result.

    Attributes
    ----------
    issues : list of str
        Validation problems; an empty list means none were found.
    """

    issues: list[str]


def save_projection_payload(
    path: PathLike,
    *,
    projections: np.ndarray,
    metadata: NXTomoMetadata,
) -> None:
    """Save a solver-facing projection payload with updated metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        NXtomo output path.
    projections : numpy.ndarray
        Projection stack in (view, v, u) order.
    metadata : NXTomoMetadata
        Updated acquisition, geometry and reconstruction metadata.

    Returns
    -------
    None
        Write the NXtomo projection payload.

    Raises
    ------
    ValueError
        If projection or volume metadata are invalid.
    OSError
        If the file cannot be written.
    """
    save_nxtomo(str(path), projections=projections, metadata=metadata)


def build_geometry_from_dataset_metadata(
    meta: Mapping[str, Any],
    *,
    grid_override: Grid | tuple[int, int, int] | list[int] | None = None,
    poses: bool = False,
    volume_shape: Sequence[int] | None = None,
) -> tuple[Grid, Detector, ScanGeometry]:
    """Build geometry objects from normalized dataset metadata.

    ``poses`` applies the per-view corrections and angle offsets saved with
    the data. Keeping this wrapper in `tomojax.io` makes the command/data dependency
    explicit and gives the `ProjectionDataset` solver contract one replacement
    point.

    Parameters
    ----------
    meta : mapping
        Normalized geometry metadata, as returned by geometry_inputs.
    grid_override : Grid, tuple of int, list of int or None, optional
        Replacement grid or (nx, ny, nz) dimensions.
    poses : bool, optional
        Apply saved per-view alignment and angle corrections; defaults to False.
    volume_shape : sequence of int or None, optional
        Fallback (nx, ny, nz) dimensions when no grid is recorded.

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
        poses=poses,
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
    projections : numpy.ndarray
        Measured projections in (view, v, u) order.
    angles : numpy.ndarray
        One rotation angle per view, in degrees.
    volume : numpy.ndarray, jax.Array or None, optional
        Reconstructed volume in (x, y, z) order, when present.
    detector : Detector or None, optional
        Detector pixel counts, pitches and physical centre.
    grid : Grid or None, optional
        Reconstruction voxel grid.
    geometry_type : str, optional
        Geometry family, such as parallel, lamino or cone.
    geometry_metadata : dict, optional
        Additional geometry and acquisition metadata.
    angle_offset_deg : numpy.ndarray or None, optional
        Per-view angular corrections in degrees.
    align_params : numpy.ndarray or None, optional
        Saved per-view alignment parameters, with convention in align_gauge.
    align_gauge : dict or None, optional
        Gauge and frame metadata for saved alignment parameters.
    misalign_spec : dict or None, optional
        Simulation misalignment record; angles already include simulated offsets.
    source_path : str or None, optional
        Original input path.
    source_format : str or None, optional
        Original container format.
    sample_name : str or None, optional
        Sample name retained on output.
    _metadata : NXTomoMetadata or None, optional
        Internal retained acquisition metadata; normally populated by from_nxtomo.
    """

    projections: np.ndarray
    angles: np.ndarray
    volume: np.ndarray | jax.Array | None = None
    detector: Detector | None = None
    grid: Grid | None = None
    geometry_type: str = "parallel"
    geometry_metadata: dict[str, Any] = field(default_factory=dict)
    angle_offset_deg: np.ndarray | None = None
    align_params: np.ndarray | None = None
    align_gauge: dict[str, JsonValue] | None = None
    # Simulated misalignment, whose angle offsets ``angles`` already include.
    misalign_spec: dict[str, JsonValue] | None = None
    source_path: str | None = None
    source_format: str | None = None
    sample_name: str | None = None
    _metadata: NXTomoMetadata | None = field(default=None, repr=False, compare=False)

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
            # load_frames can apply an explicit detector before materialisation.
            detector = None
        grid = metadata.grid
        return cls(
            projections=np.asarray(payload.projections),
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
            else (_grid_from_mapping(grid) if grid is not None else None),
            geometry_type=str(metadata.geometry_type),
            geometry_metadata=dict(metadata.geometry_meta or {}),
            angle_offset_deg=(
                None
                if metadata.angle_offset_deg is None
                else np.asarray(metadata.angle_offset_deg, dtype=np.float32)
            ),
            align_params=(
                None if metadata.align_params is None else np.asarray(metadata.align_params)
            ),
            align_gauge=None if metadata.align_gauge is None else dict(metadata.align_gauge),
            misalign_spec=None if metadata.misalign_spec is None else dict(metadata.misalign_spec),
            source_path=None if source_path is None else str(source_path),
            source_format="nxtomo",
            sample_name=metadata.sample_name,
            _metadata=metadata,
        )

    def to_nxtomo_metadata(self) -> NXTomoMetadata:
        """Return NXtomo metadata for saving this public dataset.

        Returns
        -------
        NXTomoMetadata
            Persistence metadata reflecting the dataset's current fields.
        """
        return self.copy_metadata()

    def copy_metadata(self) -> NXTomoMetadata:
        """Return writable persistence metadata for save/update workflows.

        Returns
        -------
        NXTomoMetadata
            Writable metadata with copied arrays and retained acquisition information.
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
                simulation_artefacts=_copied(stored.simulation_artefacts),
            )
        metadata.angles = np.array(self.angles, dtype=np.float32, copy=True)
        metadata.volume = None if self.volume is None else np.array(self.volume, copy=True)
        metadata.grid = self.grid
        metadata.detector = self.detector
        metadata.geometry_type = self.geometry_type
        metadata.geometry_meta = dict(self.geometry_metadata)
        metadata.angle_offset_deg = (
            None
            if self.angle_offset_deg is None
            else np.array(self.angle_offset_deg, dtype=np.float32, copy=True)
        )
        metadata.align_params = (
            None if self.align_params is None else np.array(self.align_params, copy=True)
        )
        metadata.align_gauge = None if self.align_gauge is None else dict(self.align_gauge)
        metadata.misalign_spec = None if self.misalign_spec is None else dict(self.misalign_spec)
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
        if (
            detector is None
            or self.geometry_metadata.get("detector_center_status") == "not_recorded"
        ):
            raise ValueError(
                "Missing detector pixel sizes or centre: pass detector=... or geometry=... "
                "to tomojax.load_frames"
            )
        if self.geometry_metadata.get("axis_status") == "not_recorded":
            raise ValueError(
                "Missing calibrated rotation axis: pass geometry=... to tomojax.load_frames"
            )
        if self.angles.shape != (self.projections.shape[0],):
            raise ValueError("Missing view angles: pass angles=... to tomojax.load_frames")
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


def load_projection_payload(path: PathLike) -> ProjectionDataset:
    """Load the solver-facing projection dataset for reconstruction/alignment.

    Parameters
    ----------
    path : str or pathlib.Path
        TomoJAX projection dataset to load.

    Returns
    -------
    ProjectionDataset
        Normalized solver-facing projection dataset.

    Raises
    ------
    ValueError
        If the container format is unsupported.
    OSError
        If the file cannot be read.
    """
    return load_dataset(path)


def load_dataset(path: PathLike) -> ProjectionDataset:
    """Load a standard TomoJAX projection dataset.

    Supported direct inputs are NXtomo/HDF5 (`.nxs`, `.h5`, `.hdf`, `.hdf5`) and TomoJAX
    `.npz` payloads. TIFF stacks need explicit angle metadata, so use
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
    if input_path.is_dir() or suffix in _TIFF_SUFFIXES:
        raise ValueError(
            "TIFF inputs require angle metadata; use load_tiff_stack(path, angles=...)"
        )
    raise ValueError(f"unsupported dataset format for {input_path}")


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
    suffix = output_path.suffix.lower()
    # The metadata copy would duplicate a large volume that the writer only reads.
    metadata = replace(dataset, volume=None).to_nxtomo_metadata()
    metadata.volume = dataset.volume
    if suffix in _HDF5_SUFFIXES:
        save_nxtomo(
            str(output_path),
            projections=np.asarray(dataset.projections),
            metadata=metadata,
        )
        return
    if suffix == ".npz":
        save_npz(
            str(output_path),
            np.asarray(dataset.projections),
            metadata=metadata,
        )
        return
    raise ValueError(f"unsupported dataset output format for {output_path}")


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
        Dataset path to inspect for structural validity.

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

    if read_acquisition(input_path) is not None:
        return []
    suffix = input_path.suffix.lower()
    if suffix in _HDF5_SUFFIXES:
        located = locate_frames(str(input_path))
        issues = []
        if located.angles is None:
            issues.append("Missing rotation angles; supply angles when loading the frames")
        if located.image_key is not None and not np.any(located.image_key == 0):
            issues.append("No sample frames (image_key == 0)")
        return issues
    if suffix == ".npz":
        try:
            load_dataset(input_path)
        except Exception as exc:
            return [str(exc)]
        return []
    if input_path.is_dir() or suffix in _TIFF_SUFFIXES:
        return [
            "TIFF stacks require an angle sidecar or explicit angles; "
            "import them with `tomojax import` or load_tiff_stack"
        ]
    return [f"unsupported dataset format for {input_path}"]


def _merge_dataset_solver_metadata(
    payload: dict[str, Any],
    dataset: ProjectionDataset,
) -> None:
    if dataset.angle_offset_deg is not None:
        payload["angle_offset_deg"] = np.asarray(dataset.angle_offset_deg, dtype=np.float32)
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
        "tilt_deg",
        "tilt_about",
        "axis_unit_lab",
        "detector_roll_deg",
        "cone_beam",
        "cone_segments",
    ):
        value = geometry_metadata.get(key)
        if value is not None:
            payload[key] = value


def _grid_from_mapping(payload: Mapping[str, Any]) -> Grid:
    vol_origin = (
        None
        if payload.get("vol_origin") is None
        else _vec3(payload["vol_origin"], name="vol_origin")
    )
    vol_center = (
        None
        if payload.get("vol_center") is None
        else _vec3(payload["vol_center"], name="vol_center")
    )
    return Grid(
        nx=int(payload["nx"]),
        ny=int(payload["ny"]),
        nz=int(payload["nz"]),
        vx=float(payload["vx"]),
        vy=float(payload["vy"]),
        vz=float(payload["vz"]),
        vol_origin=vol_origin,
        vol_center=vol_center,
    )


def _vec3(value: object, *, name: str) -> tuple[float, float, float]:
    items = tuple(float(v) for v in cast("Sequence[float | int | str]", value))
    if len(items) != 3:
        raise ValueError(f"{name} must have length 3")
    return items[0], items[1], items[2]


def _copied(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if value is None else copy.deepcopy(value)
