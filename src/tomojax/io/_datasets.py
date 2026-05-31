"""Public projection dataset loading and saving facade."""
# pyright: reportUnknownMemberType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import imageio.v3 as iio
import numpy as np

from tomojax._data import (
    LoadedGeometryMeta,
    LoadedNXTomo,
    NXTomoMetadata,
    build_geometry_from_meta,
    load_npz,
    load_nxtomo,
    save_npz,
    save_nxtomo,
    validate_nxtomo,
)
from tomojax.core.geometry import Detector, Geometry, Grid
from tomojax.io._tiff import TIFF_SUFFIXES, tiff_files

__all__ = [
    "LoadedNXTomo",
    "NXTomoMetadata",
    "ProjectionDataset",
    "ValidationReport",
    "build_geometry_from_dataset_metadata",
    "convert_dataset",
    "load_dataset",
    "load_nxtomo",
    "load_projection_payload",
    "load_tiff_stack",
    "save_dataset",
    "save_nxtomo",
    "save_projection_payload",
    "validate_dataset",
    "validate_nxtomo",
]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tomojax.io._json import JsonValue

type PathLike = str | Path

_HDF5_SUFFIXES = {".nxs", ".h5", ".hdf5"}
_TIFF_SUFFIXES = TIFF_SUFFIXES


class ValidationReport(TypedDict):
    """Public dataset validation result."""

    issues: list[str]


def save_projection_payload(
    path: PathLike,
    *,
    projections: np.ndarray,
    metadata: NXTomoMetadata,
) -> None:
    """Save a solver-facing projection payload with updated metadata."""
    save_nxtomo(str(path), projections=projections, metadata=metadata)


def build_geometry_from_dataset_metadata(
    meta: Mapping[str, Any],
    *,
    grid_override: Grid | tuple[int, int, int] | list[int] | None = None,
    apply_saved_alignment: bool = False,
    volume_shape: Sequence[int] | None = None,
) -> tuple[Grid, Detector, Geometry]:
    """Build geometry objects from normalized dataset metadata.

    Keeping this wrapper in `tomojax.io` makes the command/data dependency
    explicit and gives the `ProjectionDataset` solver contract one replacement
    point.
    """
    return build_geometry_from_meta(
        cast("LoadedGeometryMeta", dict(meta)),
        grid_override=grid_override,
        apply_saved_alignment=apply_saved_alignment,
        volume_shape=volume_shape,
    )


@dataclass(slots=True)
class ProjectionDataset:
    """Standard public in-memory projection dataset.

    This is the current IO boundary. It contains measured projection data,
    normalized metadata needed by reconstruction/alignment code, and the
    optional reconstructed volume when a processed container stores one.
    """

    projections: np.ndarray
    angles_deg: np.ndarray
    volume: np.ndarray | None = None
    detector: Detector | None = None
    grid: Grid | None = None
    geometry_type: str = "parallel"
    geometry_metadata: dict[str, Any] = field(default_factory=dict)
    angle_offset_deg: np.ndarray | None = None
    align_params: np.ndarray | None = None
    align_gauge: dict[str, JsonValue] | None = None
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
        """Build a public dataset from the lower-level NXtomo payload."""
        metadata = payload.metadata
        detector = metadata.detector
        grid = metadata.grid
        return cls(
            projections=np.asarray(payload.projections),
            angles_deg=np.asarray(
                np.zeros(payload.projections.shape[0], dtype=np.float32)
                if metadata.thetas_deg is None
                else metadata.thetas_deg,
                dtype=np.float32,
            ),
            volume=None if metadata.volume is None else np.asarray(metadata.volume),
            detector=detector
            if isinstance(detector, Detector)
            else (_detector_from_mapping(detector) if detector is not None else None),
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
            source_path=None if source_path is None else str(source_path),
            source_format="nxtomo",
            sample_name=metadata.sample_name,
            _metadata=metadata,
        )

    def to_nxtomo_metadata(self) -> NXTomoMetadata:
        """Return NXtomo metadata for saving this public dataset."""
        if self._metadata is not None:
            metadata = self.copy_metadata()
            metadata.thetas_deg = np.array(self.angles_deg, dtype=np.float32, copy=True)
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
            metadata.sample_name = self.sample_name or metadata.sample_name or "sample"
            return metadata
        return NXTomoMetadata(
            thetas_deg=np.array(self.angles_deg, dtype=np.float32, copy=True),
            volume=None if self.volume is None else np.array(self.volume, copy=True),
            grid=self.grid,
            detector=self.detector,
            geometry_type=self.geometry_type,
            geometry_meta=dict(self.geometry_metadata),
            angle_offset_deg=(
                None
                if self.angle_offset_deg is None
                else np.array(self.angle_offset_deg, dtype=np.float32, copy=True)
            ),
            align_params=(
                None if self.align_params is None else np.array(self.align_params, copy=True)
            ),
            align_gauge=None if self.align_gauge is None else dict(self.align_gauge),
            sample_name=self.sample_name or "sample",
        )

    def copy_metadata(self) -> NXTomoMetadata:
        """Return writable persistence metadata for save/update workflows."""
        if self._metadata is None:
            metadata = NXTomoMetadata()
        else:
            metadata = NXTomoMetadata.from_dataset(
                LoadedNXTomo(
                    projections=np.asarray(self.projections),
                    metadata=self._metadata,
                ).to_dataset_dict()
            )
        metadata.thetas_deg = np.array(self.angles_deg, dtype=np.float32, copy=True)
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
            None
            if self.align_params is None
            else np.array(
                self.align_params,
                copy=True,
            )
        )
        metadata.align_gauge = None if self.align_gauge is None else dict(self.align_gauge)
        metadata.sample_name = self.sample_name or metadata.sample_name or "sample"
        return metadata

    def geometry_inputs(self) -> dict[str, Any]:
        """Return normalized geometry metadata for reconstruction/alignment."""
        detector = self.detector
        if detector is None:
            raise ValueError("projection dataset is missing detector metadata")
        payload: dict[str, Any] = {
            "detector": detector.to_dict(),
            "thetas_deg": np.asarray(self.angles_deg, dtype=np.float32),
            "geometry_type": self.geometry_type,
        }
        if self.grid is not None:
            payload["grid"] = self.grid.to_dict()
        if self._metadata is not None:
            _merge_optional_geometry_metadata(payload, self._metadata)
        else:
            _merge_dataset_solver_metadata(payload, self)
            _merge_geometry_metadata_dict(payload, self.geometry_metadata)
        return payload


def load_projection_payload(path: PathLike) -> ProjectionDataset:
    """Load the solver-facing projection dataset for reconstruction/alignment."""
    return load_dataset(path)


def load_dataset(path: PathLike) -> ProjectionDataset:
    """Load a standard TomoJAX projection dataset.

    Supported direct inputs are NXtomo/HDF5 (`.nxs`, `.h5`, `.hdf5`) and TomoJAX
    `.npz` payloads. TIFF stacks need explicit angle metadata, so use
    :func:`load_tiff_stack`.
    """
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in _HDF5_SUFFIXES:
        return ProjectionDataset.from_nxtomo(load_nxtomo(str(input_path)), source_path=input_path)
    if suffix == ".npz":
        return ProjectionDataset.from_nxtomo(load_npz(str(input_path)), source_path=input_path)
    if input_path.is_dir() or suffix in _TIFF_SUFFIXES:
        raise ValueError(
            "TIFF inputs require angle metadata; use load_tiff_stack(path, angles_deg=...)"
        )
    raise ValueError(f"unsupported dataset format for {input_path}")


def save_dataset(path: PathLike, dataset: ProjectionDataset) -> None:
    """Save a public projection dataset as NXtomo/HDF5 or NPZ."""
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix in _HDF5_SUFFIXES:
        save_nxtomo(
            str(output_path),
            projections=np.asarray(dataset.projections),
            metadata=dataset.to_nxtomo_metadata(),
        )
        return
    if suffix == ".npz":
        save_npz(
            str(output_path),
            np.asarray(dataset.projections),
            metadata=dataset.to_nxtomo_metadata(),
        )
        return
    raise ValueError(f"unsupported dataset output format for {output_path}")


def convert_dataset(input_path: PathLike, output_path: PathLike) -> None:
    """Convert between supported TomoJAX dataset container formats."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    input_suffix = in_path.suffix.lower()
    output_suffix = out_path.suffix.lower()
    if input_suffix == ".npz" and output_suffix in _HDF5_SUFFIXES:
        data = load_npz(str(in_path))
        save_nxtomo(
            str(out_path),
            data.projections,
            metadata=data.copy_metadata(),
        )
        return
    if input_suffix in _HDF5_SUFFIXES and output_suffix == ".npz":
        data = load_nxtomo(str(in_path))
        save_npz(str(out_path), data.projections, metadata=data.copy_metadata())
        return
    raise ValueError("Unsupported conversion. Use .npz <-> .nxs/.h5/.hdf5")


def load_tiff_stack(
    path: PathLike,
    *,
    angles_deg: Sequence[float] | np.ndarray,
    detector: Detector | None = None,
    grid: Grid | None = None,
    geometry_type: str = "parallel",
    geometry_metadata: Mapping[str, Any] | None = None,
) -> ProjectionDataset:
    """Load a TIFF projection stack with explicit angle metadata."""
    input_path = Path(path)
    files = tiff_files(input_path)
    if not files:
        raise ValueError(f"no TIFF files found under {input_path}")

    projections = np.stack(
        [np.asarray(cast("object", iio.imread(file)), dtype=np.float32) for file in files],
        axis=0,
    )
    angles = np.asarray(angles_deg, dtype=np.float32)
    if angles.ndim != 1:
        raise ValueError("angles_deg must be one-dimensional")
    if angles.shape[0] != projections.shape[0]:
        raise ValueError(
            f"angles_deg length {angles.shape[0]} does not match projection count "
            f"{projections.shape[0]}"
        )

    return ProjectionDataset(
        projections=projections,
        angles_deg=angles,
        detector=detector,
        grid=grid,
        geometry_type=geometry_type,
        geometry_metadata=dict(geometry_metadata or {}),
        source_path=str(input_path),
        source_format="tiff_stack",
    )


def validate_dataset(path: PathLike) -> ValidationReport:
    """Validate a user dataset path using the public IO boundary."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in _HDF5_SUFFIXES:
        report = validate_nxtomo(str(input_path))
        return {"issues": [str(issue) for issue in report.get("issues", [])]}
    if suffix == ".npz":
        try:
            load_dataset(input_path)
        except Exception as exc:
            return {"issues": [str(exc)]}
        return {"issues": []}
    if input_path.is_dir() or suffix in _TIFF_SUFFIXES:
        return {
            "issues": [
                "TIFF stacks require an angle sidecar or explicit angles; "
                "use load_tiff_stack/ingest"
            ]
        }
    return {"issues": [f"unsupported dataset format for {input_path}"]}


def _merge_optional_geometry_metadata(payload: dict[str, Any], metadata: NXTomoMetadata) -> None:
    if metadata.angle_offset_deg is not None:
        payload["angle_offset_deg"] = np.asarray(metadata.angle_offset_deg)
    if metadata.misalign_spec is not None:
        payload["misalign_spec"] = metadata.misalign_spec
    if metadata.align_params is not None:
        payload["align_params"] = np.asarray(metadata.align_params)
    if metadata.align_gauge is not None:
        payload["align_gauge"] = metadata.align_gauge
    _merge_geometry_metadata_dict(payload, metadata.geometry_meta or {})


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


def _merge_geometry_metadata_dict(
    payload: dict[str, Any],
    geometry_metadata: Mapping[str, Any],
) -> None:
    for key in (
        "tilt_deg",
        "tilt_about",
        "axis_unit_lab",
        "detector_roll_deg",
    ):
        value = geometry_metadata.get(key)
        if value is not None:
            payload[key] = value


def _detector_from_mapping(payload: Mapping[str, Any]) -> Detector:
    det_center = _vec2(payload.get("det_center", (0.0, 0.0)), name="det_center")
    return Detector(
        nu=int(payload["nu"]),
        nv=int(payload["nv"]),
        du=float(payload["du"]),
        dv=float(payload["dv"]),
        det_center=det_center,
    )


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


def _vec2(value: object, *, name: str) -> tuple[float, float]:
    items = tuple(float(v) for v in cast("Sequence[float | int | str]", value))
    if len(items) != 2:
        raise ValueError(f"{name} must have length 2")
    return items[0], items[1]


def _vec3(value: object, *, name: str) -> tuple[float, float, float]:
    items = tuple(float(v) for v in cast("Sequence[float | int | str]", value))
    if len(items) != 3:
        raise ValueError(f"{name} must have length 3")
    return items[0], items[1], items[2]
