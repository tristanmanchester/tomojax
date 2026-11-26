from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from tomojax.core.geometry.base import Detector, DetectorDict, Grid, GridDict

from ._io_frames import Hdf5Frames

if TYPE_CHECKING:
    import jax

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


type JsonObject = dict[str, JsonValue]


class SourceInfo(TypedDict, total=False):
    """Source metadata persisted in NXtomo files."""

    name: str | None
    type: str | None
    probe: str | None


type DatasetValue = np.ndarray | Hdf5Frames | JsonValue | GridDict | DetectorDict | SourceInfo


type LoadedDataset = dict[str, DatasetValue]


def _json_object(data: Mapping[str, DatasetValue], key: str) -> JsonObject | None:
    """A saved JSON-object field, checked as it leaves the generic loaded mapping."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"saved {key} must be a JSON object, not {type(value).__name__}")
    return cast("JsonObject", value)


def _source_record(value: DatasetValue | None) -> SourceInfo | None:
    if not isinstance(value, dict):
        return None
    fields: Mapping[str, object] = value

    def text(key: str) -> str | None:
        item = fields.get(key)
        return None if item is None else str(item)

    info = SourceInfo()
    if "name" in fields:
        info["name"] = text("name")
    if "type" in fields:
        info["type"] = text("type")
    if "probe" in fields:
        info["probe"] = text("probe")
    return info


def _copy_array_metadata(value: object) -> np.ndarray | None:
    if value is None:
        return None
    return np.array(value, copy=True)


# Geometry metadata keys copied into geometry inputs, with their conversions.
_GEOMETRY_PASSTHROUGH = (
    "tilt",
    "tilt_about",
    "cone_beam",
    "cone_segments",
    "trajectory_lengths",
    "axis_unit_lab",
    "detector_roll",
)


@dataclass(slots=True)
class NXTomoMetadata:
    """Portable metadata bundle for NXtomo persistence.

    Parameters
    ----------
    angles : numpy.ndarray or None, optional
        Per-frame rotation angles in degrees.
    image_key : numpy.ndarray or None, optional
        Frame labels: 0 sample, 1 flat, 2 dark, 3 other.
    grid : Grid, dict or None, optional
        Reconstruction voxel grid in physical units.
    detector : Detector, dict or None, optional
        Detector dimensions, pitches and physical center.
    geometry_type : str, optional
        Geometry family; defaults to parallel.
    geometry_meta : dict or None, optional
        Additional geometry and acquisition metadata.
    volume : numpy.ndarray, jax.Array or None, optional
        Reconstructed volume in (z, y, x) order.
    align_params : numpy.ndarray or None, optional
        Saved per-view alignment parameters.
    align_gauge : dict or None, optional
        Gauge and frame conventions for alignment parameters.
    geometry_calibration : dict or None, optional
        Calibration state and provenance.
    angle_offset : numpy.ndarray or None, optional
        Saved per-view angle corrections in degrees.
    misalign_spec : dict or None, optional
        Simulation misalignment metadata.
    simulation_artifacts : dict or None, optional
        Simulation artifact settings and provenance.
    frame : str or None, optional
        Coordinate-frame label.
    sample_name : str or None, optional
        Sample name; defaults to sample.
    source_name : str or None, optional
        Radiation source name; defaults to TomoJAX source.
    source_type : str or None, optional
        Radiation source type.
    source_probe : str or None, optional
        Probe label; defaults to x-ray.
    """

    angles: np.ndarray | None = None
    image_key: np.ndarray | None = None
    grid: Grid | GridDict | None = None
    detector: Detector | DetectorDict | None = None
    geometry_type: str = "parallel"
    geometry_meta: JsonObject | None = None
    volume: np.ndarray | jax.Array | None = None
    align_params: np.ndarray | None = None
    align_gauge: JsonObject | None = None
    geometry_calibration: JsonObject | None = None
    angle_offset: np.ndarray | None = None
    misalign_spec: JsonObject | None = None
    simulation_artifacts: JsonObject | None = None
    frame: str | None = None
    sample_name: str | None = "sample"
    source_name: str | None = "TomoJAX source"
    source_type: str | None = None
    source_probe: str | None = "x-ray"

    @classmethod
    def from_dataset(cls, data: Mapping[str, DatasetValue]) -> NXTomoMetadata:
        """Build metadata from a generic loaded dataset mapping.

        Parameters
        ----------
        data : mapping
            Loaded dataset fields and metadata.

        Returns
        -------
        NXTomoMetadata
            Typed dataset metadata or payload with copied array metadata.
        """
        source_info = _source_record(data.get("source"))
        source_name = data.get("source_name")
        source_type = data.get("source_type")
        source_probe = data.get("source_probe")
        if source_info is not None:
            if source_name is None:
                source_name = source_info.get("name")
            if source_type is None:
                source_type = source_info.get("type")
            if source_probe is None:
                source_probe = source_info.get("probe")

        geometry_type = data.get("geometry_type")
        return cls(
            angles=_copy_array_metadata(data.get("angles")),
            image_key=_copy_array_metadata(data.get("image_key")),
            grid=cast("GridDict | None", _json_object(data, "grid")),
            # An incomplete saved detector stays a mapping: load_frames may complete it.
            detector=cast("DetectorDict | None", _json_object(data, "detector")),
            geometry_type="parallel" if geometry_type is None else str(geometry_type),
            geometry_meta=_json_object(data, "geometry_meta"),
            volume=_copy_array_metadata(data.get("volume")),
            align_params=_copy_array_metadata(data.get("align_params")),
            align_gauge=_json_object(data, "align_gauge"),
            geometry_calibration=_json_object(data, "geometry_calibration"),
            angle_offset=_copy_array_metadata(data.get("angle_offset")),
            misalign_spec=_json_object(data, "misalign_spec"),
            simulation_artifacts=_json_object(data, "simulation_artifacts"),
            frame=None if data.get("frame") is None else str(data.get("frame")),
            sample_name=(None if data.get("sample_name") is None else str(data.get("sample_name"))),
            source_name=None if source_name is None else str(source_name),
            source_type=None if source_type is None else str(source_type),
            source_probe=None if source_probe is None else str(source_probe),
        )


_NXTOMO_METADATA_FIELDS = frozenset(NXTomoMetadata.__dataclass_fields__)


@dataclass(slots=True)
class LoadedNXTomo:
    """Typed NXtomo payload returned by ``load_nxtomo()``.

    Parameters
    ----------
    projections : numpy.ndarray or lazy HDF5 stack
        Projection or detector-frame stack in (view, v, u) order.
    metadata : NXTomoMetadata
        Persisted acquisition and reconstruction metadata.
    source : dict or None, optional
        Radiation source metadata.
    disk_volume_axes_order : str or None, optional
        Original on-disk volume axis order.
    """

    projections: np.ndarray | Hdf5Frames | None
    metadata: NXTomoMetadata
    source: SourceInfo | None = None
    disk_volume_axes_order: str | None = None

    @classmethod
    def from_dataset(cls, data: Mapping[str, DatasetValue]) -> LoadedNXTomo:
        """Build a loaded payload from a generic dataset mapping.

        Parameters
        ----------
        data : mapping
            Loaded dataset fields and metadata.

        Returns
        -------
        LoadedNXTomo
            Typed dataset metadata or payload with copied array metadata.

        Raises
        ------
        KeyError
            If projections are absent.
        """
        return cls(
            projections=None
            if data.get("projections") is None
            else _frames_or_array(data["projections"]),
            metadata=NXTomoMetadata.from_dataset(data),
            source=_source_record(data.get("source")),
            disk_volume_axes_order=(
                None
                if data.get("disk_volume_axes_order") is None
                else str(data.get("disk_volume_axes_order"))
            ),
        )

    def __getattr__(self, name: str) -> object:
        if name in _NXTOMO_METADATA_FIELDS:
            return getattr(self.metadata, name)
        raise AttributeError(name)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key == "projections":
            return True
        if key in ("source", "disk_volume_axes_order"):
            return getattr(self, key) is not None
        if key in _NXTOMO_METADATA_FIELDS:
            return getattr(self.metadata, key) is not None
        return key in (self.metadata.geometry_meta or {})

    def __getitem__(self, key: str) -> DatasetValue:
        value = self.get(key)
        if value is None and key not in self:
            raise KeyError(key)
        return value

    def get(self, key: str, default: DatasetValue | None = None) -> DatasetValue | None:
        """Return a metadata or payload value by mapping-style key.

        Parameters
        ----------
        key : str
            Payload, metadata or additional geometry key.
        default : object, optional
            Value for an absent metadata or geometry key; defaults to None.

        Returns
        -------
        object
            Stored value, or the default when that metadata key is absent.
        """
        if key == "projections":
            return self.projections
        if key == "source":
            return self.source
        if key == "disk_volume_axes_order":
            return self.disk_volume_axes_order
        if key in _NXTOMO_METADATA_FIELDS:
            value = getattr(self.metadata, key)
            return default if value is None else value
        geom_meta = self.metadata.geometry_meta or {}
        return geom_meta.get(key, default)

    def to_dataset_dict(self) -> LoadedDataset:
        """Return this payload as a generic dataset mapping.

        Returns
        -------
        dict
            Dataset mapping with projections materialized and typed geometry converted to mappings.
        """
        data: LoadedDataset = {}
        if self.projections is not None:  # none in a reconstruction saved without its scan
            data["projections"] = np.asarray(self.projections)
        for field_name in _NXTOMO_METADATA_FIELDS:
            value = getattr(self.metadata, field_name)
            if value is None:
                continue
            if isinstance(value, Grid | Detector):
                data[field_name] = value.to_dict()
            else:
                data[field_name] = value
        if self.source is not None:
            data["source"] = self.source.copy()
        if self.disk_volume_axes_order is not None:
            data["disk_volume_axes_order"] = self.disk_volume_axes_order
        return data

    def copy_metadata(self) -> NXTomoMetadata:
        """Return a writable metadata copy suitable for save/update workflows.

        Returns
        -------
        NXTomoMetadata
            Writable persistence metadata with copied array fields.
        """
        return NXTomoMetadata.from_dataset(self.to_dataset_dict())

    def geometry_inputs(self) -> dict[str, DatasetValue]:
        """Return the subset of metadata needed to materialize geometry.

        Returns
        -------
        dict
            Detector, angles, grid and additional geometry metadata for reconstruction.

        Raises
        ------
        ValueError
            If detector metadata or rotation angles are absent.
        """
        detector = self.metadata.detector
        if detector is None:
            raise ValueError("NXtomo payload is missing detector metadata")
        angles = self.metadata.angles
        if angles is None:
            raise ValueError("NXtomo payload is missing rotation angles")

        payload: dict[str, DatasetValue] = {
            "detector": detector.to_dict() if isinstance(detector, Detector) else detector,
            "angles": np.asarray(angles, dtype=np.float32),
            "geometry_type": self.metadata.geometry_type,
        }
        grid = self.metadata.grid
        if grid is not None:
            payload["grid"] = grid.to_dict() if isinstance(grid, Grid) else grid
        arrays = {
            "angle_offset": self.metadata.angle_offset,
            "align_params": self.metadata.align_params,
        }
        records = {
            "misalign_spec": self.metadata.misalign_spec,
            "align_gauge": self.metadata.align_gauge,
        }
        payload.update(
            {name: np.asarray(array) for name, array in arrays.items() if array is not None}
        )
        payload.update({name: record for name, record in records.items() if record is not None})
        geom_meta = self.metadata.geometry_meta or {}
        for key in _GEOMETRY_PASSTHROUGH:
            value = geom_meta.get(key)
            if value is None:
                continue
            if key in ("tilt", "detector_roll"):
                if not isinstance(value, str | int | float):
                    raise TypeError(f"saved {key} must be a number")
                value = float(value)
            elif key == "tilt_about":
                value = str(value)
            payload[key] = value
        return payload


class ValidationReport(TypedDict):
    """Lightweight validation report for an NXtomo file."""

    issues: list[str]


def _frames_or_array(value: object) -> np.ndarray | Hdf5Frames:
    """Lazily read frames as they are; anything else as an array."""
    if isinstance(value, Hdf5Frames):
        return value
    array = np.asarray(value)
    return array[:, None, :] if array.ndim == 2 else array
