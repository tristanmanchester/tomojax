from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from tomojax.core.geometry.base import Detector, DetectorDict, Grid, GridDict
from tomojax.geometry import DISK_VOLUME_AXES

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


type JsonObject = dict[str, JsonValue]


class SourceInfo(TypedDict, total=False):
    """Source metadata persisted in NXtomo files."""

    name: str | None
    type: str | None
    probe: str | None


type DatasetValue = np.ndarray | JsonValue | GridDict | DetectorDict | SourceInfo


type LoadedDataset = dict[str, DatasetValue]


def _copy_array_metadata(value: object) -> np.ndarray | None:
    if value is None:
        return None
    return np.array(value, copy=True)


@dataclass(slots=True)
class NXTomoMetadata:
    """Portable metadata bundle for NXtomo persistence."""

    thetas_deg: np.ndarray | None = None
    image_key: np.ndarray | None = None
    grid: Grid | GridDict | None = None
    detector: Detector | DetectorDict | None = None
    geometry_type: str = "parallel"
    geometry_meta: JsonObject | None = None
    volume: np.ndarray | None = None
    align_params: np.ndarray | None = None
    align_gauge: JsonObject | None = None
    geometry_calibration: JsonObject | None = None
    angle_offset_deg: np.ndarray | None = None
    misalign_spec: JsonObject | None = None
    simulation_artefacts: JsonObject | None = None
    frame: str | None = None
    sample_name: str | None = "sample"
    source_name: str | None = "TomoJAX source"
    source_type: str | None = None
    source_probe: str | None = "x-ray"
    volume_axes_order: str = DISK_VOLUME_AXES

    @classmethod
    def from_dataset(cls, data: Mapping[str, DatasetValue]) -> NXTomoMetadata:
        """Build metadata from a generic loaded dataset mapping."""
        source_info = data.get("source")
        source_name = data.get("source_name")
        source_type = data.get("source_type")
        source_probe = data.get("source_probe")
        if isinstance(source_info, dict):
            if source_name is None:
                source_name = source_info.get("name")
            if source_type is None:
                source_type = source_info.get("type")
            if source_probe is None:
                source_probe = source_info.get("probe")

        geometry_type = data.get("geometry_type")
        volume_axes_order = data.get("volume_axes_order")
        disk_volume_axes_order = data.get("disk_volume_axes_order")
        if volume_axes_order is None and disk_volume_axes_order in {
            "xyz",
            "xzy",
            "yxz",
            "yzx",
            "zxy",
            "zyx",
            "unknown",
        }:
            volume_axes_order = disk_volume_axes_order
        return cls(
            thetas_deg=_copy_array_metadata(data.get("thetas_deg")),
            image_key=_copy_array_metadata(data.get("image_key")),
            grid=data.get("grid"),
            detector=data.get("detector"),
            geometry_type="parallel" if geometry_type is None else str(geometry_type),
            geometry_meta=data.get("geometry_meta"),
            volume=_copy_array_metadata(data.get("volume")),
            align_params=_copy_array_metadata(data.get("align_params")),
            align_gauge=data.get("align_gauge"),
            geometry_calibration=data.get("geometry_calibration"),
            angle_offset_deg=_copy_array_metadata(data.get("angle_offset_deg")),
            misalign_spec=data.get("misalign_spec"),
            simulation_artefacts=data.get("simulation_artefacts"),
            frame=None if data.get("frame") is None else str(data.get("frame")),
            sample_name=(None if data.get("sample_name") is None else str(data.get("sample_name"))),
            source_name=None if source_name is None else str(source_name),
            source_type=None if source_type is None else str(source_type),
            source_probe=None if source_probe is None else str(source_probe),
            volume_axes_order=(
                DISK_VOLUME_AXES if volume_axes_order is None else str(volume_axes_order)
            ),
        )


_NXTOMO_METADATA_FIELDS = frozenset(NXTomoMetadata.__dataclass_fields__)


@dataclass(slots=True)
class LoadedNXTomo:
    """Typed NXtomo payload returned by ``load_nxtomo()``."""

    projections: np.ndarray
    metadata: NXTomoMetadata
    source: SourceInfo | None = None
    disk_volume_axes_order: str | None = None
    volume_axes_source: str | None = None

    @classmethod
    def from_dataset(cls, data: Mapping[str, DatasetValue]) -> LoadedNXTomo:
        """Build a loaded payload from a generic dataset mapping."""
        source_info = data.get("source")
        return cls(
            projections=np.asarray(data["projections"]),
            metadata=NXTomoMetadata.from_dataset(data),
            source=source_info if isinstance(source_info, dict) else None,
            disk_volume_axes_order=(
                None
                if data.get("disk_volume_axes_order") is None
                else str(data.get("disk_volume_axes_order"))
            ),
            volume_axes_source=(
                None
                if data.get("volume_axes_source") is None
                else str(data.get("volume_axes_source"))
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
        if key == "source":
            return self.source is not None
        if key == "disk_volume_axes_order":
            return self.disk_volume_axes_order is not None
        if key == "volume_axes_source":
            return self.volume_axes_source is not None
        if key in _NXTOMO_METADATA_FIELDS:
            return getattr(self.metadata, key) is not None
        geom_meta = self.metadata.geometry_meta or {}
        return key in geom_meta

    def __getitem__(self, key: str) -> DatasetValue:
        value = self.get(key)
        if value is None and key not in self:
            raise KeyError(key)
        return value

    def get(self, key: str, default: DatasetValue | None = None) -> DatasetValue | None:
        """Return a metadata or payload value by mapping-style key."""
        if key == "projections":
            return self.projections
        if key == "source":
            return self.source
        if key == "disk_volume_axes_order":
            return self.disk_volume_axes_order
        if key == "volume_axes_source":
            return self.volume_axes_source
        if key in _NXTOMO_METADATA_FIELDS:
            value = getattr(self.metadata, key)
            return default if value is None else value
        geom_meta = self.metadata.geometry_meta or {}
        return geom_meta.get(key, default)

    def to_dataset_dict(self) -> LoadedDataset:
        """Return this payload as a generic dataset mapping."""
        data: LoadedDataset = {"projections": np.asarray(self.projections)}
        for field_name in _NXTOMO_METADATA_FIELDS:
            value = getattr(self.metadata, field_name)
            if value is None:
                continue
            if isinstance(value, Grid | Detector):
                data[field_name] = value.to_dict()
            else:
                data[field_name] = value
        if self.source is not None:
            data["source"] = dict(self.source)
        if self.disk_volume_axes_order is not None:
            data["disk_volume_axes_order"] = self.disk_volume_axes_order
        if self.volume_axes_source is not None:
            data["volume_axes_source"] = self.volume_axes_source
        return data

    def copy_metadata(self) -> NXTomoMetadata:
        """Return a writable metadata copy suitable for save/update workflows."""
        return NXTomoMetadata.from_dataset(self.to_dataset_dict())

    def geometry_inputs(self) -> dict[str, DatasetValue]:
        """Return the subset of metadata needed to materialize geometry."""
        detector = self.metadata.detector
        if detector is None:
            raise ValueError("NXtomo payload is missing detector metadata")
        thetas_deg = self.metadata.thetas_deg
        if thetas_deg is None:
            raise ValueError("NXtomo payload is missing rotation angles")

        payload: dict[str, DatasetValue] = {
            "detector": detector.to_dict() if isinstance(detector, Detector) else detector,
            "thetas_deg": np.asarray(thetas_deg, dtype=np.float32),
            "geometry_type": self.metadata.geometry_type,
        }
        grid = self.metadata.grid
        if grid is not None:
            payload["grid"] = grid.to_dict() if isinstance(grid, Grid) else grid
        if self.metadata.angle_offset_deg is not None:
            payload["angle_offset_deg"] = np.asarray(self.metadata.angle_offset_deg)
        if self.metadata.misalign_spec is not None:
            payload["misalign_spec"] = self.metadata.misalign_spec
        if self.metadata.align_params is not None:
            payload["align_params"] = np.asarray(self.metadata.align_params)
        if self.metadata.align_gauge is not None:
            payload["align_gauge"] = self.metadata.align_gauge
        geom_meta = self.metadata.geometry_meta or {}
        tilt_deg = geom_meta.get("tilt_deg")
        if tilt_deg is not None:
            payload["tilt_deg"] = float(tilt_deg)
        tilt_about = geom_meta.get("tilt_about")
        if tilt_about is not None:
            payload["tilt_about"] = str(tilt_about)
        axis_unit_lab = geom_meta.get("axis_unit_lab")
        if axis_unit_lab is not None:
            payload["axis_unit_lab"] = axis_unit_lab
        detector_roll_deg = geom_meta.get("detector_roll_deg")
        if detector_roll_deg is not None:
            payload["detector_roll_deg"] = float(detector_roll_deg)
        return payload


class ValidationReport(TypedDict):
    """Lightweight validation report for an NXtomo file."""

    issues: list[str]
