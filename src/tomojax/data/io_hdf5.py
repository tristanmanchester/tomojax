"""HDF5/NXtomo IO for TomoJAX v2.

Provides utilities to read/write datasets in HDF5 using the NeXus (NXtomo)
conventions with TomoJAX-specific extras under /entry/processing/tomojax.

This module focuses on accessibility at beamlines and interop with existing
pipelines, while keeping simulation-friendly metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from typing import Mapping, TypedDict

import h5py
import numpy as np

from ..core.geometry.base import Detector, DetectorDict, Grid, GridDict
from ..utils.axes import (
    DISK_VOLUME_AXES,
    INTERNAL_VOLUME_AXES,
    VOLUME_AXES_ATTR,
    axes_to_perm,
    infer_disk_axes,
    transpose_volume,
)


LOG = logging.getLogger(__name__)

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class SourceInfo(TypedDict, total=False):
    name: str | None
    type: str | None
    probe: str | None


type DatasetValue = np.ndarray | JsonValue | GridDict | DetectorDict | SourceInfo
type LoadedDataset = dict[str, DatasetValue]


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
        return cls(
            thetas_deg=data.get("thetas_deg"),
            image_key=data.get("image_key"),
            grid=data.get("grid"),
            detector=data.get("detector"),
            geometry_type="parallel" if geometry_type is None else str(geometry_type),
            geometry_meta=data.get("geometry_meta"),
            volume=data.get("volume"),
            align_params=data.get("align_params"),
            align_gauge=data.get("align_gauge"),
            angle_offset_deg=data.get("angle_offset_deg"),
            misalign_spec=data.get("misalign_spec"),
            simulation_artefacts=data.get("simulation_artefacts"),
            frame=None if data.get("frame") is None else str(data.get("frame")),
            sample_name=(
                None if data.get("sample_name") is None else str(data.get("sample_name"))
            ),
            source_name=None if source_name is None else str(source_name),
            source_type=None if source_type is None else str(source_type),
            source_probe=None if source_probe is None else str(source_probe),
            volume_axes_order=(
                DISK_VOLUME_AXES
                if data.get("volume_axes_order") is None
                else str(data.get("volume_axes_order"))
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

    def __getattr__(self, name: str):
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
        if key == "projections":
            return self.projections
        if key == "source":
            return self.source
        if key == "disk_volume_axes_order":
            return self.disk_volume_axes_order
        if key == "volume_axes_source":
            return self.volume_axes_source
        if key in _NXTOMO_METADATA_FIELDS:
            return getattr(self.metadata, key)
        geom_meta = self.metadata.geometry_meta or {}
        return geom_meta.get(key, default)

    def to_dataset_dict(self) -> LoadedDataset:
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
        return payload


class ValidationReport(TypedDict):
    issues: list[str]


def _axes_warnings_silenced() -> bool:
    return os.environ.get("TOMOJAX_AXES_SILENCE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _axes_log_warning(message: str, *args: object) -> None:
    if not _axes_warnings_silenced():
        LOG.warning(message, *args)


def _attr_to_str(v: object, default: str | None = None) -> str | None:
    """Robustly convert an HDF5 attribute to a Python string.

    Handles h5py special string dtypes, numpy scalars/arrays, bytes, and plain str.
    """
    if v is None:
        return default
    try:
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
                return _attr_to_str(v, default)
            # 1-D array of length 1 or more — take first element
            if v.size >= 1:
                return _attr_to_str(v.flat[0], default)
        return str(v)
    except Exception:
        return default


def _ensure_group(root: h5py.Group, name: str, nx_class: str | None = None) -> h5py.Group:
    g = root.require_group(name)
    if nx_class:
        g.attrs["NX_class"] = nx_class
    return g


def _write_string_attr(obj: h5py.Group | h5py.Dataset, key: str, value: str) -> None:
    obj.attrs[key] = np.array(value, dtype=h5py.string_dtype(encoding="utf-8"))


def _normalize_geometry_type(geometry_type: str | None) -> str:
    gtype = "parallel" if geometry_type is None else str(geometry_type).strip().lower()
    if gtype == "parallel":
        return gtype
    if gtype in {"lamino", "laminography"}:
        return "lamino"
    raise ValueError(
        f"Unsupported geometry_type {geometry_type!r}; expected 'parallel' or 'lamino'"
    )


def _load_json_mapping_attr(
    raw_attr: object,
    *,
    path: str,
    context: str,
) -> JsonObject | None:
    if raw_attr is None:
        return None
    payload = _attr_to_str(raw_attr)
    if not payload:
        return None
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        _axes_log_warning(
            "load_nxtomo: ignoring malformed %s JSON for %s: %s",
            context,
            path,
            exc,
        )
        return None
    if not isinstance(loaded, dict):
        _axes_log_warning(
            "load_nxtomo: ignoring non-object %s JSON for %s",
            context,
            path,
        )
        return None
    return loaded


def _default_detector_meta(projections: np.ndarray) -> DetectorDict:
    nv, nu = int(projections.shape[1]), int(projections.shape[2])
    return {
        "nu": nu,
        "nv": nv,
        "du": 1.0,
        "dv": 1.0,
        "det_center": [0.0, 0.0],
    }


def _detector_group(entry: h5py.Group) -> h5py.Group | None:
    inst_grp = entry.get("instrument")
    if inst_grp is None:
        return None
    return inst_grp.get("detector")


def _load_image_key(entry: h5py.Group, *, n_views: int, path: str) -> np.ndarray:
    det_grp = _detector_group(entry)
    if det_grp is None or "image_key" not in det_grp:
        _axes_log_warning(
            "load_nxtomo: missing image_key for %s; defaulting to zeros",
            path,
        )
        return np.zeros((n_views,), dtype=np.int32)
    return np.asarray(det_grp["image_key"][...], dtype=np.int32)


def _load_rotation_angles(entry: h5py.Group, *, n_views: int, path: str) -> np.ndarray:
    sample_grp = entry.get("sample")
    trans_grp = None if sample_grp is None else sample_grp.get("transformations")
    if trans_grp is None or "rotation_angle" not in trans_grp:
        _axes_log_warning(
            "load_nxtomo: missing rotation_angle for %s; defaulting to zeros",
            path,
        )
        return np.zeros((n_views,), dtype=np.float32)
    return np.asarray(trans_grp["rotation_angle"][...], dtype=np.float32)


def _load_geometry_metadata(
    out: LoadedDataset,
    entry: h5py.Group,
    *,
    path: str,
) -> None:
    geom = entry.get("geometry")
    geom_type = "parallel"
    if geom is not None and "type" in geom.attrs:
        raw_type = _attr_to_str(geom.attrs.get("type"), default="parallel")
        geom_type = _normalize_geometry_type(raw_type)
    out["geometry_type"] = geom_type
    if geom is None:
        return
    meta_dict = _load_json_mapping_attr(
        geom.attrs.get("geometry_meta_json"),
        path=path,
        context="geometry metadata",
    )
    if meta_dict is None:
        return
    out["geometry_meta"] = meta_dict
    for key, val in meta_dict.items():
        out.setdefault(key, val)


def _load_grid_metadata(out: LoadedDataset, entry: h5py.Group, *, path: str) -> None:
    grid_dict = _load_json_mapping_attr(
        entry.attrs.get("grid_meta_json"),
        path=path,
        context="grid metadata",
    )
    if grid_dict is not None:
        out["grid"] = grid_dict


def _load_detector_metadata(
    out: LoadedDataset,
    entry: h5py.Group,
    projections: np.ndarray,
    *,
    path: str,
) -> None:
    det_grp = _detector_group(entry)
    detector_dict = None
    if det_grp is not None:
        detector_dict = _load_json_mapping_attr(
            det_grp.attrs.get("detector_meta_json"),
            path=path,
            context="detector metadata",
        )
    if detector_dict is None:
        _axes_log_warning(
            "load_nxtomo: missing detector metadata for %s; synthesizing unit detector from projection shape",
            path,
        )
        detector_dict = _default_detector_meta(projections)
    out["detector"] = detector_dict


def _load_source_metadata(out: LoadedDataset, entry: h5py.Group) -> None:
    inst_grp = entry.get("instrument")
    if inst_grp is None:
        return
    source_grp = inst_grp.get("SOURCE") or inst_grp.get("source")
    if source_grp is None:
        return
    source_info: SourceInfo = {}
    for key in ("name", "type", "probe"):
        if key in source_grp:
            source_info[key] = _attr_to_str(source_grp[key][()])
    if not source_info:
        return
    out["source"] = source_info
    if "name" in source_info:
        out["source_name"] = source_info["name"]
    if "type" in source_info:
        out["source_type"] = source_info["type"]
    if "probe" in source_info:
        out["source_probe"] = source_info["probe"]


def _load_processing_metadata(
    out: LoadedDataset,
    entry: h5py.Group,
    *,
    path: str,
) -> tuple[np.ndarray | None, str | None]:
    processing_grp = entry.get("processing")
    tomojax_grp = None if processing_grp is None else processing_grp.get("tomojax")
    if tomojax_grp is None:
        return None, None

    volume_raw = tomojax_grp["volume"][...] if "volume" in tomojax_grp else None
    volume_axes_attr = _attr_to_str(tomojax_grp.attrs.get(VOLUME_AXES_ATTR))

    frame_attr = tomojax_grp.attrs.get("frame")
    if frame_attr is not None:
        frame = _attr_to_str(frame_attr)
        if frame:
            out["frame"] = frame

    align_grp = tomojax_grp.get("align")
    if align_grp is not None:
        if "thetas" in align_grp:
            out["align_params"] = align_grp["thetas"][...]
        align_gauge = _load_json_mapping_attr(
            align_grp.attrs.get("gauge_fix_json"),
            path=path,
            context="alignment gauge fix",
        )
        if align_gauge is not None:
            out["align_gauge"] = align_gauge
        if "angle_offset_deg" in align_grp:
            out["angle_offset_deg"] = align_grp["angle_offset_deg"][...]
        misalign_spec = _load_json_mapping_attr(
            align_grp.attrs.get("misalign_spec_json"),
            path=path,
            context="misalignment spec",
        )
        if misalign_spec is not None:
            out["misalign_spec"] = misalign_spec

    simulation_grp = tomojax_grp.get("simulation")
    if simulation_grp is not None:
        simulation_artefacts = _load_json_mapping_attr(
            simulation_grp.attrs.get("artefacts_json"),
            path=path,
            context="simulation artefacts",
        )
        if simulation_artefacts is not None:
            out["simulation_artefacts"] = simulation_artefacts

    return volume_raw, volume_axes_attr


def _load_sample_metadata(out: LoadedDataset, entry: h5py.Group) -> None:
    sample_grp = entry.get("sample")
    if sample_grp is not None and "name" in sample_grp:
        out["sample_name"] = _attr_to_str(sample_grp["name"][()], default=None)


def save_nxtomo(
    path: str,
    projections: np.ndarray,
    *,
    metadata: NXTomoMetadata | None = None,
    compression: str = "lzf",
    overwrite: bool = True,
) -> None:
    """Write a dataset to an HDF5 file using NXtomo + TomoJAX extras.

    `metadata` owns the persistence contract: geometry, detector/grid definitions,
    optional reconstructed volume, and TomoJAX-specific extras. Volume inputs in
    `metadata.volume` are always expected in internal `xyz` order; set
    `metadata.volume_axes_order` to control how that internal volume is serialized
    on disk.
    """
    # Ensure parent directory exists
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    meta = metadata or NXTomoMetadata()
    if overwrite:
        mode = "w"
    else:
        mode = "x"

    proj = np.asarray(projections)
    if proj.ndim != 3:
        raise ValueError("projections must be (n_views, nv, nu)")
    n_views, nv, nu = proj.shape
    geometry_type_norm = _normalize_geometry_type(meta.geometry_type)

    disk_axes = meta.volume_axes_order.lower()
    try:
        axes_to_perm(INTERNAL_VOLUME_AXES, disk_axes)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            "metadata.volume_axes_order must be a permutation of 'xyz', "
            f"got {meta.volume_axes_order!r}"
        ) from exc

    with h5py.File(path, mode) as f:
        entry = _ensure_group(f, "entry", "NXentry")
        _write_string_attr(entry, "definition", "NXtomo")

        # Geometry metadata (mark as NXcollection so common viewers display it)
        geom = _ensure_group(entry, "geometry", "NXcollection")
        _write_string_attr(geom, "type", geometry_type_norm)
        if meta.geometry_meta:
            geom.attrs["geometry_meta_json"] = json.dumps(meta.geometry_meta)

        # Instrument / detector
        inst = _ensure_group(entry, "instrument", "NXinstrument")
        det = _ensure_group(inst, "detector", "NXdetector")
        det.create_dataset(
            "data",
            data=proj,
            chunks=(1, min(256, nv), min(256, nu)),
            compression=compression,
        )
        det["data"].attrs["long_name"] = "projections"
        if meta.image_key is None:
            im_key = np.zeros((n_views,), dtype=np.int32)
        else:
            im_key = np.asarray(meta.image_key, dtype=np.int32)
            if im_key.shape != (n_views,):
                raise ValueError("image_key must be shape (n_views,)")
        det.create_dataset("image_key", data=im_key, dtype=np.int32)

        # Pixel geometry if provided
        if meta.detector is not None:
            det_dict = meta.detector if isinstance(meta.detector, dict) else meta.detector.to_dict()
            det.attrs["detector_meta_json"] = json.dumps(det_dict)
            # Store basic pixel sizes (units arbitrary for sims)
            det.create_dataset("x_pixel_size", data=np.asarray(det_dict.get("du", 1.0)))
            det["x_pixel_size"].attrs["units"] = "pixel"
            det.create_dataset("y_pixel_size", data=np.asarray(det_dict.get("dv", 1.0)))
            det["y_pixel_size"].attrs["units"] = "pixel"

        # Sample and transformations (angles)
        sample = _ensure_group(entry, "sample", "NXsample")
        trans = _ensure_group(sample, "transformations", "NXtransformations")
        if meta.thetas_deg is None:
            thetas_deg = np.zeros((n_views,), dtype=np.float32)
        else:
            thetas_deg = np.asarray(meta.thetas_deg, dtype=np.float32)
            if thetas_deg.shape != (n_views,):
                raise ValueError("thetas_deg must be (n_views,)")
        d_angle = trans.create_dataset("rotation_angle", data=thetas_deg)
        d_angle.attrs["units"] = "degree"
        _write_string_attr(d_angle, "transformation_type", "rotation")
        # Rotation around +z by default (matching phi in current code)
        trans.create_dataset("rotation_axis", data=np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        _write_string_attr(trans, "depends_on", "rotation_angle")
        sample_name_str = "sample" if meta.sample_name is None else str(meta.sample_name)
        sample.create_dataset(
            "name",
            data=np.array(sample_name_str, dtype=h5py.string_dtype(encoding="utf-8")),
        )

        # NXdata linking for default plot (optional)
        data_grp = _ensure_group(entry, "data", "NXdata")
        # store projections also under /entry/data/projections for convenience
        data_grp["projections"] = det["data"]
        _write_string_attr(data_grp, "signal", "projections")

        # Source metadata (optional but recommended)
        src_grp = _ensure_group(inst, "SOURCE", "NXsource")
        if meta.source_name is not None:
            src_grp.create_dataset(
                "name",
                data=np.array(str(meta.source_name), dtype=h5py.string_dtype(encoding="utf-8")),
            )
        if meta.source_type is not None:
            src_grp.create_dataset(
                "type",
                data=np.array(str(meta.source_type), dtype=h5py.string_dtype(encoding="utf-8")),
            )
        if meta.source_probe is not None:
            src_grp.create_dataset(
                "probe",
                data=np.array(str(meta.source_probe), dtype=h5py.string_dtype(encoding="utf-8")),
            )

        # Grid metadata
        if meta.grid is not None:
            gdict = meta.grid if isinstance(meta.grid, dict) else meta.grid.to_dict()
            entry.attrs["grid_meta_json"] = json.dumps(gdict)

        # Optional GT / reconstructed volume and TomoJAX metadata
        if meta.volume is not None:
            processing = _ensure_group(entry, "processing", "NXprocess")
            tj = _ensure_group(processing, "tomojax", "NXcollection")
            vol_data = np.asarray(meta.volume)
            if vol_data.ndim != 3:
                raise ValueError("volume must be 3D (nx, ny, nz)")
            if disk_axes != INTERNAL_VOLUME_AXES:
                vol_data = np.asarray(transpose_volume(vol_data, INTERNAL_VOLUME_AXES, disk_axes))
            vol = tj.create_dataset(
                "volume",
                data=vol_data,
                chunks=True,
                compression=compression,
            )
            vol.attrs["long_name"] = "ground_truth_volume"
            _write_string_attr(tj, VOLUME_AXES_ATTR, disk_axes)
            # Persist volume frame metadata if provided (e.g., 'sample' or 'lab')
            if meta.frame is not None:
                _write_string_attr(tj, "frame", str(meta.frame))

        # Optional alignment params and misalignment metadata
        if (
            meta.align_params is not None
            or meta.align_gauge is not None
            or meta.angle_offset_deg is not None
            or meta.misalign_spec is not None
        ):
            processing = _ensure_group(entry, "processing", "NXprocess")
            tj = _ensure_group(processing, "tomojax", "NXcollection")
            align_grp = _ensure_group(tj, "align", "NXcollection")
            if meta.align_params is not None:
                dset = align_grp.create_dataset(
                    "thetas",
                    data=np.asarray(meta.align_params, dtype=np.float32),
                    chunks=True,
                    compression=compression,
                )
                dset.attrs["columns"] = np.array(
                    ["alpha", "beta", "phi", "dx", "dz"], dtype=h5py.string_dtype()
                )
            if meta.align_gauge is not None:
                align_grp.attrs["gauge_fix_json"] = json.dumps(meta.align_gauge)
            if meta.angle_offset_deg is not None:
                align_grp.create_dataset(
                    "angle_offset_deg",
                    data=np.asarray(meta.angle_offset_deg, dtype=np.float32),
                    chunks=True,
                    compression=compression,
                )
            if meta.misalign_spec is not None:
                align_grp.attrs["misalign_spec_json"] = json.dumps(meta.misalign_spec)

        # Optional simulation artefact metadata
        if meta.simulation_artefacts is not None:
            processing = _ensure_group(entry, "processing", "NXprocess")
            tj = _ensure_group(processing, "tomojax", "NXcollection")
            simulation_grp = _ensure_group(tj, "simulation", "NXcollection")
            simulation_grp.attrs["artefacts_json"] = json.dumps(meta.simulation_artefacts)


def load_nxtomo(path: str) -> LoadedNXTomo:
    """Load an NXtomo dataset and TomoJAX extras.

    Returns a ``LoadedNXTomo`` payload with typed accessors for projections and
    persisted metadata. ``grid`` is present when the file stores grid metadata
    or when a saved volume allows a unit-grid fallback. Volumes are returned in
    internal ``xyz`` order.
    """
    out: LoadedDataset = {}
    volume_raw: np.ndarray | None = None
    volume_axes_attr: str | None = None

    with h5py.File(path, "r") as f:
        entry = f["/entry"]
        proj = None
        det_grp = _detector_group(entry)
        if det_grp is not None and "data" in det_grp:
            proj = det_grp["data"][...]
        elif "data" in entry and "projections" in entry["data"]:
            proj = entry["data/projections"][...]
        elif "projections" in entry:
            proj = entry["projections"][...]
        if proj is None:
            raise KeyError("Could not find projections dataset under /entry")
        out["projections"] = proj
        n_views = proj.shape[0]
        out["image_key"] = _load_image_key(entry, n_views=n_views, path=path)
        out["thetas_deg"] = _load_rotation_angles(entry, n_views=n_views, path=path)
        _load_geometry_metadata(out, entry, path=path)
        _load_grid_metadata(out, entry, path=path)
        _load_detector_metadata(out, entry, proj, path=path)
        _load_source_metadata(out, entry)
        volume_raw, volume_axes_attr = _load_processing_metadata(out, entry, path=path)
        _load_sample_metadata(out, entry)

        # Normalize volume axes if present
        if volume_raw is not None:
            grid_hint = out.get("grid")
            attr_norm = None
            if volume_axes_attr:
                try:
                    axes_to_perm(INTERNAL_VOLUME_AXES, volume_axes_attr)
                    attr_norm = volume_axes_attr.lower()
                except ValueError:
                    _axes_log_warning(
                        "Ignoring malformed volume_axes_order=%r on %s",
                        volume_axes_attr,
                        path,
                    )
            disk_axes = attr_norm or infer_disk_axes(volume_raw.shape, grid_hint)
            source = "attr" if attr_norm else "heuristic"
            if disk_axes is None and source == "heuristic":
                # Preserve legacy behavior for old files that omitted both the
                # explicit volume axis attribute and grid metadata.
                disk_axes = INTERNAL_VOLUME_AXES
            volume_np = np.asarray(volume_raw)
            disk_order: str
            if disk_axes == DISK_VOLUME_AXES:
                if source == "heuristic":
                    _axes_log_warning(
                        "load_nxtomo: inferred disk volume axes zyx for %s; transposing to internal xyz",
                        path,
                    )
                volume_np = np.asarray(
                    transpose_volume(volume_np, DISK_VOLUME_AXES, INTERNAL_VOLUME_AXES)
                )
                disk_order = DISK_VOLUME_AXES
            elif disk_axes == INTERNAL_VOLUME_AXES:
                if source == "heuristic":
                    _axes_log_warning(
                        "load_nxtomo: assuming legacy xyz disk volume order for %s",
                        path,
                    )
                disk_order = INTERNAL_VOLUME_AXES if source == "attr" else "xyz_legacy"
            elif disk_axes is None:
                disk_order = "unknown"
                _axes_log_warning(
                    "load_nxtomo: unable to infer volume axis order for %s (shape %s)",
                    path,
                    volume_raw.shape,
                )
            else:
                volume_np = np.asarray(transpose_volume(volume_np, disk_axes, INTERNAL_VOLUME_AXES))
                disk_order = disk_axes
            out["volume"] = volume_np
            out["volume_axes_order"] = INTERNAL_VOLUME_AXES
            out["disk_volume_axes_order"] = disk_order
            out["volume_axes_source"] = source

        # Grid fallback from volume if grid meta missing
        if "grid" not in out and volume_raw is not None and "volume" in out:
            vol = out["volume"]
            nx, ny, nz = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
            _axes_log_warning(
                "load_nxtomo: missing grid metadata for %s; synthesizing unit grid from loaded volume shape",
                path,
            )
            out["grid"] = {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "vx": 1.0,
                "vy": 1.0,
                "vz": 1.0,
            }
    return LoadedNXTomo.from_dataset(out)


def validate_nxtomo(path: str) -> ValidationReport:
    """Lightweight schema checks. Returns a report dict; empty `issues` means OK."""
    report: ValidationReport = {"issues": []}
    try:
        with h5py.File(path, "r") as f:
            if "/entry" not in f:
                report["issues"].append("Missing /entry")
                return report
            e = f["/entry"]
            definition = _attr_to_str(e.attrs.get("definition"))
            if definition != "NXtomo":
                report["issues"].append("entry/definition must be 'NXtomo'")
            # Required datasets
            if "instrument/detector/data" not in e:
                report["issues"].append("Missing instrument/detector/data")
            if "sample/transformations/rotation_angle" not in e:
                report["issues"].append("Missing sample/transformations/rotation_angle")
            if "instrument/detector/image_key" not in e:
                report["issues"].append("Missing instrument/detector/image_key")
            if "sample" not in e or "name" not in e["sample"]:
                report["issues"].append("Missing sample/name")
            # Basic shapes
            n_views = None
            if "instrument/detector/data" in e:
                d = e["instrument/detector/data"]
                if d.ndim != 3:
                    report["issues"].append("instrument/detector/data must be 3D (n_views, nv, nu)")
                else:
                    n_views = d.shape[0]
            if "instrument/detector/image_key" in e:
                ik = e["instrument/detector/image_key"]
                if ik.ndim != 1:
                    report["issues"].append("instrument/detector/image_key must be 1D (n_views,)")
                if n_views is not None and ik.shape[0] != n_views:
                    report["issues"].append("image_key length must match #views in detector/data")
                if ik.dtype.kind not in {"i", "u"}:
                    report["issues"].append("image_key must use integer dtype")
            if "sample/transformations/rotation_angle" in e:
                ang = e["sample/transformations/rotation_angle"]
                if ang.ndim != 1:
                    report["issues"].append("rotation_angle must be 1D (n_views,)")
                if n_views is not None and ang.shape[0] != n_views:
                    report["issues"].append(
                        "rotation_angle length must match #views in detector/data"
                    )
                units = _attr_to_str(ang.attrs.get("units"))
                if units != "degree":
                    report["issues"].append("rotation_angle units attr should be 'degree'")
    except Exception as exc:  # pragma: no cover (defensive)
        report["issues"].append(f"Exception during validation: {exc}")
    return report


def save_npz(path: str, projections: np.ndarray, **meta: DatasetValue) -> None:
    """Simple NPZ saver for tiny tests or interop."""
    np.savez_compressed(path, projections=projections, **meta)


def load_npz(path: str) -> LoadedDataset:
    with np.load(path, allow_pickle=True) as z:
        out: LoadedDataset = {}
        for k in z.files:
            val = z[k]
            if isinstance(val, np.ndarray) and val.shape == () and val.dtype == object:
                out[k] = val.item()
            else:
                out[k] = val
        return out


def convert(in_path: str, out_path: str) -> None:
    """Convert between .npz and .nxs based on file extension."""
    if in_path.endswith(".npz") and out_path.endswith((".nxs", ".h5", ".hdf5")):
        data = load_npz(in_path)
        save_nxtomo(
            out_path,
            np.asarray(data["projections"]),
            metadata=NXTomoMetadata.from_dataset(data),
        )
    elif in_path.endswith((".nxs", ".h5", ".hdf5")) and out_path.endswith(".npz"):
        data = load_nxtomo(in_path)
        save_npz(out_path, **data.to_dataset_dict())
    else:
        raise ValueError("Unsupported conversion. Use .npz <-> .nxs/.h5/.hdf5")
