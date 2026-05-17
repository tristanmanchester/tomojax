from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from ._preprocess_config import _IMAGE_KEY_PATH
from .io_hdf5 import NXTomoMetadata


def _attr_to_str(value: object, default: str | None = None) -> str | None:
    if value is None:
        return default
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return _attr_to_str(value.item(), default)
            if value.size >= 1:
                return _attr_to_str(value.flat[0], default)
        return str(value)
    except Exception:
        return default


def _json_mapping_attr(value: object) -> dict[str, Any] | None:
    text = _attr_to_str(value)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _dataset_at(file: h5py.File, path: str) -> h5py.Dataset | None:
    obj = file.get(path)
    return obj if isinstance(obj, h5py.Dataset) else None


def _resolve_dataset(
    file: h5py.File,
    *,
    explicit_path: str | None,
    default_paths: tuple[str, ...],
    label: str,
) -> tuple[str, h5py.Dataset]:
    if explicit_path is not None:
        dataset = _dataset_at(file, explicit_path)
        if dataset is None:
            raise KeyError(f"Could not find {label} dataset at {explicit_path!r}")
        return explicit_path, dataset

    for path in default_paths:
        dataset = _dataset_at(file, path)
        if dataset is not None:
            return path, dataset
    raise KeyError(f"Could not find {label} dataset in expected NXtomo locations")


def _find_unique_image_key(file: h5py.File) -> tuple[str, h5py.Dataset]:
    default = _dataset_at(file, _IMAGE_KEY_PATH)
    if default is not None:
        return _IMAGE_KEY_PATH, default

    matches: list[tuple[str, h5py.Dataset]] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset) and name.rsplit("/", 1)[-1] == "image_key":
            matches.append(("/" + name.lstrip("/"), obj))

    file.visititems(visitor)
    if not matches:
        raise KeyError(
            "Could not find image_key dataset at /entry/instrument/detector/image_key "
            "or by unique dataset name"
        )
    if len(matches) > 1:
        paths = ", ".join(path for path, _dataset in matches)
        raise KeyError(
            f"Found multiple image_key datasets; pass --image-key-path explicitly: {paths}"
        )
    return matches[0]


def _resolve_image_key_dataset(
    file: h5py.File,
    explicit_path: str | None,
) -> tuple[str, h5py.Dataset]:
    if explicit_path is not None:
        dataset = _dataset_at(file, explicit_path)
        if dataset is None:
            raise KeyError(f"Could not find image_key dataset at {explicit_path!r}")
        return explicit_path, dataset
    return _find_unique_image_key(file)


def _read_scalar_dataset(group: h5py.Group | None, name: str) -> float | None:
    if group is None:
        return None
    dataset = group.get(name)
    if not isinstance(dataset, h5py.Dataset):
        return None
    try:
        return float(np.asarray(dataset[...]).reshape(-1)[0])
    except (TypeError, ValueError, OverflowError, IndexError) as exc:
        raise ValueError(
            f"Could not read scalar detector metadata dataset {dataset.name!r}"
        ) from exc


def _detector_meta_from_raw(
    entry: h5py.Group,
    *,
    nv: int,
    nu: int,
    crop_bounds: tuple[int, int, int, int] | None = None,
    original_nv: int | None = None,
    original_nu: int | None = None,
) -> dict[str, Any]:
    detector = entry.get("instrument/detector")
    meta = (
        _json_mapping_attr(detector.attrs.get("detector_meta_json"))
        if isinstance(detector, h5py.Group)
        else None
    )
    if meta is None:
        meta = {}
    detector_group = detector if isinstance(detector, h5py.Group) else None
    x_pixel_size = _read_scalar_dataset(detector_group, "x_pixel_size")
    y_pixel_size = _read_scalar_dataset(detector_group, "y_pixel_size")
    du = float(meta.get("du", 1.0 if x_pixel_size is None else x_pixel_size))
    dv = float(meta.get("dv", 1.0 if y_pixel_size is None else y_pixel_size))
    det_center = meta.get("det_center", [0.0, 0.0])
    try:
        cx = float(det_center[0])
        cz = float(det_center[1])
    except (TypeError, ValueError, IndexError):
        cx, cz = 0.0, 0.0

    if crop_bounds is not None:
        y0, y1, x0, x1 = crop_bounds
        old_nv = int(original_nv if original_nv is not None else meta.get("nv", nv))
        old_nu = int(original_nu if original_nu is not None else meta.get("nu", nu))
        cx = cx + ((float(x0) + float(x1) - float(old_nu)) / 2.0) * du
        cz = cz + ((float(y0) + float(y1) - float(old_nv)) / 2.0) * dv

    return {
        "nu": int(nu),
        "nv": int(nv),
        "du": du,
        "dv": dv,
        "det_center": [cx, cz],
    }


def _source_info(entry: h5py.Group) -> dict[str, str | None]:
    instrument = entry.get("instrument")
    if not isinstance(instrument, h5py.Group):
        return {}
    source = instrument.get("SOURCE") or instrument.get("source")
    if not isinstance(source, h5py.Group):
        return {}
    out: dict[str, str | None] = {}
    for key in ("name", "type", "probe"):
        if key in source:
            out[key] = _attr_to_str(source[key][()], default=None)
    return out


def _metadata_from_raw(
    entry: h5py.Group,
    *,
    thetas_deg: np.ndarray,
    output_image_key: np.ndarray,
    nv: int,
    nu: int,
    crop_bounds: tuple[int, int, int, int] | None = None,
    original_nv: int | None = None,
    original_nu: int | None = None,
    align_params: np.ndarray | None = None,
    angle_offset_deg: np.ndarray | None = None,
) -> NXTomoMetadata:
    geometry_type = "parallel"
    geometry_meta = None
    geometry = entry.get("geometry")
    if isinstance(geometry, h5py.Group):
        geometry_type = _attr_to_str(geometry.attrs.get("type"), default="parallel") or "parallel"
        geometry_meta = _json_mapping_attr(geometry.attrs.get("geometry_meta_json"))

    grid = _json_mapping_attr(entry.attrs.get("grid_meta_json"))
    source = _source_info(entry)
    sample = entry.get("sample")
    sample_name = "sample"
    if isinstance(sample, h5py.Group) and "name" in sample:
        sample_name = _attr_to_str(sample["name"][()], default="sample") or "sample"

    return NXTomoMetadata(
        thetas_deg=np.asarray(thetas_deg, dtype=np.float32),
        image_key=output_image_key,
        grid=grid,
        detector=_detector_meta_from_raw(
            entry,
            nv=nv,
            nu=nu,
            crop_bounds=crop_bounds,
            original_nv=original_nv,
            original_nu=original_nu,
        ),
        geometry_type=geometry_type,
        geometry_meta=geometry_meta,
        align_params=align_params,
        angle_offset_deg=angle_offset_deg,
        sample_name=sample_name,
        source_name=source.get("name", "TomoJAX preprocess"),
        source_type=source.get("type"),
        source_probe=source.get("probe", "x-ray"),
    )


def _optional_sample_metadata(
    entry: h5py.Group,
    *,
    n_sample_views: int,
    selected_sample_view_indices: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, bool]]:
    align = entry.get("processing/tomojax/align")
    if not isinstance(align, h5py.Group):
        return None, None, {"align_params": False, "angle_offset_deg": False}

    align_params = None
    angle_offset = None
    found = {"align_params": False, "angle_offset_deg": False}
    params_dataset = align.get("thetas")
    if isinstance(params_dataset, h5py.Dataset) and params_dataset.shape[0] == n_sample_views:
        align_params = np.asarray(params_dataset[...], dtype=np.float32)[
            selected_sample_view_indices
        ]
        found["align_params"] = True
    offset_dataset = align.get("angle_offset_deg")
    if isinstance(offset_dataset, h5py.Dataset) and offset_dataset.shape[0] == n_sample_views:
        angle_offset = np.asarray(offset_dataset[...], dtype=np.float32)[
            selected_sample_view_indices
        ]
        found["angle_offset_deg"] = True
    return align_params, angle_offset, found
