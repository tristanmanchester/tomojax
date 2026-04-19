"""Raw NXtomo preprocessing with flat/dark correction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .io_hdf5 import NXTomoMetadata, save_nxtomo


LOG = logging.getLogger(__name__)

PREPROCESS_SCHEMA_VERSION = 1

_PROJECTION_PATHS = (
    "/entry/instrument/detector/data",
    "/entry/data/projections",
    "/entry/projections",
)
_ANGLE_PATH = "/entry/sample/transformations/rotation_angle"
_IMAGE_KEY_PATH = "/entry/instrument/detector/image_key"
_OUTPUT_DTYPES = {"float32": np.float32, "float64": np.float64}


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for raw NXtomo flat/dark correction."""

    log: bool = False
    epsilon: float = 1e-6
    clip_min: float | None = None
    output_dtype: str = "float32"
    data_path: str | None = None
    angles_path: str | None = None
    image_key_path: str | None = None
    assume_dark_field: float | None = None
    assume_flat_field: float | None = None


@dataclass(slots=True)
class PreprocessResult:
    """Summary of a preprocessing run."""

    sample_count: int
    flat_count: int
    dark_count: int
    output_shape: tuple[int, int, int]
    output_domain: str
    warning_counts: dict[str, int]
    provenance: dict[str, Any]


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
    try:
        obj = file.get(path)
    except Exception:
        return None
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
            "Found multiple image_key datasets; pass --image-key-path explicitly: "
            f"{paths}"
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
    if group is None or name not in group:
        return None
    try:
        return float(np.asarray(group[name][...]).reshape(-1)[0])
    except Exception:
        return None


def _detector_meta_from_raw(
    entry: h5py.Group,
    *,
    nv: int,
    nu: int,
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
    return {
        "nu": int(meta.get("nu", nu)),
        "nv": int(meta.get("nv", nv)),
        "du": float(meta.get("du", 1.0 if x_pixel_size is None else x_pixel_size)),
        "dv": float(meta.get("dv", 1.0 if y_pixel_size is None else y_pixel_size)),
        "det_center": meta.get("det_center", [0.0, 0.0]),
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
) -> NXTomoMetadata:
    geometry_type = "parallel"
    geometry_meta = None
    geometry = entry.get("geometry")
    if isinstance(geometry, h5py.Group):
        geometry_type = (
            _attr_to_str(geometry.attrs.get("type"), default="parallel") or "parallel"
        )
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
        detector=_detector_meta_from_raw(entry, nv=nv, nu=nu),
        geometry_type=geometry_type,
        geometry_meta=geometry_meta,
        sample_name=sample_name,
        source_name=source.get("name", "TomoJAX preprocess"),
        source_type=source.get("type"),
        source_probe=source.get("probe", "x-ray"),
    )


def _validate_config(config: PreprocessConfig) -> np.dtype:
    if not np.isfinite(config.epsilon) or config.epsilon <= 0:
        raise ValueError("epsilon must be a positive finite value")
    if config.clip_min is not None and (
        not np.isfinite(config.clip_min) or config.clip_min <= 0
    ):
        raise ValueError("clip_min must be a positive finite value when provided")
    if config.output_dtype not in _OUTPUT_DTYPES:
        allowed = ", ".join(sorted(_OUTPUT_DTYPES))
        raise ValueError(f"output_dtype must be one of: {allowed}")
    return np.dtype(_OUTPUT_DTYPES[config.output_dtype])


def _constant_field(value: float, shape: tuple[int, int], *, label: str) -> np.ndarray:
    if not np.isfinite(value):
        raise ValueError(f"{label} override must be finite")
    return np.full(shape, float(value), dtype=np.float64)


def _warn_count(warning_counts: dict[str, int], key: str, message: str, count: int) -> None:
    warning_counts[key] = int(count)
    if count > 0:
        LOG.warning("%s: %d", message, int(count))


def _correct_frames(
    frames: np.ndarray,
    image_key: np.ndarray,
    *,
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, Any]]:
    sample_mask = image_key == 0
    flat_mask = image_key == 1
    dark_mask = image_key == 2

    samples = frames[sample_mask].astype(np.float64, copy=False)
    flats = frames[flat_mask].astype(np.float64, copy=False)
    darks = frames[dark_mask].astype(np.float64, copy=False)

    if samples.size == 0:
        raise ValueError("No sample frames found (image_key==0)")

    field_shape = tuple(int(v) for v in frames.shape[1:])
    flat_override_used = False
    dark_override_used = False
    if flats.size == 0:
        if config.assume_flat_field is None:
            raise ValueError(
                "No flat fields found (image_key==1); pass --assume-flat-field VALUE "
                "to use an explicit constant flat field"
            )
        flat_mean = _constant_field(config.assume_flat_field, field_shape, label="flat field")
        flat_override_used = True
    else:
        flat_mean = np.mean(flats, axis=0, dtype=np.float64)

    if darks.size == 0:
        if config.assume_dark_field is None:
            raise ValueError(
                "No dark fields found (image_key==2); pass --assume-dark-field VALUE "
                "to use an explicit constant dark field"
            )
        dark_mean = _constant_field(config.assume_dark_field, field_shape, label="dark field")
        dark_override_used = True
    else:
        dark_mean = np.mean(darks, axis=0, dtype=np.float64)

    warning_counts: dict[str, int] = {}
    denominator_raw = flat_mean - dark_mean
    _warn_count(
        warning_counts,
        "nonpositive_flat_denominator",
        "Flat-dark denominator values were zero or negative before epsilon clipping",
        int(np.count_nonzero(denominator_raw <= 0.0)),
    )
    denominator = np.maximum(denominator_raw, float(config.epsilon))

    sample_dark_corrected = samples - dark_mean
    transmission_raw = sample_dark_corrected / denominator
    _warn_count(
        warning_counts,
        "nonpositive_transmission",
        "Transmission values were zero or negative before clipping/log safeguards",
        int(np.count_nonzero(transmission_raw <= 0.0)),
    )

    transmission = transmission_raw
    if config.clip_min is not None:
        transmission = np.maximum(transmission, float(config.clip_min))

    if config.log:
        log_min = max(float(config.epsilon), float(config.clip_min or config.epsilon))
        output = -np.log(np.maximum(transmission, log_min))
        output_domain = "absorption"
    else:
        output = transmission
        output_domain = "transmission"

    nonfinite = int(np.count_nonzero(~np.isfinite(output)))
    _warn_count(
        warning_counts,
        "nonfinite_output",
        "Corrected output contained non-finite values and was repaired",
        nonfinite,
    )
    if nonfinite:
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

    correction_meta = {
        "flat_override_used": flat_override_used,
        "dark_override_used": dark_override_used,
        "sample_count": int(sample_mask.sum()),
        "flat_count": int(flat_mask.sum()),
        "dark_count": int(dark_mask.sum()),
        "output_domain": output_domain,
    }
    return output, flat_mean, dark_mean, warning_counts, correction_meta


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(val) for val in value]
    return value


def _write_preprocess_provenance(
    output_path: str | Path,
    *,
    provenance: dict[str, Any],
    flat_mean: np.ndarray,
    dark_mean: np.ndarray,
) -> None:
    with h5py.File(output_path, "r+") as file:
        entry = file.require_group("entry")
        processing = entry.require_group("processing")
        processing.attrs["NX_class"] = "NXprocess"
        tomojax = processing.require_group("tomojax")
        tomojax.attrs["NX_class"] = "NXcollection"
        if "preprocess" in tomojax:
            del tomojax["preprocess"]
        group = tomojax.create_group("preprocess")
        group.attrs["NX_class"] = "NXcollection"
        for key, value in provenance.items():
            if isinstance(value, dict | list | tuple):
                group.attrs[key] = json.dumps(_json_safe(value), sort_keys=True)
            elif value is None:
                group.attrs[key] = "null"
            else:
                group.attrs[key] = value
        group.create_dataset(
            "flat_mean",
            data=np.asarray(flat_mean),
            chunks=True,
            compression="lzf",
        )
        group.create_dataset(
            "dark_mean",
            data=np.asarray(dark_mean),
            chunks=True,
            compression="lzf",
        )


def preprocess_nxtomo(
    input_path: str | Path,
    output_path: str | Path,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """Preprocess raw NXtomo sample/flat/dark frames into corrected projections."""

    cfg = config or PreprocessConfig()
    output_dtype = _validate_config(cfg)

    input_path = Path(input_path)
    output_path = Path(output_path)

    with h5py.File(input_path, "r") as file:
        frame_path, frame_dataset = _resolve_dataset(
            file,
            explicit_path=cfg.data_path,
            default_paths=_PROJECTION_PATHS,
            label="raw frame",
        )
        if frame_dataset.ndim != 3:
            raise ValueError(
                "raw frame dataset must be 3D (n_frames, nv, nu), "
                f"got {frame_dataset.shape}"
            )
        frames = np.asarray(frame_dataset[...])
        n_frames, nv, nu = (int(v) for v in frames.shape)

        key_path, key_dataset = _resolve_image_key_dataset(file, cfg.image_key_path)
        image_key = np.asarray(key_dataset[...], dtype=np.int32).reshape(-1)
        if image_key.shape != (n_frames,):
            raise ValueError(
                f"image_key length must match raw frame count ({n_frames}), got {image_key.shape[0]}"
            )

        angle_path, angle_dataset = _resolve_dataset(
            file,
            explicit_path=cfg.angles_path,
            default_paths=(_ANGLE_PATH,),
            label="rotation angle",
        )
        angles_all = np.asarray(angle_dataset[...], dtype=np.float32).reshape(-1)
        if angles_all.shape[0] != n_frames:
            raise ValueError(
                f"rotation_angle length must match raw frame count ({n_frames}), got {angles_all.shape[0]}"
            )

        corrected, flat_mean, dark_mean, warning_counts, correction_meta = _correct_frames(
            frames,
            image_key,
            config=cfg,
        )
        output = np.asarray(corrected, dtype=output_dtype)
        sample_mask = image_key == 0
        output_angles = angles_all[sample_mask]
        output_image_key = np.zeros((int(sample_mask.sum()),), dtype=np.int32)
        metadata = _metadata_from_raw(
            file["/entry"],
            thetas_deg=output_angles,
            output_image_key=output_image_key,
            nv=nv,
            nu=nu,
        )

    save_nxtomo(str(output_path), output, metadata=metadata)

    provenance: dict[str, Any] = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_path": str(input_path),
        "data_path": frame_path,
        "angles_path": angle_path,
        "image_key_path": key_path,
        "frame_counts": {
            "sample": int(correction_meta["sample_count"]),
            "flat": int(correction_meta["flat_count"]),
            "dark": int(correction_meta["dark_count"]),
        },
        "output_domain": correction_meta["output_domain"],
        "epsilon": float(cfg.epsilon),
        "clip_min": None if cfg.clip_min is None else float(cfg.clip_min),
        "output_dtype": str(output_dtype),
        "correction_formula": (
            "transmission=(sample-mean(dark))/max(mean(flat)-mean(dark),epsilon); "
            "absorption=-log(max(transmission,log_min)) when log=true"
        ),
        "log": bool(cfg.log),
        "assume_dark_field": cfg.assume_dark_field,
        "assume_flat_field": cfg.assume_flat_field,
        "dark_override_used": bool(correction_meta["dark_override_used"]),
        "flat_override_used": bool(correction_meta["flat_override_used"]),
        "warning_counts": dict(warning_counts),
    }
    _write_preprocess_provenance(
        output_path,
        provenance=provenance,
        flat_mean=flat_mean.astype(output_dtype, copy=False),
        dark_mean=dark_mean.astype(output_dtype, copy=False),
    )

    return PreprocessResult(
        sample_count=int(correction_meta["sample_count"]),
        flat_count=int(correction_meta["flat_count"]),
        dark_count=int(correction_meta["dark_count"]),
        output_shape=tuple(int(v) for v in output.shape),
        output_domain=str(correction_meta["output_domain"]),
        warning_counts=dict(warning_counts),
        provenance=provenance,
    )


__all__ = [
    "PreprocessConfig",
    "PreprocessResult",
    "preprocess_nxtomo",
]
