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
    select_views: str | None = None
    reject_views: str | None = None
    select_views_file: str | Path | None = None
    reject_views_file: str | Path | None = None
    auto_reject: str = "off"
    outlier_z_threshold: float = 6.0
    crop: str | None = None


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


def _validate_config(config: PreprocessConfig) -> np.dtype:
    if not np.isfinite(config.epsilon) or config.epsilon <= 0:
        raise ValueError("epsilon must be a positive finite value")
    if config.clip_min is not None and (not np.isfinite(config.clip_min) or config.clip_min <= 0):
        raise ValueError("clip_min must be a positive finite value when provided")
    if config.output_dtype not in _OUTPUT_DTYPES:
        allowed = ", ".join(sorted(_OUTPUT_DTYPES))
        raise ValueError(f"output_dtype must be one of: {allowed}")
    auto_reject = str(config.auto_reject).strip().lower()
    if auto_reject not in {"off", "nonfinite", "outliers", "both"}:
        raise ValueError("auto_reject must be one of: off, nonfinite, outliers, both")
    if not np.isfinite(config.outlier_z_threshold) or config.outlier_z_threshold <= 0:
        raise ValueError("outlier_z_threshold must be a positive finite value")
    return np.dtype(_OUTPUT_DTYPES[config.output_dtype])


def _view_tokens_from_text(text: str) -> list[str]:
    tokens: list[str] = []
    for line in str(text).splitlines():
        line = line.split("#", 1)[0]
        if not line.strip():
            continue
        tokens.extend(part for part in line.replace(",", " ").split() if part)
    return tokens


def _read_view_spec_file(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read view index file {path!s}: {exc}") from exc


def _parse_nonnegative_int(text: str, *, label: str, token: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"invalid {label} {text!r} in view spec token {token!r}") from exc
    if value < 0:
        raise ValueError(f"negative view indices are not supported in token {token!r}")
    return value


def _parse_view_spec(text: str | None, *, n_views: int, label: str) -> np.ndarray:
    if text is None or not str(text).strip():
        return np.asarray([], dtype=np.int64)

    selected: dict[int, None] = {}
    for token in _view_tokens_from_text(str(text)):
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"malformed {label} range {token!r}")
            if parts[0] == "" or parts[1] == "":
                raise ValueError(f"{label} ranges must provide explicit start and stop: {token!r}")
            start = _parse_nonnegative_int(parts[0], label="range start", token=token)
            stop = _parse_nonnegative_int(parts[1], label="range stop", token=token)
            step = 1
            if len(parts) == 3:
                if parts[2] == "":
                    raise ValueError(f"{label} ranges must provide an explicit step: {token!r}")
                step = _parse_nonnegative_int(parts[2], label="range step", token=token)
            if step <= 0:
                raise ValueError(f"{label} range step must be positive in token {token!r}")
            if stop <= start:
                raise ValueError(f"{label} range must be non-empty in token {token!r}")
            if start >= n_views or stop > n_views:
                raise ValueError(
                    f"{label} range {token!r} is out of bounds for {n_views} sample views"
                )
            values = range(start, stop, step)
            for value in values:
                selected[int(value)] = None
        else:
            value = _parse_nonnegative_int(token, label="view index", token=token)
            if value >= n_views:
                raise ValueError(
                    f"{label} index {value} is out of bounds for {n_views} sample views"
                )
            selected[int(value)] = None

    return np.asarray(sorted(selected), dtype=np.int64)


def _combine_view_specs(
    spec: str | None,
    file_path: str | Path | None,
    *,
    n_views: int,
    label: str,
) -> np.ndarray | None:
    parts: list[str] = []
    if spec is not None and str(spec).strip():
        parts.append(str(spec))
    if file_path is not None:
        parts.append(_read_view_spec_file(file_path))
    if not parts:
        return None
    return _parse_view_spec("\n".join(parts), n_views=n_views, label=label)


def _resolve_sample_view_indices(
    *,
    n_sample_views: int,
    config: PreprocessConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    select = _combine_view_specs(
        config.select_views,
        config.select_views_file,
        n_views=n_sample_views,
        label="select-views",
    )
    reject = _combine_view_specs(
        config.reject_views,
        config.reject_views_file,
        n_views=n_sample_views,
        label="reject-views",
    )

    if select is None:
        candidate = np.arange(n_sample_views, dtype=np.int64)
    else:
        candidate = select.astype(np.int64, copy=False)

    explicit_rejected = np.asarray([], dtype=np.int64)
    if reject is not None and reject.size:
        reject_set = set(int(v) for v in reject.tolist())
        keep_mask = np.asarray([int(v) not in reject_set for v in candidate], dtype=bool)
        explicit_rejected = candidate[~keep_mask]
        candidate = candidate[keep_mask]

    if candidate.size == 0:
        raise ValueError("view selection/rejection removed all sample views")

    meta = {
        "select_views": None if select is None else select.tolist(),
        "reject_views": [] if reject is None else reject.tolist(),
        "explicit_rejected_sample_view_indices": explicit_rejected.tolist(),
        "candidate_sample_view_indices": candidate.tolist(),
    }
    return candidate, meta


def _parse_crop_spec(
    crop: str | None,
    *,
    nv: int,
    nu: int,
) -> tuple[int, int, int, int] | None:
    if crop is None or not str(crop).strip():
        return None
    text = str(crop).strip()
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("--crop must be formatted as y0:y1,x0:x1")

    def parse_axis(axis_text: str, limit: int, axis_name: str) -> tuple[int, int]:
        axis_parts = axis_text.split(":")
        if len(axis_parts) != 2 or axis_parts[0] == "" or axis_parts[1] == "":
            raise ValueError(f"--crop {axis_name} range must be formatted as start:stop")
        start = _parse_nonnegative_int(axis_parts[0], label=f"{axis_name} start", token=axis_text)
        stop = _parse_nonnegative_int(axis_parts[1], label=f"{axis_name} stop", token=axis_text)
        if stop <= start:
            raise ValueError(f"--crop {axis_name} range must be non-empty")
        if stop > limit:
            raise ValueError(
                f"--crop {axis_name} range {axis_text!r} is out of bounds for size {limit}"
            )
        return start, stop

    y0, y1 = parse_axis(parts[0].strip(), nv, "y")
    x0, x1 = parse_axis(parts[1].strip(), nu, "x")
    return y0, y1, x0, x1


def _coverage_stats(angles: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return {
            "count": int(values.size),
            "finite_count": 0,
            "min_deg": None,
            "max_deg": None,
            "span_deg": None,
            "max_gap_deg": None,
        }
    if finite.size >= 2:
        max_gap = float(np.max(np.diff(finite)))
    else:
        max_gap = 0.0
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "min_deg": float(finite[0]),
        "max_deg": float(finite[-1]),
        "span_deg": float(finite[-1] - finite[0]),
        "max_gap_deg": max_gap,
    }


def _coverage_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for key in ("min_deg", "max_deg", "span_deg"):
        if before.get(key) is None or after.get(key) is None:
            if before.get(key) != after.get(key):
                return True
        elif not np.isclose(float(before[key]), float(after[key]), rtol=0.0, atol=1e-6):
            return True
    before_gap = before.get("max_gap_deg")
    after_gap = after.get("max_gap_deg")
    if before_gap is not None and after_gap is not None:
        if float(after_gap) > float(before_gap) + 1e-6:
            return True
    return False


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

    correction_meta = {
        "flat_override_used": flat_override_used,
        "dark_override_used": dark_override_used,
        "sample_count": int(sample_mask.sum()),
        "flat_count": int(flat_mask.sum()),
        "dark_count": int(dark_mask.sum()),
        "output_domain": output_domain,
    }
    return output, flat_mean, dark_mean, warning_counts, correction_meta


def _auto_reject_views(
    output: np.ndarray,
    sample_view_indices: np.ndarray,
    *,
    mode: str,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode_norm = str(mode).strip().lower()
    keep = np.ones((int(output.shape[0]),), dtype=bool)
    reason_by_index: dict[int, list[str]] = {}
    outlier_meta: dict[str, Any] = {
        "ran": mode_norm in {"outliers", "both"},
        "skipped": False,
        "skip_reason": None,
        "view_medians": [],
        "median": None,
        "mad": None,
        "robust_scale": None,
    }

    if mode_norm in {"nonfinite", "both"}:
        nonfinite = np.any(~np.isfinite(output), axis=(1, 2))
        for pos in np.flatnonzero(nonfinite):
            sample_idx = int(sample_view_indices[int(pos)])
            reason_by_index.setdefault(sample_idx, []).append("nonfinite")
        keep &= ~nonfinite

    if mode_norm in {"outliers", "both"}:
        medians = np.full((int(output.shape[0]),), np.nan, dtype=np.float64)
        for i, view in enumerate(output):
            finite = np.asarray(view, dtype=np.float64)[np.isfinite(view)]
            if finite.size:
                medians[i] = float(np.median(finite))
        finite_medians = medians[np.isfinite(medians)]
        outlier_meta["view_medians"] = [
            None if not np.isfinite(value) else float(value) for value in medians
        ]
        if finite_medians.size < 3:
            outlier_meta["skipped"] = True
            outlier_meta["skip_reason"] = "fewer than 3 finite per-view medians"
        else:
            center = float(np.median(finite_medians))
            mad = float(np.median(np.abs(finite_medians - center)))
            scale = 1.4826 * mad
            outlier_meta["median"] = center
            outlier_meta["mad"] = mad
            outlier_meta["robust_scale"] = scale
            if scale <= 0.0 or not np.isfinite(scale):
                outlier_meta["skipped"] = True
                outlier_meta["skip_reason"] = "zero or non-finite MAD"
            else:
                robust_z = np.abs((medians - center) / scale)
                outliers = np.isfinite(robust_z) & (robust_z > float(threshold))
                for pos in np.flatnonzero(outliers):
                    sample_idx = int(sample_view_indices[int(pos)])
                    reason_by_index.setdefault(sample_idx, []).append("outlier")
                keep &= ~outliers

    if not np.any(keep):
        raise ValueError("automatic rejection removed all sample views")

    rejected = sorted(reason_by_index)
    meta = {
        "mode": mode_norm,
        "outlier_z_threshold": float(threshold),
        "rejected_sample_view_indices": rejected,
        "rejected_reasons": {str(k): v for k, v in sorted(reason_by_index.items())},
        "outlier": outlier_meta,
    }
    return keep, meta


def _repair_nonfinite_output(
    output: np.ndarray,
    warning_counts: dict[str, int],
) -> np.ndarray:
    nonfinite = int(np.count_nonzero(~np.isfinite(output)))
    _warn_count(
        warning_counts,
        "nonfinite_output",
        "Corrected output contained non-finite values and was repaired",
        nonfinite,
    )
    if nonfinite:
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    return output


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
                f"raw frame dataset must be 3D (n_frames, nv, nu), got {frame_dataset.shape}"
            )
        n_frames, raw_nv, raw_nu = (int(v) for v in frame_dataset.shape)
        crop_bounds = _parse_crop_spec(cfg.crop, nv=raw_nv, nu=raw_nu)
        if crop_bounds is None:
            y0, y1, x0, x1 = 0, raw_nv, 0, raw_nu
        else:
            y0, y1, x0, x1 = crop_bounds
        frames = np.asarray(frame_dataset[:, y0:y1, x0:x1])
        cropped_nv, cropped_nu = int(y1 - y0), int(x1 - x0)

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

        sample_mask = image_key == 0
        flat_mask = image_key == 1
        dark_mask = image_key == 2
        sample_raw_indices = np.flatnonzero(sample_mask)
        flat_raw_indices = np.flatnonzero(flat_mask)
        dark_raw_indices = np.flatnonzero(dark_mask)
        raw_frame_counts = {
            "sample": int(sample_raw_indices.size),
            "flat": int(flat_raw_indices.size),
            "dark": int(dark_raw_indices.size),
        }
        if sample_raw_indices.size == 0:
            raise ValueError("No sample frames found (image_key==0)")

        candidate_sample_indices, view_selection_meta = _resolve_sample_view_indices(
            n_sample_views=int(sample_raw_indices.size),
            config=cfg,
        )
        candidate_sample_raw_indices = sample_raw_indices[candidate_sample_indices]
        reduced_raw_indices = np.concatenate(
            [candidate_sample_raw_indices, flat_raw_indices, dark_raw_indices]
        )
        reduced_image_key = np.concatenate(
            [
                np.zeros((int(candidate_sample_indices.size),), dtype=np.int32),
                np.ones((int(flat_raw_indices.size),), dtype=np.int32),
                np.full((int(dark_raw_indices.size),), 2, dtype=np.int32),
            ]
        )
        reduced_frames = frames[reduced_raw_indices]
        candidate_angles = angles_all[candidate_sample_raw_indices]

        corrected, flat_mean, dark_mean, warning_counts, correction_meta = _correct_frames(
            reduced_frames,
            reduced_image_key,
            config=cfg,
        )

        auto_keep, auto_reject_meta = _auto_reject_views(
            corrected,
            candidate_sample_indices,
            mode=cfg.auto_reject,
            threshold=float(cfg.outlier_z_threshold),
        )
        output_unrepaired = corrected[auto_keep]
        final_sample_indices = candidate_sample_indices[auto_keep]
        final_raw_sample_indices = candidate_sample_raw_indices[auto_keep]
        output_angles = candidate_angles[auto_keep]
        output_repaired = _repair_nonfinite_output(output_unrepaired, warning_counts)
        output = np.asarray(output_repaired, dtype=output_dtype)
        output_image_key = np.zeros((int(output.shape[0]),), dtype=np.int32)

        coverage_before = _coverage_stats(angles_all[sample_raw_indices])
        coverage_after = _coverage_stats(output_angles)
        if _coverage_changed(coverage_before, coverage_after):
            LOG.warning(
                "View rejection changed angular coverage: before=%s after=%s",
                coverage_before,
                coverage_after,
            )

        align_params, angle_offset_deg, optional_meta_found = _optional_sample_metadata(
            file["/entry"],
            n_sample_views=int(sample_raw_indices.size),
            selected_sample_view_indices=final_sample_indices,
        )
        metadata = _metadata_from_raw(
            file["/entry"],
            thetas_deg=output_angles,
            output_image_key=output_image_key,
            nv=cropped_nv,
            nu=cropped_nu,
            crop_bounds=crop_bounds,
            original_nv=raw_nv,
            original_nu=raw_nu,
            align_params=align_params,
            angle_offset_deg=angle_offset_deg,
        )

    save_nxtomo(str(output_path), output, metadata=metadata)

    provenance: dict[str, Any] = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "input_path": str(input_path),
        "data_path": frame_path,
        "angles_path": angle_path,
        "image_key_path": key_path,
        "frame_counts": raw_frame_counts,
        "processing_frame_counts": {
            "candidate_sample": int(correction_meta["sample_count"]),
            "final_sample": int(output.shape[0]),
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
        "select_views": cfg.select_views,
        "reject_views": cfg.reject_views,
        "select_views_file": None if cfg.select_views_file is None else str(cfg.select_views_file),
        "reject_views_file": None if cfg.reject_views_file is None else str(cfg.reject_views_file),
        "view_selection": view_selection_meta,
        "final_sample_view_indices": final_sample_indices.tolist(),
        "final_raw_frame_indices": final_raw_sample_indices.tolist(),
        "auto_reject": auto_reject_meta,
        "crop": cfg.crop,
        "crop_bounds": None
        if crop_bounds is None
        else {
            "y0": int(y0),
            "y1": int(y1),
            "x0": int(x0),
            "x1": int(x1),
        },
        "original_projection_shape": [int(n_frames), int(raw_nv), int(raw_nu)],
        "cropped_projection_shape": [int(n_frames), int(cropped_nv), int(cropped_nu)],
        "final_projection_shape": [int(v) for v in output.shape],
        "angular_coverage_before": coverage_before,
        "angular_coverage_after": coverage_after,
        "optional_sample_metadata_filtered": optional_meta_found,
    }
    _write_preprocess_provenance(
        output_path,
        provenance=provenance,
        flat_mean=flat_mean.astype(output_dtype, copy=False),
        dark_mean=dark_mean.astype(output_dtype, copy=False),
    )

    return PreprocessResult(
        sample_count=int(output.shape[0]),
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
