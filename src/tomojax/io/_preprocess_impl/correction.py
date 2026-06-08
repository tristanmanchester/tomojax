from __future__ import annotations

import logging
from typing import cast

import numpy as np
import numpy.typing as npt

from tomojax.io._json import JsonValue
from tomojax.io._preprocess_impl.config import PreprocessConfig

LOG = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
Metadata = dict[str, JsonValue]


def _constant_field(value: float, shape: tuple[int, int], *, label: str) -> FloatArray:
    if not np.isfinite(value):
        raise ValueError(f"{label} override must be finite")
    return np.full(shape, float(value), dtype=np.float64)


def _array_sum_int(values: BoolArray) -> int:
    return int(np.sum(values, dtype=np.int64))


def _array_dim(values: npt.NDArray[np.generic], axis: int) -> int:
    shape = cast("tuple[int, ...]", values.shape)
    return int(shape[axis])


def _array_float_at(values: npt.NDArray[np.generic], index: int) -> float:
    value = cast("object", values[index])
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"expected numeric array scalar, got {type(value).__name__}")
    return float(value)


def _array_int_at(values: npt.NDArray[np.generic], index: int) -> int:
    value = cast("object", values[index])
    if isinstance(value, np.integer):
        return int(cast("int", value))
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError(f"expected integer array scalar, got {type(value).__name__}")
    return int(value)


def _median_float(values: FloatArray) -> float:
    value = cast("object", np.median(values))
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"expected numeric median scalar, got {type(value).__name__}")
    return float(value)


def _finite_float(value: np.float64) -> float:
    return float(value)


def _warn_count(warning_counts: dict[str, int], key: str, message: str, count: int) -> None:
    warning_counts[key] = int(count)
    if count > 0:
        LOG.warning("%s: %d", message, int(count))


def _flat_dark_to_transmission_np(
    projections: FloatArray,
    flats: FloatArray,
    darks: FloatArray,
    *,
    denominator_min: float,
    clip_min: float | None,
) -> FloatArray:
    denom = np.maximum(flats - darks, float(denominator_min))
    norm = (projections - darks) / denom
    if clip_min is not None:
        norm = np.maximum(norm, float(clip_min))
    return np.asarray(norm, dtype=np.float64)


def _flat_dark_to_absorption_np(
    projections: FloatArray,
    flats: FloatArray,
    darks: FloatArray,
    *,
    min_intensity: float,
    transmission_min: float,
) -> FloatArray:
    transmission = _flat_dark_to_transmission_np(
        projections,
        flats,
        darks,
        denominator_min=float(min_intensity),
        clip_min=float(transmission_min),
    )
    return np.asarray(-np.log(np.maximum(transmission, float(transmission_min))), dtype=np.float64)


def correct_nxtomo_frames(
    frames: FloatArray,
    image_key: npt.NDArray[np.int32],
    *,
    config: PreprocessConfig,
    output_domain: str,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, int], Metadata]:
    """Correct sample frames using NXtomo flat/dark frame groups."""
    sample_mask: BoolArray = np.asarray(image_key == 0, dtype=np.bool_)
    flat_mask: BoolArray = np.asarray(image_key == 1, dtype=np.bool_)
    dark_mask: BoolArray = np.asarray(image_key == 2, dtype=np.bool_)

    samples: FloatArray = np.asarray(frames[sample_mask], dtype=np.float64)
    flats: FloatArray = np.asarray(frames[flat_mask], dtype=np.float64)
    darks: FloatArray = np.asarray(frames[dark_mask], dtype=np.float64)

    if samples.size == 0:
        raise ValueError("No sample frames found (image_key==0)")

    field_shape = (_array_dim(frames, 1), _array_dim(frames, 2))
    flat_override_used = False
    dark_override_used = False
    if flats.size == 0:
        if config.assume_flat_field is None:
            raise ValueError(
                "No flat fields found (image_key==1); pass --assume-flat-field VALUE "
                "to use an explicit constant flat field"
            )
        flat_mean: FloatArray = _constant_field(
            config.assume_flat_field, field_shape, label="flat field"
        )
        flat_override_used = True
    else:
        flat_mean = np.asarray(np.mean(flats, axis=0, dtype=np.float64), dtype=np.float64)

    if darks.size == 0:
        if config.assume_dark_field is None:
            raise ValueError(
                "No dark fields found (image_key==2); pass --assume-dark-field VALUE "
                "to use an explicit constant dark field"
            )
        dark_mean: FloatArray = _constant_field(
            config.assume_dark_field, field_shape, label="dark field"
        )
        dark_override_used = True
    else:
        dark_mean = np.asarray(np.mean(darks, axis=0, dtype=np.float64), dtype=np.float64)

    warning_counts: dict[str, int] = {}
    denominator_raw = flat_mean - dark_mean
    _warn_count(
        warning_counts,
        "nonpositive_flat_denominator",
        "Flat-dark denominator values were zero or negative before epsilon clipping",
        int(np.count_nonzero(denominator_raw <= 0.0)),
    )

    sample_dark_corrected = samples - dark_mean
    denominator = np.maximum(denominator_raw, float(config.epsilon))
    transmission_raw = sample_dark_corrected / denominator
    _warn_count(
        warning_counts,
        "nonpositive_transmission",
        "Transmission values were zero or negative before clipping/log safeguards",
        int(np.count_nonzero(transmission_raw <= 0.0)),
    )

    if output_domain == "absorption":
        output = _flat_dark_to_absorption_np(
            samples,
            flat_mean,
            dark_mean,
            min_intensity=float(config.epsilon),
            transmission_min=max(
                float(config.epsilon),
                float(config.clip_min or config.epsilon),
            ),
        )
    else:
        output = _flat_dark_to_transmission_np(
            samples,
            flat_mean,
            dark_mean,
            denominator_min=float(config.epsilon),
            clip_min=None if config.clip_min is None else float(config.clip_min),
        )

    correction_meta: Metadata = {
        "flat_override_used": flat_override_used,
        "dark_override_used": dark_override_used,
        "sample_count": _array_sum_int(sample_mask),
        "flat_count": _array_sum_int(flat_mask),
        "dark_count": _array_sum_int(dark_mask),
        "output_domain": output_domain,
    }
    return (
        np.asarray(output, dtype=np.float64),
        flat_mean,
        dark_mean,
        warning_counts,
        correction_meta,
    )


def repair_nonfinite_preprocess_output(
    output: FloatArray,
    warning_counts: dict[str, int],
) -> FloatArray:
    """Replace non-finite corrected pixels after recording the warning count."""
    nonfinite = int(np.count_nonzero(~np.isfinite(output)))
    _warn_count(
        warning_counts,
        "nonfinite_output",
        "Corrected output contained non-finite values and was repaired",
        nonfinite,
    )
    if nonfinite:
        return np.asarray(np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
    return output


def _coverage_stats(angles: npt.NDArray[np.floating]) -> dict[str, float | int | None]:
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
    max_gap = float(np.max(np.diff(finite))) if finite.size >= 2 else 0.0
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "min_deg": _array_float_at(finite, 0),
        "max_deg": _array_float_at(finite, -1),
        "span_deg": _array_float_at(finite, -1) - _array_float_at(finite, 0),
        "max_gap_deg": max_gap,
    }


def _coverage_changed(
    before: dict[str, float | int | None],
    after: dict[str, float | int | None],
) -> bool:
    for key in ("min_deg", "max_deg", "span_deg"):
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value is None or after_value is None:
            if before_value != after_value:
                return True
        elif not np.isclose(float(before_value), float(after_value), rtol=0.0, atol=1e-6):
            return True
    before_gap = before.get("max_gap_deg")
    after_gap = after.get("max_gap_deg")
    if before_gap is None or after_gap is None:
        if before_gap != after_gap:
            return True
    elif not np.isclose(float(before_gap), float(after_gap), rtol=0.0, atol=1e-6):
        return True
    return False


def _auto_reject_views(
    output: FloatArray,
    sample_view_indices: IntArray,
    *,
    mode: str,
    threshold: float,
) -> tuple[BoolArray, Metadata]:
    mode_norm = str(mode).strip().lower()
    keep: BoolArray = np.ones((_array_dim(output, 0),), dtype=np.bool_)
    reason_by_index: dict[int, list[str]] = {}
    outlier_meta: Metadata = {
        "ran": mode_norm in {"outliers", "both"},
        "skipped": False,
        "skip_reason": None,
        "view_medians": [],
        "median": None,
        "mad": None,
        "robust_scale": None,
    }

    if mode_norm in {"nonfinite", "both"}:
        nonfinite: BoolArray = np.asarray(np.any(~np.isfinite(output), axis=(1, 2)), dtype=np.bool_)
        for pos in np.flatnonzero(nonfinite):
            sample_idx = _array_int_at(sample_view_indices, int(pos))
            reason_by_index.setdefault(sample_idx, []).append("nonfinite")
        keep &= ~nonfinite

    if mode_norm in {"outliers", "both"}:
        medians: FloatArray = np.full((_array_dim(output, 0),), np.nan, dtype=np.float64)
        for i in range(_array_dim(output, 0)):
            view_values: FloatArray = np.asarray(output[i], dtype=np.float64)
            finite = view_values[np.isfinite(view_values)]
            if finite.size:
                medians[i] = _median_float(np.asarray(finite, dtype=np.float64))
        finite_medians = medians[np.isfinite(medians)]
        view_medians: list[JsonValue] = []
        for i in range(_array_dim(medians, 0)):
            value = _array_float_at(medians, i)
            view_medians.append(
                None if not np.isfinite(value) else _finite_float(np.float64(value))
            )
        outlier_meta["view_medians"] = view_medians
        if finite_medians.size < 3:
            outlier_meta["skipped"] = True
            outlier_meta["skip_reason"] = "fewer than 3 finite per-view medians"
        else:
            center = _median_float(np.asarray(finite_medians, dtype=np.float64))
            mad = _median_float(np.asarray(np.abs(finite_medians - center), dtype=np.float64))
            scale = 1.4826 * mad
            outlier_meta["median"] = center
            outlier_meta["mad"] = mad
            outlier_meta["robust_scale"] = scale
            if scale <= 0.0 or not np.isfinite(scale):
                outlier_meta["skipped"] = True
                outlier_meta["skip_reason"] = "zero or non-finite MAD"
            else:
                robust_z = np.abs((medians - center) / scale)
                outliers: BoolArray = np.asarray(
                    np.isfinite(robust_z) & (robust_z > float(threshold)),
                    dtype=np.bool_,
                )
                for pos in np.flatnonzero(outliers):
                    sample_idx = _array_int_at(sample_view_indices, int(pos))
                    reason_by_index.setdefault(sample_idx, []).append("outlier")
                keep &= ~outliers

    if not np.any(keep):
        raise ValueError("automatic rejection removed all sample views")

    rejected = sorted(reason_by_index)
    rejected_indices: list[JsonValue] = [cast("JsonValue", index) for index in rejected]
    rejected_reasons: dict[str, JsonValue] = {
        str(k): [cast("JsonValue", reason) for reason in v]
        for k, v in sorted(reason_by_index.items())
    }
    meta: Metadata = {
        "mode": mode_norm,
        "outlier_z_threshold": float(threshold),
        "rejected_sample_view_indices": rejected_indices,
        "rejected_reasons": rejected_reasons,
        "outlier": outlier_meta,
    }
    return keep, meta
