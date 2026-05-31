from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tomojax.io._preprocess_impl.config import PreprocessConfig

LOG = logging.getLogger(__name__)


def _constant_field(value: float, shape: tuple[int, int], *, label: str) -> np.ndarray:
    if not np.isfinite(value):
        raise ValueError(f"{label} override must be finite")
    return np.full(shape, float(value), dtype=np.float64)


def _warn_count(warning_counts: dict[str, int], key: str, message: str, count: int) -> None:
    warning_counts[key] = int(count)
    if count > 0:
        LOG.warning("%s: %d", message, int(count))


def _flat_dark_to_transmission_np(
    projections: np.ndarray,
    flats: np.ndarray,
    darks: np.ndarray,
    *,
    denominator_min: float,
    clip_min: float | None,
) -> np.ndarray:
    denom = np.maximum(flats - darks, float(denominator_min))
    norm = (projections - darks) / denom
    if clip_min is not None:
        norm = np.maximum(norm, float(clip_min))
    return norm


def _flat_dark_to_absorption_np(
    projections: np.ndarray,
    flats: np.ndarray,
    darks: np.ndarray,
    *,
    min_intensity: float,
    transmission_min: float,
) -> np.ndarray:
    transmission = _flat_dark_to_transmission_np(
        projections,
        flats,
        darks,
        denominator_min=float(min_intensity),
        clip_min=float(transmission_min),
    )
    return -np.log(np.maximum(transmission, float(transmission_min)))


def correct_nxtomo_frames(
    frames: np.ndarray,
    image_key: np.ndarray,
    *,
    config: PreprocessConfig,
    output_domain: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, Any]]:
    """Correct sample frames using NXtomo flat/dark frame groups."""
    sample_mask = image_key == 0
    flat_mask = image_key == 1
    dark_mask = image_key == 2

    samples = frames[sample_mask].astype(np.float64, copy=False)
    flats = frames[flat_mask].astype(np.float64, copy=False)
    darks = frames[dark_mask].astype(np.float64, copy=False)

    if samples.size == 0:
        raise ValueError("No sample frames found (image_key==0)")

    field_shape = (int(frames.shape[1]), int(frames.shape[2]))
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
        flat_mean = np.asarray(np.mean(flats, axis=0, dtype=np.float64), dtype=np.float64)

    if darks.size == 0:
        if config.assume_dark_field is None:
            raise ValueError(
                "No dark fields found (image_key==2); pass --assume-dark-field VALUE "
                "to use an explicit constant dark field"
            )
        dark_mean = _constant_field(config.assume_dark_field, field_shape, label="dark field")
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

    correction_meta = {
        "flat_override_used": flat_override_used,
        "dark_override_used": dark_override_used,
        "sample_count": int(sample_mask.sum()),
        "flat_count": int(flat_mask.sum()),
        "dark_count": int(dark_mask.sum()),
        "output_domain": output_domain,
    }
    return output, flat_mean, dark_mean, warning_counts, correction_meta


def repair_nonfinite_preprocess_output(
    output: np.ndarray,
    warning_counts: dict[str, int],
) -> np.ndarray:
    """Replace non-finite corrected pixels after recording the warning count."""
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
    max_gap = float(np.max(np.diff(finite))) if finite.size >= 2 else 0.0
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
    if before_gap is None or after_gap is None:
        if before_gap != after_gap:
            return True
    elif not np.isclose(float(before_gap), float(after_gap), rtol=0.0, atol=1e-6):
        return True
    return False


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
