"""Result-quality helpers for article alignment benchmark artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np


def array_finite_summary(name: str, value: np.ndarray | None) -> dict[str, Any]:
    """Summarize finite/non-finite values for a volume-like array."""
    if value is None:
        return {
            "name": name,
            "present": False,
            "all_finite": False,
            "finite_fraction": 0.0,
        }
    arr = np.asarray(value)
    finite = np.isfinite(arr)
    total = int(arr.size)
    summary: dict[str, Any] = {
        "name": name,
        "present": True,
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "size": total,
        "finite_count": int(finite.sum()),
        "finite_fraction": float(finite.mean()) if total else 1.0,
        "nan_count": int(np.isnan(arr).sum()),
        "posinf_count": int(np.isposinf(arr).sum()),
        "neginf_count": int(np.isneginf(arr).sum()),
        "all_finite": bool(finite.all()),
    }
    if total and not bool(finite.all()):
        first = np.argwhere(~finite)[0]
        summary["first_nonfinite_index"] = [int(v) for v in first]
        summary["first_nonfinite_value"] = repr(arr[tuple(first)])
    if bool(finite.any()):
        vals = arr[finite].astype(np.float64, copy=False)
        summary.update(
            {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "rms": float(np.sqrt(np.mean(vals * vals))),
            }
        )
    return summary


def scalar_finite_summary(name: str, value: Any) -> dict[str, Any]:
    """Summarize finite/non-finite state for a scalar-like value."""
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError):
        return {
            "name": name,
            "present": value is not None,
            "all_finite": False,
            "value": repr(value),
        }
    return {"name": name, "present": True, "all_finite": bool(np.isfinite(scalar)), "value": scalar}


def article_scenario_finite_report(result: Any) -> dict[str, Any]:
    """Build the finite-value report for one article alignment scenario result."""
    volume_summaries = [
        array_finite_summary("naive_fbp", result.naive_fbp),
        array_finite_summary("calibrated_fbp", result.calibrated_fbp),
        array_finite_summary("aligned_tv", result.aligned_tv),
    ]
    metric_summaries = [
        scalar_finite_summary(name, value) for name, value in sorted(result.metrics.items())
    ]
    if result.provenance == "naive_only":
        required = ["naive_fbp"]
    else:
        required = ["naive_fbp", "calibrated_fbp", "aligned_tv"]
    required_by_name = {
        summary["name"]: bool(summary.get("all_finite", False)) for summary in volume_summaries
    }
    all_required_finite = all(required_by_name.get(name, False) for name in required) and all(
        bool(summary.get("all_finite", False)) for summary in metric_summaries
    )
    first_nonfinite = next(
        (
            summary
            for summary in volume_summaries + metric_summaries
            if not bool(summary.get("all_finite", False))
        ),
        None,
    )
    return {
        "schema_version": 1,
        "all_required_finite": bool(all_required_finite),
        "required_arrays": required,
        "first_nonfinite": first_nonfinite,
        "volumes": volume_summaries,
        "metrics": metric_summaries,
    }


__all__ = ["array_finite_summary", "article_scenario_finite_report", "scalar_finite_summary"]
