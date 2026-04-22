from __future__ import annotations

from collections.abc import Mapping
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ALIGNMENT_PARAMS_SCHEMA = "tomojax.alignment_params.v1"
PARAMETER_ORDER = ("alpha", "beta", "phi", "dx", "dz")
CSV_FIELDNAMES = (
    "view_index",
    "alpha_rad",
    "beta_rad",
    "phi_rad",
    "dx_world",
    "dz_world",
    "dx_px",
    "dz_px",
)
PARAMETER_UNITS = {
    "alpha": "rad",
    "beta": "rad",
    "phi": "rad",
    "dx": "world",
    "dz": "world",
    "dx_px": "pixel",
    "dz_px": "pixel",
}


type AlignmentParamRecord = dict[str, int | float]


def _normalize_params5(params5: np.ndarray) -> np.ndarray:
    arr = np.asarray(params5, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"params5 must have shape (n_views, 5), got {arr.shape}")
    return arr


def _validate_detector_spacing(*, du: float, dv: float) -> tuple[float, float]:
    du_f = float(du)
    dv_f = float(dv)
    if not np.isfinite(du_f) or du_f <= 0.0:
        raise ValueError("detector du must be positive finite to export dx_px")
    if not np.isfinite(dv_f) or dv_f <= 0.0:
        raise ValueError("detector dv must be positive finite to export dz_px")
    return du_f, dv_f


def alignment_param_records(
    params5: np.ndarray,
    *,
    du: float,
    dv: float,
) -> list[AlignmentParamRecord]:
    """Return per-view named alignment records for JSON/CSV export."""
    arr = _normalize_params5(params5)
    du_f, dv_f = _validate_detector_spacing(du=du, dv=dv)

    records: list[AlignmentParamRecord] = []
    for view_index, row in enumerate(arr):
        alpha, beta, phi, dx, dz = (float(v) for v in row)
        records.append(
            {
                "view_index": int(view_index),
                "alpha_rad": alpha,
                "beta_rad": beta,
                "phi_rad": phi,
                "dx_world": dx,
                "dz_world": dz,
                "dx_px": dx / du_f,
                "dz_px": dz / dv_f,
            }
        )
    return records


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str | int | float | bool | type(None)):
        return value

    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return value

    if arr.shape == ():
        if arr.dtype == object:
            return value
        return _json_native(arr.item())
    return _json_native(arr.tolist())


def alignment_params_payload(
    params5: np.ndarray,
    *,
    du: float,
    dv: float,
    gauge_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the versioned JSON payload for exported alignment parameters."""
    du_f, dv_f = _validate_detector_spacing(du=du, dv=dv)
    payload = {
        "schema": ALIGNMENT_PARAMS_SCHEMA,
        "parameter_order": list(PARAMETER_ORDER),
        "units": dict(PARAMETER_UNITS),
        "detector_spacing": {"du": du_f, "dv": dv_f},
        "views": alignment_param_records(params5, du=du_f, dv=dv_f),
    }
    if gauge_metadata is not None:
        payload["gauge_fix"] = _json_native(gauge_metadata)
    return payload


def _ensure_parent(path: str | Path) -> Path:
    out_path = Path(path)
    if out_path.parent and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def save_alignment_params_json(
    path: str | Path,
    params5: np.ndarray,
    *,
    du: float,
    dv: float,
    gauge_metadata: dict[str, Any] | None = None,
) -> None:
    """Write per-view alignment parameters as a named JSON sidecar."""
    out_path = _ensure_parent(path)
    payload = alignment_params_payload(
        params5,
        du=du,
        dv=dv,
        gauge_metadata=gauge_metadata,
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def save_alignment_params_csv(
    path: str | Path,
    params5: np.ndarray,
    *,
    du: float,
    dv: float,
) -> None:
    """Write per-view alignment parameters as a pandas-readable CSV sidecar."""
    out_path = _ensure_parent(path)
    records = alignment_param_records(params5, du=du, dv=dv)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
