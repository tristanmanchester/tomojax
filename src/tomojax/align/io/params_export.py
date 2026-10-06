"""Export per-view alignment parameters as JSON or CSV sidecars."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tomojax.align._geometry.parametrizations import pad_pose_params

ALIGNMENT_PARAMS_SCHEMA = "tomojax.alignment_params"
PARAMETER_ORDER = ("alpha", "beta", "phi", "dx", "dz", "dy")
CSV_FIELDNAMES = (
    "view_index",
    "alpha_rad",
    "beta_rad",
    "phi_rad",
    "dx_world",
    "dz_world",
    "dx_px",
    "dz_px",
    "dy_world",
)
PARAMETER_UNITS = {
    "alpha": "rad",
    "beta": "rad",
    "phi": "rad",
    "dx": "world",
    "dz": "world",
    "dy": "world",
    "dx_px": "pixel",
    "dz_px": "pixel",
}


type AlignmentParamRecord = dict[str, int | float]


def _normalize_params5(params5: np.ndarray) -> np.ndarray:
    try:
        return pad_pose_params(params5)
    except ValueError as exc:
        raise ValueError(f"params5 must have shape (n_views, 5 or 6): {exc}") from None


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
        alpha, beta, phi, dx, dz, dy = (float(v) for v in row)
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
                "dy_world": dy,
            }
        )
    return records


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_native(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str | int | float | bool | type(None)):
        return value

    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return value

    return _json_native(arr.item()) if arr.shape == () else _json_native(arr.tolist())


def alignment_params_payload(
    params5: np.ndarray,
    *,
    du: float,
    dv: float,
    gauge_metadata: dict[str, Any] | None = None,
    translation_frame: str = "object",
) -> dict[str, Any]:
    """Build the JSON payload for exported alignment parameters."""
    du_f, dv_f = _validate_detector_spacing(du=du, dv=dv)
    _validate_translation_frame(translation_frame)
    payload = {
        "schema": ALIGNMENT_PARAMS_SCHEMA,
        "parameter_order": list(PARAMETER_ORDER),
        "pose_translation_frame": translation_frame,
        "units": dict(PARAMETER_UNITS),
        "detector_spacing": {"du": du_f, "dv": dv_f},
        "views": alignment_param_records(params5, du=du_f, dv=dv_f),
    }
    if gauge_metadata is not None:
        payload["gauge_fix"] = _json_native(gauge_metadata)
    return payload


def _validate_translation_frame(frame: str) -> None:
    if frame not in {"object", "detector"}:
        raise ValueError("translation_frame must be 'object' or 'detector'")


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
    translation_frame: str = "object",
) -> None:
    """Write per-view alignment parameters as a named JSON sidecar."""
    out_path = _ensure_parent(path)
    payload = alignment_params_payload(
        params5,
        du=du,
        dv=dv,
        gauge_metadata=gauge_metadata,
        translation_frame=translation_frame,
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
    translation_frame: str = "object",
) -> None:
    """Write a CSV sidecar, adding an explicit frame column for detector poses."""
    _validate_translation_frame(translation_frame)
    out_path = _ensure_parent(path)
    records = alignment_param_records(params5, du=du, dv=dv)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fields = CSV_FIELDNAMES + (
            ("pose_translation_frame",) if translation_frame == "detector" else ()
        )
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row: dict[str, int | float | str] = dict(record)
            if translation_frame == "detector":
                row["pose_translation_frame"] = translation_frame
            writer.writerow(row)
