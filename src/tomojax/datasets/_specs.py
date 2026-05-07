"""Synthetic benchmark specifications."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

GeometryMode = Literal["parallel_tomography", "parallel_laminography"]


@dataclass(frozen=True)
class SyntheticDatasetSpec:
    name: str
    purpose: str
    mode: GeometryMode
    phantom_seed: int
    pose_seed: int | None
    detector_shape: tuple[int, int]
    views: int
    theta_range_deg: tuple[float, float]
    true_setup: dict[str, float | str]
    true_pose: dict[str, float | str]
    true_object_motion: dict[str, float | str]
    nuisance: dict[str, float | int | str | bool]
    pass_criteria: dict[str, float | int | str | bool]
    laminography_tilt_deg: float | None = None


def default_benchmark_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "docs" / "tomojax-v2" / "benchmark_manifest.yaml"


def load_synthetic128_specs(manifest_path: Path | None = None) -> dict[str, SyntheticDatasetSpec]:
    path = manifest_path or default_benchmark_manifest_path()
    with path.open("r", encoding="utf-8") as fh:
        raw = cast("dict[str, Any]", yaml.safe_load(fh))

    suite = cast("dict[str, Any]", raw["synthetic128_suite"])
    datasets = cast("dict[str, dict[str, Any]]", suite["datasets"])
    return {name: _parse_spec(name, item) for name, item in datasets.items()}


def synthetic128_spec(name: str, manifest_path: Path | None = None) -> SyntheticDatasetSpec:
    specs = load_synthetic128_specs(manifest_path)
    try:
        return specs[name]
    except KeyError as exc:
        available = ", ".join(sorted(specs))
        raise KeyError(f"unknown synthetic128 dataset {name!r}; available: {available}") from exc


def _parse_spec(name: str, item: dict[str, Any]) -> SyntheticDatasetSpec:
    detector_shape = _pair_ints(item["detector_shape"], field="detector_shape")
    theta_range = _pair_floats(item["theta_range_deg"], field="theta_range_deg")
    mode = cast("GeometryMode", item["mode"])
    if mode not in {"parallel_tomography", "parallel_laminography"}:
        raise ValueError(f"unsupported synthetic mode {mode!r}")

    return SyntheticDatasetSpec(
        name=name,
        purpose=str(item["purpose"]),
        mode=mode,
        phantom_seed=int(item["phantom_seed"]),
        pose_seed=int(item["pose_seed"]) if "pose_seed" in item else None,
        detector_shape=detector_shape,
        views=int(item["views"]),
        theta_range_deg=theta_range,
        true_setup=_numberish_mapping(item["true_setup"]),
        true_pose=_numberish_mapping(item.get("true_pose", {})),
        true_object_motion=_numberish_mapping(item.get("true_object_motion", {})),
        nuisance=_numberish_mapping(item.get("nuisance", {})),
        pass_criteria=_numberish_mapping(item.get("pass_criteria", {})),
        laminography_tilt_deg=(
            float(item["laminography_tilt_deg"]) if "laminography_tilt_deg" in item else None
        ),
    )


def _pair_ints(value: object, *, field: str) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{field} must contain two values")
    return (int(value[0]), int(value[1]))


def _pair_floats(value: object, *, field: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{field} must contain two values")
    return (float(value[0]), float(value[1]))


def _numberish_mapping(value: object) -> dict[str, float | int | str | bool]:
    if not isinstance(value, dict):
        raise ValueError("expected mapping")
    out: dict[str, float | int | str | bool] = {}
    for key, item in value.items():
        if isinstance(item, bool | int | float | str):
            out[str(key)] = item
        else:
            out[str(key)] = str(item)
    return out
