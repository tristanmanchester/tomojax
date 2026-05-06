"""Synthetic dataset artifact loading."""
# pyright: reportAny=false

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

import numpy as np

from tomojax.geometry import GeometryState, read_geometry_json, read_pose_params_csv


@dataclass(frozen=True)
class SyntheticArrayMetadata:
    """Shape and dtype metadata for a generated NumPy artifact."""

    path: Path
    shape: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable metadata payload."""
        return {
            "path": str(self.path),
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class SyntheticDatasetSidecars:
    """Readback view over generated synthetic dataset artifacts."""

    dataset_dir: Path
    manifest: dict[str, object]
    artifacts: dict[str, Path]
    nominal_geometry: GeometryState
    corrupted_geometry: GeometryState
    true_geometry: GeometryState
    volume: SyntheticArrayMetadata
    projections: SyntheticArrayMetadata
    mask: SyntheticArrayMetadata


def load_synthetic_dataset_sidecars(dataset_dir: Path) -> SyntheticDatasetSidecars:
    """Load manifest-indexed v2 sidecars from a generated synthetic dataset."""
    root = Path(dataset_dir)
    manifest_path = root / "dataset_manifest.json"
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    artifacts = _resolve_artifacts(root, manifest)
    nominal_geometry = _read_indexed_geometry(
        artifacts,
        geometry_key="v2_nominal_geometry_json",
        pose_key="v2_nominal_pose_params_csv",
    )
    corrupted_geometry = _read_indexed_geometry(
        artifacts,
        geometry_key="v2_corrupted_geometry_json",
        pose_key="v2_corrupted_pose_params_csv",
    )
    true_geometry = _read_indexed_geometry(
        artifacts,
        geometry_key="v2_true_geometry_json",
        pose_key="v2_true_pose_params_csv",
    )
    volume = _read_array_metadata(artifacts, "ground_truth_volume_npy")
    projections = _read_array_metadata(artifacts, "projections_npy")
    mask = _read_array_metadata(artifacts, "mask_npy")
    return SyntheticDatasetSidecars(
        dataset_dir=root,
        manifest=manifest,
        artifacts=artifacts,
        nominal_geometry=nominal_geometry,
        corrupted_geometry=corrupted_geometry,
        true_geometry=true_geometry,
        volume=volume,
        projections=projections,
        mask=mask,
    )


def _resolve_artifacts(root: Path, manifest: dict[str, object]) -> dict[str, Path]:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise ValueError("dataset_manifest.json must contain an artifacts mapping")
    artifact_items = cast("dict[object, object]", raw_artifacts)
    resolved: dict[str, Path] = {}
    for key, value in artifact_items.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("dataset_manifest.json artifacts must map strings to strings")
        resolved[key] = root / value
    return resolved


def _read_indexed_geometry(
    artifacts: dict[str, Path],
    *,
    geometry_key: str,
    pose_key: str,
) -> GeometryState:
    try:
        geometry_path = artifacts[geometry_key]
        pose_path = artifacts[pose_key]
    except KeyError as exc:
        raise ValueError(
            f"dataset_manifest.json missing required artifact {exc.args[0]!r}"
        ) from exc
    pose = read_pose_params_csv(pose_path)
    return read_geometry_json(geometry_path, pose)


def _read_array_metadata(
    artifacts: dict[str, Path],
    key: str,
) -> SyntheticArrayMetadata:
    try:
        path = artifacts[key]
    except KeyError as exc:
        raise ValueError(f"dataset_manifest.json missing required artifact {key!r}") from exc
    array = np.load(path, mmap_mode="r")
    return SyntheticArrayMetadata(
        path=path,
        shape=tuple(int(dim) for dim in array.shape),
        dtype=str(array.dtype),
    )
