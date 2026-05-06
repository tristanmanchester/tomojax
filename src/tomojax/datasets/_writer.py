"""Synthetic dataset artifact writer."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from tomojax.datasets._phantoms import make_benchmark_phantom
from tomojax.datasets._specs import SyntheticDatasetSpec, synthetic128_spec

if TYPE_CHECKING:
    from numpy.typing import NDArray

SyntheticSize = Literal[32, 128]


@dataclass(frozen=True)
class SyntheticArtifactPaths:
    dataset_dir: Path
    manifest: Path
    volume: Path
    projections: Path
    mask: Path
    nominal_geometry: Path
    corrupted_geometry: Path
    true_geometry: Path
    true_pose: Path
    true_motion: Path
    nuisance_truth: Path
    noise_truth: Path


def generate_synthetic_dataset(
    name: str,
    output_dir: Path,
    *,
    size: SyntheticSize = 32,
    clean: bool = False,
    views: int | None = None,
) -> SyntheticArtifactPaths:
    """Write deterministic synthetic dataset artifacts for a manifest spec."""
    spec = synthetic128_spec(name)
    dataset_dir = output_dir / f"{name}_{size}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        _remove_known_artifacts(dataset_dir)

    n_views = int(views or (16 if size == 32 else spec.views))
    detector_shape = (
        max(16, spec.detector_shape[0] * size // 128),
        max(16, spec.detector_shape[1] * size // 128),
    )
    theta = np.linspace(
        spec.theta_range_deg[0],
        spec.theta_range_deg[1],
        n_views,
        endpoint=False,
        dtype=np.float32,
    )

    volume = make_benchmark_phantom(size, spec.phantom_seed)
    true_pose = _make_pose_table(spec, theta)
    projections = _project_smoke(
        volume,
        detector_shape=detector_shape,
        theta_deg=theta,
        pose=true_pose,
    )
    mask = np.ones(projections.shape, dtype=bool)

    paths = _paths(dataset_dir)
    np.save(paths.volume, volume)
    np.save(paths.projections, projections)
    np.save(paths.mask, mask)
    nominal_geometry = _nominal_geometry(spec, detector_shape, theta)
    _write_json(paths.nominal_geometry, nominal_geometry)
    _write_json(paths.corrupted_geometry, nominal_geometry)
    _write_json(paths.true_geometry, _true_geometry(spec, detector_shape, theta))
    _write_pose_csv(paths.true_pose, true_pose)
    _write_motion_csv(paths.true_motion, n_views)
    _write_json(paths.nuisance_truth, spec.nuisance)
    _write_json(paths.noise_truth, _noise_truth(spec))
    _write_json(paths.manifest, _dataset_manifest(spec, size, detector_shape, n_views))
    return paths


def _paths(dataset_dir: Path) -> SyntheticArtifactPaths:
    return SyntheticArtifactPaths(
        dataset_dir=dataset_dir,
        manifest=dataset_dir / "dataset_manifest.json",
        volume=dataset_dir / "ground_truth_volume.npy",
        projections=dataset_dir / "projections.npy",
        mask=dataset_dir / "mask.npy",
        nominal_geometry=dataset_dir / "nominal_geometry.json",
        corrupted_geometry=dataset_dir / "corrupted_geometry.json",
        true_geometry=dataset_dir / "true_geometry.json",
        true_pose=dataset_dir / "true_pose.csv",
        true_motion=dataset_dir / "true_motion.csv",
        nuisance_truth=dataset_dir / "nuisance_truth.json",
        noise_truth=dataset_dir / "noise_truth.json",
    )


def _remove_known_artifacts(dataset_dir: Path) -> None:
    for path in _paths(dataset_dir).__dict__.values():
        if isinstance(path, Path) and path.is_file():
            path.unlink()


def _project_smoke(
    volume: NDArray[np.float32],
    *,
    detector_shape: tuple[int, int],
    theta_deg: NDArray[np.float32],
    pose: dict[str, NDArray[np.float32]],
) -> NDArray[np.float32]:
    base_views: list[NDArray[np.float32]] = []
    for idx, theta in enumerate(theta_deg):
        quadrant = int(np.floor((float(theta) % 180.0) / 45.0)) % 4
        rotated = np.rot90(volume, k=quadrant, axes=(0, 1))
        projection = rotated.sum(axis=1, dtype=np.float32)
        projection = _resize_nearest(projection, detector_shape)
        projection = np.roll(projection, round(float(pose["dx_px"][idx])), axis=1)
        projection = np.roll(projection, round(float(pose["dz_px"][idx])), axis=0)
        base_views.append(projection.astype(np.float32))
    return np.stack(base_views, axis=0).astype(np.float32)


def _resize_nearest(image: NDArray[np.float32], shape: tuple[int, int]) -> NDArray[np.float32]:
    rows = np.linspace(0, image.shape[0] - 1, shape[0]).round().astype(np.intp)
    cols = np.linspace(0, image.shape[1] - 1, shape[1]).round().astype(np.intp)
    return image[np.ix_(rows, cols)].astype(np.float32)


def _make_pose_table(
    spec: SyntheticDatasetSpec,
    theta_deg: NDArray[np.float32],
) -> dict[str, NDArray[np.float32]]:
    rng = np.random.default_rng(spec.pose_seed or spec.phantom_seed)
    n_views = theta_deg.shape[0]
    theta_rad = np.deg2rad(theta_deg.astype(np.float32))
    return {
        "view": np.arange(n_views, dtype=np.float32),
        "theta_deg": theta_deg.astype(np.float32),
        "alpha_deg": _pose_component(
            spec.true_pose.get("alpha_deg", "zero"), rng, theta_rad, n_views, scale=2.0
        ),
        "beta_deg": _pose_component(
            spec.true_pose.get("beta_deg", "zero"), rng, theta_rad, n_views, scale=2.0
        ),
        "phi_residual_deg": _pose_component(
            spec.true_pose.get("phi_residual_deg", "zero"), rng, theta_rad, n_views, scale=10.0
        ),
        "dx_px": _pose_component(
            spec.true_pose.get("dx_px", "zero"), rng, theta_rad, n_views, scale=20.0
        ),
        "dz_px": _pose_component(
            spec.true_pose.get("dz_px", "zero"), rng, theta_rad, n_views, scale=20.0
        ),
    }


def _pose_component(
    value: float | int | str | bool | None,
    rng: np.random.Generator,
    theta_rad: NDArray[np.float32],
    n_views: int,
    *,
    scale: float,
) -> NDArray[np.float32]:
    out = np.zeros(n_views, dtype=np.float32)
    if value in (None, "zero", False):
        pass
    elif isinstance(value, int | float):
        out = np.full(n_views, float(value), dtype=np.float32)
    elif (text := str(value)).startswith("uniform_"):
        parts = text.split("_")
        out = rng.uniform(float(parts[1]), float(parts[2]), size=n_views).astype(np.float32)
    elif "sin" in text:
        out = (np.float32(scale * 0.15) * np.sin(theta_rad)).astype(np.float32)
    elif "cos" in text:
        out = (np.float32(scale * 0.10) * np.cos(theta_rad)).astype(np.float32)
    elif text.startswith("normal_"):
        parts = text.split("_")
        out = rng.normal(float(parts[1]), float(parts[2]), size=n_views).astype(np.float32)
    return out


def _nominal_geometry(
    spec: SyntheticDatasetSpec, detector_shape: tuple[int, int], theta_deg: NDArray[np.float32]
) -> dict[str, object]:
    return {
        "model": spec.mode,
        "detector_shape": list(detector_shape),
        "theta_deg": [float(v) for v in theta_deg],
        "setup": {
            "det_u_px": 0.0,
            "det_v_px": 0.0,
            "detector_roll_deg": 0.0,
            "axis_rot_x_deg": 0.0,
            "axis_rot_y_deg": 0.0,
            "theta_offset_deg": 0.0,
            "theta_scale": 1.0,
        },
    }


def _true_geometry(
    spec: SyntheticDatasetSpec, detector_shape: tuple[int, int], theta_deg: NDArray[np.float32]
) -> dict[str, object]:
    geom = _nominal_geometry(spec, detector_shape, theta_deg)
    geom["setup"] = spec.true_setup
    if spec.laminography_tilt_deg is not None:
        geom["laminography_tilt_deg"] = spec.laminography_tilt_deg
    return geom


def _dataset_manifest(
    spec: SyntheticDatasetSpec, size: int, detector_shape: tuple[int, int], views: int
) -> dict[str, object]:
    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "mode": spec.mode,
        "volume_shape": [size, size, size],
        "detector_shape": list(detector_shape),
        "views": views,
        "phantom_seed": spec.phantom_seed,
        "pose_seed": spec.pose_seed,
        "artifact_contract": "tomojax-v2.synthetic-dataset.v1",
        "recovery_tolerances": spec.pass_criteria,
    }


def _noise_truth(spec: SyntheticDatasetSpec) -> dict[str, object]:
    return {"source": "benchmark_manifest", "nuisance": spec.nuisance}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_pose_csv(path: Path, pose: dict[str, NDArray[np.float32]]) -> None:
    fieldnames = [
        "view",
        "theta_deg",
        "alpha_deg",
        "beta_deg",
        "phi_residual_deg",
        "dx_px",
        "dz_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(pose["view"].shape[0]):
            writer.writerow({field: float(pose[field][idx]) for field in fieldnames})


def _write_motion_csv(path: Path, n_views: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["view", "tx_obj_px", "ty_obj_px", "tz_obj_px", "rot_obj_z_deg"],
        )
        writer.writeheader()
        for idx in range(n_views):
            writer.writerow(
                {
                    "view": idx,
                    "tx_obj_px": 0.0,
                    "ty_obj_px": 0.0,
                    "tz_obj_px": 0.0,
                    "rot_obj_z_deg": 0.0,
                }
            )
