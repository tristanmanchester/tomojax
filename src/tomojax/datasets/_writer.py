"""Synthetic dataset artifact writer."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from tomojax.datasets._phantoms import make_benchmark_phantom
from tomojax.datasets._specs import SyntheticDatasetSpec, synthetic128_spec
from tomojax.geometry import (
    GeometryState,
    PoseParameters,
    write_geometry_json,
    write_pose_params_csv,
)

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
    v2_nominal_geometry: Path
    v2_corrupted_geometry: Path
    v2_true_geometry: Path
    v2_nominal_pose: Path
    v2_corrupted_pose: Path
    v2_true_pose: Path
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
    nuisance = _realize_nuisance(spec, n_views, applied_to_projections=not clean)
    if not clean:
        projections = _apply_nuisance(projections, nuisance)
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
    nominal_state = _geometry_state_from_spec(spec, n_views=n_views, pose=None)
    true_state = _geometry_state_from_spec(spec, n_views=n_views, pose=true_pose)
    write_geometry_json(paths.v2_nominal_geometry, nominal_state)
    write_geometry_json(paths.v2_corrupted_geometry, nominal_state)
    write_geometry_json(paths.v2_true_geometry, true_state)
    write_pose_params_csv(paths.v2_nominal_pose, nominal_state.pose)
    write_pose_params_csv(paths.v2_corrupted_pose, nominal_state.pose)
    write_pose_params_csv(paths.v2_true_pose, true_state.pose)
    _write_json(paths.nuisance_truth, nuisance)
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
        v2_nominal_geometry=dataset_dir / "v2_nominal_geometry.json",
        v2_corrupted_geometry=dataset_dir / "v2_corrupted_geometry.json",
        v2_true_geometry=dataset_dir / "v2_true_geometry.json",
        v2_nominal_pose=dataset_dir / "v2_nominal_pose_params.csv",
        v2_corrupted_pose=dataset_dir / "v2_corrupted_pose_params.csv",
        v2_true_pose=dataset_dir / "v2_true_pose_params.csv",
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


def _realize_nuisance(
    spec: SyntheticDatasetSpec,
    n_views: int,
    *,
    applied_to_projections: bool,
) -> dict[str, object]:
    gain = _gain_drift(spec.nuisance, n_views)
    offset = _background_offset(spec.nuisance, n_views)
    vertical_gradient = _background_vertical_gradient(spec.nuisance, n_views)
    return {
        "schema": "tomojax.synthetic_nuisance_truth.v1",
        "source": "benchmark_manifest",
        "applied_to_projections": applied_to_projections,
        "spec": dict(spec.nuisance),
        "applied_terms": {
            "gain": gain is not None,
            "offset": offset is not None,
            "background_vertical_gradient": vertical_gradient is not None,
        },
        "gain": None if gain is None else [float(value) for value in gain],
        "offset": None if offset is None else [float(value) for value in offset],
        "background_vertical_gradient": (
            None if vertical_gradient is None else [float(value) for value in vertical_gradient]
        ),
    }


def _apply_nuisance(
    projections: NDArray[np.float32],
    nuisance: dict[str, object],
) -> NDArray[np.float32]:
    out = np.asarray(projections, dtype=np.float32).copy()
    raw_gain = nuisance.get("gain")
    if isinstance(raw_gain, list):
        gain = np.asarray(raw_gain, dtype=np.float32)
        out *= gain[:, None, None]
    raw_offset = nuisance.get("offset")
    if isinstance(raw_offset, list):
        offset = np.asarray(raw_offset, dtype=np.float32)
        out += offset[:, None, None]
    raw_gradient = nuisance.get("background_vertical_gradient")
    if isinstance(raw_gradient, list):
        gradient = np.asarray(raw_gradient, dtype=np.float32)
        vertical = np.linspace(-1.0, 1.0, out.shape[1], dtype=np.float32)
        out += gradient[:, None, None] * vertical[None, :, None]
    return out.astype(np.float32)


def _gain_drift(
    nuisance: dict[str, float | int | str | bool],
    n_views: int,
) -> NDArray[np.float32] | None:
    if "gain_drift" in nuisance:
        text = str(nuisance["gain_drift"])
        t = np.linspace(0.0, 1.0, n_views, dtype=np.float32)
        if text == "linear_0.98_to_1.03":
            return np.linspace(0.98, 1.03, n_views, dtype=np.float32)
        if text == "0.97_to_1.04_plus_sinusoid":
            return (
                np.linspace(0.97, 1.04, n_views, dtype=np.float32)
                + np.float32(0.01) * np.sin(np.float32(2.0 * np.pi) * t)
            ).astype(np.float32)
    if "gain_drift_fraction" in nuisance:
        fraction = float(nuisance["gain_drift_fraction"])
        theta = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False, dtype=np.float32)
        return (1.0 + np.float32(fraction) * np.sin(theta)).astype(np.float32)
    return None


def _background_offset(
    nuisance: dict[str, float | int | str | bool],
    n_views: int,
) -> NDArray[np.float32] | None:
    if nuisance.get("background_offset") == "small_linear":
        return np.linspace(-0.015, 0.015, n_views, dtype=np.float32)
    return None


def _background_vertical_gradient(
    nuisance: dict[str, float | int | str | bool],
    n_views: int,
) -> NDArray[np.float32] | None:
    if nuisance.get("background_drift") == "low_frequency_vertical_gradient":
        return (np.float32(0.02) * np.sin(np.linspace(0.0, np.pi, n_views))).astype(np.float32)
    return None


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


def _geometry_state_from_spec(
    spec: SyntheticDatasetSpec,
    *,
    n_views: int,
    pose: dict[str, NDArray[np.float32]] | None,
) -> GeometryState:
    state = GeometryState.zeros(n_views)
    setup_values = spec.true_setup if pose is not None else {}
    setup = state.setup
    det_v_active = abs(float(spec.true_setup.get("det_v_px", 0.0))) > 0.0
    setup = setup.replace_parameter(
        "det_v_px",
        replace(
            setup.det_v_px.with_value(_setup_value(setup_values, "det_v_px")),
            active=det_v_active,
        ),
    )
    setup = setup.replace_parameter(
        "det_u_px",
        setup.det_u_px.with_value(_setup_value(setup_values, "det_u_px")),
    )
    setup = setup.replace_parameter(
        "detector_roll_rad",
        setup.detector_roll_rad.with_value(
            np.deg2rad(_setup_value(setup_values, "detector_roll_deg"))
        ),
    )
    setup = setup.replace_parameter(
        "axis_rot_x_rad",
        setup.axis_rot_x_rad.with_value(np.deg2rad(_setup_value(setup_values, "axis_rot_x_deg"))),
    )
    setup = setup.replace_parameter(
        "axis_rot_y_rad",
        setup.axis_rot_y_rad.with_value(np.deg2rad(_setup_value(setup_values, "axis_rot_y_deg"))),
    )
    setup = setup.replace_parameter(
        "theta_offset_rad",
        setup.theta_offset_rad.with_value(
            np.deg2rad(_setup_value(setup_values, "theta_offset_deg"))
        ),
    )
    setup = setup.replace_parameter(
        "theta_scale",
        setup.theta_scale.with_value(_setup_value(setup_values, "theta_scale", default=1.0)),
    )
    return GeometryState(setup=setup, pose=_pose_params_from_table(pose, n_views=n_views))


def _setup_value(
    values: dict[str, float | str],
    name: str,
    *,
    default: float = 0.0,
) -> float:
    raw = values.get(name, default)
    return float(raw) if isinstance(raw, int | float) else default


def _pose_params_from_table(
    pose: dict[str, NDArray[np.float32]] | None,
    *,
    n_views: int,
) -> PoseParameters:
    if pose is None:
        return PoseParameters.zeros(n_views)
    return PoseParameters(
        alpha_rad=np.deg2rad(pose["alpha_deg"].astype(np.float64)),
        beta_rad=np.deg2rad(pose["beta_deg"].astype(np.float64)),
        phi_residual_rad=np.deg2rad(pose["phi_residual_deg"].astype(np.float64)),
        dx_px=pose["dx_px"].astype(np.float64),
        dz_px=pose["dz_px"].astype(np.float64),
    )


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
