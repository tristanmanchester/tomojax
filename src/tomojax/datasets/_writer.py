"""Synthetic dataset artifact writer."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np

from tomojax.datasets._phantoms import make_benchmark_phantom
from tomojax.datasets._specs import SyntheticDatasetSpec, synthetic128_spec
from tomojax.forward import (
    PROJECTION_OPERATOR,
    core_projection_geometry_from_state,
    project_parallel_reference,
)
from tomojax.geometry import (
    AcquisitionParameters,
    GeometryState,
    PoseParameters,
    write_geometry_json,
    write_pose_params_csv,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

SyntheticSize = Literal[32, 64, 128]


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
    supported_only: bool = False,
) -> SyntheticArtifactPaths:
    """Write deterministic synthetic dataset artifacts for a manifest spec."""
    spec = synthetic128_spec(name)
    dataset_suffix = f"{size}_supported_only" if supported_only else str(size)
    dataset_dir = output_dir / f"{name}_{dataset_suffix}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        _remove_known_artifacts(dataset_dir)

    n_views = int(views or (16 if size == 32 else spec.views))
    detector_shape = _detector_shape_for_size(spec, size)
    theta = np.linspace(
        spec.theta_range_deg[0],
        spec.theta_range_deg[1],
        n_views,
        endpoint=False,
        dtype=np.float32,
    )

    volume = make_benchmark_phantom(size, spec.phantom_seed)
    pixel_scale = float(size) / 128.0
    true_pose = _make_pose_table(spec, theta, pixel_scale=pixel_scale)
    setup_override = _supported_only_setup(spec.true_setup) if supported_only else None
    nominal_state = _geometry_state_from_spec(
        spec,
        n_views=n_views,
        pose=None,
        theta_deg=theta,
        pixel_scale=pixel_scale,
        setup_override=setup_override,
    )
    true_state = _geometry_state_from_spec(
        spec,
        n_views=n_views,
        pose=true_pose,
        theta_deg=theta,
        pixel_scale=pixel_scale,
        setup_override=setup_override,
    )
    unsupported_dofs = _unsupported_dofs_not_evaluated(spec, supported_only=supported_only)
    projected_true_state = _core_projectable_state(true_state)
    projections = _project_v2_smoke(volume, projected_true_state)
    core_geometry = core_projection_geometry_from_state(volume.shape, projected_true_state)
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
    _write_json(
        paths.true_geometry,
        _true_geometry(spec, detector_shape, theta, setup_override=setup_override),
    )
    _write_pose_csv(paths.true_pose, true_pose)
    _write_motion_csv(paths.true_motion, spec.true_object_motion, n_views)
    write_geometry_json(paths.v2_nominal_geometry, nominal_state)
    write_geometry_json(paths.v2_corrupted_geometry, nominal_state)
    write_geometry_json(paths.v2_true_geometry, projected_true_state)
    write_pose_params_csv(paths.v2_nominal_pose, nominal_state.pose)
    write_pose_params_csv(paths.v2_corrupted_pose, nominal_state.pose)
    write_pose_params_csv(paths.v2_true_pose, projected_true_state.pose)
    _write_json(paths.nuisance_truth, nuisance)
    _write_json(paths.noise_truth, _noise_truth(spec))
    _write_json(
        paths.manifest,
        _dataset_manifest(
            spec,
            size,
            detector_shape,
            n_views,
            paths,
            supported_only=supported_only,
            operator_provenance=core_geometry.provenance(),
            unsupported_dofs_not_evaluated=unsupported_dofs,
        ),
    )
    return paths


def _detector_shape_for_size(spec: SyntheticDatasetSpec, size: int) -> tuple[int, int]:
    _ = spec
    return (int(size), int(size))


def _project_v2_smoke(
    volume: NDArray[np.float32],
    geometry: GeometryState,
) -> NDArray[np.float32]:
    return np.asarray(project_parallel_reference(jnp.asarray(volume), geometry), dtype=np.float32)


def _core_projectable_state(state: GeometryState) -> GeometryState:
    setup = state.setup
    pose = PoseParameters(
        alpha_rad=state.pose.alpha_rad,
        beta_rad=state.pose.beta_rad,
        theta_nominal_rad=state.pose.theta_nominal_rad,
        phi_residual_rad=state.pose.phi_residual_rad,
        dx_px=state.pose.dx_px,
        dz_px=state.pose.dz_px,
    )
    return GeometryState(setup=setup, pose=pose, acquisition=state.acquisition)


def _unsupported_dofs_not_evaluated(
    spec: SyntheticDatasetSpec,
    *,
    supported_only: bool,
) -> list[str]:
    if supported_only:
        return []
    unsupported: list[str] = []
    if spec.true_object_motion:
        unsupported.append("object_motion")
    unsupported.extend(_unsupported_pose_terms(spec.true_pose))
    unsupported.extend(_unsupported_nuisance_terms(spec.nuisance))
    return unsupported


def _unsupported_pose_terms(pose: dict[str, float | str]) -> list[str]:
    unsupported: list[str] = []
    for name, value in pose.items():
        text = str(value)
        if "sparse_jumps" in text:
            unsupported.append(f"pose.{name}.sparse_jumps")
    return unsupported


def _unsupported_nuisance_terms(
    nuisance: dict[str, float | int | str | bool],
) -> list[str]:
    unsupported: list[str] = []
    supported_keys = {
        "background_drift",
        "background_offset",
        "gain_drift",
        "gain_drift_fraction",
        "noise",
    }
    for key, value in nuisance.items():
        if key in supported_keys:
            continue
        if key in {
            "bad_views",
            "dead_pixels_fraction",
            "gaussian_noise_fraction",
            "hot_pixels_fraction",
            "partial_fov",
            "partial_fov_invalid_edge_fraction",
            "stripe_bias_columns",
        }:
            unsupported.append(f"nuisance.{key}")
            continue
        if isinstance(value, str) and value not in {"standard", "hard"}:
            unsupported.append(f"nuisance.{key}")
    return unsupported


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
    *,
    pixel_scale: float,
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
        "dx_px": pixel_scale
        * _pose_component(spec.true_pose.get("dx_px", "zero"), rng, theta_rad, n_views, scale=20.0),
        "dz_px": pixel_scale
        * _pose_component(spec.true_pose.get("dz_px", "zero"), rng, theta_rad, n_views, scale=20.0),
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
    theta_deg: NDArray[np.float32],
    pixel_scale: float,
    setup_override: dict[str, float | str] | None = None,
) -> GeometryState:
    state = GeometryState.zeros(n_views)
    setup_values = (
        setup_override
        if setup_override is not None and pose is not None
        else spec.true_setup
        if pose is not None
        else {}
    )
    setup = state.setup
    reference_setup = setup_override if setup_override is not None else spec.true_setup
    det_v_active = abs(float(reference_setup.get("det_v_px", 0.0))) > 0.0
    setup = setup.replace_parameter(
        "det_v_px",
        replace(
            setup.det_v_px.with_value(pixel_scale * _setup_value(setup_values, "det_v_px")),
            active=det_v_active,
        ),
    )
    setup = setup.replace_parameter(
        "det_u_px",
        setup.det_u_px.with_value(pixel_scale * _setup_value(setup_values, "det_u_px")),
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
    acquisition = (
        AcquisitionParameters.parallel_laminography(
            tilt_rad=float(np.deg2rad(spec.laminography_tilt_deg)),
            tilt_about="x",
        )
        if spec.laminography_tilt_deg is not None
        else AcquisitionParameters.parallel()
    )
    return GeometryState(
        setup=setup,
        pose=_pose_params_from_table(pose, n_views=n_views, theta_deg=theta_deg),
        acquisition=acquisition,
    )


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
    theta_deg: NDArray[np.float32],
) -> PoseParameters:
    if pose is None:
        return PoseParameters.zeros(n_views).with_updates(
            theta_nominal_rad=np.deg2rad(theta_deg.astype(np.float64))
        )
    return PoseParameters(
        alpha_rad=np.deg2rad(pose["alpha_deg"].astype(np.float64)),
        beta_rad=np.deg2rad(pose["beta_deg"].astype(np.float64)),
        theta_nominal_rad=np.deg2rad(theta_deg.astype(np.float64)),
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
    spec: SyntheticDatasetSpec,
    detector_shape: tuple[int, int],
    theta_deg: NDArray[np.float32],
    *,
    setup_override: dict[str, float | str] | None = None,
) -> dict[str, object]:
    geom = _nominal_geometry(spec, detector_shape, theta_deg)
    geom["setup"] = setup_override if setup_override is not None else spec.true_setup
    if spec.laminography_tilt_deg is not None:
        geom["laminography_tilt_deg"] = spec.laminography_tilt_deg
    return geom


def _dataset_manifest(
    spec: SyntheticDatasetSpec,
    size: int,
    detector_shape: tuple[int, int],
    views: int,
    paths: SyntheticArtifactPaths,
    supported_only: bool,
    operator_provenance: dict[str, object],
    unsupported_dofs_not_evaluated: list[str],
) -> dict[str, object]:
    pass_criteria = dict(spec.pass_criteria)
    if supported_only:
        pass_criteria = {
            key: value
            for key, value in pass_criteria.items()
            if key in {"det_u_error_px_lt", "det_v_error_px_lt", "theta_offset_error_deg_lt"}
        }
    manifest: dict[str, object] = {
        "name": spec.name,
        "purpose": spec.purpose,
        "mode": spec.mode,
        "volume_shape": [size, size, size],
        "detector_shape": list(detector_shape),
        "views": views,
        "phantom_seed": spec.phantom_seed,
        "pose_seed": spec.pose_seed,
        "artifact_contract": "tomojax-v2.synthetic-dataset.v1",
        "projection_operator": PROJECTION_OPERATOR,
        "operator_provenance": operator_provenance,
        "variant": "supported_only" if supported_only else "manifest",
        "unsupported_dofs_not_evaluated": unsupported_dofs_not_evaluated,
        "unsupported_dof_status": (
            "unsupported_dof_not_evaluated" if unsupported_dofs_not_evaluated else "all_supported"
        ),
        "artifacts": _manifest_artifact_map(paths),
        "recovery_tolerances": pass_criteria,
    }
    if spec.detector_grid is not None:
        manifest["detector_grid"] = spec.detector_grid
    return manifest


def _supported_only_setup(values: dict[str, float | str]) -> dict[str, float | str]:
    supported = dict(values)
    supported["detector_roll_deg"] = 0.0
    supported["axis_rot_x_deg"] = 0.0
    supported["axis_rot_y_deg"] = 0.0
    supported["theta_scale"] = 1.0
    return supported


def _manifest_artifact_map(paths: SyntheticArtifactPaths) -> dict[str, str]:
    return {
        "ground_truth_volume_npy": paths.volume.name,
        "projections_npy": paths.projections.name,
        "mask_npy": paths.mask.name,
        "nominal_geometry_json": paths.nominal_geometry.name,
        "corrupted_geometry_json": paths.corrupted_geometry.name,
        "true_geometry_json": paths.true_geometry.name,
        "true_pose_csv": paths.true_pose.name,
        "true_motion_csv": paths.true_motion.name,
        "v2_nominal_geometry_json": paths.v2_nominal_geometry.name,
        "v2_corrupted_geometry_json": paths.v2_corrupted_geometry.name,
        "v2_true_geometry_json": paths.v2_true_geometry.name,
        "v2_nominal_pose_params_csv": paths.v2_nominal_pose.name,
        "v2_corrupted_pose_params_csv": paths.v2_corrupted_pose.name,
        "v2_true_pose_params_csv": paths.v2_true_pose.name,
        "nuisance_truth_json": paths.nuisance_truth.name,
        "noise_truth_json": paths.noise_truth.name,
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


def _write_motion_csv(
    path: Path,
    motion: dict[str, float | str],
    n_views: int,
) -> None:
    values = _make_object_motion_table(motion, n_views)
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
                    "tx_obj_px": float(values["tx_obj_px"][idx]),
                    "ty_obj_px": float(values["ty_obj_px"][idx]),
                    "tz_obj_px": float(values["tz_obj_px"][idx]),
                    "rot_obj_z_deg": float(values["rot_obj_z_deg"][idx]),
                }
            )


def _make_object_motion_table(
    motion: dict[str, float | str],
    n_views: int,
) -> dict[str, NDArray[np.float32]]:
    return {
        name: _object_motion_component(motion.get(name, 0.0), n_views)
        for name in ("tx_obj_px", "ty_obj_px", "tz_obj_px", "rot_obj_z_deg")
    }


def _object_motion_component(value: float | str, n_views: int) -> NDArray[np.float32]:
    if n_views <= 0:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(value, int | float):
        return np.full(n_views, float(value), dtype=np.float32)
    text = str(value).replace(" ", "")
    t = (
        np.linspace(0.0, 1.0, n_views, dtype=np.float32)
        if n_views > 1
        else np.zeros(1, dtype=np.float32)
    )
    smooth = (3.0 * t**2 - 2.0 * t**3).astype(np.float32)
    if text.endswith("*smoothstep(t)"):
        return np.float32(float(text.removesuffix("*smoothstep(t)"))) * smooth
    if text.endswith("*sin(2*pi*t)"):
        return (
            np.float32(float(text.removesuffix("*sin(2*pi*t)")))
            * np.sin(np.float32(2.0 * np.pi) * t)
        ).astype(np.float32)
    if text.endswith("*t"):
        return np.float32(float(text.removesuffix("*t"))) * t
    return np.zeros(n_views, dtype=np.float32)
