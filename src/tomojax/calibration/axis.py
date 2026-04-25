from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import math
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Grid, RotationAxisGeometry
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device
from tomojax.recon.fbp import _default_fbp_scale, _run_fbp_fast_path

from ._json import JsonValue, normalize_json
from .axis_geometry import (
    AXIS_DIRECTION_DOFS,
    axis_pose_stack,
    axis_unit_from_active,
    axis_values_from_rotations,
    default_active_axis_dofs,
    nominal_axis_unit_from_inputs,
)
from .gauge import validate_calibration_gauges
from .manifest import build_calibration_manifest
from .objectives import MetricSpec, ObjectiveCard
from .state import CalibrationState, CalibrationVariable


@dataclass(frozen=True)
class AxisDirectionCalibrationConfig:
    """Configuration for scanner rotation-axis direction Gauss-Newton calibration."""

    active_axis_dofs: tuple[str, ...] | None = None
    initial_axis_rot_x_deg: float = 0.0
    initial_axis_rot_y_deg: float = 0.0
    outer_iters: int = 12
    gn_damping: float = 1e-3
    gn_accept_tol: float = 0.0
    max_step_deg: float = 2.0
    heldout_stride: int = 8
    filter_name: str = "ramp"
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "auto"

    def resolved_active_axis_dofs(
        self, geometry_inputs: Mapping[str, object]
    ) -> tuple[str, ...]:
        active = self.active_axis_dofs
        if active is None:
            active = default_active_axis_dofs(geometry_inputs)
        active = tuple(str(name) for name in active)
        if not active:
            raise ValueError("active_axis_dofs must not be empty")
        unknown = sorted(set(active) - set(AXIS_DIRECTION_DOFS))
        if unknown:
            raise ValueError(f"Unknown axis-direction DOFs: {unknown}")
        if len(set(active)) != len(active):
            raise ValueError("active_axis_dofs must not contain duplicates")
        return active

    def __post_init__(self) -> None:
        if self.active_axis_dofs is not None:
            active = tuple(str(name) for name in self.active_axis_dofs)
            unknown = sorted(set(active) - set(AXIS_DIRECTION_DOFS))
            if not active:
                raise ValueError("active_axis_dofs must not be empty")
            if unknown:
                raise ValueError(f"Unknown axis-direction DOFs: {unknown}")
            if len(set(active)) != len(active):
                raise ValueError("active_axis_dofs must not contain duplicates")
            object.__setattr__(self, "active_axis_dofs", active)
        if int(self.outer_iters) < 1:
            raise ValueError("outer_iters must be >= 1")
        if not math.isfinite(float(self.gn_damping)) or float(self.gn_damping) < 0.0:
            raise ValueError("gn_damping must be finite and >= 0")
        if not math.isfinite(float(self.gn_accept_tol)) or float(self.gn_accept_tol) < 0.0:
            raise ValueError("gn_accept_tol must be finite and >= 0")
        if not math.isfinite(float(self.max_step_deg)) or float(self.max_step_deg) <= 0.0:
            raise ValueError("max_step_deg must be finite and > 0")
        if int(self.heldout_stride) < 2:
            raise ValueError("heldout_stride must be >= 2")
        if int(self.views_per_batch) < 1:
            raise ValueError("views_per_batch must be >= 1")

    def to_dict(self, *, active_axis_dofs: Sequence[str] | None = None) -> dict[str, JsonValue]:
        return {
            "active_axis_dofs": list(active_axis_dofs or self.active_axis_dofs or ()),
            "initial_axis_rot_x_deg": float(self.initial_axis_rot_x_deg),
            "initial_axis_rot_y_deg": float(self.initial_axis_rot_y_deg),
            "outer_iters": int(self.outer_iters),
            "gn_damping": float(self.gn_damping),
            "gn_accept_tol": float(self.gn_accept_tol),
            "max_step_deg": float(self.max_step_deg),
            "heldout_stride": int(self.heldout_stride),
            "filter_name": str(self.filter_name),
            "views_per_batch": int(self.views_per_batch),
            "projector_unroll": int(self.projector_unroll),
            "checkpoint_projector": bool(self.checkpoint_projector),
            "gather_dtype": str(self.gather_dtype),
        }


@dataclass(frozen=True)
class AxisDirectionIteration:
    iteration: int
    axis_rot_x_deg: float
    axis_rot_y_deg: float
    axis_unit_lab: tuple[float, float, float]
    loss_before: float
    loss_after: float
    accepted: bool
    raw_step_deg: tuple[float, ...]
    applied_step_deg: tuple[float, ...]
    step_scale: float
    gradient_norm: float
    curvature: tuple[tuple[float, ...], ...]
    validation_mode: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "iteration": int(self.iteration),
            "axis_rot_x_deg": float(self.axis_rot_x_deg),
            "axis_rot_y_deg": float(self.axis_rot_y_deg),
            "axis_unit_lab": [float(v) for v in self.axis_unit_lab],
            "loss_before": float(self.loss_before),
            "loss_after": float(self.loss_after),
            "accepted": bool(self.accepted),
            "raw_step_deg": [float(v) for v in self.raw_step_deg],
            "applied_step_deg": [float(v) for v in self.applied_step_deg],
            "step_scale": float(self.step_scale),
            "gradient_norm": float(self.gradient_norm),
            "curvature": [[float(v) for v in row] for row in self.curvature],
            "validation_mode": self.validation_mode,
        }


@dataclass(frozen=True)
class AxisDirectionCalibrationResult:
    axis_rot_x_deg: float
    axis_rot_y_deg: float
    axis_unit_lab: tuple[float, float, float]
    calibrated_geometry: RotationAxisGeometry
    final_volume: np.ndarray
    iterations: tuple[AxisDirectionIteration, ...]
    objective_card: ObjectiveCard
    calibration_state: CalibrationState
    manifest: dict[str, JsonValue]
    confidence: dict[str, JsonValue]
    artifact_paths: dict[str, str] = field(default_factory=dict)


def _split_views(n_views: int, heldout_stride: int) -> tuple[np.ndarray, np.ndarray, str]:
    all_views = np.arange(int(n_views), dtype=np.int32)
    heldout = all_views[:: int(heldout_stride)]
    heldout_set = {int(v) for v in heldout.tolist()}
    train = np.asarray([int(v) for v in all_views if int(v) not in heldout_set], dtype=np.int32)
    if len(heldout) == 0 or len(train) == 0:
        return all_views, all_views, "insample_projection_nmse"
    return train, heldout, "heldout_projection_nmse"


def _loss_from_residual(residual: jnp.ndarray) -> float:
    return float(jnp.mean(jnp.square(residual)))


def _reconstruct_with_axis(
    *,
    poses: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    view_indices: np.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    config: AxisDirectionCalibrationConfig,
) -> jnp.ndarray:
    subset_projections = jnp.asarray(projections[view_indices], dtype=jnp.float32)
    n_views = int(subset_projections.shape[0])
    batch_size = max(1, min(int(config.views_per_batch), n_views))
    volume = _run_fbp_fast_path(
        poses[jnp.asarray(view_indices, dtype=jnp.int32)],
        subset_projections,
        batch_size=batch_size,
        grid=grid,
        detector=detector,
        filter_name=str(config.filter_name),
        projector_unroll=int(config.projector_unroll),
        checkpoint_projector=bool(config.checkpoint_projector),
        gather_dtype=str(config.gather_dtype),
        det_grid=det_grid,
    )
    return volume * _default_fbp_scale(n_views)


def _residual_vector(
    *,
    poses: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    score_indices: np.ndarray,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    config: AxisDirectionCalibrationConfig,
) -> jnp.ndarray:
    score_idx = jnp.asarray(score_indices, dtype=jnp.int32)
    score_poses = poses[score_idx]
    preds = jnp.stack(
        [
            forward_project_view_T(
                score_poses[i],
                grid,
                detector,
                volume,
                gather_dtype=str(config.gather_dtype),
                det_grid=det_grid,
            )
            for i in range(len(score_indices))
        ],
        axis=0,
    )
    measured = projections[score_indices]
    denom = jnp.sqrt(jnp.maximum(jnp.mean(measured.astype(jnp.float32) ** 2), jnp.float32(1e-6)))
    return ((preds - measured) / denom).ravel()


def _axis_rotations_from_values(
    values: jnp.ndarray,
    *,
    active_names: Sequence[str],
    fixed_x: float,
    fixed_y: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rot_x = jnp.asarray(fixed_x, dtype=jnp.float32)
    rot_y = jnp.asarray(fixed_y, dtype=jnp.float32)
    for idx, name in enumerate(active_names):
        if name == "axis_rot_x_deg":
            rot_x = values[idx]
        elif name == "axis_rot_y_deg":
            rot_y = values[idx]
    return rot_x, rot_y


def _confidence_diagnostics(
    iterations: Sequence[AxisDirectionIteration],
) -> dict[str, JsonValue]:
    accepted = [it for it in iterations if it.accepted]
    final = iterations[-1] if iterations else None
    level = "low"
    final_update_norm = None
    final_gradient_norm = None
    curvature_min_eig = None
    loss_rel_drop = None
    if final is not None:
        final_update_norm = float(np.linalg.norm(np.asarray(final.applied_step_deg)))
        final_gradient_norm = float(final.gradient_norm)
        curvature = np.asarray(final.curvature, dtype=np.float64)
        if curvature.size:
            try:
                curvature_min_eig = float(np.min(np.linalg.eigvalsh(curvature)))
            except np.linalg.LinAlgError:
                curvature_min_eig = None
        first = float(iterations[0].loss_before)
        last = float(final.loss_after)
        loss_rel_drop = (first - last) / max(abs(first), 1e-12)
        if (
            accepted
            and curvature_min_eig is not None
            and curvature_min_eig > 0.0
            and final_update_norm <= 0.05
            and final_gradient_norm <= 1e-2
        ):
            level = "high"
        elif accepted and curvature_min_eig is not None and curvature_min_eig > 0.0:
            level = "medium"
    return {
        "level": level,
        "accepted_steps": int(len(accepted)),
        "final_update_norm_deg": final_update_norm,
        "final_gradient_norm": final_gradient_norm,
        "curvature_min_eig": curvature_min_eig,
        "loss_rel_drop": loss_rel_drop,
    }


def _calibration_state(
    *,
    active_names: Sequence[str],
    axis_rot_x_deg: float,
    axis_rot_y_deg: float,
    axis_unit_lab: Sequence[float],
    confidence: Mapping[str, object] | None,
) -> CalibrationState:
    return CalibrationState(
        scan=(
            CalibrationVariable(
                name="axis_rot_x_deg",
                value=float(axis_rot_x_deg),
                unit="deg",
                status="estimated" if "axis_rot_x_deg" in active_names else "frozen",
                frame="scan",
                gauge="rotation_axis_direction",
                uncertainty=normalize_json(confidence),
            ),
            CalibrationVariable(
                name="axis_rot_y_deg",
                value=float(axis_rot_y_deg),
                unit="deg",
                status="estimated" if "axis_rot_y_deg" in active_names else "frozen",
                frame="scan",
                gauge="rotation_axis_direction",
                uncertainty=normalize_json(confidence),
            ),
            CalibrationVariable(
                name="axis_unit_lab",
                value=[float(v) for v in axis_unit_lab],
                unit="unit_vector",
                status="derived",
                frame="scan",
                gauge="rotation_axis_direction",
                description="Lab-frame scanner rotation-axis unit vector derived from axis DOFs.",
            ),
        )
    )


def _write_iterations_csv(path: Path, iterations: Sequence[AxisDirectionIteration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "iteration",
        "axis_rot_x_deg",
        "axis_rot_y_deg",
        "axis_unit_lab",
        "loss_before",
        "loss_after",
        "accepted",
        "raw_step_deg",
        "applied_step_deg",
        "step_scale",
        "gradient_norm",
        "validation_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for iteration in iterations:
            row = iteration.to_dict()
            row["axis_unit_lab"] = json.dumps(row["axis_unit_lab"])
            row["raw_step_deg"] = json.dumps(row["raw_step_deg"])
            row["applied_step_deg"] = json.dumps(row["applied_step_deg"])
            row.pop("curvature", None)
            writer.writerow(row)


def _write_iterations_json(path: Path, iterations: Sequence[AxisDirectionIteration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([iteration.to_dict() for iteration in iterations], fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_artifacts(
    workdir: Path,
    *,
    iterations: Sequence[AxisDirectionIteration],
) -> dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    iterations_csv = workdir / "axis_iterations.csv"
    iterations_json = workdir / "axis_iterations.json"
    _write_iterations_csv(iterations_csv, iterations)
    _write_iterations_json(iterations_json, iterations)
    return {
        "iterations_csv": str(iterations_csv),
        "iterations_json": str(iterations_json),
    }


def calibrate_axis_direction(
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: object,
    config: AxisDirectionCalibrationConfig | None = None,
    workdir: str | Path | None = None,
) -> AxisDirectionCalibrationResult:
    """Estimate scanner rotation-axis direction with damped Gauss-Newton."""
    cfg = config or AxisDirectionCalibrationConfig()
    active_names = cfg.resolved_active_axis_dofs(geometry_inputs)
    nominal_axis = nominal_axis_unit_from_inputs(geometry_inputs)
    initial_state = _calibration_state(
        active_names=active_names,
        axis_rot_x_deg=float(cfg.initial_axis_rot_x_deg),
        axis_rot_y_deg=float(cfg.initial_axis_rot_y_deg),
        axis_unit_lab=nominal_axis,
        confidence=None,
    )
    validate_calibration_gauges(initial_state)

    y = jnp.asarray(projections, dtype=jnp.float32)
    if y.ndim != 3:
        raise ValueError(f"projections must have shape (n_views, nv, nu), got {y.shape}")
    if tuple(y.shape[1:]) != (int(detector.nv), int(detector.nu)):
        raise ValueError(
            "projection detector shape does not match detector metadata: "
            f"projection={tuple(y.shape[1:])}, detector={(detector.nv, detector.nu)}"
        )
    thetas = np.asarray(geometry_inputs["thetas_deg"], dtype=np.float32)
    if thetas.shape != (int(y.shape[0]),):
        raise ValueError(
            f"thetas_deg must have shape ({int(y.shape[0])},), got {thetas.shape}"
        )

    train_indices, score_indices, validation_mode = _split_views(
        int(y.shape[0]),
        int(cfg.heldout_stride),
    )
    fixed_x = float(cfg.initial_axis_rot_x_deg)
    fixed_y = float(cfg.initial_axis_rot_y_deg)
    active_values = axis_values_from_rotations(
        active_names=active_names,
        axis_rot_x_deg=fixed_x,
        axis_rot_y_deg=fixed_y,
    )
    det_grid = get_detector_grid_device(detector)
    iterations: list[AxisDirectionIteration] = []

    def axis_for_values(values: jnp.ndarray) -> jnp.ndarray:
        return axis_unit_from_active(
            values,
            active_names=active_names,
            nominal_axis_unit=nominal_axis,
            fixed_axis_rot_x_deg=fixed_x,
            fixed_axis_rot_y_deg=fixed_y,
        )

    def poses_for_values(values: jnp.ndarray) -> jnp.ndarray:
        return axis_pose_stack(thetas, axis_for_values(values))

    for iteration_idx in range(1, int(cfg.outer_iters) + 1):
        def residual_for_values(values: jnp.ndarray) -> jnp.ndarray:
            poses = poses_for_values(values)
            volume = _reconstruct_with_axis(
                poses=poses,
                grid=grid,
                detector=detector,
                projections=y,
                view_indices=train_indices,
                det_grid=det_grid,
                config=cfg,
            )
            return _residual_vector(
                poses=poses,
                grid=grid,
                detector=detector,
                projections=y,
                score_indices=score_indices,
                volume=volume,
                det_grid=det_grid,
                config=cfg,
            )

        residual = residual_for_values(active_values)
        loss_before = _loss_from_residual(residual)
        _, pullback = jax.vjp(residual_for_values, active_values)
        gradient = pullback(residual)[0] * jnp.float32(2.0 / max(int(residual.size), 1))
        eye = jnp.eye(int(active_values.size), dtype=jnp.float32)

        def jvp_col(direction: jnp.ndarray) -> jnp.ndarray:
            return jax.jvp(residual_for_values, (active_values,), (direction,))[1]

        jac_cols = jax.vmap(jvp_col)(eye)
        curvature = (jac_cols @ jac_cols.T) * jnp.float32(2.0 / max(int(residual.size), 1))
        system = curvature + jnp.eye(int(active_values.size), dtype=jnp.float32) * jnp.float32(
            cfg.gn_damping
        )
        raw_step = jnp.linalg.solve(system, -gradient)
        raw_norm = jnp.linalg.norm(raw_step)
        max_step = jnp.asarray(float(cfg.max_step_deg), dtype=jnp.float32)
        clipped_step = jnp.where(
            raw_norm > max_step,
            raw_step * (max_step / jnp.maximum(raw_norm, jnp.float32(1e-6))),
            raw_step,
        )

        best_values = active_values
        best_loss = loss_before
        best_step = jnp.zeros_like(active_values)
        best_scale = 0.0
        for scale in (1.0, 0.5, 0.25):
            trial_step = clipped_step * jnp.float32(scale)
            trial_values = active_values + trial_step
            trial_loss = _loss_from_residual(residual_for_values(trial_values))
            improvement = loss_before - trial_loss
            threshold = float(cfg.gn_accept_tol) * max(abs(loss_before), 1e-12)
            if math.isfinite(trial_loss) and improvement >= threshold and trial_loss < best_loss:
                best_values = trial_values
                best_loss = trial_loss
                best_step = trial_step
                best_scale = float(scale)
                break

        active_values = jnp.asarray(best_values, dtype=jnp.float32)
        rot_x, rot_y = _axis_rotations_from_values(
            active_values,
            active_names=active_names,
            fixed_x=fixed_x,
            fixed_y=fixed_y,
        )
        axis = axis_for_values(active_values)
        iterations.append(
            AxisDirectionIteration(
                iteration=iteration_idx,
                axis_rot_x_deg=float(rot_x),
                axis_rot_y_deg=float(rot_y),
                axis_unit_lab=tuple(float(v) for v in np.asarray(axis)),
                loss_before=float(loss_before),
                loss_after=float(best_loss),
                accepted=bool(best_scale > 0.0),
                raw_step_deg=tuple(float(v) for v in np.asarray(raw_step)),
                applied_step_deg=tuple(float(v) for v in np.asarray(best_step)),
                step_scale=float(best_scale),
                gradient_norm=float(jnp.linalg.norm(gradient)),
                curvature=tuple(
                    tuple(float(v) for v in row) for row in np.asarray(curvature)
                ),
                validation_mode=validation_mode,
            )
        )
        if float(jnp.linalg.norm(best_step)) <= 1e-4:
            break

    final_axis = np.asarray(axis_for_values(active_values), dtype=np.float64)
    final_axis_tuple = tuple(float(v) for v in final_axis)
    final_rot_x, final_rot_y = _axis_rotations_from_values(
        active_values,
        active_names=active_names,
        fixed_x=fixed_x,
        fixed_y=fixed_y,
    )
    final_rot_x_f = float(final_rot_x)
    final_rot_y_f = float(final_rot_y)
    final_poses = poses_for_values(active_values)
    all_indices = np.arange(int(y.shape[0]), dtype=np.int32)
    final_volume = np.asarray(
        _reconstruct_with_axis(
            poses=final_poses,
            grid=grid,
            detector=detector,
            projections=y,
            view_indices=all_indices,
            det_grid=det_grid,
            config=cfg,
        )
    )
    calibrated_geometry = RotationAxisGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=thetas,
        axis_unit_lab=final_axis_tuple,
    )
    confidence = _confidence_diagnostics(iterations)
    state = _calibration_state(
        active_names=active_names,
        axis_rot_x_deg=final_rot_x_f,
        axis_rot_y_deg=final_rot_y_f,
        axis_unit_lab=final_axis_tuple,
        confidence=confidence,
    )
    artifact_paths: dict[str, str] = {}
    if workdir is not None:
        artifact_paths = _write_artifacts(Path(workdir), iterations=iterations)
    objective = ObjectiveCard(
        primary_metric=MetricSpec(
            name=validation_mode,
            direction="minimize",
            domain="projection",
            description="Projection-domain normalized mean squared error.",
        ),
        validation_split={
            "mode": validation_mode,
            "heldout_stride": int(cfg.heldout_stride),
            "train_indices": train_indices.tolist(),
            "score_indices": score_indices.tolist(),
        },
        curvature={"axis_direction": confidence.get("curvature_min_eig")},
        contact_sheet=artifact_paths.get("contact_sheet"),
    )
    manifest = build_calibration_manifest(
        calibration_state=state,
        objective_card=objective,
        calibrated_geometry={
            "detector": detector.to_dict(),
            "axis_unit_lab": [float(v) for v in final_axis_tuple],
            "input_axis_unit_lab": [float(v) for v in nominal_axis],
            "gauge": "rotation_axis_direction",
        },
        source={
            "geometry_type": normalize_json(geometry_inputs.get("geometry_type", "parallel")),
            "n_views": int(y.shape[0]),
            "projection_shape": list(map(int, y.shape)),
        },
        extra={
            "config": cfg.to_dict(active_axis_dofs=active_names),
            "confidence": confidence,
            "iterations": [iteration.to_dict() for iteration in iterations],
            "artifact_paths": artifact_paths,
            "wording": (
                "axis_unit_lab is the calibrated lab-frame scanner rotation-axis "
                "direction. It is instrument geometry, not object-frame pose residual."
            ),
        },
    )
    return AxisDirectionCalibrationResult(
        axis_rot_x_deg=final_rot_x_f,
        axis_rot_y_deg=final_rot_y_f,
        axis_unit_lab=final_axis_tuple,
        calibrated_geometry=calibrated_geometry,
        final_volume=final_volume,
        iterations=tuple(iterations),
        objective_card=objective,
        calibration_state=state,
        manifest=manifest,
        confidence=confidence,
        artifact_paths=artifact_paths,
    )
