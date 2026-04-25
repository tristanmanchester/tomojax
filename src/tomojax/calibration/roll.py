from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import math
from typing import Mapping, Sequence

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Geometry, Grid
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import forward_project_view_T
from tomojax.data.geometry_meta import build_geometry_from_meta
from tomojax.recon.fbp import _default_fbp_scale, _run_fbp_fast_path, fbp
from tomojax.recon.quicklook import extract_central_slice, scale_to_uint8

from ._json import JsonValue, normalize_json
from .detector_grid import detector_grid_from_detector_roll
from .gauge import validate_calibration_gauges
from .manifest import build_calibration_manifest
from .objectives import MetricSpec, ObjectiveCard
from .state import CalibrationState, CalibrationVariable


DETECTOR_ROLL_DOFS: tuple[str, ...] = ("detector_roll_deg",)


@dataclass(frozen=True)
class DetectorRollCalibrationConfig:
    """Configuration for detector-plane roll Gauss-Newton calibration."""

    initial_detector_roll_deg: float = 0.0
    outer_iters: int = 12
    gn_damping: float = 1e-3
    gn_accept_tol: float = 0.0
    max_step_deg: float = 1.0
    heldout_stride: int = 8
    filter_name: str = "ramp"
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "auto"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.initial_detector_roll_deg)):
            raise ValueError("initial_detector_roll_deg must be finite")
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

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "initial_detector_roll_deg": float(self.initial_detector_roll_deg),
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
class DetectorRollIteration:
    iteration: int
    detector_roll_deg: float
    loss_before: float
    loss_after: float
    accepted: bool
    raw_step_deg: float
    applied_step_deg: float
    step_scale: float
    gradient_norm: float
    curvature: float
    validation_mode: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "iteration": int(self.iteration),
            "detector_roll_deg": float(self.detector_roll_deg),
            "loss_before": float(self.loss_before),
            "loss_after": float(self.loss_after),
            "accepted": bool(self.accepted),
            "raw_step_deg": float(self.raw_step_deg),
            "applied_step_deg": float(self.applied_step_deg),
            "step_scale": float(self.step_scale),
            "gradient_norm": float(self.gradient_norm),
            "curvature": float(self.curvature),
            "validation_mode": self.validation_mode,
        }


@dataclass(frozen=True)
class DetectorRollCalibrationResult:
    detector_roll_deg: float
    final_volume: np.ndarray
    iterations: tuple[DetectorRollIteration, ...]
    objective_card: ObjectiveCard
    calibration_state: CalibrationState
    manifest: dict[str, JsonValue]
    confidence: dict[str, JsonValue]
    artifact_paths: dict[str, str] = field(default_factory=dict)


def _geometry_from_inputs(
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    thetas_deg: Sequence[float],
) -> Geometry:
    payload = dict(geometry_inputs)
    payload["grid"] = grid.to_dict()
    payload["detector"] = detector.to_dict()
    payload["thetas_deg"] = np.asarray(thetas_deg, dtype=np.float32)
    _, _, geometry = build_geometry_from_meta(payload)
    return geometry


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


def _reconstruct_with_roll(
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    thetas_deg: np.ndarray,
    view_indices: np.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    config: DetectorRollCalibrationConfig,
) -> jnp.ndarray:
    subset_thetas = thetas_deg[view_indices]
    geometry = _geometry_from_inputs(
        geometry_inputs,
        grid=grid,
        detector=detector,
        thetas_deg=subset_thetas,
    )
    subset_projections = jnp.asarray(projections[view_indices], dtype=jnp.float32)
    n_views = int(subset_projections.shape[0])
    batch_size = max(1, min(int(config.views_per_batch), n_views))
    volume = _run_fbp_fast_path(
        stack_view_poses(geometry, n_views),
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
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    thetas_deg: np.ndarray,
    score_indices: np.ndarray,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    config: DetectorRollCalibrationConfig,
) -> jnp.ndarray:
    score_geometry = _geometry_from_inputs(
        geometry_inputs,
        grid=grid,
        detector=detector,
        thetas_deg=thetas_deg[score_indices],
    )
    poses = stack_view_poses(score_geometry, len(score_indices))
    preds = jnp.stack(
        [
            forward_project_view_T(
                poses[i],
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


def _confidence_diagnostics(
    iterations: Sequence[DetectorRollIteration],
) -> dict[str, JsonValue]:
    accepted = [it for it in iterations if it.accepted]
    final = iterations[-1] if iterations else None
    level = "low"
    final_update_norm = None
    final_gradient_norm = None
    curvature = None
    loss_rel_drop = None
    if final is not None:
        final_update_norm = abs(float(final.applied_step_deg))
        final_gradient_norm = float(final.gradient_norm)
        curvature = float(final.curvature)
        first = float(iterations[0].loss_before)
        last = float(final.loss_after)
        loss_rel_drop = (first - last) / max(abs(first), 1e-12)
        if accepted and curvature > 0.0 and final_update_norm <= 0.02 and final_gradient_norm <= 1e-2:
            level = "high"
        elif accepted and curvature > 0.0:
            level = "medium"
    return {
        "level": level,
        "accepted_steps": int(len(accepted)),
        "final_update_norm_deg": final_update_norm,
        "final_gradient_norm": final_gradient_norm,
        "curvature_min_eig": curvature,
        "loss_rel_drop": loss_rel_drop,
    }


def _write_iterations_csv(path: Path, iterations: Sequence[DetectorRollIteration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(DetectorRollIteration.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for iteration in iterations:
            writer.writerow(iteration.to_dict())


def _write_iterations_json(path: Path, iterations: Sequence[DetectorRollIteration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([iteration.to_dict() for iteration in iterations], fh, indent=2, sort_keys=True)
        fh.write("\n")


def _volume_preview(
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    thetas_deg: np.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    config: DetectorRollCalibrationConfig,
) -> np.ndarray:
    all_views = np.arange(len(thetas_deg), dtype=np.int32)
    volume = _reconstruct_with_roll(
        geometry_inputs,
        grid=grid,
        detector=detector,
        projections=projections,
        thetas_deg=thetas_deg,
        view_indices=all_views,
        det_grid=det_grid,
        config=config,
    )
    return scale_to_uint8(extract_central_slice(np.asarray(volume)))


def _write_contact_sheet(
    path: Path,
    images: Sequence[np.ndarray],
    *,
    pad: int = 4,
) -> None:
    if not images:
        return
    height = max(int(image.shape[0]) for image in images)
    width = sum(int(image.shape[1]) for image in images) + pad * (len(images) - 1)
    sheet = np.zeros((height, width), dtype=np.uint8)
    x = 0
    for image in images:
        h, w = image.shape
        sheet[:h, x : x + w] = image
        x += w + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, sheet)


def _write_artifacts(
    workdir: Path,
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    thetas_deg: np.ndarray,
    iterations: Sequence[DetectorRollIteration],
    final_detector_roll_deg: float,
    config: DetectorRollCalibrationConfig,
) -> dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    nominal_grid = detector_grid_from_detector_roll(
        detector,
        detector_roll_deg=float(config.initial_detector_roll_deg),
    )
    final_grid = detector_grid_from_detector_roll(
        detector,
        detector_roll_deg=float(final_detector_roll_deg),
    )
    nominal_preview = _volume_preview(
        geometry_inputs,
        grid=grid,
        detector=detector,
        projections=projections,
        thetas_deg=thetas_deg,
        det_grid=nominal_grid,
        config=config,
    )
    final_preview = _volume_preview(
        geometry_inputs,
        grid=grid,
        detector=detector,
        projections=projections,
        thetas_deg=thetas_deg,
        det_grid=final_grid,
        config=config,
    )
    final_preview_path = workdir / "final_detector_roll.png"
    iio.imwrite(final_preview_path, final_preview)
    contact_sheet = workdir / "detector_roll_before_after.png"
    _write_contact_sheet(contact_sheet, (nominal_preview, final_preview))
    iterations_csv = workdir / "iterations.csv"
    iterations_json = workdir / "iterations.json"
    _write_iterations_csv(iterations_csv, iterations)
    _write_iterations_json(iterations_json, iterations)
    return {
        "contact_sheet": str(contact_sheet),
        "final_preview": str(final_preview_path),
        "iterations_csv": str(iterations_csv),
        "iterations_json": str(iterations_json),
    }


def _calibration_state_for_roll(
    *,
    detector_roll_deg: float,
    confidence: Mapping[str, object] | None = None,
) -> CalibrationState:
    return CalibrationState(
        detector=(
            CalibrationVariable(
                name="detector_roll_deg",
                value=float(detector_roll_deg),
                unit="deg",
                status="estimated",
                frame="detector",
                gauge="detector_plane_roll",
                uncertainty=normalize_json(confidence),
                description="Detector-plane roll around the detector centre.",
            ),
        )
    )


def calibrate_detector_roll(
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: object,
    config: DetectorRollCalibrationConfig | None = None,
    workdir: str | Path | None = None,
) -> DetectorRollCalibrationResult:
    """Estimate detector-plane roll with damped Gauss-Newton."""
    cfg = config or DetectorRollCalibrationConfig()
    validate_calibration_gauges(
        _calibration_state_for_roll(detector_roll_deg=float(cfg.initial_detector_roll_deg))
    )

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
    roll_value = jnp.asarray([float(cfg.initial_detector_roll_deg)], dtype=jnp.float32)
    iterations: list[DetectorRollIteration] = []

    def det_grid_for_roll(value: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return detector_grid_from_detector_roll(detector, detector_roll_deg=value[0])

    for iteration_idx in range(1, int(cfg.outer_iters) + 1):
        def residual_for_roll(value: jnp.ndarray) -> jnp.ndarray:
            det_grid = det_grid_for_roll(value)
            volume = _reconstruct_with_roll(
                geometry_inputs,
                grid=grid,
                detector=detector,
                projections=y,
                thetas_deg=thetas,
                view_indices=train_indices,
                det_grid=det_grid,
                config=cfg,
            )
            return _residual_vector(
                geometry_inputs,
                grid=grid,
                detector=detector,
                projections=y,
                thetas_deg=thetas,
                score_indices=score_indices,
                volume=volume,
                det_grid=det_grid,
                config=cfg,
            )

        residual = residual_for_roll(roll_value)
        loss_before = _loss_from_residual(residual)
        _, pullback = jax.vjp(residual_for_roll, roll_value)
        gradient = pullback(residual)[0] * jnp.float32(2.0 / max(int(residual.size), 1))
        direction = jnp.asarray([1.0], dtype=jnp.float32)
        jac_col = jax.jvp(residual_for_roll, (roll_value,), (direction,))[1]
        curvature = (
            jnp.sum(jac_col * jac_col) * jnp.float32(2.0 / max(int(residual.size), 1))
        )
        raw_step = -gradient / (curvature + jnp.float32(cfg.gn_damping))
        raw_step = jnp.asarray(raw_step, dtype=jnp.float32)
        raw_norm = jnp.linalg.norm(raw_step)
        max_step = jnp.asarray(float(cfg.max_step_deg), dtype=jnp.float32)
        clipped_step = jnp.where(
            raw_norm > max_step,
            raw_step * (max_step / jnp.maximum(raw_norm, jnp.float32(1e-6))),
            raw_step,
        )

        best_value = roll_value
        best_loss = loss_before
        best_step = jnp.zeros_like(roll_value)
        best_scale = 0.0
        for scale in (1.0, 0.5, 0.25):
            trial_step = clipped_step * jnp.float32(scale)
            trial_value = roll_value + trial_step
            trial_loss = _loss_from_residual(residual_for_roll(trial_value))
            improvement = loss_before - trial_loss
            threshold = float(cfg.gn_accept_tol) * max(abs(loss_before), 1e-12)
            if math.isfinite(trial_loss) and improvement >= threshold and trial_loss < best_loss:
                best_value = trial_value
                best_loss = trial_loss
                best_step = trial_step
                best_scale = float(scale)
                break

        roll_value = jnp.asarray(best_value, dtype=jnp.float32)
        iterations.append(
            DetectorRollIteration(
                iteration=iteration_idx,
                detector_roll_deg=float(roll_value[0]),
                loss_before=float(loss_before),
                loss_after=float(best_loss),
                accepted=bool(best_scale > 0.0),
                raw_step_deg=float(raw_step[0]),
                applied_step_deg=float(best_step[0]),
                step_scale=float(best_scale),
                gradient_norm=float(jnp.linalg.norm(gradient)),
                curvature=float(curvature),
                validation_mode=validation_mode,
            )
        )
        if float(jnp.linalg.norm(best_step)) <= 1e-4:
            break

    final_roll = float(roll_value[0])
    final_det_grid = detector_grid_from_detector_roll(detector, detector_roll_deg=final_roll)
    final_geometry = _geometry_from_inputs(
        geometry_inputs,
        grid=grid,
        detector=detector,
        thetas_deg=thetas,
    )
    final_volume = np.asarray(
        fbp(
            final_geometry,
            grid,
            detector,
            y,
            filter_name=str(cfg.filter_name),
            views_per_batch=int(cfg.views_per_batch),
            projector_unroll=int(cfg.projector_unroll),
            checkpoint_projector=bool(cfg.checkpoint_projector),
            gather_dtype=str(cfg.gather_dtype),
            det_grid=final_det_grid,
        )
    )
    confidence = _confidence_diagnostics(iterations)
    artifact_paths: dict[str, str] = {}
    if workdir is not None:
        artifact_paths = _write_artifacts(
            Path(workdir),
            geometry_inputs,
            grid=grid,
            detector=detector,
            projections=y,
            thetas_deg=thetas,
            iterations=iterations,
            final_detector_roll_deg=final_roll,
            config=cfg,
        )
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
        curvature={"detector_roll": confidence.get("curvature_min_eig")},
        contact_sheet=artifact_paths.get("contact_sheet"),
    )
    state = _calibration_state_for_roll(detector_roll_deg=final_roll, confidence=confidence)
    manifest = build_calibration_manifest(
        calibration_state=state,
        objective_card=objective,
        calibrated_geometry={
            "detector": detector.to_dict(),
            "input_detector": detector.to_dict(),
            "detector_roll_deg": final_roll,
            "gauge": "detector_plane_roll",
        },
        source={
            "geometry_type": normalize_json(geometry_inputs.get("geometry_type", "parallel")),
            "n_views": int(y.shape[0]),
            "projection_shape": list(map(int, y.shape)),
        },
        extra={
            "config": cfg.to_dict(),
            "confidence": confidence,
            "iterations": [iteration.to_dict() for iteration in iterations],
            "artifact_paths": artifact_paths,
        },
    )
    return DetectorRollCalibrationResult(
        detector_roll_deg=final_roll,
        final_volume=final_volume,
        iterations=tuple(iterations),
        objective_card=objective,
        calibration_state=state,
        manifest=manifest,
        confidence=confidence,
        artifact_paths=artifact_paths,
    )
