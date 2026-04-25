from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device
from tomojax.data.geometry_meta import build_geometry_from_meta
from tomojax.recon.fbp import _default_fbp_scale, _run_fbp_fast_path, fbp
from tomojax.recon.quicklook import extract_central_slice, scale_to_uint8

from ._json import JsonValue, normalize_json
from .detector_grid import offset_detector_grid
from .gauge import validate_calibration_gauges
from .manifest import build_calibration_manifest
from .objectives import MetricSpec, ObjectiveCard
from .state import CalibrationState, CalibrationVariable


DETECTOR_CENTER_DOFS: tuple[str, ...] = ("det_u_px", "det_v_px")


@dataclass(frozen=True)
class DetectorCenterCalibrationConfig:
    """Configuration for detector/ray-grid centre Gauss-Newton calibration."""

    initial_det_u_px: float = 0.0
    det_v_px: float = 0.0
    det_v_status: str = "frozen"
    active_detector_dofs: tuple[str, ...] = ("det_u_px",)
    outer_iters: int = 6
    gn_damping: float = 1e-3
    gn_accept_tol: float = 0.0
    max_step_px: float = 2.0
    heldout_stride: int = 8
    filter_name: str = "ramp"
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "auto"

    def __post_init__(self) -> None:
        if self.det_v_status not in {"frozen", "supplied"}:
            raise ValueError("det_v_status must be 'frozen' or 'supplied'")
        active = tuple(str(name) for name in self.active_detector_dofs)
        if not active:
            raise ValueError("active_detector_dofs must not be empty")
        unknown = sorted(set(active) - set(DETECTOR_CENTER_DOFS))
        if unknown:
            raise ValueError(f"Unknown detector-centre DOFs: {unknown}")
        if len(set(active)) != len(active):
            raise ValueError("active_detector_dofs must not contain duplicates")
        object.__setattr__(self, "active_detector_dofs", active)

        if int(self.outer_iters) < 1:
            raise ValueError("outer_iters must be >= 1")
        if not math.isfinite(float(self.gn_damping)) or float(self.gn_damping) < 0.0:
            raise ValueError("gn_damping must be finite and >= 0")
        if not math.isfinite(float(self.gn_accept_tol)) or float(self.gn_accept_tol) < 0.0:
            raise ValueError("gn_accept_tol must be finite and >= 0")
        if not math.isfinite(float(self.max_step_px)) or float(self.max_step_px) <= 0.0:
            raise ValueError("max_step_px must be finite and > 0")
        if int(self.heldout_stride) < 2:
            raise ValueError("heldout_stride must be >= 2")
        if int(self.views_per_batch) < 1:
            raise ValueError("views_per_batch must be >= 1")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "initial_det_u_px": float(self.initial_det_u_px),
            "det_v_px": float(self.det_v_px),
            "det_v_status": str(self.det_v_status),
            "active_detector_dofs": [str(name) for name in self.active_detector_dofs],
            "outer_iters": int(self.outer_iters),
            "gn_damping": float(self.gn_damping),
            "gn_accept_tol": float(self.gn_accept_tol),
            "max_step_px": float(self.max_step_px),
            "heldout_stride": int(self.heldout_stride),
            "filter_name": str(self.filter_name),
            "views_per_batch": int(self.views_per_batch),
            "projector_unroll": int(self.projector_unroll),
            "checkpoint_projector": bool(self.checkpoint_projector),
            "gather_dtype": str(self.gather_dtype),
        }


@dataclass(frozen=True)
class DetectorCenterIteration:
    iteration: int
    det_u_px: float
    det_v_px: float
    loss_before: float
    loss_after: float
    accepted: bool
    raw_step_px: tuple[float, ...]
    applied_step_px: tuple[float, ...]
    step_scale: float
    gradient_norm: float
    curvature: tuple[tuple[float, ...], ...]
    validation_mode: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "iteration": int(self.iteration),
            "det_u_px": float(self.det_u_px),
            "det_v_px": float(self.det_v_px),
            "loss_before": float(self.loss_before),
            "loss_after": float(self.loss_after),
            "accepted": bool(self.accepted),
            "raw_step_px": [float(v) for v in self.raw_step_px],
            "applied_step_px": [float(v) for v in self.applied_step_px],
            "step_scale": float(self.step_scale),
            "gradient_norm": float(self.gradient_norm),
            "curvature": [[float(v) for v in row] for row in self.curvature],
            "validation_mode": self.validation_mode,
        }


@dataclass(frozen=True)
class DetectorCenterCalibrationResult:
    best_det_u_px: float
    det_v_px: float
    calibrated_detector: Detector
    final_volume: np.ndarray
    iterations: tuple[DetectorCenterIteration, ...]
    objective_card: ObjectiveCard
    calibration_state: CalibrationState
    manifest: dict[str, JsonValue]
    confidence: dict[str, JsonValue]
    artifact_paths: dict[str, str] = field(default_factory=dict)


def detector_with_center_offset(
    detector: Detector,
    *,
    det_u_px: float,
    det_v_px: float = 0.0,
) -> Detector:
    """Return a detector whose centre is shifted by native detector pixel offsets."""
    return replace(
        detector,
        det_center=(
            float(detector.det_center[0]) + float(det_u_px) * float(detector.du),
            float(detector.det_center[1]) + float(det_v_px) * float(detector.dv),
        ),
    )


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


def _det_grid_from_offsets(
    base_grid: tuple[jnp.ndarray, jnp.ndarray],
    detector: Detector,
    *,
    det_u_px: object,
    det_v_px: object,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return offset_detector_grid(
        base_grid,
        det_u_px=det_u_px,
        det_v_px=det_v_px,
        native_du=float(detector.du),
        native_dv=float(detector.dv),
    )


def _active_values_from_offsets(
    *,
    active_names: Sequence[str],
    det_u_px: float,
    det_v_px: float,
) -> jnp.ndarray:
    values = []
    for name in active_names:
        if name == "det_u_px":
            values.append(float(det_u_px))
        elif name == "det_v_px":
            values.append(float(det_v_px))
    return jnp.asarray(values, dtype=jnp.float32)


def _det_u_v_from_active(
    active_values: jnp.ndarray,
    *,
    active_names: Sequence[str],
    fixed_det_u_px: float,
    fixed_det_v_px: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    values = jnp.asarray(active_values, dtype=jnp.float32)
    det_u = jnp.asarray(fixed_det_u_px, dtype=jnp.float32)
    det_v = jnp.asarray(fixed_det_v_px, dtype=jnp.float32)
    for idx, name in enumerate(active_names):
        if name == "det_u_px":
            det_u = values[idx]
        elif name == "det_v_px":
            det_v = values[idx]
    return det_u, det_v


def _reconstruct_with_detector_center(
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    thetas_deg: np.ndarray,
    view_indices: np.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    config: DetectorCenterCalibrationConfig,
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
    # Use the differentiable FBP core directly here. The public FBP wrapper blocks
    # for runtime/progress diagnostics, which is invalid under JAX JVP/VJP tracing.
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
    config: DetectorCenterCalibrationConfig,
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


def _loss_from_residual(residual: jnp.ndarray) -> float:
    return float(jnp.mean(jnp.square(residual)))


def _confidence_diagnostics(
    iterations: Sequence[DetectorCenterIteration],
) -> dict[str, JsonValue]:
    accepted = [it for it in iterations if it.accepted]
    final = iterations[-1] if iterations else None
    final_update_norm = None
    final_gradient_norm = None
    curvature_min_eig = None
    loss_rel_drop = None
    level = "low"
    if final is not None:
        final_update_norm = float(np.linalg.norm(np.asarray(final.applied_step_px)))
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
        "final_update_norm_px": final_update_norm,
        "final_gradient_norm": final_gradient_norm,
        "curvature_min_eig": curvature_min_eig,
        "loss_rel_drop": loss_rel_drop,
    }


def _write_iterations_csv(path: Path, iterations: Sequence[DetectorCenterIteration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "iteration",
        "det_u_px",
        "det_v_px",
        "loss_before",
        "loss_after",
        "accepted",
        "raw_step_px",
        "applied_step_px",
        "step_scale",
        "gradient_norm",
        "validation_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for iteration in iterations:
            row = iteration.to_dict()
            row["raw_step_px"] = json.dumps(row["raw_step_px"])
            row["applied_step_px"] = json.dumps(row["applied_step_px"])
            row.pop("curvature", None)
            writer.writerow(row)


def _write_iterations_json(path: Path, iterations: Sequence[DetectorCenterIteration]) -> None:
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
    config: DetectorCenterCalibrationConfig,
) -> np.ndarray:
    all_views = np.arange(len(thetas_deg), dtype=np.int32)
    volume = _reconstruct_with_detector_center(
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
    iterations: Sequence[DetectorCenterIteration],
    final_det_u_px: float,
    final_det_v_px: float,
    config: DetectorCenterCalibrationConfig,
) -> dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    base_grid = get_detector_grid_device(detector)
    nominal_grid = _det_grid_from_offsets(
        base_grid,
        detector,
        det_u_px=0.0,
        det_v_px=float(config.det_v_px),
    )
    final_grid = _det_grid_from_offsets(
        base_grid,
        detector,
        det_u_px=float(final_det_u_px),
        det_v_px=float(final_det_v_px),
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
    final_preview_path = workdir / "final_detector_center.png"
    iio.imwrite(final_preview_path, final_preview)
    contact_sheet = workdir / "detector_center_before_after.png"
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


def _calibration_state_for_config(
    cfg: DetectorCenterCalibrationConfig,
    *,
    det_u_px: float,
    det_v_px: float,
    confidence: Mapping[str, object] | None = None,
) -> CalibrationState:
    det_u_status = "estimated" if "det_u_px" in cfg.active_detector_dofs else "frozen"
    det_v_status = "estimated" if "det_v_px" in cfg.active_detector_dofs else cfg.det_v_status
    return CalibrationState(
        detector=(
            CalibrationVariable(
                name="det_u_px",
                value=float(det_u_px),
                unit="native_detector_px",
                status=det_u_status,  # type: ignore[arg-type]
                frame="detector",
                gauge="detector_ray_grid_center",
                uncertainty=normalize_json(confidence),
                description=(
                    "Detector/ray-grid horizontal centre representation of a static "
                    "COR-like offset under the detector-centre gauge."
                ),
            ),
            CalibrationVariable(
                name="det_v_px",
                value=float(det_v_px),
                unit="native_detector_px",
                status=det_v_status,  # type: ignore[arg-type]
                frame="detector",
                gauge="detector_ray_grid_center",
            ),
        )
    )


def calibrate_detector_center(
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: object,
    config: DetectorCenterCalibrationConfig | None = None,
    workdir: str | Path | None = None,
) -> DetectorCenterCalibrationResult:
    """Estimate static detector/ray-grid centre offsets with damped Gauss-Newton."""
    cfg = config or DetectorCenterCalibrationConfig()
    validate_calibration_gauges(_calibration_state_for_config(cfg, det_u_px=0.0, det_v_px=0.0))

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
    active_names = tuple(cfg.active_detector_dofs)
    fixed_det_u = float(cfg.initial_det_u_px)
    fixed_det_v = float(cfg.det_v_px)
    active_values = _active_values_from_offsets(
        active_names=active_names,
        det_u_px=fixed_det_u,
        det_v_px=fixed_det_v,
    )
    base_grid = get_detector_grid_device(detector)
    iterations: list[DetectorCenterIteration] = []

    def det_grid_for_values(values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        det_u, det_v = _det_u_v_from_active(
            values,
            active_names=active_names,
            fixed_det_u_px=fixed_det_u,
            fixed_det_v_px=fixed_det_v,
        )
        return _det_grid_from_offsets(base_grid, detector, det_u_px=det_u, det_v_px=det_v)

    for iteration_idx in range(1, int(cfg.outer_iters) + 1):
        def residual_for_values(values: jnp.ndarray) -> jnp.ndarray:
            det_grid = det_grid_for_values(values)
            # Detector-centre is an instrument geometry parameter, so the volume estimate
            # depends on the candidate detector grid. Differentiating only the held-out
            # reprojection with a fixed volume gives the wrong local direction.
            volume = _reconstruct_with_detector_center(
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
        max_step = jnp.asarray(float(cfg.max_step_px), dtype=jnp.float32)
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
        det_u, det_v = _det_u_v_from_active(
            active_values,
            active_names=active_names,
            fixed_det_u_px=fixed_det_u,
            fixed_det_v_px=fixed_det_v,
        )
        iterations.append(
            DetectorCenterIteration(
                iteration=iteration_idx,
                det_u_px=float(det_u),
                det_v_px=float(det_v),
                loss_before=float(loss_before),
                loss_after=float(best_loss),
                accepted=bool(best_scale > 0.0),
                raw_step_px=tuple(float(v) for v in np.asarray(raw_step)),
                applied_step_px=tuple(float(v) for v in np.asarray(best_step)),
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

    final_det_u, final_det_v = _det_u_v_from_active(
        active_values,
        active_names=active_names,
        fixed_det_u_px=fixed_det_u,
        fixed_det_v_px=fixed_det_v,
    )
    final_det_u_f = float(final_det_u)
    final_det_v_f = float(final_det_v)
    calibrated_detector = detector_with_center_offset(
        detector,
        det_u_px=final_det_u_f,
        det_v_px=final_det_v_f,
    )
    final_geometry = _geometry_from_inputs(
        geometry_inputs,
        grid=grid,
        detector=calibrated_detector,
        thetas_deg=thetas,
    )
    final_volume = np.asarray(
        fbp(
            final_geometry,
            grid,
            calibrated_detector,
            y,
            filter_name=str(cfg.filter_name),
            views_per_batch=int(cfg.views_per_batch),
            projector_unroll=int(cfg.projector_unroll),
            checkpoint_projector=bool(cfg.checkpoint_projector),
            gather_dtype=str(cfg.gather_dtype),
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
            final_det_u_px=final_det_u_f,
            final_det_v_px=final_det_v_f,
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
        curvature={"detector_center": confidence.get("curvature_min_eig")},
        contact_sheet=artifact_paths.get("contact_sheet"),
    )
    state = _calibration_state_for_config(
        cfg,
        det_u_px=final_det_u_f,
        det_v_px=final_det_v_f,
        confidence=confidence,
    )
    manifest = build_calibration_manifest(
        calibration_state=state,
        objective_card=objective,
        calibrated_geometry={
            "detector": calibrated_detector.to_dict(),
            "input_detector": detector.to_dict(),
            "gauge": "detector_ray_grid_center",
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
            "wording": (
                "det_u_px is the detector/ray-grid centre representation of a static "
                "COR-like offset under the detector-centre gauge."
            ),
        },
    )
    return DetectorCenterCalibrationResult(
        best_det_u_px=final_det_u_f,
        det_v_px=final_det_v_f,
        calibrated_detector=calibrated_detector,
        final_volume=final_volume,
        iterations=tuple(iterations),
        objective_card=objective,
        calibration_state=state,
        manifest=manifest,
        confidence=confidence,
        artifact_paths=artifact_paths,
    )
