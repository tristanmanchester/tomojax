from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import csv
import json
import math
from typing import Mapping, Sequence

import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Geometry, Grid
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device
from tomojax.data.geometry_meta import build_geometry_from_meta
from tomojax.recon.fbp import fbp
from tomojax.recon.quicklook import extract_central_slice, scale_to_uint8

from ._json import JsonValue, normalize_json
from .detector_grid import offset_detector_grid
from .manifest import build_calibration_manifest
from .objectives import CandidateScore, MetricSpec, ObjectiveCard
from .state import CalibrationState, CalibrationVariable


DEFAULT_SEARCH_PASSES: tuple[tuple[float, float], ...] = (
    (10.0, 2.0),
    (2.0, 0.5),
    (0.5, 0.1),
)


@dataclass(frozen=True)
class DetectorCenterCalibrationConfig:
    """Configuration for one-dimensional detector/ray-grid centre calibration."""

    initial_det_u_px: float = 0.0
    det_v_px: float = 0.0
    det_v_status: str = "frozen"
    search_passes: tuple[tuple[float, float], ...] = DEFAULT_SEARCH_PASSES
    heldout_stride: int = 8
    top_k: int = 5
    filter_name: str = "ramp"
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "auto"

    def __post_init__(self) -> None:
        if self.det_v_status not in {"frozen", "supplied"}:
            raise ValueError("det_v_status must be 'frozen' or 'supplied'")
        if int(self.heldout_stride) < 2:
            raise ValueError("heldout_stride must be >= 2")
        if int(self.top_k) < 1:
            raise ValueError("top_k must be >= 1")
        if int(self.views_per_batch) < 1:
            raise ValueError("views_per_batch must be >= 1")
        passes = tuple((float(radius), float(step)) for radius, step in self.search_passes)
        if not passes:
            raise ValueError("search_passes must not be empty")
        for radius, step in passes:
            if not math.isfinite(radius) or radius < 0.0:
                raise ValueError(f"search pass radius must be finite and >= 0, got {radius!r}")
            if not math.isfinite(step) or step <= 0.0:
                raise ValueError(f"search pass step must be finite and > 0, got {step!r}")
        object.__setattr__(self, "search_passes", passes)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "initial_det_u_px": float(self.initial_det_u_px),
            "det_v_px": float(self.det_v_px),
            "det_v_status": str(self.det_v_status),
            "search_passes": [[float(r), float(s)] for r, s in self.search_passes],
            "heldout_stride": int(self.heldout_stride),
            "top_k": int(self.top_k),
            "filter_name": str(self.filter_name),
            "views_per_batch": int(self.views_per_batch),
            "projector_unroll": int(self.projector_unroll),
            "checkpoint_projector": bool(self.checkpoint_projector),
            "gather_dtype": str(self.gather_dtype),
        }


@dataclass(frozen=True)
class DetectorCenterCandidate:
    det_u_px: float
    det_v_px: float
    pass_index: int
    score: float
    validation_mode: str
    detector_center: tuple[float, float]
    rank: int | None = None
    artifact_path: str | None = None

    def ranked(self, rank: int) -> "DetectorCenterCandidate":
        return replace(self, rank=int(rank))

    def with_artifact(self, path: str) -> "DetectorCenterCandidate":
        return replace(self, artifact_path=str(path))

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "rank": self.rank,
            "pass_index": int(self.pass_index),
            "det_u_px": float(self.det_u_px),
            "det_v_px": float(self.det_v_px),
            "score": float(self.score),
            "validation_mode": self.validation_mode,
            "detector_center": [float(v) for v in self.detector_center],
        }
        if self.artifact_path is not None:
            payload["artifact_path"] = self.artifact_path
        return payload

    def to_candidate_score(self) -> CandidateScore:
        return CandidateScore(
            parameters={
                "det_u_px": float(self.det_u_px),
                "det_v_px": float(self.det_v_px),
                "detector_center": [float(v) for v in self.detector_center],
            },
            score=float(self.score),
            rank=self.rank,
            artifacts=(
                {"preview": self.artifact_path}
                if self.artifact_path is not None
                else None
            ),
        )


@dataclass(frozen=True)
class DetectorCenterCalibrationResult:
    best_det_u_px: float
    det_v_px: float
    calibrated_detector: Detector
    final_volume: np.ndarray
    candidates: tuple[DetectorCenterCandidate, ...]
    objective_card: ObjectiveCard
    calibration_state: CalibrationState
    manifest: dict[str, JsonValue]
    confidence: dict[str, JsonValue]
    artifact_paths: dict[str, str] = field(default_factory=dict)

    @property
    def best_candidate(self) -> DetectorCenterCandidate:
        return min(self.candidates, key=lambda candidate: candidate.score)


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


def candidate_values(center: float, radius: float, step: float) -> tuple[float, ...]:
    """Return sorted inclusive candidate values for one search pass."""
    radius = float(radius)
    step = float(step)
    if radius == 0.0:
        return (round(float(center), 10),)
    count = int(math.floor((2.0 * radius) / step + 1e-9)) + 1
    start = float(center) - radius
    values = [start + idx * step for idx in range(count)]
    end = float(center) + radius
    if not values or abs(values[-1] - end) > step * 1e-6:
        values.append(end)
    return tuple(sorted({round(float(value), 10) for value in values}))


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


def _candidate_det_grid(
    base_grid: tuple[jnp.ndarray, jnp.ndarray],
    detector: Detector,
    *,
    det_u_px: float,
    det_v_px: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return offset_detector_grid(
        base_grid,
        det_u_px=float(det_u_px),
        det_v_px=float(det_v_px),
        native_du=float(detector.du),
        native_dv=float(detector.dv),
    )


def _reconstruct_candidate(
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
    return fbp(
        geometry,
        grid,
        detector,
        projections[view_indices],
        filter_name=str(config.filter_name),
        views_per_batch=int(config.views_per_batch),
        projector_unroll=int(config.projector_unroll),
        checkpoint_projector=bool(config.checkpoint_projector),
        gather_dtype=str(config.gather_dtype),
        det_grid=det_grid,
    )


def _score_candidate(
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    thetas_deg: np.ndarray,
    train_indices: np.ndarray,
    score_indices: np.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    config: DetectorCenterCalibrationConfig,
) -> float:
    volume = _reconstruct_candidate(
        geometry_inputs,
        grid=grid,
        detector=detector,
        projections=projections,
        thetas_deg=thetas_deg,
        view_indices=train_indices,
        det_grid=det_grid,
        config=config,
    )
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
    denom = jnp.maximum(jnp.mean(measured.astype(jnp.float32) ** 2), jnp.float32(1e-6))
    return float(jnp.mean((preds - measured) ** 2) / denom)


def _rank_candidates(
    candidates: Sequence[DetectorCenterCandidate],
) -> tuple[DetectorCenterCandidate, ...]:
    ordered = sorted(candidates, key=lambda candidate: (candidate.score, candidate.det_u_px))
    return tuple(candidate.ranked(idx + 1) for idx, candidate in enumerate(ordered))


def _confidence_diagnostics(
    candidates: Sequence[DetectorCenterCandidate],
    *,
    final_pass_values: Sequence[float],
) -> dict[str, JsonValue]:
    ranked = _rank_candidates(candidates)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = None
    if second is not None:
        margin = (float(second.score) - float(best.score)) / max(abs(float(best.score)), 1e-12)
    values = tuple(float(v) for v in final_pass_values)
    best_on_boundary = bool(values and best.det_u_px in {min(values), max(values)})

    level = "low"
    if margin is not None and not best_on_boundary:
        if margin >= 0.10:
            level = "high"
        elif margin >= 0.03:
            level = "medium"

    curvature = None
    by_value = {float(candidate.det_u_px): float(candidate.score) for candidate in candidates}
    lower = sorted(v for v in by_value if v < best.det_u_px)
    upper = sorted(v for v in by_value if v > best.det_u_px)
    if lower and upper:
        left = lower[-1]
        right = upper[0]
        step = (right - left) / 2.0
        if step > 0:
            curvature = (by_value[left] - 2.0 * best.score + by_value[right]) / (step * step)

    return {
        "level": level,
        "score_margin_rel": margin,
        "best_on_boundary": best_on_boundary,
        "curvature": curvature,
    }


def _write_candidates_csv(path: Path, candidates: Sequence[DetectorCenterCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "rank",
        "pass_index",
        "det_u_px",
        "det_v_px",
        "score",
        "validation_mode",
        "detector_center_u",
        "detector_center_v",
        "artifact_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "rank": candidate.rank,
                    "pass_index": candidate.pass_index,
                    "det_u_px": candidate.det_u_px,
                    "det_v_px": candidate.det_v_px,
                    "score": candidate.score,
                    "validation_mode": candidate.validation_mode,
                    "detector_center_u": candidate.detector_center[0],
                    "detector_center_v": candidate.detector_center[1],
                    "artifact_path": candidate.artifact_path or "",
                }
            )


def _write_candidates_json(path: Path, candidates: Sequence[DetectorCenterCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([candidate.to_dict() for candidate in candidates], fh, indent=2, sort_keys=True)
        fh.write("\n")


def _candidate_preview(
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
    volume = _reconstruct_candidate(
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
    candidates: Sequence[DetectorCenterCandidate],
    config: DetectorCenterCalibrationConfig,
) -> tuple[tuple[DetectorCenterCandidate, ...], dict[str, str]]:
    workdir.mkdir(parents=True, exist_ok=True)
    ranked = _rank_candidates(candidates)
    top = ranked[: int(config.top_k)]
    base_grid = get_detector_grid_device(detector)
    preview_dir = workdir / "candidate_previews"
    images = []
    updated: list[DetectorCenterCandidate] = []

    nominal_grid = _candidate_det_grid(
        base_grid,
        detector,
        det_u_px=0.0,
        det_v_px=float(config.det_v_px),
    )
    images.append(
        _candidate_preview(
            geometry_inputs,
            grid=grid,
            detector=detector,
            projections=projections,
            thetas_deg=thetas_deg,
            det_grid=nominal_grid,
            config=config,
        )
    )

    for candidate in ranked:
        if candidate.rank is not None and candidate.rank <= int(config.top_k):
            det_grid = _candidate_det_grid(
                base_grid,
                detector,
                det_u_px=candidate.det_u_px,
                det_v_px=candidate.det_v_px,
            )
            image = _candidate_preview(
                geometry_inputs,
                grid=grid,
                detector=detector,
                projections=projections,
                thetas_deg=thetas_deg,
                det_grid=det_grid,
                config=config,
            )
            rel_path = preview_dir / f"rank_{candidate.rank:02d}_det_u_{candidate.det_u_px:+.3f}.png"
            rel_path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(rel_path, image)
            images.append(image)
            updated.append(candidate.with_artifact(str(rel_path)))
        else:
            updated.append(candidate)

    contact_sheet = workdir / "top_candidates.png"
    _write_contact_sheet(contact_sheet, images)
    candidates_csv = workdir / "candidates.csv"
    candidates_json = workdir / "candidates.json"
    updated_ranked = _rank_candidates(updated)
    _write_candidates_csv(candidates_csv, updated_ranked)
    _write_candidates_json(candidates_json, updated_ranked)
    return updated_ranked, {
        "contact_sheet": str(contact_sheet),
        "candidates_csv": str(candidates_csv),
        "candidates_json": str(candidates_json),
    }


def calibrate_detector_center(
    geometry_inputs: Mapping[str, object],
    *,
    grid: Grid,
    detector: Detector,
    projections: object,
    config: DetectorCenterCalibrationConfig | None = None,
    workdir: str | Path | None = None,
) -> DetectorCenterCalibrationResult:
    """Estimate static detector/ray-grid horizontal centre offset ``det_u_px``."""
    cfg = config or DetectorCenterCalibrationConfig()
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
    base_grid = get_detector_grid_device(detector)
    candidates: list[DetectorCenterCandidate] = []
    evaluated: set[float] = set()
    center = float(cfg.initial_det_u_px)
    final_pass_values: tuple[float, ...] = ()

    for pass_index, (radius, step) in enumerate(cfg.search_passes, start=1):
        values = candidate_values(center, radius, step)
        final_pass_values = values
        for det_u_px in values:
            key = round(float(det_u_px), 10)
            if key in evaluated:
                continue
            evaluated.add(key)
            det_grid = _candidate_det_grid(
                base_grid,
                detector,
                det_u_px=det_u_px,
                det_v_px=float(cfg.det_v_px),
            )
            score = _score_candidate(
                geometry_inputs,
                grid=grid,
                detector=detector,
                projections=y,
                thetas_deg=thetas,
                train_indices=train_indices,
                score_indices=score_indices,
                det_grid=det_grid,
                config=cfg,
            )
            calibrated = detector_with_center_offset(
                detector,
                det_u_px=det_u_px,
                det_v_px=float(cfg.det_v_px),
            )
            candidates.append(
                DetectorCenterCandidate(
                    det_u_px=float(det_u_px),
                    det_v_px=float(cfg.det_v_px),
                    pass_index=pass_index,
                    score=score,
                    validation_mode=validation_mode,
                    detector_center=tuple(float(v) for v in calibrated.det_center),
                )
            )
        center = min(candidates, key=lambda candidate: candidate.score).det_u_px

    ranked = _rank_candidates(candidates)
    artifact_paths: dict[str, str] = {}
    if workdir is not None:
        ranked, artifact_paths = _write_artifacts(
            Path(workdir),
            geometry_inputs,
            grid=grid,
            detector=detector,
            projections=y,
            thetas_deg=thetas,
            candidates=ranked,
            config=cfg,
        )

    best = ranked[0]
    calibrated_detector = detector_with_center_offset(
        detector,
        det_u_px=best.det_u_px,
        det_v_px=best.det_v_px,
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
    confidence = _confidence_diagnostics(ranked, final_pass_values=final_pass_values)
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
        top_candidates=tuple(candidate.to_candidate_score() for candidate in ranked[: cfg.top_k]),
        curvature={"det_u_px": confidence.get("curvature")},
        contact_sheet=artifact_paths.get("contact_sheet"),
    )
    state = CalibrationState(
        detector=(
            CalibrationVariable(
                name="det_u_px",
                value=float(best.det_u_px),
                unit="native_detector_px",
                status="estimated",
                frame="detector",
                gauge="detector_ray_grid_center",
                uncertainty=confidence,
                description=(
                    "Detector/ray-grid horizontal centre representation of a static "
                    "COR-like offset under the detector-centre gauge."
                ),
            ),
            CalibrationVariable(
                name="det_v_px",
                value=float(best.det_v_px),
                unit="native_detector_px",
                status=cfg.det_v_status,  # type: ignore[arg-type]
                frame="detector",
                gauge="detector_ray_grid_center",
            ),
        )
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
            "candidates": [candidate.to_dict() for candidate in ranked],
            "artifact_paths": artifact_paths,
            "wording": (
                "det_u_px is the detector/ray-grid centre representation of a static "
                "COR-like offset under the detector-centre gauge."
            ),
        },
    )
    return DetectorCenterCalibrationResult(
        best_det_u_px=float(best.det_u_px),
        det_v_px=float(best.det_v_px),
        calibrated_detector=calibrated_detector,
        final_volume=final_volume,
        candidates=tuple(ranked),
        objective_card=objective,
        calibration_state=state,
        manifest=manifest,
        confidence=confidence,
        artifact_paths=artifact_paths,
    )
