from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Geometry, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.core.geometry.axis import RotationAxisGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import forward_project_view_T
from tomojax.recon.fista_tv import FistaConfig, fista_tv

from .geometry_blocks import GeometryCalibrationState, geometry_with_axis_state, level_detector_grid
from .losses import AlignmentLossSpec, LossAdapter, build_loss_adapter


@dataclass(frozen=True, slots=True)
class DetectorCenterSeed:
    """Projection-domain detector-u seed in native detector pixels."""

    det_u_px: float
    intercept_px: float
    amplitude_px: float
    status: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "det_u_px": float(self.det_u_px),
            "intercept_px": float(self.intercept_px),
            "amplitude_px": float(self.amplitude_px),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class HeldoutDetectorCenterResult:
    """Result of reduced held-out detector-centre scoring."""

    state: GeometryCalibrationState
    seed: DetectorCenterSeed
    candidate_values: tuple[float, ...]
    candidate_losses: tuple[float, ...]
    train_indices: tuple[int, ...]
    heldout_indices: tuple[int, ...]
    loss_kind: str
    objective: str
    status: str
    warnings: tuple[str, ...] = ()

    @property
    def estimate(self) -> float:
        return float(self.state.det_u_px)

    @property
    def best_loss(self) -> float:
        finite = [value for value in self.candidate_losses if math.isfinite(float(value))]
        return float(min(finite)) if finite else math.inf

    def to_metadata(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "loss_kind": self.loss_kind,
            "seed": self.seed.to_dict(),
            "estimate_det_u_px": self.estimate,
            "candidate_values": list(self.candidate_values),
            "candidate_losses": list(self.candidate_losses),
            "train_indices": list(self.train_indices),
            "heldout_indices": list(self.heldout_indices),
            "status": self.status,
            "warnings": list(self.warnings),
        }


def train_heldout_view_indices(
    n_views: int,
    *,
    holdout_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic interleaved train/held-out view indices."""
    n = int(n_views)
    if n < 4:
        raise ValueError("held-out detector-centre objective requires at least 4 views")
    stride = max(3, int(holdout_stride))
    all_indices = np.arange(n, dtype=np.int32)
    heldout = all_indices[(all_indices % stride) == (stride // 2)]
    if heldout.size == 0:
        heldout = all_indices[:: max(2, n // 4)]
    train_mask = np.ones((n,), dtype=bool)
    train_mask[heldout] = False
    train = all_indices[train_mask]
    if train.size == 0 or heldout.size == 0:
        raise ValueError("held-out detector-centre split produced an empty partition")
    return train, heldout


def projection_com_det_u_seed(
    projections: jnp.ndarray,
    geometry: Geometry,
    loss_adapter: LossAdapter,
) -> DetectorCenterSeed:
    """Estimate a cheap detector-u seed from projection centre-of-mass evidence.

    The constant term is treated only as a seed, not as the final calibration. Object
    centering and truncation can bias it, so the held-out objective always scores a
    wider candidate window around both the seed and the current value.
    """
    y = np.asarray(projections, dtype=np.float32)
    n_views, _nv, nu = y.shape
    u = np.arange(nu, dtype=np.float32) - (float(nu) - 1.0) * 0.5
    mask = getattr(loss_adapter.state, "mask", None)
    if mask is not None:
        weights = np.asarray(mask, dtype=np.float32)
    else:
        shifted = y - np.min(y, axis=(1, 2), keepdims=True)
        weights = np.maximum(shifted, 0.0)
    denom = np.sum(weights, axis=(1, 2))
    numerator = np.sum(weights * u[None, None, :], axis=(1, 2))
    valid = denom > 1e-6
    if int(np.count_nonzero(valid)) < 3:
        return DetectorCenterSeed(
            det_u_px=0.0,
            intercept_px=0.0,
            amplitude_px=0.0,
            status="insufficient_projection_mass",
        )
    com = np.zeros((n_views,), dtype=np.float32)
    com[valid] = numerator[valid] / denom[valid]
    theta = np.deg2rad(np.asarray(getattr(geometry, "thetas_deg"), dtype=np.float32))
    design = np.stack(
        [
            np.ones_like(theta[valid]),
            np.cos(theta[valid]),
            np.sin(theta[valid]),
        ],
        axis=1,
    )
    coeffs, *_ = np.linalg.lstsq(design, com[valid], rcond=None)
    intercept = float(coeffs[0])
    amplitude = float(np.hypot(coeffs[1], coeffs[2]))
    return DetectorCenterSeed(
        det_u_px=intercept,
        intercept_px=intercept,
        amplitude_px=amplitude,
        status="ok",
    )


def calibrate_detector_u_heldout(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    state: GeometryCalibrationState,
    factor: int,
    recon_iters: int,
    lambda_tv: float,
    regulariser: str,
    huber_delta: float,
    tv_prox_iters: int,
    views_per_batch: int,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
    loss_spec: AlignmentLossSpec,
    loss_adapter: LossAdapter,
    candidate_radius_px: float = 8.0,
    candidate_step_px: float = 1.0,
    holdout_stride: int = 4,
) -> HeldoutDetectorCenterResult:
    """Discover detector-u centre by reconstructing train views and scoring held-out views."""
    y = jnp.asarray(projections, dtype=jnp.float32)
    n_views = int(y.shape[0])
    train_idx, heldout_idx = train_heldout_view_indices(n_views, holdout_stride=holdout_stride)
    seed = projection_com_det_u_seed(y, geometry, loss_adapter)
    candidates = _candidate_values(
        current=float(state.det_u_px),
        seed=float(seed.det_u_px),
        radius=float(candidate_radius_px),
        step=float(candidate_step_px),
    )
    train_y = y[jnp.asarray(train_idx)]
    heldout_y = y[jnp.asarray(heldout_idx)]
    heldout_loss_adapter = build_loss_adapter(loss_spec, heldout_y)

    losses: list[float] = []
    for candidate_value in candidates:
        candidate_state = replace(state, det_u_px=float(candidate_value))
        train_geometry = _subset_geometry(
            geometry_with_axis_state(geometry, grid, detector, candidate_state),
            train_idx,
        )
        train_det_grid = level_detector_grid(detector, state=candidate_state, factor=factor)
        x_candidate, _ = fista_tv(
            train_geometry,
            grid,
            detector,
            train_y,
            init_x=None,
            config=FistaConfig(
                iters=int(recon_iters),
                lambda_tv=float(lambda_tv),
                regulariser=regulariser,  # type: ignore[arg-type]
                huber_delta=float(huber_delta),
                views_per_batch=max(1, int(views_per_batch)),
                projector_unroll=int(projector_unroll),
                checkpoint_projector=bool(checkpoint_projector),
                gather_dtype=str(gather_dtype),
                tv_prox_iters=int(tv_prox_iters),
            ),
            det_grid=train_det_grid,
        )
        heldout_geometry = _subset_geometry(
            geometry_with_axis_state(geometry, grid, detector, candidate_state),
            heldout_idx,
        )
        heldout_det_grid = level_detector_grid(detector, state=candidate_state, factor=factor)
        loss = _score_heldout(
            geometry=heldout_geometry,
            grid=grid,
            detector=detector,
            volume=x_candidate,
            targets=heldout_y,
            det_grid=heldout_det_grid,
            loss_adapter=heldout_loss_adapter,
            views_per_batch=max(1, int(views_per_batch)),
            projector_unroll=int(projector_unroll),
            checkpoint_projector=bool(checkpoint_projector),
            gather_dtype=str(gather_dtype),
        )
        losses.append(float(loss))

    loss_arr = np.asarray(losses, dtype=np.float64)
    finite = np.isfinite(loss_arr)
    warnings: list[str] = []
    if not np.any(finite):
        warnings.append("no_finite_heldout_candidates")
        estimate = float(state.det_u_px)
        status = "ill_conditioned"
    else:
        best_idx = int(np.nanargmin(np.where(finite, loss_arr, np.inf)))
        estimate = float(candidates[best_idx])
        status = "converged"
        if best_idx == 0 or best_idx == len(candidates) - 1:
            status = "underconverged"
            warnings.append("best_candidate_on_search_boundary")
        if _is_flat_or_weak(loss_arr):
            status = "ill_conditioned"
            warnings.append("flat_heldout_detector_center_objective")
    next_state = replace(state, det_u_px=estimate)
    return HeldoutDetectorCenterResult(
        state=next_state,
        seed=seed,
        candidate_values=tuple(float(value) for value in candidates),
        candidate_losses=tuple(float(value) for value in losses),
        train_indices=tuple(int(v) for v in train_idx),
        heldout_indices=tuple(int(v) for v in heldout_idx),
        loss_kind=heldout_loss_adapter.name,
        objective="heldout_reprojection",
        status=status,
        warnings=tuple(warnings),
    )


def _candidate_values(
    *,
    current: float,
    seed: float,
    radius: float,
    step: float,
) -> tuple[float, ...]:
    radius = max(float(radius), abs(float(seed) - float(current)) + 2.0)
    step = max(float(step), 0.25)
    lower = min(current, seed) - radius
    upper = max(current, seed) + radius
    values = np.arange(lower, upper + 0.5 * step, step, dtype=np.float32)
    values = np.concatenate([values, np.asarray([current, seed], dtype=np.float32)])
    values = np.unique(np.round(values.astype(np.float64), 4))
    return tuple(float(v) for v in values)


def _score_heldout(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    targets: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    loss_adapter: LossAdapter,
    views_per_batch: int,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
) -> float:
    n_views = int(targets.shape[0])
    poses = stack_view_poses(geometry, n_views)
    predictions = []
    for start in range(0, n_views, max(1, int(views_per_batch))):
        stop = min(start + max(1, int(views_per_batch)), n_views)
        pose_chunk = poses[start:stop]
        pred_chunk = jax.vmap(
            lambda T: forward_project_view_T(
                T,
                grid,
                detector,
                volume,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
            )
        )(pose_chunk)
        predictions.append(pred_chunk)
    pred = jnp.concatenate(predictions, axis=0)
    mask = getattr(loss_adapter.state, "mask", None)
    losses = loss_adapter.per_view_loss(
        pred,
        targets,
        mask,
        view_indices=jnp.arange(n_views, dtype=jnp.int32),
    )
    return float(jnp.sum(losses))


def _subset_geometry(geometry: Geometry, indices: Sequence[int]) -> Geometry:
    thetas = np.asarray(getattr(geometry, "thetas_deg"), dtype=np.float32)[np.asarray(indices)]
    if isinstance(geometry, LaminographyGeometry):
        return LaminographyGeometry(
            grid=geometry.grid,
            detector=geometry.detector,
            thetas_deg=thetas,
            tilt_deg=float(geometry.tilt_deg),
            tilt_about=str(geometry.tilt_about),
        )
    if isinstance(geometry, RotationAxisGeometry):
        return RotationAxisGeometry(
            grid=geometry.grid,
            detector=geometry.detector,
            thetas_deg=thetas,
            axis_unit_lab=geometry.axis_unit_lab,
        )
    if isinstance(geometry, ParallelGeometry):
        return ParallelGeometry(
            grid=geometry.grid,
            detector=geometry.detector,
            thetas_deg=thetas,
        )
    raise TypeError(f"Unsupported geometry type for view subset: {type(geometry).__name__}")


def _is_flat_or_weak(losses: np.ndarray) -> bool:
    finite = losses[np.isfinite(losses)]
    if finite.size < 3:
        return True
    spread = float(np.max(finite) - np.min(finite))
    scale = max(float(np.median(np.abs(finite))), 1e-12)
    return (spread / scale) < 1e-3
