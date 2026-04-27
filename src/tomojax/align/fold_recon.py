from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Geometry, Grid, RotationAxisGeometry
from tomojax.core.geometry.lamino import LaminographyGeometry
from tomojax.core.geometry.parallel import ParallelGeometry
from tomojax.recon.fista_tv import FistaConfig, fista_tv

from .geometry_applier import apply_setup_to_detector_grid, setup_axis_unit
from .state import AlignmentState


@dataclass(frozen=True, slots=True)
class FoldReconstructionConfig:
    iters: int
    lambda_tv: float
    regulariser: str
    huber_delta: float
    tv_prox_iters: int
    positivity: bool
    views_per_batch: int = 1
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    L: float | None = None


def reconstruct_train_fold_nograd(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    state: AlignmentState,
    train_idx: jnp.ndarray,
    train_mask: jnp.ndarray,
    init_x: jnp.ndarray | None,
    level_factor: int,
    cfg: FoldReconstructionConfig,
) -> tuple[jnp.ndarray, dict[str, object]]:
    """Reconstruct a train-fold volume with setup fixed and no setup AD path."""
    idx_np = np.asarray(train_idx, dtype=np.int32).reshape(-1)
    mask_np = np.asarray(train_mask, dtype=np.float32).reshape(-1)
    valid_idx = idx_np[mask_np > 0.0]
    if valid_idx.size == 0:
        raise ValueError("train fold must contain at least one active view")

    fold_geometry = _geometry_for_setup_subset(geometry, grid, detector, state, valid_idx)
    det_grid = apply_setup_to_detector_grid(
        detector,
        state.setup,
        level_factor=max(1, int(level_factor)),
    )
    y_fold = jnp.asarray(projections, dtype=jnp.float32)[jnp.asarray(valid_idx, dtype=jnp.int32)]
    x, info = fista_tv(
        fold_geometry,
        grid,
        detector,
        y_fold,
        init_x=init_x,
        config=FistaConfig(
            iters=max(1, int(cfg.iters)),
            lambda_tv=float(cfg.lambda_tv),
            regulariser=cfg.regulariser,
            huber_delta=float(cfg.huber_delta),
            tv_prox_iters=int(cfg.tv_prox_iters),
            L=cfg.L,
            views_per_batch=max(1, int(cfg.views_per_batch)),
            projector_unroll=int(cfg.projector_unroll),
            checkpoint_projector=bool(cfg.checkpoint_projector),
            gather_dtype=str(cfg.gather_dtype),
            grad_mode="stream",
            positivity=bool(cfg.positivity),
        ),
        det_grid=det_grid,
    )
    metadata = {
        "train_view_count": int(valid_idx.size),
        "train_indices": [int(v) for v in valid_idx],
        "recon_sensitivity": "stopped",
        "inner_recon_algo": "fista_tv",
        "inner_regulariser": str(cfg.regulariser),
        "inner_iters": int(cfg.iters),
        "views_per_batch": max(1, int(cfg.views_per_batch)),
        "info": info,
    }
    return x, metadata


def _geometry_for_setup_subset(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    state: AlignmentState,
    indices: Sequence[int],
) -> Geometry:
    idx = np.asarray(indices, dtype=np.int32)
    thetas = np.asarray(getattr(geometry, "thetas_deg"), dtype=np.float32)[idx]
    axis = tuple(float(v) for v in np.asarray(setup_axis_unit(state.setup), dtype=np.float32))
    axis_changed = (
        np.linalg.norm(np.asarray(axis, dtype=np.float32) - np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
        > 1e-7
    )
    if isinstance(geometry, RotationAxisGeometry) or axis_changed:
        return RotationAxisGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            axis_unit_lab=axis,
        )
    if isinstance(geometry, LaminographyGeometry):
        return LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            tilt_deg=float(geometry.tilt_deg),
            tilt_about=str(geometry.tilt_about),
        )
    return ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)
