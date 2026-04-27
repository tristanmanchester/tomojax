from __future__ import annotations

"""Compatibility facade for alignment configuration and execution."""

from typing import Iterable

import jax.numpy as jnp

from ..core.geometry.base import Detector, Geometry, Grid
from ..recon.fista_tv import fista_tv
from ..utils.fov import cylindrical_mask_xy
from ._config import (
    AlignConfig,
    _active_dof_mask_for_cfg,
    _active_dofs_for_cfg,
    _active_geometry_dofs_for_cfg,
    _resolved_schedule_for_cfg,
    _scoped_dofs_for_cfg,
)
from ._observer import (
    LegacyObserverCallback,
    ObserverAction,
    ObserverCallback,
    OuterStat,
    OuterStatValue,
    _normalize_observer_action,
    adapt_legacy_observer,
)
from . import _pose_stage as _pose_stage_mod
from . import _reconstruction_stage as _reconstruction_stage_mod
from . import _stage_loop as _stage_loop_mod
from ._pose_stage import (
    _evaluate_align_loss,
    _is_expected_align_eval_failure,
    _second_difference_gram,
    _select_gn_candidate,
    _should_prefer_gn_candidate,
    _smooth_gn_candidate,
)
from ._results import (
    AlignCheckpointCallback,
    AlignInfo,
    AlignMultiresCheckpointCallback,
    AlignMultiresInfo,
    AlignMultiresResumeState,
    AlignResumeState,
)
from ._stage_loop import MultiresLevel


def _sync_compatibility_overrides() -> None:
    """Forward monkeypatches applied to this facade into extracted owners."""
    _pose_stage_mod._evaluate_align_loss = _evaluate_align_loss
    _pose_stage_mod.cylindrical_mask_xy = cylindrical_mask_xy
    _reconstruction_stage_mod.fista_tv = fista_tv
    _stage_loop_mod.align = align


def align(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    cfg: AlignConfig | None = None,
    init_x: jnp.ndarray | None = None,
    init_params5: jnp.ndarray | None = None,
    observer: ObserverCallback | None = None,
    resume_state: AlignResumeState | None = None,
    checkpoint_callback: AlignCheckpointCallback | None = None,
    det_grid_override: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignInfo]:
    _sync_compatibility_overrides()
    return _pose_stage_mod.align(
        geometry,
        grid,
        detector,
        projections,
        cfg=cfg,
        init_x=init_x,
        init_params5=init_params5,
        observer=observer,
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback,
        det_grid_override=det_grid_override,
    )


def align_multires(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    factors: Iterable[int] = (2, 1),
    cfg: AlignConfig | None = None,
    observer: ObserverCallback | None = None,
    resume_state: AlignMultiresResumeState | None = None,
    checkpoint_callback: AlignMultiresCheckpointCallback | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignMultiresInfo]:
    _sync_compatibility_overrides()
    return _stage_loop_mod.align_multires(
        geometry,
        grid,
        detector,
        projections,
        factors=factors,
        cfg=cfg,
        observer=observer,
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback,
    )


def _build_alignment_volume_mask(
    grid: Grid,
    detector: Detector,
    *,
    mask_vol: str,
) -> jnp.ndarray | None:
    _sync_compatibility_overrides()
    return _pose_stage_mod._build_alignment_volume_mask(
        grid,
        detector,
        mask_vol=mask_vol,
    )

__all__ = [
    "AlignCheckpointCallback",
    "AlignConfig",
    "AlignInfo",
    "AlignMultiresCheckpointCallback",
    "AlignMultiresInfo",
    "AlignMultiresResumeState",
    "AlignResumeState",
    "LegacyObserverCallback",
    "MultiresLevel",
    "ObserverAction",
    "ObserverCallback",
    "OuterStat",
    "OuterStatValue",
    "adapt_legacy_observer",
    "align",
    "align_multires",
    "cylindrical_mask_xy",
    "fista_tv",
    "_active_dof_mask_for_cfg",
    "_active_dofs_for_cfg",
    "_active_geometry_dofs_for_cfg",
    "_build_alignment_volume_mask",
    "_evaluate_align_loss",
    "_is_expected_align_eval_failure",
    "_normalize_observer_action",
    "_resolved_schedule_for_cfg",
    "_scoped_dofs_for_cfg",
    "_second_difference_gram",
    "_select_gn_candidate",
    "_should_prefer_gn_candidate",
    "_smooth_gn_candidate",
]
