from __future__ import annotations

"""Compatibility facade for alignment configuration and execution.

This module preserves the stable alignment entry points while the implementation
lives in private stage modules. Tests and internal callers that need stage
details should import those owner modules directly.
"""

from typing import Iterable

import jax.numpy as jnp

from ..core.geometry.base import Detector, Geometry, Grid
from ._config import AlignConfig
from ._observer import (
    LegacyObserverCallback,
    ObserverAction,
    ObserverCallback,
    OuterStat,
    OuterStatValue,
    adapt_legacy_observer,
)
from . import _pose_stage as _pose_stage_mod
from . import _stage_loop as _stage_loop_mod
from ._results import (
    AlignCheckpointCallback,
    AlignInfo,
    AlignMultiresCheckpointCallback,
    AlignMultiresInfo,
    AlignMultiresResumeState,
    AlignResumeState,
)
from ._stage_loop import MultiresLevel


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
]
