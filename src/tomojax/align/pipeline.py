"""Public alignment configuration and execution entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._config import AlignConfig
from ._observer import (
    ObserverAction,
    ObserverCallback,
    OuterStat,
    OuterStatValue,
)
from ._pose._pose_loop import align as _align_pose
from ._results import (
    AlignCheckpointCallback,
    AlignInfo,
    AlignMultiresCheckpointCallback,
    AlignMultiresInfo,
    AlignMultiresResumeState,
    AlignResumeState,
)
from ._stages._stage_multires import align_multires as _align_multires
from ._stages._stage_types import MultiresLevel

if TYPE_CHECKING:
    from collections.abc import Iterable

    import jax.numpy as jnp

    from tomojax.core.geometry.base import Detector, Geometry, Grid


def align(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    config: AlignConfig | None = None,
    init_x: jnp.ndarray | None = None,
    init_params5: jnp.ndarray | None = None,
    observer: ObserverCallback | None = None,
    resume_state: AlignResumeState | None = None,
    checkpoint_callback: AlignCheckpointCallback | None = None,
    det_grid_override: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignInfo]:
    """Run single-resolution alignment."""
    return _align_pose(
        geometry,
        grid,
        detector,
        projections,
        cfg=config,
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
    config: AlignConfig | None = None,
    observer: ObserverCallback | None = None,
    resume_state: AlignMultiresResumeState | None = None,
    checkpoint_callback: AlignMultiresCheckpointCallback | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignMultiresInfo]:
    """Run multiresolution alignment."""
    return _align_multires(
        geometry,
        grid,
        detector,
        projections,
        factors=factors,
        cfg=config,
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
    "MultiresLevel",
    "ObserverAction",
    "ObserverCallback",
    "OuterStat",
    "OuterStatValue",
    "align",
    "align_multires",
]
