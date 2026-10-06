"""Public alignment configuration and execution entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.cone import beam_of
from tomojax.geometry import stack_view_poses

from ._config import AlignConfig
from ._gauge import least_motion_estimate
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

    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from ._geometry.parametrizations import PoseTranslationFrame


def align(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    config: AlignConfig | None = None,
    init_x: jnp.ndarray | None = None,
    init_pose_params: jnp.ndarray | None = None,
    observer: ObserverCallback | None = None,
    resume_state: AlignResumeState | None = None,
    checkpoint_callback: AlignCheckpointCallback | None = None,
    det_grid_override: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, AlignInfo]:
    """Run single-resolution alignment, returning its least-motion estimate.

    See :func:`align_multires`; ``init_x`` only starts the volume estimate.
    """
    x, params, info = _align_pose(
        geometry,
        grid,
        detector,
        projections,
        cfg=config,
        init_x=init_x,
        init_pose_params=init_pose_params,
        observer=observer,
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback,
        det_grid_override=det_grid_override,
    )
    if info["active_geometry_dofs"]:
        # Setup changes the nominal poses the gauge is measured against;
        # align_multires accounts for it.
        return x, params, info
    volume, moved, gauge = least_motion_estimate(
        np.asarray(x),
        np.asarray(params),
        nominal=np.asarray(stack_view_poses(geometry, int(params.shape[0]))),
        grid=grid,
        translation_frame=cast("PoseTranslationFrame", info["pose_translation_frame"]),
        active=info["active_pose_dofs"],
        beam=beam_of(geometry) is not None,
    )
    if gauge is not None:
        info["gauge"] = gauge.to_dict()
    return jnp.asarray(volume), jnp.asarray(moved), info


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
    """Run multiresolution alignment.

    Moving the object rigidly, and every pose with the inverse motion, predicts
    the same data, so the poses are only determined up to that motion. The
    result is the estimate with the least per-view motion: no common rotation
    and no rigid shift left in the poses (``info["gauge"]`` records what was
    removed). In a parallel beam with detector-frame poses, a schedule that
    reports the detector centre also takes the poses' constant u shift into it.
    """
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
