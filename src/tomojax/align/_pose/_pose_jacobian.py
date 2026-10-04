"""Shared physical pose Jacobian for fixed-volume and coupled GN."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import math
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from tomojax.align._geometry.parametrizations import apply_pose_update
from tomojax.core.projector import forward_project_view_T

from ._pose_context import _PoseObjectiveContext

if TYPE_CHECKING:
    from tomojax.align._config import AlignConfig
    from tomojax.align._geometry.parametrizations import PoseTranslationFrame
    from tomojax.geometry import Detector, Grid


@dataclass(frozen=True)
class PoseJacobianOptions:
    """Hashable projector choices, separate from a scan's changing arrays."""

    difference_step: float
    translation_frame: PoseTranslationFrame
    checkpoint: bool
    unroll: int
    gather_dtype: str
    integrator: Literal["sampled", "exact"]
    jacobian: Literal["central", "autodiff"]

    @classmethod
    def from_config(cls, cfg: AlignConfig) -> PoseJacobianOptions:
        return cls(
            float(cfg.gn_difference_step),
            cfg.pose_translation_frame,
            bool(cfg.checkpoint_projector),
            int(cfg.projector_unroll),
            cfg.gather_dtype,
            cfg.ray_integrator,
            cfg.gn_jacobian,
        )


def build_pose_prediction_and_columns(ctx: _PoseObjectiveContext) -> Callable:
    return partial(
        pose_prediction_and_columns,
        grid=ctx.grid,
        detector=ctx.detector,
        det_grid=ctx.det_grid,
        options=PoseJacobianOptions.from_config(ctx.cfg),
    )


def pose_prediction_and_columns(
    p5_i,
    t_nom_i,
    vol,
    target,
    weight,
    *,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jax.Array, jax.Array] | None,
    options: PoseJacobianOptions,
):
    """Evaluate weighted columns with all scan-dependent arrays as arguments."""
    # Keep pose finite differences in physical units across anisotropic grids
    # and resolution levels. The angular step moves a bounding-sphere surface
    # by approximately the same distance as the translation step.
    displacement = options.difference_step * min(grid.vx, grid.vy, grid.vz)
    radius = 0.5 * math.sqrt(
        (grid.nx * grid.vx) ** 2 + (grid.ny * grid.vy) ** 2 + (grid.nz * grid.vz) ** 2
    )
    difference_steps = jnp.asarray(
        [displacement / radius] * 3 + [displacement] * 2, dtype=jnp.float32
    )

    def _pred_flat(t_i: jnp.ndarray, masked_vol: jnp.ndarray) -> jnp.ndarray:
        return forward_project_view_T(
            t_i,
            grid,
            detector,
            masked_vol,
            use_checkpoint=options.checkpoint,
            unroll=options.unroll,
            gather_dtype=options.gather_dtype,
            det_grid=det_grid,
            ray_integrator=options.integrator,
        ).ravel()

    def f(p5):
        pose = apply_pose_update(t_nom_i, p5, translation_frame=options.translation_frame)
        return weight.ravel() * (_pred_flat(pose, vol) - target.ravel())

    prediction = f(p5_i)
    directions = jnp.eye(5, dtype=jnp.float32)
    if options.jacobian == "central":

        def column(direction, step):
            return (f(p5_i + step * direction) - f(p5_i - step * direction)) / (2 * step)

        columns = jax.vmap(column)(directions, difference_steps)
    else:
        columns = jax.vmap(lambda d: jax.jvp(f, (p5_i,), (d,))[1])(directions)
    return prediction, columns
