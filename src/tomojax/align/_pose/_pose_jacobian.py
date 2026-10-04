"""Shared physical pose Jacobian for fixed-volume and coupled GN."""

from __future__ import annotations

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp

from tomojax.align._geometry.parametrizations import apply_pose_update
from tomojax.core.projector import forward_project_view_T

from ._pose_context import _PoseObjectiveContext


def build_pose_prediction_and_columns(ctx: _PoseObjectiveContext) -> Callable:
    # Keep pose finite differences in physical units across anisotropic grids
    # and resolution levels. The angular step moves a bounding-sphere surface
    # by approximately the same distance as the translation step.
    displacement = ctx.cfg.gn_difference_step * min(ctx.grid.vx, ctx.grid.vy, ctx.grid.vz)
    radius = 0.5 * math.sqrt(
        (ctx.grid.nx * ctx.grid.vx) ** 2
        + (ctx.grid.ny * ctx.grid.vy) ** 2
        + (ctx.grid.nz * ctx.grid.vz) ** 2
    )
    difference_steps = jnp.asarray(
        [displacement / radius] * 3 + [displacement] * 2, dtype=jnp.float32
    )

    def _pred_flat(t_i: jnp.ndarray, masked_vol: jnp.ndarray) -> jnp.ndarray:
        return forward_project_view_T(
            t_i,
            ctx.grid,
            ctx.detector,
            masked_vol,
            use_checkpoint=ctx.cfg.checkpoint_projector,
            unroll=int(ctx.cfg.projector_unroll),
            gather_dtype=ctx.cfg.gather_dtype,
            det_grid=ctx.det_grid,
            ray_integrator=ctx.cfg.ray_integrator,
        ).ravel()

    def predict_and_columns(p5_i, t_nom_i, vol, target, weight):
        def f(p5):
            pose = apply_pose_update(t_nom_i, p5, translation_frame=ctx.cfg.pose_translation_frame)
            return weight.ravel() * (_pred_flat(pose, vol) - target.ravel())

        prediction = f(p5_i)
        directions = jnp.eye(5, dtype=jnp.float32)
        if ctx.cfg.gn_jacobian == "central":

            def column(direction, step):
                return (f(p5_i + step * direction) - f(p5_i - step * direction)) / (2 * step)

            columns = jax.vmap(column)(directions, difference_steps)
        else:
            columns = jax.vmap(lambda d: jax.jvp(f, (p5_i,), (d,))[1])(directions)
        return prediction, columns

    return predict_and_columns
