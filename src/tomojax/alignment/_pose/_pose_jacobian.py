"""Shared physical pose Jacobian for fixed-volume and coupled GN."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
import math
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from tomojax.alignment._geometry.parametrizations import apply_pose_update
from tomojax.core.projector import forward_project_view_T

from ._pose_context import _PoseObjectiveContext

if TYPE_CHECKING:
    from tomojax.alignment._config import AlignConfig
    from tomojax.alignment._geometry.parametrizations import PoseTranslationFrame
    from tomojax.geometry import Detector, Grid


@dataclass(frozen=True)
class PoseJacobianOptions:
    """Hashable projector choices, separate from a scan's changing arrays."""

    difference_step: float
    translation_frame: PoseTranslationFrame
    checkpoint: bool
    unroll: int
    gather_dtype: str
    integrator: Literal["sampled", "exact", "joseph", "joseph_cubic"]
    jacobian: Literal["central", "autodiff"]
    # Cone beams project each view in its lab frame (``frame_i`` below).
    cone: bool = False

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


def _cuda() -> bool:
    version = jax.devices()[0].client.platform_version.lower()
    return jax.default_backend() == "gpu" and "cuda" in version


def build_pose_prediction_and_columns(ctx: _PoseObjectiveContext) -> Callable:
    return partial(
        pose_prediction_and_columns,
        grid=ctx.grid,
        detector=ctx.detector,
        det_grid=ctx.det_grid,
        options=replace(PoseJacobianOptions.from_config(ctx.cfg), cone=ctx.cone is not None),
    )


def pose_prediction_and_columns(
    p5_i,
    t_nom_i,
    frame_i,
    vol,
    target,
    weight,
    *,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jax.Array, jax.Array] | None,
    options: PoseJacobianOptions,
):
    """Evaluate weighted columns with all scan-dependent arrays as arguments.

    ``frame_i`` is the view's cone-beam lab frame, or None for a parallel beam.
    """
    # Keep pose finite differences in physical units across anisotropic grids
    # and resolution levels. The angular step moves a bounding-sphere surface
    # by approximately the same distance as the translation step.
    displacement = options.difference_step * min(grid.vx, grid.vy, grid.vz)
    radius = 0.5 * math.sqrt(
        (grid.nx * grid.vx) ** 2 + (grid.ny * grid.vy) ** 2 + (grid.nz * grid.vz) ** 2
    )
    width = int(p5_i.shape[-1])
    difference_steps = jnp.asarray(
        [displacement / radius] * 3 + [displacement] * (width - 3), dtype=jnp.float32
    )

    joseph = options.integrator.startswith("joseph")
    # Cone views use central differences of the CUDA forward, which has no pose
    # derivative; autodiff columns use the differentiable JAX reference.
    central = options.jacobian == "central" or (options.cone and _cuda())
    cone_backend = "jax" if options.cone and not central else "pallas"

    def _pred_flat(t_i: jnp.ndarray, masked_vol: jnp.ndarray) -> jnp.ndarray:
        if options.cone:
            return forward_project_view_T(
                t_i, grid, detector, masked_vol, projector_backend=cone_backend, frames=frame_i
            ).ravel()
        return forward_project_view_T(
            t_i,
            grid,
            detector,
            masked_vol,
            use_checkpoint=options.checkpoint,
            unroll=options.unroll,
            gather_dtype=options.gather_dtype,
            det_grid=det_grid,
            # Joseph CUDA kernels' pose derivatives avoid a per-plane JAX tape.
            projector_backend="pallas" if joseph and _cuda() else "jax",
            ray_integrator=options.integrator,
        ).ravel()

    def f(p5):
        pose = apply_pose_update(t_nom_i, p5, translation_frame=options.translation_frame)
        return weight.ravel() * (_pred_flat(pose, vol) - target.ravel())

    prediction = f(p5_i)
    directions = jnp.eye(width, dtype=jnp.float32)
    if central:

        def column(direction, step):
            return (f(p5_i + step * direction) - f(p5_i - step * direction)) / (2 * step)

        columns = jax.vmap(column)(directions, difference_steps)
    else:
        columns = jax.vmap(lambda d: jax.jvp(f, (p5_i,), (d,))[1])(directions)
    return prediction, columns
