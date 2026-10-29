"""Shared physical pose Jacobian for fixed-volume and coupled GN."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from tomojax.alignment._geometry.parametrizations import apply_pose_update
from tomojax.core.backend_policy import cuda_backend
from tomojax.core.cone import cone_pose_columns, cone_project, frame_coefficients, use_cuda_cone
from tomojax.core.joseph import plane_coefficients, plane_pose_columns
from tomojax.core.projector import forward_project_view_T

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
    cone_beam: bool = False

    @classmethod
    def from_config(cls, cfg: AlignConfig) -> PoseJacobianOptions:
        return cls(
            float(cfg.gn_difference_step),
            cfg.pose_translation_frame,
            bool(cfg.checkpoint_projector),
            1,
            cfg.gather_dtype,
            cfg.ray_integrator,
            cfg.gn_jacobian,
        )


def _cone_autodiff_columns(
    params, nominal, frame, volume, target, weight, indices, *, grid, detector, translation_frame
):
    """Analytic columns of the same piecewise-smooth cone Joseph model."""

    def coefficients(p):
        pose = apply_pose_update(nominal, p, translation_frame=translation_frame)
        return frame_coefficients(pose[None], frame, grid, detector)[0, :12]

    # Forward rays do not need the adjoint's detector-plane coefficients or
    # separability flag. Padding only the ray geometry lets JAX drop that work.
    coeff = jnp.pad(coefficients(params), (0, 13))
    tangent = jax.jacfwd(coefficients)(params).T[indices]
    images = cone_pose_columns(
        volume, coeff[None], tangent=tangent[None], grid=grid, detector=detector
    )[0]
    images = images.reshape(1 + len(indices), -1)
    return weight.ravel() * (images[0] - target.ravel()), weight.ravel() * images[1:]


def _cone_central_columns(
    params,
    nominal,
    frame,
    volume,
    target,
    weight,
    steps,
    indices,
    *,
    grid,
    detector,
    translation_frame,
):
    """The same central differences, with all perturbations in one CUDA launch."""
    width = len(indices)
    offsets = jnp.eye(params.shape[-1], dtype=jnp.float32)[indices] * steps[:, None]

    def coefficients(p):
        pose = apply_pose_update(nominal, p, translation_frame=translation_frame)
        return frame_coefficients(pose[None], frame, grid, detector)

    # Keep the legacy (direction, single-view) geometry layout. Combining pose
    # rows before their matrix products can alter FP32 rounding, amplified by
    # the small finite-difference steps on large grids. Batch only the launch.
    coeff = jnp.concatenate(
        [
            coefficients(params),
            jax.vmap(coefficients)(params[None] + offsets).reshape(width, -1),
            jax.vmap(coefficients)(params[None] - offsets).reshape(width, -1),
        ]
    )
    images = cone_project(volume, coeff, grid, detector, backend="cuda").reshape(1 + 2 * width, -1)
    residual = weight.ravel() * (images[0] - target.ravel())
    # The target is constant in pose: subtract predictions first, so a large
    # target cannot round away their difference. Weights are fixed here too.
    columns = weight.ravel() * (images[1 : width + 1] - images[width + 1 :]) / (2 * steps[:, None])
    return residual, columns


def _difference_steps(grid: Grid, fraction: float, width: int) -> jax.Array:
    """Translation steps in physical units, with rotations moving the bounding sphere as far."""
    displacement = fraction * min(grid.vx, grid.vy, grid.vz)
    radius = 0.5 * math.sqrt(
        (grid.nx * grid.vx) ** 2 + (grid.ny * grid.vy) ** 2 + (grid.nz * grid.vz) ** 2
    )
    return jnp.asarray([displacement / radius] * 3 + [displacement] * (width - 3), jnp.float32)


def _parallel_autodiff_columns(
    params, nominal, volume, target, weight, indices, *, grid, detector, det_grid, options
):
    """Share a physical ray traversal among the active coefficient tangents.

    Mapping the complete projector's JVP launches a separate ray traversal for
    every pose coordinate. Only the small coefficient map needs that transform;
    the ray kernel can accumulate all its directions from each voxel sample.
    """

    def coefficients(p):
        pose = apply_pose_update(nominal, p, translation_frame=options.translation_frame)
        return plane_coefficients(pose[None], grid, detector, det_grid)[0]

    coeff = coefficients(params)
    directions = jnp.eye(params.size, dtype=params.dtype)[indices]
    tangent = jax.vmap(lambda d: jax.jvp(coefficients, (params,), (d,))[1])(directions)

    images = plane_pose_columns(
        coeff[None],
        volume,
        tangent[None],
        grid,
        detector,
        backend="cuda",
        interpolation="cubic" if options.integrator == "joseph_cubic" else "linear",
    )[0].reshape(1 + len(indices), -1)
    return weight.ravel() * (images[0] - target.ravel()), weight.ravel() * images[1:]


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
    column_indices: tuple[int, ...] | None = None,
):
    """Evaluate weighted columns with all scan-dependent arrays as arguments.

    ``frame_i`` is the view's cone-beam lab frame, or None for a parallel beam.
    ``column_indices`` selects the requested parameter derivatives, in that order.
    """
    width = int(p5_i.shape[-1])
    indices = jnp.asarray(
        tuple(range(width)) if column_indices is None else column_indices, jnp.int32
    )
    difference_steps = _difference_steps(grid, options.difference_step, width)[indices]

    joseph = options.integrator.startswith("joseph")
    central = options.jacobian == "central"
    if joseph and not central and not options.cone_beam and cuda_backend():
        return _parallel_autodiff_columns(
            p5_i,
            t_nom_i,
            vol,
            target,
            weight,
            indices,
            grid=grid,
            detector=detector,
            det_grid=det_grid,
            options=options,
        )
    cone_backend = "jax" if options.cone_beam and not central else "cuda"
    if options.cone_beam and use_cuda_cone():
        if not central:
            return _cone_autodiff_columns(
                p5_i,
                t_nom_i,
                frame_i,
                vol,
                target,
                weight,
                indices,
                grid=grid,
                detector=detector,
                translation_frame=options.translation_frame,
            )
        return _cone_central_columns(
            p5_i,
            t_nom_i,
            frame_i,
            vol,
            target,
            weight,
            difference_steps,
            indices,
            grid=grid,
            detector=detector,
            translation_frame=options.translation_frame,
        )

    def _pred_flat(t_i: jnp.ndarray, masked_vol: jnp.ndarray) -> jnp.ndarray:
        if options.cone_beam:
            return forward_project_view_T(
                t_i, grid, detector, masked_vol, backend=cone_backend, frames=frame_i
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
            backend="cuda" if joseph and cuda_backend() else "jax",
            ray_integrator=options.integrator,
        ).ravel()

    def predict(p5):
        pose = apply_pose_update(t_nom_i, p5, translation_frame=options.translation_frame)
        return _pred_flat(pose, vol)

    return _reference_columns(
        predict, p5_i, target, weight, indices, difference_steps, central=central
    )


def _reference_columns(predict, params, target, weight, indices, steps, *, central):
    """Differentiate the physical prediction, weighting after finite differences."""

    def residual(p):
        return weight.ravel() * (predict(p) - target.ravel())

    prediction = residual(params)
    directions = jnp.eye(params.shape[-1], dtype=jnp.float32)[indices]
    if central:

        def column(direction, step):
            return (
                weight.ravel()
                * (predict(params + step * direction) - predict(params - step * direction))
                / (2 * step)
            )

        columns = jax.vmap(column)(directions, steps)
    else:
        columns = jax.vmap(lambda d: jax.jvp(residual, (params,), (d,))[1])(directions)
    return prediction, columns
