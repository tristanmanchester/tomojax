"""Probability-aware Schur step sizes in physical measurement coordinates.

Positive row and column sums bound every block without a spectral iteration.
The three scalar sizes follow diagonal primal-dual preconditioning, conservatively
collapsed to scalars so no sinogram or volume-sized preconditioner is retained.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.backend_policy import cuda_backend
from tomojax.core.projector import project_views, sum_backproject_views_T

from ._projection import chunk_operators
from ._tv_ops import gradient_norm_squared

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid

    from ._projection import ConeModel
    from ._tv_metric import TVMetric
    from .spdhg_tv import SPDHGConfig

_SPDHGProjectChunk = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
_SPDHGBackprojectChunk = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SPDHGStepSizes:
    tau: float
    sigma_data: float
    sigma_tv: float
    data_norm: float | None
    grad_norm: float


def make_spdhg_chunk_operators(
    grid: Grid,
    detector: Detector,
    poses: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    *,
    projector: tuple[str | ConeModel, str] | None,
    views_per_batch: int,
    ray_settings: tuple[bool, int, str, str],
    absolute_weights: bool = False,
) -> tuple[_SPDHGProjectChunk, _SPDHGBackprojectChunk]:
    """Slice full-scan coefficients along with poses, retaining each arrangement."""
    checkpoint, unroll, gather_dtype, ray_integrator = ray_settings
    if projector is not None:
        model, backend = projector
        return chunk_operators(poses, grid, detector, backend, views_per_batch, model)

    if absolute_weights and ray_integrator == "joseph_cubic":
        # Only the bound uses |A|. Preserve explicit detector coordinates and
        # use absolute interpolation on BOTH sides, not only the transpose.
        from tomojax.core.joseph import (
            forward_project_planes,
            plane_coefficients,
            sum_backproject_planes,
        )

        coefficients = plane_coefficients(poses, grid, detector, det_grid)
        backend = "cuda" if cuda_backend() else "jax"

        def project_absolute(vol, start, chunk):
            coeff = jax.lax.dynamic_slice_in_dim(coefficients, start, chunk.shape[0])
            return forward_project_planes(
                coeff,
                vol,
                grid,
                detector,
                backend=backend,
                interpolation="cubic",
                absolute_weights=True,
            )

        def backproject_absolute(images, start, chunk, total):
            coeff = jax.lax.dynamic_slice_in_dim(coefficients, start, chunk.shape[0])
            return sum_backproject_planes(
                coeff,
                images,
                grid,
                detector,
                backend=backend,
                interpolation="cubic",
                absolute_weights=True,
                accumulate=total,
            )

        return project_absolute, backproject_absolute

    def project_chunk(vol: jnp.ndarray, _start: jnp.ndarray, T_chunk: jnp.ndarray) -> jnp.ndarray:
        return project_views(
            T_chunk,
            grid,
            detector,
            vol,
            use_checkpoint=checkpoint,
            unroll=unroll,
            gather_dtype=gather_dtype,
            det_grid=det_grid,
            ray_integrator=ray_integrator,
        )

    def backproject_chunk(
        images: jnp.ndarray, _start: jnp.ndarray, T_chunk: jnp.ndarray, total: jnp.ndarray
    ) -> jnp.ndarray:
        return total + sum_backproject_views_T(
            T_chunk,
            grid,
            detector,
            images,
            unroll=unroll,
            gather_dtype=gather_dtype,
            det_grid=det_grid,
            ray_integrator=ray_integrator,
        )

    return project_chunk, backproject_chunk


@functools.partial(
    jax.jit,
    static_argnames=("grid", "detector", "projector", "views_per_batch", "ray_settings"),
)
def _block_row_column_bounds(
    poses: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    *,
    grid: Grid,
    detector: Detector,
    projector: tuple[str | ConeModel, str] | None,
    views_per_batch: int,
    ray_settings: tuple[bool, int, str, str],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = poses.shape[0]
    batch = min(views_per_batch, n)
    project, backproject = make_spdhg_chunk_operators(
        grid,
        detector,
        poses,
        det_grid,
        projector=projector,
        views_per_batch=batch,
        ray_settings=ray_settings,
        absolute_weights=True,
    )
    shape = (grid.nz, grid.ny, grid.nx)
    zero, one = jnp.zeros(shape, jnp.float32), jnp.ones(shape, jnp.float32)

    def bound(block, maxima):
        start = block * batch
        valid = jnp.minimum(batch, n - start)
        shifted = jnp.minimum(start, n - batch)
        chunk = jax.lax.dynamic_slice_in_dim(poses, shifted, batch)
        mask = jnp.broadcast_to(
            (jnp.arange(batch) >= batch - valid)[:, None, None],
            (batch, detector.nv, detector.nu),
        ).astype(jnp.float32)
        # Duplicate tail rows cannot enlarge the global row maximum; they MUST
        # be excluded from the column sum of the actual partial block.
        rows = project(one, shifted, chunk)
        columns = backproject(mask, shifted, chunk, zero)
        return jnp.maximum(maxima[0], jnp.max(rows)), jnp.maximum(maxima[1], jnp.max(columns))

    return jax.lax.fori_loop(0, (n + batch - 1) // batch, bound, (jnp.float32(0), jnp.float32(0)))


def resolve_spdhg_step_sizes(
    grid: Grid,
    detector: Detector,
    poses: jnp.ndarray,
    config: SPDHGConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    projector: tuple[str | ConeModel, str] | None = None,
    *,
    metric: TVMetric,
) -> SPDHGStepSizes:
    rho = 0.99
    c = metric.gradient_scale_xyz
    grad_norm = float(
        np.sqrt(gradient_norm_squared((grid.nz, grid.ny, grid.nx), gradient_scale_xyz=c))
    )
    grad_norm = grad_norm if config.tv_weight != 0 else 0.0
    length = metric.length_scale
    ratio = float(metric.weight_over_contrast)
    sigma_tv = rho * length * ratio / (2 * max(c))
    data_norm = None
    if config.tau is None or config.sigma_data is None:
        row, column = _block_row_column_bounds(
            poses,
            det_grid,
            grid=grid,
            detector=detector,
            projector=projector,
            views_per_batch=config.views_per_batch,
            ray_settings=(
                config.checkpoint_projector,
                config.projector_unroll,
                config.gather_dtype,
                config.ray_integrator,
            ),
        )
        row, column = float(row), float(column)
        batch = min(config.views_per_batch, poses.shape[0])
        blocks = (poses.shape[0] + batch - 1) // batch
        observed = float(metric.observed_count)
        mean_weight = float(metric.weight_sum) / observed if observed else 1.0
        # Schur: ||A_i||² <= R*B. Together with ||G||² <=4*max(c)*sum(c),
        # tau*(sigma_data/p*A_i.T*A_i + sigma_tv*G.T*G) <= rho² I.
        sigma_data = rho * length * mean_weight / (blocks * (row or length))
        denominator = length * (mean_weight * column + 2 * sum(c) * ratio)
        tau = rho / (denominator or length**2)
        data_norm = float(np.sqrt(blocks * row * column))
    else:
        tau, sigma_data = config.tau, config.sigma_data
    tau = tau if config.tau is None else float(config.tau)
    sigma_data = sigma_data if config.sigma_data is None else float(config.sigma_data)
    sigma_tv = sigma_tv if config.sigma_tv is None else float(config.sigma_tv)
    return SPDHGStepSizes(tau, sigma_data, sigma_tv, data_norm, grad_norm)
