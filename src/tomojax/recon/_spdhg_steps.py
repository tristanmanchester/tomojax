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

from ._projection import chunk_operators
from ._tv_ops import gradient_norm_squared
from ._volume_storage import volume_shape

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid

    from ._tv_metric import TVMetric
    from ._volume_storage import Model
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
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    *,
    projector: tuple[Model, str],
    views_per_batch: int,
) -> tuple[_SPDHGProjectChunk, _SPDHGBackprojectChunk]:
    """Slice full-scan coefficients along with poses, retaining each arrangement."""
    model, backend = projector
    return chunk_operators(
        poses, grid, detector, backend, views_per_batch, model, det_grid=det_grid
    )


@functools.partial(
    jax.jit,
    static_argnames=("grid", "detector", "projector", "views_per_batch"),
)
def _block_row_column_bounds(
    poses: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    *,
    grid: Grid,
    detector: Detector,
    projector: tuple[Model, str],
    views_per_batch: int,
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
    )
    shape = volume_shape(projector[0], grid)
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
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    projector: tuple[Model, str],
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
