"""Resolve SPDHG step sizes without discarding independently supplied overrides.

Norm estimation is needed only when a data-dependent step size is missing.
Sizes bound every stochastic block's squared norm divided by its selection
probability, which stays safe when the scan's information sits in one block.
Batched positive-kernel projectors need three Collatz passes per block;
reference rays use a one-pass absolute row-sum bound.
This module owns that policy; the iteration kernel consumes the resolved sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.operator_norm import estimate_normal_norm

from ._projection import chunk_operators, positive_normal_bound
from ._tv_ops import gradient_norm_squared

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from ._projection import ConeModel
    from .spdhg_tv import SPDHGConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SPDHGStepSizes:
    tau: float
    sigma_data_base: float
    sigma_data_eff: float
    sigma_tv: float
    data_norm: float | None
    grad_norm: float


def _estimate_norm_A2(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections_shape: tuple[int, int, int],
    T_all: jnp.ndarray,
    *,
    views_per_batch: int,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
    ray_integrator: str = "sampled",
    safety: float = 1.05,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> float:
    """Bound reference blocks by absolute row sums and their selection probability."""
    del geometry
    n_views = projections_shape[0]
    norm_squared = _reference_block_norm_squared(
        T_all,
        det_grid,
        grid=grid,
        detector=detector,
        batch_size=max(1, min(views_per_batch, n_views)),
        unroll=int(projector_unroll),
        checkpoint=checkpoint_projector,
        gather_dtype=gather_dtype,
        ray_integrator=ray_integrator,
    )
    return float(norm_squared) * float(safety**2)


@functools.partial(
    jax.jit,
    static_argnames=(
        "grid",
        "detector",
        "batch_size",
        "unroll",
        "checkpoint",
        "gather_dtype",
        "ray_integrator",
    ),
)
def _reference_block_norm_squared(
    poses: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    *,
    grid: Grid,
    detector: Detector,
    batch_size: int,
    unroll: int,
    checkpoint: bool,
    gather_dtype: str,
    ray_integrator: str,
) -> jnp.ndarray:
    blocks = (poses.shape[0] + batch_size - 1) // batch_size
    initial = jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    def bound(block: jnp.ndarray, largest: jnp.ndarray) -> jnp.ndarray:
        start = jnp.minimum(block * batch_size, poses.shape[0] - batch_size)
        chunk = jax.lax.dynamic_slice_in_dim(poses, start, batch_size)
        # A shifted short tail includes extra rays: its bound is conservative
        # for the actual tail, without changing any measured ray.
        norm = estimate_normal_norm(
            chunk,
            initial,
            det_grid,
            None,
            grid=grid,
            detector=detector,
            batch_size=batch_size,
            iterations=1,
            unroll=unroll,
            checkpoint=checkpoint,
            gather_dtype=gather_dtype,
            upper_bound=True,
            ray_integrator=ray_integrator,
        )
        return jnp.maximum(largest, norm)

    return blocks * jax.lax.fori_loop(0, blocks, bound, jnp.float32(0))


@functools.partial(
    jax.jit, static_argnames=("grid", "detector", "projector", "iterations", "views_per_batch")
)
def _batched_norm_squared(
    poses: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    projector: tuple[str | ConeModel, str],
    iterations: int,
    views_per_batch: int,
) -> jnp.ndarray:
    model, backend = projector
    n = int(poses.shape[0])
    batch = max(1, min(views_per_batch, n))
    blocks = (n + batch - 1) // batch
    project, backproject = chunk_operators(poses, grid, detector, backend, batch, model)
    shape = (grid.nx, grid.ny, grid.nz)

    def bound(block: jnp.ndarray, largest: jnp.ndarray) -> jnp.ndarray:
        start = block * batch
        valid = jnp.minimum(batch, n - start)
        shifted = jnp.minimum(start, n - batch)
        chunk = jax.lax.dynamic_slice_in_dim(poses, shifted, batch)
        mask = (jnp.arange(batch) >= batch - valid)[:, None, None]

        def forward(x: jnp.ndarray) -> jnp.ndarray:
            return project(x, shifted, chunk) * mask

        def adjoint(y: jnp.ndarray) -> jnp.ndarray:
            return backproject(y * mask, shifted, chunk, jnp.zeros(shape, jnp.float32))

        norm = positive_normal_bound(forward, adjoint, shape, iterations=iterations)
        return jnp.maximum(largest, norm)

    # For p_i=1/blocks, this bounds max_i ||A_i||²/p_i, and also ||A||².
    return blocks * jax.lax.fori_loop(0, blocks, bound, jnp.float32(0))


def resolve_spdhg_step_sizes(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data_shape: tuple[int, int, int],
    poses: jnp.ndarray,
    config: SPDHGConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    projector: tuple[str | ConeModel, str] | None = None,
) -> SPDHGStepSizes:
    grad_norm = (
        float(np.sqrt(gradient_norm_squared((grid.nx, grid.ny, grid.nz))))
        if config.tv_weight > 0
        else 0.0
    )
    grad_scale = grad_norm or 1.0
    rho = 0.99
    if config.tau is not None and config.sigma_data is not None:
        return SPDHGStepSizes(
            tau=float(config.tau),
            sigma_data_base=float(config.sigma_data),
            sigma_data_eff=float(config.sigma_data),
            sigma_tv=rho / grad_scale if config.sigma_tv is None else float(config.sigma_tv),
            data_norm=None,
            grad_norm=grad_norm,
        )
    if projector is not None:
        norm_sq = _batched_norm_squared(
            poses,
            grid=grid,
            detector=detector,
            projector=projector,
            iterations=3,
            views_per_batch=config.views_per_batch,
        )
        data_norm_sq = float(norm_sq) * 1.05**2
    else:
        data_norm_sq = _estimate_norm_A2(
            geometry,
            grid,
            detector,
            data_shape,
            poses,
            views_per_batch=max(1, config.views_per_batch),
            projector_unroll=config.projector_unroll,
            checkpoint_projector=config.checkpoint_projector,
            gather_dtype=config.gather_dtype,
            safety=1.05,
            det_grid=det_grid,
            ray_integrator=config.ray_integrator,
        )
    if not np.isfinite(data_norm_sq) or data_norm_sq < 0:
        raise ValueError("spdhg_tv: estimated projection norm must be finite and nonnegative")
    data_norm = float(np.sqrt(data_norm_sq))
    tau = rho / (data_norm + grad_norm or 1.0) if config.tau is None else float(config.tau)
    sigma_data_base = (
        rho / (data_norm or 1.0) if config.sigma_data is None else float(config.sigma_data)
    )
    sigma_tv = rho / grad_scale if config.sigma_tv is None else float(config.sigma_tv)
    return SPDHGStepSizes(
        tau=tau,
        sigma_data_base=sigma_data_base,
        sigma_data_eff=sigma_data_base,
        sigma_tv=sigma_tv,
        data_norm=data_norm,
        grad_norm=grad_norm,
    )
