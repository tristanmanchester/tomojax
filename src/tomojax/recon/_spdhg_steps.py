"""Resolve SPDHG step sizes without discarding independently supplied overrides.

Norm estimation is needed only when a data-dependent step size is missing.
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

from ._projection import normal_operator_norm, projection_operators

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Geometry, Grid

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
    key: jax.Array | None = None,
    power_iterations: int = 20,
    safety: float = 1.05,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> float:
    """Estimate the squared projection-operator norm by power iteration."""
    del geometry
    n_views = projections_shape[0]
    if key is None:
        key = jax.random.key(0)
    initial = jax.random.normal(key, (grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    norm_squared = estimate_normal_norm(
        T_all,
        initial,
        det_grid,
        None,
        grid=grid,
        detector=detector,
        batch_size=max(1, min(views_per_batch, n_views)),
        iterations=int(power_iterations),
        unroll=int(projector_unroll),
        checkpoint=checkpoint_projector,
        gather_dtype=gather_dtype,
        ray_integrator=ray_integrator,
    )
    return max(float(norm_squared) * float(safety**2), 1e-6)


@functools.partial(jax.jit, static_argnames=("grid", "detector", "projector", "iterations"))
def _batched_norm_squared(
    poses: jnp.ndarray,
    *,
    grid: Grid,
    detector: Detector,
    projector: tuple[str, str],
    iterations: int,
) -> jnp.ndarray:
    model, backend = projector
    batch = min(64, int(poses.shape[0]))
    forward, adjoint = projection_operators(poses, grid, detector, None, backend, batch, model)
    shape = (grid.nx, grid.ny, grid.nz)
    return normal_operator_norm(forward, adjoint, shape, iterations=iterations)


def resolve_spdhg_step_sizes(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data_shape: tuple[int, int, int],
    poses: jnp.ndarray,
    config: SPDHGConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    projector: tuple[str, str] | None = None,
) -> SPDHGStepSizes:
    grad_norm = float(np.sqrt(12.0))
    rho = 0.99
    if config.tau is not None and config.sigma_data is not None:
        return SPDHGStepSizes(
            tau=float(config.tau),
            sigma_data_base=float(config.sigma_data),
            sigma_data_eff=float(config.sigma_data),
            sigma_tv=rho / grad_norm if config.sigma_tv is None else float(config.sigma_tv),
            data_norm=None,
            grad_norm=grad_norm,
        )
    if projector is not None:
        norm_sq = _batched_norm_squared(
            poses, grid=grid, detector=detector, projector=projector, iterations=20
        )
        data_norm_sq = max(float(norm_sq) * 1.05**2, 1e-6)
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
            key=jax.random.key(config.seed),
            power_iterations=20,
            safety=1.05,
            det_grid=det_grid,
            ray_integrator=config.ray_integrator,
        )
    data_norm = float(np.sqrt(data_norm_sq))
    tau = rho / (data_norm + grad_norm) if config.tau is None else float(config.tau)
    sigma_data_base = (
        rho / max(data_norm, 1e-6) if config.sigma_data is None else float(config.sigma_data)
    )
    sigma_tv = rho / grad_norm if config.sigma_tv is None else float(config.sigma_tv)
    return SPDHGStepSizes(
        tau=tau,
        sigma_data_base=sigma_data_base,
        sigma_data_eff=sigma_data_base,
        sigma_tv=sigma_tv,
        data_norm=data_norm,
        grad_norm=grad_norm,
    )
