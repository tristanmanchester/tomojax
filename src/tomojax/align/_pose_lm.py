"""Pose-only LM/GN solver for the v2 reference path."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.forward import project_parallel_reference_arrays, pseudo_huber_weights, residual_loss
from tomojax.geometry import CanonicalizedGeometry, GeometryState, canonicalize_geometry_gauges

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class PoseOnlyLMConfig:
    max_iterations: int = 6
    damping: float = 1e-2
    sigma: float = 1.0
    delta: float = 1.0
    finite_difference_step: float = 1e-3


@dataclass(frozen=True)
class PoseOnlyLMResult:
    geometry: GeometryState
    canonicalized_geometry: CanonicalizedGeometry
    initial_loss: float
    final_loss: float
    iterations: int
    active_dofs: tuple[str, ...]
    frozen_dofs: tuple[str, ...]


def solve_pose_only_lm(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    *,
    mask: jax.Array | None = None,
    config: PoseOnlyLMConfig | None = None,
) -> PoseOnlyLMResult:
    """Solve per-view detector shifts with damped Gauss-Newton/LM."""
    cfg = config or PoseOnlyLMConfig()
    vol = jnp.asarray(volume, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    theta = jnp.asarray(geometry.setup.theta_offset_rad.value + geometry.pose.phi_residual_rad)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    setup_shift = jnp.asarray([geometry.setup.det_u_px.value, dz_setup], dtype=jnp.float32)
    params = _pack_pose(geometry)

    initial_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
    iterations = 0
    for _ in range(max(0, int(cfg.max_iterations))):
        residual = _residual_for_params(
            vol,
            obs,
            theta,
            setup_shift,
            params,
            mask=mask,
            sigma=cfg.sigma,
        )
        weights = jnp.sqrt(pseudo_huber_weights(residual, delta=cfg.delta)).reshape(-1)

        def weighted_residual(
            candidate: jax.Array,
            weights_current: jax.Array = weights,
        ) -> jax.Array:
            raw = _residual_for_params(
                vol,
                obs,
                theta,
                setup_shift,
                candidate,
                mask=mask,
                sigma=cfg.sigma,
            )
            return raw.reshape(-1) * weights_current

        r = weighted_residual(params)
        jacobian = _finite_difference_jacobian(
            weighted_residual,
            params,
            step_size=cfg.finite_difference_step,
        )
        lhs = jacobian.T @ jacobian + jnp.eye(params.shape[0], dtype=jnp.float32) * jnp.asarray(
            cfg.damping,
            dtype=jnp.float32,
        )
        rhs = -(jacobian.T @ r)
        step = jnp.linalg.solve(lhs, rhs)
        candidate = params + step
        candidate_loss = _loss_for_params(vol, obs, geometry, candidate, mask=mask, cfg=cfg)
        current_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
        params = jnp.where(candidate_loss <= current_loss, candidate, params)
        iterations += 1
        if float(jnp.linalg.norm(step)) < 1e-5:
            break

    solved = _geometry_with_params(geometry, params)
    canonicalized = canonicalize_geometry_gauges(solved)
    final_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
    return PoseOnlyLMResult(
        geometry=solved,
        canonicalized_geometry=canonicalized,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        iterations=iterations,
        active_dofs=("dx_px", "dz_px"),
        frozen_dofs=("alpha_rad", "beta_rad", "phi_residual_rad"),
    )


def _pack_pose(geometry: GeometryState) -> jax.Array:
    dx = jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32)
    dz = jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32)
    return jnp.concatenate([dx, dz], axis=0)


def _geometry_with_params(geometry: GeometryState, params: jax.Array) -> GeometryState:
    n_views = geometry.pose.n_views
    dx = np.asarray(params[:n_views], dtype=np.float64)
    dz = np.asarray(params[n_views:], dtype=np.float64)
    return GeometryState(setup=geometry.setup, pose=geometry.pose.with_updates(dx_px=dx, dz_px=dz))


def _finite_difference_jacobian(
    function: Callable[[jax.Array], jax.Array],
    params: jax.Array,
    *,
    step_size: float,
) -> jax.Array:
    eye = jnp.eye(params.shape[0], dtype=jnp.float32)

    def one_column(direction: jax.Array) -> jax.Array:
        step = jnp.asarray(step_size, dtype=jnp.float32) * direction
        return (function(params + step) - function(params - step)) / (
            2.0 * jnp.asarray(step_size, dtype=jnp.float32)
        )

    return jax.vmap(one_column)(eye).T


def _split_params(params: jax.Array) -> tuple[jax.Array, jax.Array]:
    half = params.shape[0] // 2
    return params[:half], params[half:]


def _residual_for_params(
    volume: jax.Array,
    observed: jax.Array,
    theta: jax.Array,
    setup_shift: jax.Array,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    sigma: float,
) -> jax.Array:
    dx_pose, dz_pose = _split_params(params)
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta,
        dx_px=setup_shift[0] + dx_pose,
        dz_px=setup_shift[1] + dz_pose,
    )
    residual = (predicted - observed) / jnp.asarray(sigma, dtype=jnp.float32)
    if mask is None:
        return residual
    return residual * jnp.asarray(mask, dtype=jnp.float32)


def _loss_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: PoseOnlyLMConfig,
) -> jax.Array:
    theta = jnp.asarray(geometry.setup.theta_offset_rad.value + geometry.pose.phi_residual_rad)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    dx_pose, dz_pose = _split_params(params)
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta,
        dx_px=geometry.setup.det_u_px.value + dx_pose,
        dz_px=dz_setup + dz_pose,
    )
    return residual_loss(predicted, observed, mask=mask, sigma=cfg.sigma, delta=cfg.delta).loss
