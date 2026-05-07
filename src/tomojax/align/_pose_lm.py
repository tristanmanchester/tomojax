"""Pose-only LM/GN solver for the v2 reference path."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._lm_numerics import finite_difference_jacobian
from tomojax.forward import (
    nominal_axis_unit_from_geometry,
    project_parallel_reference_arrays,
    pseudo_huber_weights,
    residual_loss,
)
from tomojax.geometry import CanonicalizedGeometry, GeometryState, canonicalize_geometry_gauges


@dataclass(frozen=True)
class PoseOnlyLMConfig:
    max_iterations: int = 6
    damping: float = 1e-2
    sigma: float = 1.0
    delta: float = 1.0
    finite_difference_step: float = 1e-3
    active_pose_dofs: tuple[str, ...] = ("phi_residual_rad", "dx_px", "dz_px")


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
    """Solve supported per-view pose DOFs with damped Gauss-Newton/LM."""
    cfg = config or PoseOnlyLMConfig()
    vol = jnp.asarray(volume, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    setup_shift = jnp.asarray([geometry.setup.det_u_px.value, dz_setup], dtype=jnp.float32)
    _validate_active_pose_dofs(cfg.active_pose_dofs)
    params = _pack_pose(geometry, active_pose_dofs=cfg.active_pose_dofs)

    initial_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
    iterations = 0
    for _ in range(max(0, int(cfg.max_iterations))):
        residual = _residual_for_params(
            vol,
            obs,
            geometry,
            setup_shift,
            params,
            mask=mask,
            sigma=cfg.sigma,
            active_pose_dofs=cfg.active_pose_dofs,
        )
        weights = jnp.sqrt(pseudo_huber_weights(residual, delta=cfg.delta)).reshape(-1)

        def weighted_residual(
            candidate: jax.Array,
            weights_current: jax.Array = weights,
        ) -> jax.Array:
            raw = _residual_for_params(
                vol,
                obs,
                geometry,
                setup_shift,
                candidate,
                mask=mask,
                sigma=cfg.sigma,
                active_pose_dofs=cfg.active_pose_dofs,
            )
            return raw.reshape(-1) * weights_current

        r = weighted_residual(params)
        jacobian = finite_difference_jacobian(
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

    solved = _geometry_with_params(geometry, params, active_pose_dofs=cfg.active_pose_dofs)
    canonicalized = canonicalize_geometry_gauges(solved)
    final_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
    return PoseOnlyLMResult(
        geometry=solved,
        canonicalized_geometry=canonicalized,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        iterations=iterations,
        active_dofs=cfg.active_pose_dofs,
        frozen_dofs=tuple(dof for dof in _POSE_DOF_ORDER if dof not in cfg.active_pose_dofs),
    )


_POSE_DOF_ORDER = ("alpha_rad", "beta_rad", "phi_residual_rad", "dx_px", "dz_px")


def _pack_pose(geometry: GeometryState, *, active_pose_dofs: tuple[str, ...]) -> jax.Array:
    arrays = {
        "alpha_rad": jnp.asarray(geometry.pose.alpha_rad, dtype=jnp.float32),
        "beta_rad": jnp.asarray(geometry.pose.beta_rad, dtype=jnp.float32),
        "phi_residual_rad": jnp.asarray(geometry.pose.phi_residual_rad, dtype=jnp.float32),
        "dx_px": jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32),
        "dz_px": jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32),
    }
    return jnp.concatenate([arrays[dof] for dof in active_pose_dofs], axis=0)


def _geometry_with_params(
    geometry: GeometryState,
    params: jax.Array,
    *,
    active_pose_dofs: tuple[str, ...],
) -> GeometryState:
    n_views = geometry.pose.n_views
    updates = _split_params(params, n_views=n_views, active_pose_dofs=active_pose_dofs)
    return GeometryState(
        setup=geometry.setup,
        pose=geometry.pose.with_updates(
            alpha_rad=_as_float64_array(updates.get("alpha_rad")),
            beta_rad=_as_float64_array(updates.get("beta_rad")),
            phi_residual_rad=_as_float64_array(updates.get("phi_residual_rad")),
            dx_px=_as_float64_array(updates.get("dx_px")),
            dz_px=_as_float64_array(updates.get("dz_px")),
        ),
        acquisition=geometry.acquisition,
    )


def _split_params(
    params: jax.Array,
    *,
    n_views: int,
    active_pose_dofs: tuple[str, ...],
) -> dict[str, jax.Array]:
    matrix = params.reshape((len(active_pose_dofs), n_views))
    return {dof: matrix[index] for index, dof in enumerate(active_pose_dofs)}


def _validate_active_pose_dofs(active_pose_dofs: tuple[str, ...]) -> None:
    if len(set(active_pose_dofs)) != len(active_pose_dofs):
        raise ValueError("PoseOnlyLMConfig.active_pose_dofs must not contain duplicates")
    unknown = tuple(dof for dof in active_pose_dofs if dof not in _POSE_DOF_ORDER)
    if unknown:
        raise ValueError(f"unknown active pose DOFs {unknown!r}")


def _as_float64_array(value: jax.Array | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _residual_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    setup_shift: jax.Array,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    sigma: float,
    active_pose_dofs: tuple[str, ...],
) -> jax.Array:
    updates = _split_params(
        params,
        n_views=geometry.pose.n_views,
        active_pose_dofs=active_pose_dofs,
    )
    alpha_pose = updates.get(
        "alpha_rad",
        jnp.asarray(geometry.pose.alpha_rad, dtype=jnp.float32),
    )
    beta_pose = updates.get("beta_rad", jnp.asarray(geometry.pose.beta_rad, dtype=jnp.float32))
    phi_pose = updates.get(
        "phi_residual_rad",
        jnp.asarray(geometry.pose.phi_residual_rad, dtype=jnp.float32),
    )
    dx_pose = updates.get("dx_px", jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32))
    dz_pose = updates.get("dz_px", jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32))
    theta = (
        jnp.asarray(geometry.setup.theta_scale.value, dtype=jnp.float32)
        * jnp.asarray(geometry.pose.theta_nominal_rad, dtype=jnp.float32)
        + jnp.asarray(geometry.setup.theta_offset_rad.value, dtype=jnp.float32)
        + phi_pose
    )
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta,
        dx_px=setup_shift[0] + dx_pose,
        dz_px=setup_shift[1] + dz_pose,
        alpha_rad=alpha_pose,
        beta_rad=beta_pose,
        nominal_axis_unit=nominal_axis_unit_from_geometry(geometry),
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
    dz_setup = geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0
    updates = _split_params(
        params,
        n_views=geometry.pose.n_views,
        active_pose_dofs=cfg.active_pose_dofs,
    )
    alpha_pose = updates.get(
        "alpha_rad",
        jnp.asarray(geometry.pose.alpha_rad, dtype=jnp.float32),
    )
    beta_pose = updates.get("beta_rad", jnp.asarray(geometry.pose.beta_rad, dtype=jnp.float32))
    phi_pose = updates.get(
        "phi_residual_rad",
        jnp.asarray(geometry.pose.phi_residual_rad, dtype=jnp.float32),
    )
    dx_pose = updates.get("dx_px", jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32))
    dz_pose = updates.get("dz_px", jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32))
    theta = (
        jnp.asarray(geometry.setup.theta_scale.value, dtype=jnp.float32)
        * jnp.asarray(geometry.pose.theta_nominal_rad, dtype=jnp.float32)
        + jnp.asarray(geometry.setup.theta_offset_rad.value, dtype=jnp.float32)
        + phi_pose
    )
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta,
        dx_px=geometry.setup.det_u_px.value + dx_pose,
        dz_px=dz_setup + dz_pose,
        alpha_rad=alpha_pose,
        beta_rad=beta_pose,
        nominal_axis_unit=nominal_axis_unit_from_geometry(geometry),
    )
    return residual_loss(predicted, observed, mask=mask, sigma=cfg.sigma, delta=cfg.delta).loss
