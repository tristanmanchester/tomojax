"""Setup-only LM/GN solver for the v2 reference path."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tomojax.align._lm_numerics import finite_difference_jacobian
from tomojax.forward import project_parallel_reference_arrays, pseudo_huber_weights, residual_loss
from tomojax.geometry import CanonicalizedGeometry, GeometryState, canonicalize_geometry_gauges


@dataclass(frozen=True)
class SetupOnlyLMConfig:
    max_iterations: int = 6
    damping: float = 1e-2
    sigma: float = 1.0
    delta: float = 1.0
    finite_difference_step: float = 1e-3


@dataclass(frozen=True)
class SetupOnlyLMResult:
    geometry: GeometryState
    canonicalized_geometry: CanonicalizedGeometry
    initial_loss: float
    final_loss: float
    iterations: int
    active_parameters: tuple[str, ...]
    frozen_parameters: tuple[str, ...]


def solve_setup_only_lm(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    *,
    mask: jax.Array | None = None,
    config: SetupOnlyLMConfig | None = None,
) -> SetupOnlyLMResult:
    """Solve supported setup parameters with damped GN/LM."""
    cfg = config or SetupOnlyLMConfig()
    vol = jnp.asarray(volume, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    phi_pose = jnp.asarray(geometry.pose.phi_residual_rad, dtype=jnp.float32)
    pose_dx = jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32)
    pose_dz = jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32)
    params = _pack_setup(geometry)
    active = _active_parameters(geometry)

    initial_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
    iterations = 0
    for _ in range(max(0, int(cfg.max_iterations))):
        residual = _residual_for_params(
            vol,
            obs,
            phi_pose,
            pose_dx,
            pose_dz,
            geometry,
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
                phi_pose,
                pose_dx,
                pose_dz,
                geometry,
                candidate,
                mask=mask,
                sigma=cfg.sigma,
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

    solved = _geometry_with_params(geometry, params)
    canonicalized = canonicalize_geometry_gauges(solved)
    final_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
    return SetupOnlyLMResult(
        geometry=solved,
        canonicalized_geometry=canonicalized,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        iterations=iterations,
        active_parameters=active,
        frozen_parameters=_frozen_parameters(active),
    )


def _active_parameters(geometry: GeometryState) -> tuple[str, ...]:
    if geometry.setup.det_v_px.active:
        return ("theta_offset_rad", "det_u_px", "det_v_px")
    return ("theta_offset_rad", "det_u_px")


def _frozen_parameters(active: tuple[str, ...]) -> tuple[str, ...]:
    all_supported = ("theta_offset_rad", "det_u_px", "det_v_px")
    unsupported = (
        "detector_roll_rad",
        "axis_rot_x_rad",
        "axis_rot_y_rad",
        "theta_scale",
    )
    return tuple(name for name in (*all_supported, *unsupported) if name not in active)


def _pack_setup(geometry: GeometryState) -> jax.Array:
    values = [geometry.setup.theta_offset_rad.value, geometry.setup.det_u_px.value]
    if geometry.setup.det_v_px.active:
        values.append(geometry.setup.det_v_px.value)
    return jnp.asarray(values, dtype=jnp.float32)


def _geometry_with_params(geometry: GeometryState, params: jax.Array) -> GeometryState:
    setup = geometry.setup.replace_parameter(
        "theta_offset_rad",
        geometry.setup.theta_offset_rad.with_value(float(params[0])),
    )
    setup = setup.replace_parameter(
        "det_u_px",
        geometry.setup.det_u_px.with_value(float(params[1])),
    )
    if geometry.setup.det_v_px.active:
        setup = setup.replace_parameter(
            "det_v_px",
            geometry.setup.det_v_px.with_value(float(params[2])),
        )
    return GeometryState(setup=setup, pose=geometry.pose)


def _setup_values(
    geometry: GeometryState, params: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    theta_offset = params[0]
    det_u = params[1]
    det_v = params[2] if geometry.setup.det_v_px.active else jnp.asarray(0.0, dtype=jnp.float32)
    return theta_offset, det_u, det_v


def _residual_for_params(
    volume: jax.Array,
    observed: jax.Array,
    phi_pose: jax.Array,
    pose_dx: jax.Array,
    pose_dz: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    sigma: float,
) -> jax.Array:
    theta_offset, det_u, det_v = _setup_values(geometry, params)
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta_offset + phi_pose,
        dx_px=det_u + pose_dx,
        dz_px=det_v + pose_dz,
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
    cfg: SetupOnlyLMConfig,
) -> jax.Array:
    theta_offset, det_u, det_v = _setup_values(geometry, params)
    theta = theta_offset + jnp.asarray(geometry.pose.phi_residual_rad, dtype=jnp.float32)
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta,
        dx_px=det_u + jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32),
        dz_px=det_v + jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32),
    )
    return residual_loss(predicted, observed, mask=mask, sigma=cfg.sigma, delta=cfg.delta).loss
