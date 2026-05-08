"""Setup-only LM/GN solver for the v2 reference path."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from tomojax.align._lm_numerics import finite_difference_jacobian
from tomojax.forward import (
    nominal_axis_unit_from_geometry,
    project_parallel_reference_arrays,
    pseudo_huber_weights,
    residual_loss,
)
from tomojax.geometry import CanonicalizedGeometry, GeometryState, canonicalize_geometry_gauges

SetupLMParameter = Literal[
    "axis_rot_x_rad",
    "axis_rot_y_rad",
    "theta_offset_rad",
    "det_u_px",
    "det_v_px",
    "detector_roll_rad",
    "theta_scale",
]


@dataclass(frozen=True)
class SetupOnlyLMConfig:
    max_iterations: int = 6
    damping: float = 1e-2
    sigma: float = 1.0
    delta: float = 1.0
    loss_mode: str = "pseudo_huber"
    finite_difference_step: float = 1e-3
    active_parameters: tuple[SetupLMParameter, ...] | None = None


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
    theta_nominal = jnp.asarray(geometry.pose.theta_nominal_rad, dtype=jnp.float32)
    phi_pose = jnp.asarray(geometry.pose.phi_residual_rad, dtype=jnp.float32)
    pose_dx = jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32)
    pose_dz = jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32)
    params = _pack_setup(geometry, cfg)
    active = _active_parameters(geometry, cfg)

    initial_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
    iterations = 0
    for _ in range(max(0, int(cfg.max_iterations))):
        residual = _residual_for_params(
            vol,
            obs,
            theta_nominal,
            phi_pose,
            pose_dx,
            pose_dz,
            geometry,
            params,
            mask=mask,
            sigma=cfg.sigma,
            config=cfg,
        )
        weights = jnp.sqrt(_residual_weights(residual, cfg=cfg)).reshape(-1)

        def weighted_residual(
            candidate: jax.Array,
            weights_current: jax.Array = weights,
        ) -> jax.Array:
            raw = _residual_for_params(
                vol,
                obs,
                theta_nominal,
                phi_pose,
                pose_dx,
                pose_dz,
                geometry,
                candidate,
                mask=mask,
                sigma=cfg.sigma,
                config=cfg,
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

    solved = _geometry_with_params(geometry, params, cfg)
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


def _active_parameters(
    geometry: GeometryState,
    config: SetupOnlyLMConfig | None = None,
) -> tuple[SetupLMParameter, ...]:
    if config is not None and config.active_parameters is not None:
        _validate_active_parameters(config.active_parameters, geometry=geometry)
        return config.active_parameters
    if geometry.setup.det_v_px.active:
        return (
            "theta_offset_rad",
            "det_u_px",
            "det_v_px",
            "detector_roll_rad",
        )
    return (
        "theta_offset_rad",
        "det_u_px",
        "detector_roll_rad",
    )


def _validate_active_parameters(
    active_parameters: tuple[SetupLMParameter, ...],
    *,
    geometry: GeometryState,
) -> None:
    if len(set(active_parameters)) != len(active_parameters):
        raise ValueError("SetupOnlyLMConfig.active_parameters must not contain duplicates")
    if "det_v_px" in active_parameters and not geometry.setup.det_v_px.active:
        raise ValueError("det_v_px cannot be active when geometry det_v_px is inactive")


def _frozen_parameters(active: tuple[str, ...]) -> tuple[str, ...]:
    all_supported = (
        "theta_offset_rad",
        "det_u_px",
        "det_v_px",
        "detector_roll_rad",
        "axis_rot_x_rad",
        "axis_rot_y_rad",
        "theta_scale",
    )
    return tuple(name for name in all_supported if name not in active)


def _setup_parameter_values(geometry: GeometryState) -> dict[SetupLMParameter, float]:
    return {
        "theta_offset_rad": geometry.setup.theta_offset_rad.value,
        "det_u_px": geometry.setup.det_u_px.value,
        "det_v_px": geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0,
        "detector_roll_rad": geometry.setup.detector_roll_rad.value,
        "axis_rot_x_rad": geometry.setup.axis_rot_x_rad.value,
        "axis_rot_y_rad": geometry.setup.axis_rot_y_rad.value,
        "theta_scale": geometry.setup.theta_scale.value,
    }


def _pack_setup(geometry: GeometryState, config: SetupOnlyLMConfig | None = None) -> jax.Array:
    values = [
        _setup_parameter_values(geometry)[parameter]
        for parameter in _active_parameters(geometry, config)
    ]
    return jnp.asarray(values, dtype=jnp.float32)


def _geometry_with_params(
    geometry: GeometryState,
    params: jax.Array,
    config: SetupOnlyLMConfig | None = None,
) -> GeometryState:
    active = _active_parameters(geometry, config)
    values = _setup_parameter_values(geometry)
    for index, parameter in enumerate(active):
        values[parameter] = float(params[index])
    setup = geometry.setup.replace_parameter(
        "theta_offset_rad",
        geometry.setup.theta_offset_rad.with_value(values["theta_offset_rad"]),
    )
    setup = setup.replace_parameter(
        "det_u_px",
        geometry.setup.det_u_px.with_value(values["det_u_px"]),
    )
    if geometry.setup.det_v_px.active:
        setup = setup.replace_parameter(
            "det_v_px",
            geometry.setup.det_v_px.with_value(values["det_v_px"]),
        )
    setup = setup.replace_parameter(
        "detector_roll_rad",
        geometry.setup.detector_roll_rad.with_value(values["detector_roll_rad"]),
    )
    setup = setup.replace_parameter(
        "axis_rot_x_rad",
        geometry.setup.axis_rot_x_rad.with_value(values["axis_rot_x_rad"]),
    )
    setup = setup.replace_parameter(
        "axis_rot_y_rad",
        geometry.setup.axis_rot_y_rad.with_value(values["axis_rot_y_rad"]),
    )
    setup = setup.replace_parameter(
        "theta_scale",
        geometry.setup.theta_scale.with_value(values["theta_scale"]),
    )
    return GeometryState(setup=setup, pose=geometry.pose, acquisition=geometry.acquisition)


def _setup_values(
    geometry: GeometryState,
    params: jax.Array,
    config: SetupOnlyLMConfig | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    active = _active_parameters(geometry, config)
    values = {
        "theta_offset_rad": jnp.asarray(geometry.setup.theta_offset_rad.value, dtype=jnp.float32),
        "det_u_px": jnp.asarray(geometry.setup.det_u_px.value, dtype=jnp.float32),
        "det_v_px": (
            jnp.asarray(geometry.setup.det_v_px.value, dtype=jnp.float32)
            if geometry.setup.det_v_px.active
            else jnp.asarray(0.0, dtype=jnp.float32)
        ),
        "detector_roll_rad": jnp.asarray(geometry.setup.detector_roll_rad.value, dtype=jnp.float32),
        "axis_rot_x_rad": jnp.asarray(geometry.setup.axis_rot_x_rad.value, dtype=jnp.float32),
        "axis_rot_y_rad": jnp.asarray(geometry.setup.axis_rot_y_rad.value, dtype=jnp.float32),
        "theta_scale": jnp.asarray(geometry.setup.theta_scale.value, dtype=jnp.float32),
    }
    for index, parameter in enumerate(active):
        values[parameter] = params[index]
    return (
        values["theta_offset_rad"],
        values["det_u_px"],
        values["det_v_px"],
        values["detector_roll_rad"],
        values["axis_rot_x_rad"],
        values["axis_rot_y_rad"],
        values["theta_scale"],
    )


def _residual_for_params(
    volume: jax.Array,
    observed: jax.Array,
    theta_nominal: jax.Array,
    phi_pose: jax.Array,
    pose_dx: jax.Array,
    pose_dz: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    sigma: float,
    config: SetupOnlyLMConfig | None = None,
) -> jax.Array:
    theta_offset, det_u, det_v, detector_roll, axis_x, axis_y, theta_scale = _setup_values(
        geometry,
        params,
        config,
    )
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta_scale * theta_nominal + theta_offset + phi_pose,
        dx_px=det_u + pose_dx,
        dz_px=det_v + pose_dz,
        alpha_rad=jnp.asarray(geometry.pose.alpha_rad, dtype=jnp.float32),
        beta_rad=jnp.asarray(geometry.pose.beta_rad, dtype=jnp.float32),
        detector_roll_rad=detector_roll,
        axis_rot_x_rad=axis_x,
        axis_rot_y_rad=axis_y,
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
    cfg: SetupOnlyLMConfig,
) -> jax.Array:
    theta_offset, det_u, det_v, detector_roll, axis_x, axis_y, theta_scale = _setup_values(
        geometry,
        params,
        cfg,
    )
    theta = (
        theta_scale * jnp.asarray(geometry.pose.theta_nominal_rad, dtype=jnp.float32)
        + theta_offset
        + jnp.asarray(geometry.pose.phi_residual_rad, dtype=jnp.float32)
    )
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta,
        dx_px=det_u + jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32),
        dz_px=det_v + jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32),
        alpha_rad=jnp.asarray(geometry.pose.alpha_rad, dtype=jnp.float32),
        beta_rad=jnp.asarray(geometry.pose.beta_rad, dtype=jnp.float32),
        detector_roll_rad=detector_roll,
        axis_rot_x_rad=axis_x,
        axis_rot_y_rad=axis_y,
        nominal_axis_unit=nominal_axis_unit_from_geometry(geometry),
    )
    return residual_loss(
        predicted,
        observed,
        mask=mask,
        sigma=cfg.sigma,
        delta=cfg.delta,
        mode="l2" if cfg.loss_mode == "l2" else "pseudo_huber",
    ).loss


def _residual_weights(residual: jax.Array, *, cfg: SetupOnlyLMConfig) -> jax.Array:
    if cfg.loss_mode == "l2":
        return jnp.ones_like(residual, dtype=jnp.float32)
    return pseudo_huber_weights(residual, delta=cfg.delta)
