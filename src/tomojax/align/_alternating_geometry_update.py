"""Geometry-update helpers for alternating smoke runs."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax

from tomojax.align._joint_schur_lm import (
    JointSchurLMConfig,
    JointSchurLMResult,
    PoseSchurDof,
    SetupSchurParameter,
    solve_joint_schur_lm,
)

if TYPE_CHECKING:
    from tomojax.align._alternating_types import GeometryUpdateVolumeSource
    from tomojax.align._continuation import ContinuationLevel
    from tomojax.forward import ResidualFilterConfig
    from tomojax.geometry import GaugeReport, GeometryState


def _run_geometry_updates(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    mask: jax.Array,
    level: ContinuationLevel,
    updates: int,
    *,
    sigma: float,
    setup_prior_strength: float | None,
    pose_prior_strength: float | None,
    active_setup_parameters: tuple[str, ...],
    pose_frozen: bool,
    active_pose_dofs: tuple[str, ...],
    fit_gain_offset_nuisance: bool,
    fit_background_nuisance: bool,
    residual_filters: tuple[ResidualFilterConfig, ...] | None = None,
    parameter_prior_strength: float | None = None,
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult]:
    setup_parameters = _active_setup_parameters(active_setup_parameters)
    pose_dofs = () if pose_frozen else _active_pose_dofs(active_pose_dofs)
    result = solve_joint_schur_lm(
        volume,
        observed,
        geometry,
        mask=mask,
        config=JointSchurLMConfig(
            max_iterations=max(1, int(updates)),
            damping=1.0e-3,
            delta=level.residual_delta,
            sigma=sigma,
            setup_trust_radius=_setup_trust_radius(
                level.trust_radius_px,
                pose_frozen=pose_frozen,
                active_setup_parameters=setup_parameters,
            ),
            pose_trust_radius=level.trust_radius_px,
            residual_filters=level.residual_filters
            if residual_filters is None
            else residual_filters,
            parameter_prior_strength=(
                level.prior_strength
                if parameter_prior_strength is None
                else parameter_prior_strength
            ),
            setup_prior_strength=setup_prior_strength,
            pose_prior_strength=pose_prior_strength,
            active_setup_parameters=setup_parameters,
            active_pose_dofs=pose_dofs,
            fit_gain_offset=fit_gain_offset_nuisance,
            fit_background_offset=fit_background_nuisance,
        ),
    )
    return result.canonicalized_geometry.state, result.canonicalized_geometry.report, result


def _active_pose_dofs(raw: tuple[str, ...]) -> tuple[PoseSchurDof, ...]:
    allowed = {"alpha_rad", "beta_rad", "phi_residual_rad", "dx_px", "dz_px"}
    if any(name not in allowed for name in raw):
        raise ValueError(f"unsupported active pose DOFs {raw!r}")
    return cast("tuple[PoseSchurDof, ...]", tuple(raw))


def _active_setup_parameters(raw: tuple[str, ...]) -> tuple[SetupSchurParameter, ...]:
    allowed = {
        "axis_rot_x_rad",
        "axis_rot_y_rad",
        "theta_offset_rad",
        "det_u_px",
        "det_v_px",
        "detector_roll_rad",
    }
    if any(name not in allowed for name in raw):
        raise ValueError(f"unsupported active setup parameters {raw!r}")
    return cast("tuple[SetupSchurParameter, ...]", tuple(raw))


def _setup_trust_radius(
    radius: float,
    *,
    pose_frozen: bool,
    active_setup_parameters: tuple[SetupSchurParameter, ...],
) -> float | None:
    if pose_frozen and active_setup_parameters == ("det_u_px",):
        return None
    return float(radius)


def _geometry_update_volume(
    *,
    truth_volume: jax.Array,
    stopped_volume: jax.Array,
    source: GeometryUpdateVolumeSource,
) -> jax.Array:
    if source == "fixed_synthetic_truth":
        return jax.lax.stop_gradient(truth_volume)
    if source == "stopped_reconstruction":
        return jax.lax.stop_gradient(stopped_volume)
    raise ValueError(f"unknown geometry update volume source {source!r}")


def _geometry_updates_for_level(
    level: ContinuationLevel,
    coarse_verified: bool,
) -> tuple[int, str | None]:
    if level.role == "final" and coarse_verified:
        return 0, "coarse_verification_passed"
    return level.geometry_updates, None
