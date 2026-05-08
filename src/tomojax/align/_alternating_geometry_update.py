"""Geometry-update helpers for alternating smoke runs."""
# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._joint_schur_lm import (
    JointSchurDiagnostics,
    JointSchurLMConfig,
    JointSchurLMResult,
    PoseSchurDof,
    SetupSchurParameter,
    solve_joint_schur_lm,
)
from tomojax.align._setup_lm import (
    SetupOnlyLMConfig,
    SetupOnlyLMResult,
    solve_setup_only_lm,
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
    pose_trust_radius: float | None,
    active_setup_parameters: tuple[str, ...],
    solver: str,
    pose_frozen: bool,
    active_pose_dofs: tuple[str, ...],
    fit_gain_offset_nuisance: bool,
    fit_background_nuisance: bool,
    residual_filters: tuple[ResidualFilterConfig, ...] | None = None,
    parameter_prior_strength: float | None = None,
) -> tuple[GeometryState, GaugeReport, JointSchurLMResult]:
    setup_parameters = _active_setup_parameters(active_setup_parameters)
    pose_dofs = (
        ()
        if pose_frozen
        else _active_pose_dofs(
            active_pose_dofs,
            geometry,
            active_setup_parameters=setup_parameters,
        )
    )
    if solver == "setup_only_lm":
        if not pose_frozen or pose_dofs:
            raise ValueError("setup_only_lm geometry updates require frozen pose DOFs")
        result = solve_setup_only_lm(
            volume,
            observed,
            geometry,
            mask=mask,
            config=SetupOnlyLMConfig(
                max_iterations=max(1, int(updates)),
                damping=1.0e-3,
                delta=level.residual_delta,
                sigma=sigma,
                active_parameters=setup_parameters,
            ),
        )
        adapted = _setup_only_result_as_schur_result(result)
        return adapted.canonicalized_geometry.state, adapted.canonicalized_geometry.report, adapted
    if solver != "joint_schur":
        raise ValueError(f"unknown geometry update solver {solver!r}")
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
            pose_trust_radius=_pose_trust_radius(
                level.trust_radius_px,
                configured=pose_trust_radius,
            ),
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


def _setup_only_result_as_schur_result(result: SetupOnlyLMResult) -> JointSchurLMResult:
    initial_loss = float(result.initial_loss)
    final_loss = float(result.final_loss)
    accepted = final_loss <= initial_loss
    diagnostics = JointSchurDiagnostics(
        schur_condition=1.0,
        setup_update_norm=abs(initial_loss - final_loss),
        pose_update_norm=0.0,
        dense_step_difference_norm=0.0,
        accepted=accepted,
        current_loss=initial_loss,
        candidate_loss=final_loss,
        predicted_reduction=max(initial_loss - final_loss, 0.0),
        actual_reduction=initial_loss - final_loss,
        reduction_ratio=1.0 if final_loss < initial_loss else None,
        residual_filter_kinds=("raw",),
    )
    return JointSchurLMResult(
        geometry=result.geometry,
        canonicalized_geometry=result.canonicalized_geometry,
        initial_loss=initial_loss,
        final_loss=final_loss,
        iterations=int(result.iterations),
        active_setup_parameters=result.active_parameters,
        active_pose_dofs=(),
        frozen_parameters=result.frozen_parameters,
        diagnostics=diagnostics,
        iteration_diagnostics=(diagnostics,),
    )


def _pose_trust_radius(level_radius: float, *, configured: float | None) -> float | None:
    if configured is None:
        return float(level_radius)
    if float(configured) < 0.0:
        return None
    return float(configured)


def _active_setup_parameters(raw: tuple[str, ...]) -> tuple[SetupSchurParameter, ...]:
    allowed = {
        "axis_rot_x_rad",
        "axis_rot_y_rad",
        "theta_offset_rad",
        "det_u_px",
        "det_v_px",
        "detector_roll_rad",
        "theta_scale",
    }
    if any(name not in allowed for name in raw):
        raise ValueError(f"unsupported active setup parameters {raw!r}")
    return cast(
        "tuple[SetupSchurParameter, ...]",
        tuple(name for name in raw if name != "theta_scale"),
    )


def _active_pose_dofs(
    raw: tuple[str, ...],
    geometry: GeometryState,
    *,
    active_setup_parameters: tuple[SetupSchurParameter, ...],
) -> tuple[PoseSchurDof, ...]:
    allowed = {"alpha_rad", "beta_rad", "phi_residual_rad", "dx_px", "dz_px"}
    if any(name not in allowed for name in raw):
        raise ValueError(f"unsupported active pose DOFs {raw!r}")
    if (
        geometry.acquisition.model == "parallel"
        and _is_global_setup_block(active_setup_parameters)
        and not _any_pose_signal(geometry, raw)
    ):
        return ()
    return cast("tuple[PoseSchurDof, ...]", tuple(raw))


def _is_global_setup_block(active_setup_parameters: tuple[SetupSchurParameter, ...]) -> bool:
    return {"detector_roll_rad", "axis_rot_x_rad", "axis_rot_y_rad"}.issubset(
        set(active_setup_parameters)
    )


def _any_pose_signal(geometry: GeometryState, names: tuple[str, ...]) -> bool:
    return any(_pose_dof_has_signal(geometry, name) for name in names)


def _pose_dof_has_signal(geometry: GeometryState, name: str) -> bool:
    values = np.asarray(getattr(geometry.pose, name), dtype=np.float64)
    return bool(np.max(np.abs(values)) > 1.0e-12)


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


def _geometry_update_volume_for_level(
    *,
    truth_volume: jax.Array,
    stopped_volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    level: ContinuationLevel,
    source: GeometryUpdateVolumeSource,
    active_setup_parameters: tuple[str, ...],
) -> jax.Array:
    volume = _geometry_update_volume(
        truth_volume=truth_volume,
        stopped_volume=stopped_volume,
        source=source,
    )
    return _anchored_geometry_update_volume(
        volume,
        observed,
        mask,
        level=level,
        source=source,
        active_setup_parameters=active_setup_parameters,
    )


def _anchored_geometry_update_volume(
    stopped_volume: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    *,
    level: ContinuationLevel,
    source: GeometryUpdateVolumeSource,
    active_setup_parameters: tuple[str, ...],
) -> jax.Array:
    if source != "stopped_reconstruction":
        return stopped_volume
    if int(level.level_factor) != 4:
        return stopped_volume
    active_setup = _active_setup_parameters(active_setup_parameters)
    if "det_u_px" not in active_setup or not _is_global_setup_block(active_setup):
        return stopped_volume
    shift_px = _det_u_recentering_shift_px(observed, mask)
    return jnp.roll(stopped_volume, shift=shift_px, axis=1)


def _det_u_recentering_shift_px(observed: jax.Array, mask: jax.Array) -> int:
    weighted = jnp.asarray(observed, dtype=jnp.float32) * jnp.asarray(mask, dtype=jnp.float32)
    mass = jnp.sum(weighted)
    if float(mass) <= 0.0:
        return 0
    cols = jnp.arange(weighted.shape[2], dtype=jnp.float32)
    col_com = jnp.sum(weighted * cols[None, None, :]) / mass
    center = jnp.asarray((int(weighted.shape[2]) - 1) / 2.0, dtype=jnp.float32)
    det_u_estimate = center - col_com
    return -int(np.rint(float(det_u_estimate)))


def _geometry_updates_for_level(
    level: ContinuationLevel,
    coarse_verified: bool,
) -> tuple[int, str | None]:
    if level.role == "final" and coarse_verified:
        return 0, "coarse_verification_passed"
    return level.geometry_updates, None
