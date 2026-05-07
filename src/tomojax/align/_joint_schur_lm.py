"""Joint setup+pose Schur LM reference solver for supported v2 DOFs."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._lm_numerics import finite_difference_jacobian
from tomojax.forward import project_parallel_reference_arrays, pseudo_huber_weights, residual_loss
from tomojax.geometry import CanonicalizedGeometry, GeometryState, canonicalize_geometry_gauges
from tomojax.nuisance import estimate_background_offset, estimate_gain_offset

if TYPE_CHECKING:
    from pathlib import Path

_POSE_DIM = 3
PoseSchurDof = Literal["phi_residual_rad", "dx_px", "dz_px"]
_POSE_DOF_ORDER: tuple[PoseSchurDof, ...] = ("phi_residual_rad", "dx_px", "dz_px")


@dataclass(frozen=True)
class JointSchurLMConfig:
    max_iterations: int = 6
    damping: float = 1e-2
    adapt_damping: bool = True
    damping_decrease_factor: float = 0.5
    damping_increase_factor: float = 2.0
    min_damping: float = 1e-6
    max_damping: float = 1e6
    adapt_trust_radii: bool = True
    trust_shrink_ratio: float = 0.25
    trust_expand_ratio: float = 0.75
    trust_shrink_factor: float = 0.5
    trust_expand_factor: float = 2.0
    min_trust_radius: float = 1e-6
    max_trust_radius: float = 1e6
    sigma: float = 1.0
    delta: float = 1.0
    finite_difference_step: float = 1e-3
    setup_trust_radius: float | None = None
    pose_trust_radius: float | None = None
    parameter_prior_strength: float = 0.0
    setup_prior_strength: float | None = None
    pose_prior_strength: float | None = None
    active_pose_dofs: tuple[PoseSchurDof, ...] = ("phi_residual_rad", "dx_px", "dz_px")
    fit_gain_offset: bool = False
    fit_background_offset: bool = False


@dataclass(frozen=True)
class JointSchurDiagnostics:
    schur_condition: float
    setup_update_norm: float
    pose_update_norm: float
    dense_step_difference_norm: float
    global_eigenvalues: tuple[float, ...] = ()
    schur_eigenvalues: tuple[float, ...] = ()
    pose_block_conditions: tuple[float, ...] = ()
    setup_correlation_matrix: tuple[tuple[float, ...], ...] = ()
    weak_mode_labels: tuple[str, ...] = ()
    trust_scale: float = 1.0
    setup_trust_scale: float = 1.0
    pose_trust_scale: float = 1.0
    trust_clipped: bool = False
    setup_update_by_parameter: tuple[float, ...] = ()
    pose_update_max_by_dof: tuple[float, ...] = ()
    damping: float = float("nan")
    next_damping: float = float("nan")
    accepted: bool = False
    current_loss: float = float("nan")
    candidate_loss: float = float("nan")
    predicted_reduction: float = float("nan")
    actual_reduction: float = float("nan")
    reduction_ratio: float | None = None
    next_setup_trust_radius: float | None = None
    next_pose_trust_radius: float | None = None
    current_loss_by_view: tuple[float, ...] = ()
    candidate_loss_by_view: tuple[float, ...] = ()
    actual_reduction_by_view: tuple[float, ...] = ()
    setup_gradient_by_view: tuple[tuple[float, ...], ...] = ()
    pose_gradient_by_view: tuple[tuple[float, ...], ...] = ()
    setup_hessian_diag_by_view: tuple[tuple[float, ...], ...] = ()
    pose_hessian_diag_by_view: tuple[tuple[float, ...], ...] = ()
    setup_pose_coupling_norm_by_view: tuple[float, ...] = ()
    parameter_prior_strength: float = 0.0
    gain_offset_fit: bool = False
    background_offset_fit: bool = False
    gain_offset_model: dict[str, object] | None = None
    background_offset_model: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schur_condition": self.schur_condition,
            "setup_update_norm": self.setup_update_norm,
            "pose_update_norm": self.pose_update_norm,
            "dense_step_difference_norm": self.dense_step_difference_norm,
            "global_eigenvalues": list(self.global_eigenvalues),
            "schur_eigenvalues": list(self.schur_eigenvalues),
            "pose_block_conditions": list(self.pose_block_conditions),
            "setup_correlation_matrix": [list(row) for row in self.setup_correlation_matrix],
            "weak_mode_labels": list(self.weak_mode_labels),
            "trust_scale": self.trust_scale,
            "setup_trust_scale": self.setup_trust_scale,
            "pose_trust_scale": self.pose_trust_scale,
            "trust_clipped": self.trust_clipped,
            "setup_update_by_parameter": list(self.setup_update_by_parameter),
            "pose_update_max_by_dof": list(self.pose_update_max_by_dof),
            "damping": self.damping,
            "next_damping": self.next_damping,
            "accepted": self.accepted,
            "current_loss": self.current_loss,
            "candidate_loss": self.candidate_loss,
            "predicted_reduction": self.predicted_reduction,
            "actual_reduction": self.actual_reduction,
            "reduction_ratio": self.reduction_ratio,
            "next_setup_trust_radius": self.next_setup_trust_radius,
            "next_pose_trust_radius": self.next_pose_trust_radius,
            "current_loss_by_view": list(self.current_loss_by_view),
            "candidate_loss_by_view": list(self.candidate_loss_by_view),
            "actual_reduction_by_view": list(self.actual_reduction_by_view),
            "setup_gradient_by_view": [list(row) for row in self.setup_gradient_by_view],
            "pose_gradient_by_view": [list(row) for row in self.pose_gradient_by_view],
            "setup_hessian_diag_by_view": [list(row) for row in self.setup_hessian_diag_by_view],
            "pose_hessian_diag_by_view": [list(row) for row in self.pose_hessian_diag_by_view],
            "setup_pose_coupling_norm_by_view": list(self.setup_pose_coupling_norm_by_view),
            "parameter_prior_strength": self.parameter_prior_strength,
            "gain_offset_fit": self.gain_offset_fit,
            "background_offset_fit": self.background_offset_fit,
            "gain_offset_model": self.gain_offset_model,
            "background_offset_model": self.background_offset_model,
        }


@dataclass(frozen=True)
class JointSchurLMResult:
    geometry: GeometryState
    canonicalized_geometry: CanonicalizedGeometry
    initial_loss: float
    final_loss: float
    iterations: int
    active_setup_parameters: tuple[str, ...]
    active_pose_dofs: tuple[str, ...]
    frozen_parameters: tuple[str, ...]
    diagnostics: JointSchurDiagnostics
    iteration_diagnostics: tuple[JointSchurDiagnostics, ...]


def solve_joint_schur_lm(  # noqa: PLR0915 - iterative solver keeps state transitions explicit.
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    *,
    mask: jax.Array | None = None,
    config: JointSchurLMConfig | None = None,
) -> JointSchurLMResult:
    """Solve supported setup and pose DOFs with a Schur LM step."""
    cfg = config or JointSchurLMConfig()
    vol = jnp.asarray(volume, dtype=jnp.float32)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    pose_dim = _pose_dim(cfg)
    params = _pack_joint(geometry, cfg)
    prior_reference = params
    n_setup = _n_setup_params(geometry)
    initial_loss = _loss_for_params(
        vol,
        obs,
        geometry,
        params,
        mask=mask,
        cfg=cfg,
        prior_reference=prior_reference,
    )
    diagnostics = JointSchurDiagnostics(
        schur_condition=float("nan"),
        setup_update_norm=0.0,
        pose_update_norm=0.0,
        dense_step_difference_norm=0.0,
        global_eigenvalues=(),
        schur_eigenvalues=(),
        pose_block_conditions=(),
        setup_correlation_matrix=(),
        weak_mode_labels=(),
        trust_scale=1.0,
        trust_clipped=False,
        setup_update_by_parameter=(),
        pose_update_max_by_dof=(),
    )
    iterations = 0
    iteration_diagnostics: list[JointSchurDiagnostics] = []
    damping = float(cfg.damping)
    setup_trust_radius = cfg.setup_trust_radius
    pose_trust_radius = cfg.pose_trust_radius

    for _ in range(max(0, int(cfg.max_iterations))):
        residual = _residual_for_params(
            vol,
            obs,
            geometry,
            params,
            mask=mask,
            cfg=cfg,
            sigma=cfg.sigma,
            fit_gain_offset=cfg.fit_gain_offset,
            fit_background_offset=cfg.fit_background_offset,
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
                candidate,
                mask=mask,
                cfg=cfg,
                sigma=cfg.sigma,
                fit_gain_offset=cfg.fit_gain_offset,
                fit_background_offset=cfg.fit_background_offset,
            )
            data_residual = raw.reshape(-1) * weights_current
            return _with_parameter_prior_residual(
                data_residual,
                candidate,
                prior_reference=prior_reference,
                cfg=cfg,
                n_setup=n_setup,
            )

        r = weighted_residual(params)
        data_rows = int(obs.size)
        jacobian = finite_difference_jacobian(
            weighted_residual,
            params,
            step_size=cfg.finite_difference_step,
        )
        schur = schur_step_from_jacobian(
            jacobian,
            r,
            n_setup=n_setup,
            n_views=geometry.pose.n_views,
            pose_dim=pose_dim,
            damping=damping,
            setup_trust_radius=setup_trust_radius,
            pose_trust_radius=pose_trust_radius,
            data_rows=data_rows,
        )
        candidate = params + schur.step
        candidate_loss = _loss_for_params(
            vol,
            obs,
            geometry,
            candidate,
            mask=mask,
            cfg=cfg,
            prior_reference=prior_reference,
        )
        current_loss = _loss_for_params(
            vol,
            obs,
            geometry,
            params,
            mask=mask,
            cfg=cfg,
            prior_reference=prior_reference,
        )
        current_loss_by_view = _loss_by_view_for_params(
            vol,
            obs,
            geometry,
            params,
            mask=mask,
            cfg=cfg,
        )
        candidate_loss_by_view = _loss_by_view_for_params(
            vol,
            obs,
            geometry,
            candidate,
            mask=mask,
            cfg=cfg,
        )
        accepted = bool(candidate_loss <= current_loss)
        actual_reduction = float(current_loss - candidate_loss)
        actual_reduction_by_view = tuple(
            current - candidate
            for current, candidate in zip(
                current_loss_by_view,
                candidate_loss_by_view,
                strict=True,
            )
        )
        predicted_reduction = schur.diagnostics.predicted_reduction
        reduction_ratio = _reduction_ratio(actual_reduction, predicted_reduction)
        params = candidate if accepted else params
        next_damping = adapt_joint_schur_damping(damping, accepted=accepted, config=cfg)
        next_setup_trust_radius = adapt_joint_schur_trust_radius(
            setup_trust_radius,
            accepted=accepted,
            reduction_ratio=reduction_ratio,
            clipped=schur.diagnostics.trust_clipped,
            config=cfg,
        )
        next_pose_trust_radius = adapt_joint_schur_trust_radius(
            pose_trust_radius,
            accepted=accepted,
            reduction_ratio=reduction_ratio,
            clipped=schur.diagnostics.trust_clipped,
            config=cfg,
        )
        iterations += 1
        diagnostics = replace(
            schur.diagnostics,
            damping=damping,
            next_damping=next_damping,
            accepted=accepted,
            current_loss=float(current_loss),
            candidate_loss=float(candidate_loss),
            predicted_reduction=predicted_reduction,
            actual_reduction=actual_reduction,
            reduction_ratio=reduction_ratio,
            next_setup_trust_radius=next_setup_trust_radius,
            next_pose_trust_radius=next_pose_trust_radius,
            current_loss_by_view=current_loss_by_view,
            candidate_loss_by_view=candidate_loss_by_view,
            actual_reduction_by_view=actual_reduction_by_view,
            parameter_prior_strength=max(float(cfg.parameter_prior_strength), 0.0),
            gain_offset_fit=bool(cfg.fit_gain_offset),
            background_offset_fit=bool(cfg.fit_background_offset),
            **_nuisance_diagnostics_for_params(
                vol,
                obs,
                geometry,
                params,
                mask=mask,
                cfg=cfg,
            ),
        )
        iteration_diagnostics.append(diagnostics)
        damping = next_damping
        setup_trust_radius = next_setup_trust_radius
        pose_trust_radius = next_pose_trust_radius
        if accepted:
            params = _pack_joint(
                canonicalize_geometry_gauges(_geometry_with_params(geometry, params, cfg)).state,
                cfg,
            )
        if float(jnp.linalg.norm(schur.step)) < 1e-5:
            break

    solved = _geometry_with_params(geometry, params, cfg)
    canonicalized = canonicalize_geometry_gauges(solved)
    final_loss = _loss_for_params(
        vol,
        obs,
        geometry,
        params,
        mask=mask,
        cfg=cfg,
        prior_reference=prior_reference,
    )
    return JointSchurLMResult(
        geometry=solved,
        canonicalized_geometry=canonicalized,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        iterations=iterations,
        active_setup_parameters=_active_setup_parameters(geometry),
        active_pose_dofs=cfg.active_pose_dofs,
        frozen_parameters=_frozen_parameters(geometry, cfg),
        diagnostics=diagnostics,
        iteration_diagnostics=tuple(iteration_diagnostics),
    )


def adapt_joint_schur_damping(
    damping: float,
    *,
    accepted: bool,
    config: JointSchurLMConfig,
) -> float:
    """Return the next LM damping value after an accepted or rejected step."""
    current = float(damping)
    if not config.adapt_damping:
        return current
    factor = config.damping_decrease_factor if accepted else config.damping_increase_factor
    candidate = current * max(float(factor), 0.0)
    return min(max(candidate, float(config.min_damping)), float(config.max_damping))


def adapt_joint_schur_trust_radius(
    radius: float | None,
    *,
    accepted: bool,
    reduction_ratio: float | None,
    clipped: bool,
    config: JointSchurLMConfig,
) -> float | None:
    """Return the next trust radius after evaluating a joint Schur candidate."""
    if radius is None or not config.adapt_trust_radii:
        return radius
    current = float(radius)
    if (
        not accepted
        or reduction_ratio is None
        or reduction_ratio < float(config.trust_shrink_ratio)
    ):
        candidate = current * max(float(config.trust_shrink_factor), 0.0)
    elif clipped and reduction_ratio > float(config.trust_expand_ratio):
        candidate = current * max(float(config.trust_expand_factor), 0.0)
    else:
        candidate = current
    return min(max(candidate, float(config.min_trust_radius)), float(config.max_trust_radius))


def joint_schur_normal_eq_summary(result: JointSchurLMResult) -> dict[str, object]:
    """Return the JSON-serializable normal-equation summary artifact."""
    return {
        "solver": "joint_schur_lm_reference",
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "iterations": result.iterations,
        "active_setup_parameters": list(result.active_setup_parameters),
        "active_pose_dofs": list(result.active_pose_dofs),
        "frozen_parameters": list(result.frozen_parameters),
        "diagnostics": result.diagnostics.to_dict(),
        "iteration_diagnostics": [
            diagnostics.to_dict() for diagnostics in result.iteration_diagnostics
        ],
    }


def write_joint_schur_normal_eq_summary(result: JointSchurLMResult, path: str | Path) -> Path:
    """Write the Phase 6 normal-equation summary artifact as JSON."""
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(
        json.dumps(joint_schur_normal_eq_summary(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


@dataclass(frozen=True)
class SchurStep:
    step: jax.Array
    dense_step: jax.Array
    diagnostics: JointSchurDiagnostics


@dataclass(frozen=True)
class _PerViewNormalBlockDiagnostics:
    setup_gradient_by_view: tuple[tuple[float, ...], ...]
    pose_gradient_by_view: tuple[tuple[float, ...], ...]
    setup_hessian_diag_by_view: tuple[tuple[float, ...], ...]
    pose_hessian_diag_by_view: tuple[tuple[float, ...], ...]
    setup_pose_coupling_norm_by_view: tuple[float, ...]


def schur_step_from_jacobian(
    jacobian: jax.Array,
    residual: jax.Array,
    *,
    n_setup: int,
    n_views: int,
    pose_dim: int,
    damping: float,
    setup_trust_radius: float | None = None,
    pose_trust_radius: float | None = None,
    data_rows: int | None = None,
) -> SchurStep:
    """Solve a damped joint normal equation by per-view Schur complement."""
    jac = jnp.asarray(jacobian, dtype=jnp.float32)
    r = jnp.asarray(residual, dtype=jnp.float32)
    n_params = n_setup + n_views * pose_dim
    diagnostic_jacobian = jac if data_rows is None else jac[: int(data_rows)]
    diagnostic_residual = r if data_rows is None else r[: int(data_rows)]
    per_view_blocks = _per_view_normal_block_diagnostics(
        diagnostic_jacobian,
        diagnostic_residual,
        n_setup=n_setup,
        n_views=n_views,
        pose_dim=pose_dim,
    )
    hessian = jac.T @ jac + jnp.eye(n_params, dtype=jnp.float32) * jnp.asarray(
        damping,
        dtype=jnp.float32,
    )
    gradient = jac.T @ r
    dense_step_unscaled = jnp.linalg.solve(hessian, -gradient)

    h_gg = hessian[:n_setup, :n_setup]
    g_g = gradient[:n_setup]
    schur_matrix = h_gg
    schur_rhs = -g_g
    pose_blocks: list[tuple[jax.Array, jax.Array, jax.Array]] = []
    pose_block_conditions: list[float] = []
    for view in range(n_views):
        start = n_setup + view * pose_dim
        stop = start + pose_dim
        h_gp = hessian[:n_setup, start:stop]
        h_pg = hessian[start:stop, :n_setup]
        h_pp = hessian[start:stop, start:stop]
        g_p = gradient[start:stop]
        inv_hpp_hpg = jnp.linalg.solve(h_pp, h_pg)
        inv_hpp_gp = jnp.linalg.solve(h_pp, g_p)
        schur_matrix = schur_matrix - h_gp @ inv_hpp_hpg
        schur_rhs = schur_rhs + h_gp @ inv_hpp_gp
        if pose_dim > 0:
            pose_blocks.append((h_pp, h_pg, g_p))
            pose_block_conditions.append(float(jnp.linalg.cond(h_pp)))

    setup_step = jnp.linalg.solve(schur_matrix, schur_rhs)
    pose_steps: list[jax.Array] = []
    if pose_dim > 0:
        for h_pp, h_pg, g_p in pose_blocks:
            pose_steps.append(jnp.linalg.solve(h_pp, -g_p - h_pg @ setup_step))
    pose_step = (
        jnp.concatenate(pose_steps, axis=0) if pose_steps else jnp.asarray([], dtype=jnp.float32)
    )
    setup_trust_scale, pose_trust_scale = _trust_scales(
        setup_step,
        pose_step,
        setup_trust_radius=setup_trust_radius,
        pose_trust_radius=pose_trust_radius,
    )
    step = jnp.concatenate([setup_step * setup_trust_scale, pose_step * pose_trust_scale], axis=0)
    dense_step = _scale_dense_step(
        dense_step_unscaled,
        n_setup=n_setup,
        setup_trust_scale=setup_trust_scale,
        pose_trust_scale=pose_trust_scale,
    )
    pose_step_matrix = (
        (pose_step * pose_trust_scale).reshape((n_views, pose_dim))
        if pose_dim > 0
        else jnp.zeros((n_views, 0), dtype=jnp.float32)
    )
    predicted_reduction = _predicted_reduction(
        gradient,
        hessian,
        step,
        data_rows=data_rows,
    )
    schur_eigenvalues = _eigvalsh_tuple(schur_matrix)
    diagnostics = JointSchurDiagnostics(
        schur_condition=float(jnp.linalg.cond(schur_matrix)),
        setup_update_norm=float(jnp.linalg.norm(setup_step * setup_trust_scale)),
        pose_update_norm=float(jnp.linalg.norm(pose_step * pose_trust_scale)),
        dense_step_difference_norm=float(jnp.linalg.norm(step - dense_step)),
        global_eigenvalues=_eigvalsh_tuple(hessian),
        schur_eigenvalues=schur_eigenvalues,
        pose_block_conditions=tuple(pose_block_conditions),
        setup_correlation_matrix=_correlation_matrix_tuple(schur_matrix),
        weak_mode_labels=_weak_mode_labels(schur_eigenvalues),
        trust_scale=float(jnp.minimum(setup_trust_scale, pose_trust_scale)),
        setup_trust_scale=float(setup_trust_scale),
        pose_trust_scale=float(pose_trust_scale),
        trust_clipped=bool(jnp.minimum(setup_trust_scale, pose_trust_scale) < 1.0),
        setup_update_by_parameter=tuple(float(value) for value in setup_step * setup_trust_scale),
        pose_update_max_by_dof=tuple(
            float(value) for value in jnp.max(jnp.abs(pose_step_matrix), axis=0)
        ),
        predicted_reduction=predicted_reduction,
        setup_gradient_by_view=per_view_blocks.setup_gradient_by_view,
        pose_gradient_by_view=per_view_blocks.pose_gradient_by_view,
        setup_hessian_diag_by_view=per_view_blocks.setup_hessian_diag_by_view,
        pose_hessian_diag_by_view=per_view_blocks.pose_hessian_diag_by_view,
        setup_pose_coupling_norm_by_view=per_view_blocks.setup_pose_coupling_norm_by_view,
    )
    return SchurStep(step=step, dense_step=dense_step, diagnostics=diagnostics)


def _per_view_normal_block_diagnostics(
    jacobian: jax.Array,
    residual: jax.Array,
    *,
    n_setup: int,
    n_views: int,
    pose_dim: int,
) -> _PerViewNormalBlockDiagnostics:
    rows_per_view = int(jacobian.shape[0]) // int(n_views)
    setup_gradient_by_view: list[tuple[float, ...]] = []
    pose_gradient_by_view: list[tuple[float, ...]] = []
    setup_hessian_diag_by_view: list[tuple[float, ...]] = []
    pose_hessian_diag_by_view: list[tuple[float, ...]] = []
    setup_pose_coupling_norm_by_view: list[float] = []
    for view in range(n_views):
        row_start = view * rows_per_view
        row_stop = row_start + rows_per_view
        pose_start = n_setup + view * pose_dim
        pose_stop = pose_start + pose_dim
        jac_view = jacobian[row_start:row_stop]
        residual_view = residual[row_start:row_stop]
        jac_setup = jac_view[:, :n_setup]
        jac_pose = jac_view[:, pose_start:pose_stop]
        setup_gradient = jac_setup.T @ residual_view
        pose_gradient = jac_pose.T @ residual_view
        setup_hessian = jac_setup.T @ jac_setup
        pose_hessian = jac_pose.T @ jac_pose
        coupling = jac_setup.T @ jac_pose
        setup_gradient_by_view.append(tuple(float(value) for value in setup_gradient))
        pose_gradient_by_view.append(tuple(float(value) for value in pose_gradient))
        setup_hessian_diag_by_view.append(tuple(float(value) for value in jnp.diag(setup_hessian)))
        pose_hessian_diag_by_view.append(tuple(float(value) for value in jnp.diag(pose_hessian)))
        setup_pose_coupling_norm_by_view.append(float(jnp.linalg.norm(coupling)))
    return _PerViewNormalBlockDiagnostics(
        setup_gradient_by_view=tuple(setup_gradient_by_view),
        pose_gradient_by_view=tuple(pose_gradient_by_view),
        setup_hessian_diag_by_view=tuple(setup_hessian_diag_by_view),
        pose_hessian_diag_by_view=tuple(pose_hessian_diag_by_view),
        setup_pose_coupling_norm_by_view=tuple(setup_pose_coupling_norm_by_view),
    )


def _trust_scales(
    setup_step: jax.Array,
    pose_step: jax.Array,
    *,
    setup_trust_radius: float | None,
    pose_trust_radius: float | None,
) -> tuple[jax.Array, jax.Array]:
    setup_scale = jnp.asarray(1.0, dtype=jnp.float32)
    pose_scale = jnp.asarray(1.0, dtype=jnp.float32)
    if setup_trust_radius is not None:
        setup_norm = jnp.linalg.norm(setup_step)
        setup_limit = jnp.asarray(max(float(setup_trust_radius), 0.0), dtype=jnp.float32)
        setup_scale = jnp.minimum(setup_scale, setup_limit / jnp.maximum(setup_norm, 1e-12))
    if pose_trust_radius is not None:
        pose_norm = jnp.linalg.norm(pose_step)
        pose_limit = jnp.asarray(max(float(pose_trust_radius), 0.0), dtype=jnp.float32)
        pose_scale = jnp.minimum(pose_scale, pose_limit / jnp.maximum(pose_norm, 1e-12))
    return setup_scale, pose_scale


def _scale_dense_step(
    dense_step: jax.Array,
    *,
    n_setup: int,
    setup_trust_scale: jax.Array,
    pose_trust_scale: jax.Array,
) -> jax.Array:
    return jnp.concatenate(
        [
            dense_step[:n_setup] * setup_trust_scale,
            dense_step[n_setup:] * pose_trust_scale,
        ],
        axis=0,
    )


def _predicted_reduction(
    gradient: jax.Array,
    hessian: jax.Array,
    step: jax.Array,
    *,
    data_rows: int | None,
) -> float:
    model_change = jnp.vdot(gradient, step).real + 0.5 * jnp.vdot(step, hessian @ step).real
    scale = 1 if data_rows is None else max(int(data_rows), 1)
    return -float(model_change) / float(scale)


def _reduction_ratio(actual_reduction: float, predicted_reduction: float) -> float | None:
    if abs(predicted_reduction) <= 1e-12:
        return None
    return float(actual_reduction / predicted_reduction)


def _eigvalsh_tuple(matrix: jax.Array) -> tuple[float, ...]:
    values = jnp.linalg.eigvalsh(matrix)
    return tuple(float(value) for value in values)


def _correlation_matrix_tuple(matrix: jax.Array) -> tuple[tuple[float, ...], ...]:
    diagonal = jnp.clip(jnp.diag(matrix), min=1e-12)
    scale = jnp.sqrt(diagonal[:, None] * diagonal[None, :])
    correlation = matrix / scale
    return tuple(tuple(float(value) for value in row) for row in correlation)


def _weak_mode_labels(eigenvalues: tuple[float, ...]) -> tuple[str, ...]:
    if not eigenvalues:
        return ()
    max_abs = max(abs(value) for value in eigenvalues)
    threshold = max(max_abs * 1e-6, 1e-9)
    return tuple(
        f"schur_eigen_{index}" for index, value in enumerate(eigenvalues) if abs(value) <= threshold
    )


def _active_setup_parameters(geometry: GeometryState) -> tuple[str, ...]:
    if geometry.setup.det_v_px.active:
        return ("theta_offset_rad", "det_u_px", "det_v_px")
    return ("theta_offset_rad", "det_u_px")


def _frozen_parameters(
    geometry: GeometryState,
    config: JointSchurLMConfig | None = None,
) -> tuple[str, ...]:
    frozen = [
        "alpha_rad",
        "beta_rad",
        "detector_roll_rad",
        "axis_rot_x_rad",
        "axis_rot_y_rad",
        "theta_scale",
    ]
    if not geometry.setup.det_v_px.active:
        frozen.append("det_v_px")
    active_pose = (
        ("phi_residual_rad", "dx_px", "dz_px") if config is None else config.active_pose_dofs
    )
    frozen.extend(dof for dof in ("phi_residual_rad", "dx_px", "dz_px") if dof not in active_pose)
    return tuple(frozen)


def _pose_dim(config: JointSchurLMConfig) -> int:
    _validate_active_pose_dofs(config.active_pose_dofs)
    return len(config.active_pose_dofs)


def _validate_active_pose_dofs(active_pose_dofs: tuple[PoseSchurDof, ...]) -> None:
    if len(set(active_pose_dofs)) != len(active_pose_dofs):
        raise ValueError("JointSchurLMConfig.active_pose_dofs must not contain duplicates")
    unknown = tuple(dof for dof in active_pose_dofs if dof not in _POSE_DOF_ORDER)
    if unknown:
        raise ValueError(f"unknown active pose DOFs {unknown!r}")


def _n_setup_params(geometry: GeometryState) -> int:
    return 3 if geometry.setup.det_v_px.active else 2


def _pack_joint(geometry: GeometryState, config: JointSchurLMConfig | None = None) -> jax.Array:
    setup_values = [geometry.setup.theta_offset_rad.value, geometry.setup.det_u_px.value]
    if geometry.setup.det_v_px.active:
        setup_values.append(geometry.setup.det_v_px.value)
    if config is not None and _pose_dim(config) == 0:
        return jnp.asarray(setup_values, dtype=jnp.float32)
    active_pose = _active_pose_arrays(
        geometry,
        config.active_pose_dofs if config else _POSE_DOF_ORDER,
    )
    pose_values = np.stack(active_pose, axis=1).reshape(-1)
    return jnp.asarray([*setup_values, *pose_values], dtype=jnp.float32)


def _active_pose_arrays(
    geometry: GeometryState,
    active_pose_dofs: tuple[PoseSchurDof, ...],
) -> tuple[np.ndarray, ...]:
    arrays = {
        "phi_residual_rad": geometry.pose.phi_residual_rad,
        "dx_px": geometry.pose.dx_px,
        "dz_px": geometry.pose.dz_px,
    }
    return tuple(arrays[dof] for dof in active_pose_dofs)


def _geometry_with_params(
    geometry: GeometryState,
    params: jax.Array,
    config: JointSchurLMConfig | None = None,
) -> GeometryState:
    n_setup = _n_setup_params(geometry)
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
    active_pose_dofs = config.active_pose_dofs if config is not None else _POSE_DOF_ORDER
    if not active_pose_dofs:
        return GeometryState(setup=setup, pose=geometry.pose)
    pose_matrix = np.asarray(params[n_setup:], dtype=np.float64).reshape(
        geometry.pose.n_views,
        len(active_pose_dofs),
    )
    pose_updates = {dof: pose_matrix[:, index] for index, dof in enumerate(active_pose_dofs)}
    return GeometryState(
        setup=setup,
        pose=geometry.pose.with_updates(
            phi_residual_rad=pose_updates.get("phi_residual_rad"),
            dx_px=pose_updates.get("dx_px"),
            dz_px=pose_updates.get("dz_px"),
        ),
    )


def _split_joint(
    geometry: GeometryState,
    params: jax.Array,
    config: JointSchurLMConfig | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    n_setup = _n_setup_params(geometry)
    theta_offset = params[0]
    det_u = params[1]
    det_v = params[2] if geometry.setup.det_v_px.active else jnp.asarray(0.0, dtype=jnp.float32)
    active_pose_dofs = config.active_pose_dofs if config is not None else _POSE_DOF_ORDER
    pose_values = {
        "phi_residual_rad": jnp.asarray(geometry.pose.phi_residual_rad, dtype=jnp.float32),
        "dx_px": jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32),
        "dz_px": jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32),
    }
    if active_pose_dofs:
        pose = params[n_setup:].reshape((geometry.pose.n_views, len(active_pose_dofs)))
        for index, dof in enumerate(active_pose_dofs):
            pose_values[dof] = pose[:, index]
    return (
        theta_offset,
        det_u,
        det_v,
        pose_values["phi_residual_rad"],
        pose_values["dx_px"],
        pose_values["dz_px"],
    )


def _residual_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
    sigma: float,
    fit_gain_offset: bool,
    fit_background_offset: bool,
) -> jax.Array:
    predicted = _predicted_for_params(volume, geometry, params, config=cfg)
    predicted = _with_fitted_nuisance(
        predicted,
        observed,
        mask=mask,
        fit_gain_offset=fit_gain_offset,
        fit_background_offset=fit_background_offset,
    )
    residual = (predicted - observed) / jnp.asarray(sigma, dtype=jnp.float32)
    if mask is None:
        return residual
    return residual * jnp.asarray(mask, dtype=jnp.float32)


def _with_parameter_prior_residual(
    data_residual: jax.Array,
    params: jax.Array,
    *,
    prior_reference: jax.Array,
    cfg: JointSchurLMConfig,
    n_setup: int,
) -> jax.Array:
    setup_strength, pose_strength = _prior_strengths(cfg)
    if setup_strength == 0.0 and pose_strength == 0.0:
        return data_residual
    prior_delta = params - jnp.asarray(prior_reference, dtype=jnp.float32)
    setup_prior = jnp.sqrt(jnp.asarray(setup_strength, dtype=jnp.float32)) * prior_delta[:n_setup]
    pose_prior = jnp.sqrt(jnp.asarray(pose_strength, dtype=jnp.float32)) * prior_delta[n_setup:]
    return jnp.concatenate([data_residual, setup_prior, pose_prior], axis=0)


def _prior_strengths(cfg: JointSchurLMConfig) -> tuple[float, float]:
    base = max(float(cfg.parameter_prior_strength), 0.0)
    setup = base if cfg.setup_prior_strength is None else max(float(cfg.setup_prior_strength), 0.0)
    pose = base if cfg.pose_prior_strength is None else max(float(cfg.pose_prior_strength), 0.0)
    return setup, pose


def _parameter_prior_loss(
    params: jax.Array,
    *,
    prior_reference: jax.Array,
    cfg: JointSchurLMConfig,
    n_setup: int,
) -> jax.Array:
    setup_strength, pose_strength = _prior_strengths(cfg)
    if setup_strength == pose_strength:
        prior_delta = params - jnp.asarray(prior_reference, dtype=jnp.float32)
        return (
            0.5
            * jnp.asarray(setup_strength, dtype=jnp.float32)
            * jnp.mean(prior_delta * prior_delta)
        )
    prior_delta = params - jnp.asarray(prior_reference, dtype=jnp.float32)
    setup_delta = prior_delta[:n_setup]
    pose_delta = prior_delta[n_setup:]
    setup_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if setup_strength > 0.0:
        setup_loss = (
            0.5
            * jnp.asarray(setup_strength, dtype=jnp.float32)
            * jnp.mean(setup_delta * setup_delta)
        )
    pose_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if pose_strength > 0.0:
        pose_loss = (
            0.5 * jnp.asarray(pose_strength, dtype=jnp.float32) * jnp.mean(pose_delta * pose_delta)
        )
    return setup_loss + pose_loss


def _predicted_for_params(
    volume: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    config: JointSchurLMConfig | None = None,
) -> jax.Array:
    theta_offset, det_u, det_v, phi_pose, dx_pose, dz_pose = _split_joint(
        geometry,
        params,
        config,
    )
    return project_parallel_reference_arrays(
        volume,
        theta_rad=(
            jnp.asarray(geometry.setup.theta_scale.value, dtype=jnp.float32)
            * jnp.asarray(geometry.pose.theta_nominal_rad, dtype=jnp.float32)
            + theta_offset
            + phi_pose
        ),
        dx_px=det_u + dx_pose,
        dz_px=det_v + dz_pose,
    )


def _with_fitted_nuisance(
    predicted: jax.Array,
    observed: jax.Array,
    *,
    mask: jax.Array | None,
    fit_gain_offset: bool,
    fit_background_offset: bool,
) -> jax.Array:
    corrected = predicted
    if fit_gain_offset:
        corrected = estimate_gain_offset(corrected, observed, mask=mask).apply(corrected)
    if fit_background_offset:
        corrected = estimate_background_offset(corrected, observed, mask=mask).apply(corrected)
    return corrected


def _nuisance_diagnostics_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
) -> dict[str, dict[str, object] | None]:
    predicted = _predicted_for_params(volume, geometry, params, config=cfg)
    gain_offset_model = None
    background_offset_model = None
    corrected = predicted
    if cfg.fit_gain_offset:
        gain_offset = estimate_gain_offset(corrected, observed, mask=mask)
        gain_offset_model = gain_offset.to_dict()
        corrected = gain_offset.apply(corrected)
    if cfg.fit_background_offset:
        background_offset = estimate_background_offset(corrected, observed, mask=mask)
        background_offset_model = background_offset.to_dict()
    return {
        "gain_offset_model": gain_offset_model,
        "background_offset_model": background_offset_model,
    }


def _loss_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
    prior_reference: jax.Array | None = None,
) -> jax.Array:
    predicted = _predicted_for_params(volume, geometry, params, config=cfg)
    predicted = _with_fitted_nuisance(
        predicted,
        observed,
        mask=mask,
        fit_gain_offset=cfg.fit_gain_offset,
        fit_background_offset=cfg.fit_background_offset,
    )
    data_loss = residual_loss(predicted, observed, mask=mask, sigma=cfg.sigma, delta=cfg.delta).loss
    if prior_reference is None:
        return data_loss
    prior_loss = _parameter_prior_loss(
        params,
        prior_reference=prior_reference,
        cfg=cfg,
        n_setup=_n_setup_params(geometry),
    )
    return data_loss + prior_loss


def _loss_by_view_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
) -> tuple[float, ...]:
    predicted = _predicted_for_params(volume, geometry, params, config=cfg)
    predicted = _with_fitted_nuisance(
        predicted,
        observed,
        mask=mask,
        fit_gain_offset=cfg.fit_gain_offset,
        fit_background_offset=cfg.fit_background_offset,
    )
    losses: list[float] = []
    mask_array = None if mask is None else jnp.asarray(mask, dtype=jnp.float32)
    for view in range(geometry.pose.n_views):
        mask_view = (
            None
            if mask_array is None
            else mask_array[view]
            if mask_array.ndim == observed.ndim
            else mask_array
        )
        loss = residual_loss(
            predicted[view],
            observed[view],
            mask=mask_view,
            sigma=cfg.sigma,
            delta=cfg.delta,
        ).loss
        losses.append(float(loss))
    return tuple(losses)
