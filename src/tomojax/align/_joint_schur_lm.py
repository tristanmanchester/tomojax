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

from tomojax.forward import (
    ResidualFilterConfig,
    apply_residual_filter_schedule,
    nominal_axis_unit_from_geometry,
    project_parallel_reference_arrays,
    pseudo_huber_loss,
    pseudo_huber_weights,
    residual_loss,
)
from tomojax.geometry import CanonicalizedGeometry, GeometryState, canonicalize_geometry_gauges
from tomojax.nuisance import estimate_background_offset, estimate_gain_offset

if TYPE_CHECKING:
    from pathlib import Path

_POSE_DIM = 5
PoseSchurDof = Literal["alpha_rad", "beta_rad", "phi_residual_rad", "dx_px", "dz_px"]
SetupSchurParameter = Literal[
    "axis_rot_x_rad",
    "axis_rot_y_rad",
    "det_u_px",
    "det_v_px",
    "detector_roll_rad",
    "theta_offset_rad",
    "theta_scale",
]
_POSE_DOF_ORDER: tuple[PoseSchurDof, ...] = (
    "alpha_rad",
    "beta_rad",
    "phi_residual_rad",
    "dx_px",
    "dz_px",
)
_FULL_STACK_LOSS_MAX_ELEMENTS = 4_000_000
_SETUP_PARAMETER_ORDER: tuple[SetupSchurParameter, ...] = (
    "theta_offset_rad",
    "det_u_px",
    "det_v_px",
    "detector_roll_rad",
    "axis_rot_x_rad",
    "axis_rot_y_rad",
    "theta_scale",
)
_NormalEquationArrayDiagnostics = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


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
    loss_mode: str = "pseudo_huber"
    finite_difference_step: float = 1e-3
    setup_trust_radius: float | None = None
    pose_trust_radius: float | None = None
    parameter_prior_strength: float = 0.0
    setup_prior_strength: float | None = None
    pose_prior_strength: float | None = None
    active_setup_parameters: tuple[SetupSchurParameter, ...] | None = None
    active_pose_dofs: tuple[PoseSchurDof, ...] = ("phi_residual_rad", "dx_px", "dz_px")
    residual_filters: tuple[ResidualFilterConfig, ...] = (ResidualFilterConfig(),)
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
    residual_filter_kinds: tuple[str, ...] = ()

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
            "residual_filter_kinds": list(self.residual_filter_kinds),
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


def solve_joint_schur_lm(
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
    n_setup = _n_setup_params(geometry, cfg)
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
    use_jitted_stream = not (
        (cfg.fit_gain_offset or cfg.fit_background_offset)
        and pose_dim == 0
        and int(obs.size) <= _FULL_STACK_LOSS_MAX_ELEMENTS
    )

    def normal_equation_arrays(
        params_arg: jax.Array,
        prior_reference_arg: jax.Array,
        damping_arg: jax.Array,
        setup_prior_strength_arg: jax.Array,
        pose_prior_strength_arg: jax.Array,
    ) -> tuple[jax.Array, jax.Array, _NormalEquationArrayDiagnostics]:
        return _stream_joint_normal_equation_arrays_for_geometry(
            vol,
            obs,
            geometry,
            params_arg,
            mask=mask,
            cfg=cfg,
            n_setup=n_setup,
            pose_dim=pose_dim,
            damping=damping_arg,
            setup_prior_strength=setup_prior_strength_arg,
            pose_prior_strength=pose_prior_strength_arg,
            prior_reference=prior_reference_arg,
        )

    normal_equation_arrays_jit = jax.jit(normal_equation_arrays)

    for _ in range(max(0, int(cfg.max_iterations))):
        setup_prior_strength, pose_prior_strength = _prior_strengths(cfg)
        if use_jitted_stream:
            hessian, gradient, per_view_arrays = normal_equation_arrays_jit(
                params,
                prior_reference,
                jnp.asarray(damping, dtype=jnp.float32),
                jnp.asarray(setup_prior_strength, dtype=jnp.float32),
                jnp.asarray(pose_prior_strength, dtype=jnp.float32),
            )
            equations = _normal_equations_from_arrays(
                hessian=hessian,
                gradient=gradient,
                per_view_arrays=per_view_arrays,
                data_rows=int(obs.size),
            )
        else:
            equations = _stream_joint_normal_equations_for_geometry(
                vol,
                obs,
                geometry,
                params,
                mask=mask,
                cfg=cfg,
                n_setup=n_setup,
                pose_dim=pose_dim,
                damping=damping,
                setup_prior_strength=setup_prior_strength,
                pose_prior_strength=pose_prior_strength,
                prior_reference=prior_reference,
            )
        schur = schur_step_from_normal_equations(
            equations.hessian,
            equations.gradient,
            per_view_blocks=equations.per_view_blocks,
            n_setup=n_setup,
            n_views=geometry.pose.n_views,
            pose_dim=pose_dim,
            active_setup_parameters=_active_setup_parameters(geometry, cfg),
            active_pose_dofs=cfg.active_pose_dofs,
            canonicalize_pose_step_gauge=True,
            setup_trust_radius=setup_trust_radius,
            pose_trust_radius=pose_trust_radius,
            data_rows=equations.data_rows,
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
            residual_filter_kinds=tuple(config.kind for config in cfg.residual_filters),
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
        active_setup_parameters=_active_setup_parameters(geometry, cfg),
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
class _NormalEquations:
    hessian: jax.Array
    gradient: jax.Array
    data_rows: int
    per_view_blocks: _PerViewNormalBlockDiagnostics


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
    active_setup_parameters: tuple[SetupSchurParameter, ...] | None = None,
    active_pose_dofs: tuple[PoseSchurDof, ...] | None = None,
    canonicalize_pose_step_gauge: bool = False,
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
        if pose_dim == 0:
            continue
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
    if canonicalize_pose_step_gauge:
        setup_step, pose_step = _canonicalize_step_gauge(
            setup_step,
            pose_step,
            n_setup=n_setup,
            n_views=n_views,
            pose_dim=pose_dim,
            active_setup_parameters=active_setup_parameters,
            active_pose_dofs=active_pose_dofs,
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


def schur_step_from_normal_equations(
    hessian: jax.Array,
    gradient: jax.Array,
    *,
    per_view_blocks: _PerViewNormalBlockDiagnostics,
    n_setup: int,
    n_views: int,
    pose_dim: int,
    active_setup_parameters: tuple[SetupSchurParameter, ...] | None = None,
    active_pose_dofs: tuple[PoseSchurDof, ...] | None = None,
    canonicalize_pose_step_gauge: bool = False,
    setup_trust_radius: float | None = None,
    pose_trust_radius: float | None = None,
    data_rows: int | None = None,
) -> SchurStep:
    """Solve a Schur step from accumulated normal equations."""
    hess = jnp.asarray(hessian, dtype=jnp.float32)
    grad = jnp.asarray(gradient, dtype=jnp.float32)
    dense_step_unscaled = jnp.linalg.solve(hess, -grad)
    if n_setup == 0:
        return _pose_only_schur_step_from_normal_equations(
            hess,
            grad,
            dense_step_unscaled=dense_step_unscaled,
            per_view_blocks=per_view_blocks,
            n_views=n_views,
            pose_dim=pose_dim,
            setup_trust_radius=setup_trust_radius,
            pose_trust_radius=pose_trust_radius,
            data_rows=data_rows,
        )

    h_gg = hess[:n_setup, :n_setup]
    g_g = grad[:n_setup]
    schur_matrix = h_gg
    schur_rhs = -g_g
    pose_blocks: list[tuple[jax.Array, jax.Array, jax.Array]] = []
    pose_block_conditions: list[float] = []
    for view in range(n_views):
        if pose_dim == 0:
            continue
        start = n_setup + view * pose_dim
        stop = start + pose_dim
        h_gp = hess[:n_setup, start:stop]
        h_pg = hess[start:stop, :n_setup]
        h_pp = hess[start:stop, start:stop]
        g_p = grad[start:stop]
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
    if canonicalize_pose_step_gauge:
        setup_step, pose_step = _canonicalize_step_gauge(
            setup_step,
            pose_step,
            n_setup=n_setup,
            n_views=n_views,
            pose_dim=pose_dim,
            active_setup_parameters=active_setup_parameters,
            active_pose_dofs=active_pose_dofs,
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
        grad,
        hess,
        step,
        data_rows=data_rows,
    )
    schur_eigenvalues = _eigvalsh_tuple(schur_matrix)
    diagnostics = JointSchurDiagnostics(
        schur_condition=float(jnp.linalg.cond(schur_matrix)),
        setup_update_norm=float(jnp.linalg.norm(setup_step * setup_trust_scale)),
        pose_update_norm=float(jnp.linalg.norm(pose_step * pose_trust_scale)),
        dense_step_difference_norm=float(jnp.linalg.norm(step - dense_step)),
        global_eigenvalues=_eigvalsh_tuple(hess),
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


def _pose_only_schur_step_from_normal_equations(
    hess: jax.Array,
    grad: jax.Array,
    *,
    dense_step_unscaled: jax.Array,
    per_view_blocks: _PerViewNormalBlockDiagnostics,
    n_views: int,
    pose_dim: int,
    setup_trust_radius: float | None,
    pose_trust_radius: float | None,
    data_rows: int | None,
) -> SchurStep:
    pose_blocks: list[tuple[jax.Array, jax.Array]] = []
    pose_block_conditions: list[float] = []
    for view in range(n_views):
        if pose_dim == 0:
            continue
        start = view * pose_dim
        stop = start + pose_dim
        h_pp = hess[start:stop, start:stop]
        g_p = grad[start:stop]
        pose_blocks.append((h_pp, g_p))
        pose_block_conditions.append(float(jnp.linalg.cond(h_pp)))
    pose_step = (
        jnp.concatenate(
            [jnp.linalg.solve(h_pp, -g_p) for h_pp, g_p in pose_blocks],
            axis=0,
        )
        if pose_blocks
        else jnp.asarray([], dtype=jnp.float32)
    )
    setup_step = jnp.asarray([], dtype=jnp.float32)
    setup_trust_scale, pose_trust_scale = _trust_scales(
        setup_step,
        pose_step,
        setup_trust_radius=setup_trust_radius,
        pose_trust_radius=pose_trust_radius,
    )
    step = pose_step * pose_trust_scale
    dense_step = dense_step_unscaled * pose_trust_scale
    pose_step_matrix = (
        step.reshape((n_views, pose_dim))
        if pose_dim > 0
        else jnp.zeros((n_views, 0), dtype=jnp.float32)
    )
    diagnostics = JointSchurDiagnostics(
        schur_condition=1.0,
        setup_update_norm=0.0,
        pose_update_norm=float(jnp.linalg.norm(step)),
        dense_step_difference_norm=float(jnp.linalg.norm(step - dense_step)),
        global_eigenvalues=_eigvalsh_tuple(hess),
        schur_eigenvalues=(),
        pose_block_conditions=tuple(pose_block_conditions),
        setup_correlation_matrix=(),
        weak_mode_labels=(),
        trust_scale=float(pose_trust_scale),
        setup_trust_scale=float(setup_trust_scale),
        pose_trust_scale=float(pose_trust_scale),
        trust_clipped=bool(pose_trust_scale < 1.0),
        setup_update_by_parameter=(),
        pose_update_max_by_dof=tuple(
            float(value) for value in jnp.max(jnp.abs(pose_step_matrix), axis=0)
        ),
        predicted_reduction=_predicted_reduction(
            grad,
            hess,
            step,
            data_rows=data_rows,
        ),
        setup_gradient_by_view=per_view_blocks.setup_gradient_by_view,
        pose_gradient_by_view=per_view_blocks.pose_gradient_by_view,
        setup_hessian_diag_by_view=per_view_blocks.setup_hessian_diag_by_view,
        pose_hessian_diag_by_view=per_view_blocks.pose_hessian_diag_by_view,
        setup_pose_coupling_norm_by_view=per_view_blocks.setup_pose_coupling_norm_by_view,
    )
    return SchurStep(step=step, dense_step=dense_step, diagnostics=diagnostics)


def _stream_joint_normal_equations_for_geometry(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
    n_setup: int,
    pose_dim: int,
    damping: float,
    setup_prior_strength: float,
    pose_prior_strength: float,
    prior_reference: jax.Array,
) -> _NormalEquations:
    if (
        (cfg.fit_gain_offset or cfg.fit_background_offset)
        and pose_dim == 0
        and int(observed.size) <= _FULL_STACK_LOSS_MAX_ELEMENTS
    ):
        residual = _residual_for_params(
            volume,
            observed,
            geometry,
            params,
            mask=mask,
            cfg=cfg,
            sigma=cfg.sigma,
            fit_gain_offset=cfg.fit_gain_offset,
            fit_background_offset=cfg.fit_background_offset,
        )
        weights = jnp.sqrt(_residual_weights(residual, cfg=cfg))
        return _small_stack_setup_normal_equations_for_geometry(
            volume,
            observed,
            geometry,
            params,
            mask=mask,
            cfg=cfg,
            weights=weights,
            n_setup=n_setup,
            damping=damping,
            setup_prior_strength=setup_prior_strength,
            pose_prior_strength=pose_prior_strength,
            prior_reference=prior_reference,
        )

    hessian, gradient, per_view_arrays = _stream_joint_normal_equation_arrays_for_geometry(
        volume,
        observed,
        geometry,
        params,
        mask=mask,
        cfg=cfg,
        n_setup=n_setup,
        pose_dim=pose_dim,
        damping=jnp.asarray(damping, dtype=jnp.float32),
        setup_prior_strength=jnp.asarray(setup_prior_strength, dtype=jnp.float32),
        pose_prior_strength=jnp.asarray(pose_prior_strength, dtype=jnp.float32),
        prior_reference=prior_reference,
    )
    return _normal_equations_from_arrays(
        hessian=hessian,
        gradient=gradient,
        per_view_arrays=per_view_arrays,
        data_rows=int(observed.size),
    )


def _stream_joint_normal_equation_arrays_for_geometry(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
    n_setup: int,
    pose_dim: int,
    damping: jax.Array,
    setup_prior_strength: jax.Array,
    pose_prior_strength: jax.Array,
    prior_reference: jax.Array,
) -> tuple[jax.Array, jax.Array, _NormalEquationArrayDiagnostics]:
    n_views = int(geometry.pose.n_views)
    n_params = n_setup + n_views * pose_dim
    step = jnp.asarray(cfg.finite_difference_step, dtype=jnp.float32)

    def view_contribution(view: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        view_i = jnp.asarray(view, dtype=jnp.int32)
        setup_indices = jnp.arange(n_setup, dtype=jnp.int32)
        pose_indices = (
            n_setup
            + view_i * jnp.asarray(pose_dim, dtype=jnp.int32)
            + jnp.arange(
                pose_dim,
                dtype=jnp.int32,
            )
        )
        local_indices = jnp.concatenate([setup_indices, pose_indices], axis=0)
        directions = jax.nn.one_hot(local_indices, n_params, dtype=jnp.float32)
        weight_view = _dynamic_view_weights_for_params(
            volume,
            observed,
            geometry,
            params,
            view_i,
            mask=mask,
            cfg=cfg,
        )

        def residual_for_direction(direction: jax.Array, sign: jax.Array) -> jax.Array:
            return _weighted_residual_dynamic_view_for_params(
                volume,
                observed,
                geometry,
                params + sign * step * direction,
                view_i,
                mask=mask,
                cfg=cfg,
                weights=weight_view,
            ).reshape(-1)

        def plus_residual(direction: jax.Array) -> jax.Array:
            return residual_for_direction(direction, jnp.float32(1.0))

        def minus_residual(direction: jax.Array) -> jax.Array:
            return residual_for_direction(direction, jnp.float32(-1.0))

        def direction_body(
            jac_rows: jax.Array,
            direction_index: jax.Array,
        ) -> tuple[jax.Array, None]:
            direction = jax.lax.dynamic_slice(
                directions,
                (direction_index, 0),
                (1, n_params),
            )[0]
            column = (plus_residual(direction) - minus_residual(direction)) / (
                jnp.float32(2.0) * step
            )
            return jac_rows.at[direction_index].set(column), None

        jac_rows_init = jnp.zeros(
            (int(directions.shape[0]), int(observed.shape[1]) * int(observed.shape[2])),
            dtype=jnp.float32,
        )
        jac_rows, _ = jax.lax.scan(
            direction_body,
            jac_rows_init,
            jnp.arange(int(directions.shape[0]), dtype=jnp.int32),
        )
        jac_local = jac_rows.T
        residual_view = _weighted_residual_dynamic_view_for_params(
            volume,
            observed,
            geometry,
            params,
            view_i,
            mask=mask,
            cfg=cfg,
            weights=weight_view,
        ).reshape(-1)
        local_gradient = jac_local.T @ residual_view
        local_hessian = jac_local.T @ jac_local
        return local_indices, local_gradient, local_hessian

    def body(
        carry: tuple[jax.Array, jax.Array],
        view: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
        hessian_acc, gradient_acc = carry
        local_indices, local_gradient, local_hessian = view_contribution(view)
        hessian_acc = hessian_acc.at[local_indices[:, None], local_indices[None, :]].add(
            local_hessian
        )
        gradient_acc = gradient_acc.at[local_indices].add(local_gradient)
        setup_gradient = local_gradient[:n_setup]
        setup_hessian = local_hessian[:n_setup, :n_setup]
        pose_gradient = local_gradient[n_setup:]
        pose_hessian = local_hessian[n_setup:, n_setup:]
        setup_pose = local_hessian[:n_setup, n_setup:]
        diagnostics = (
            setup_gradient,
            pose_gradient,
            jnp.diag(setup_hessian),
            jnp.diag(pose_hessian),
            jnp.linalg.norm(setup_pose),
        )
        return (hessian_acc, gradient_acc), diagnostics

    init = (
        jnp.zeros((n_params, n_params), dtype=jnp.float32),
        jnp.zeros((n_params,), dtype=jnp.float32),
    )
    (hessian, gradient), diagnostics = jax.lax.scan(
        body,
        init,
        jnp.arange(n_views, dtype=jnp.int32),
    )
    (
        setup_gradient_by_view_arr,
        pose_gradient_by_view_arr,
        setup_hessian_diag_by_view_arr,
        pose_hessian_diag_by_view_arr,
        setup_pose_coupling_norm_by_view_arr,
    ) = diagnostics

    prior_delta = params - jnp.asarray(prior_reference, dtype=jnp.float32)
    prior_diag = jnp.concatenate(
        [
            jnp.full((n_setup,), setup_prior_strength, dtype=jnp.float32),
            jnp.full((n_params - n_setup,), pose_prior_strength, dtype=jnp.float32),
        ],
        axis=0,
    )
    hessian = hessian + jnp.diag(prior_diag + damping)
    gradient = gradient + prior_diag * prior_delta
    return (
        hessian,
        gradient,
        (
            setup_gradient_by_view_arr,
            pose_gradient_by_view_arr,
            setup_hessian_diag_by_view_arr,
            pose_hessian_diag_by_view_arr,
            setup_pose_coupling_norm_by_view_arr,
        ),
    )


def _normal_equations_from_arrays(
    *,
    hessian: jax.Array,
    gradient: jax.Array,
    per_view_arrays: _NormalEquationArrayDiagnostics,
    data_rows: int,
) -> _NormalEquations:
    (
        setup_gradient_by_view_arr,
        pose_gradient_by_view_arr,
        setup_hessian_diag_by_view_arr,
        pose_hessian_diag_by_view_arr,
        setup_pose_coupling_norm_by_view_arr,
    ) = per_view_arrays
    return _NormalEquations(
        hessian=hessian,
        gradient=gradient,
        data_rows=data_rows,
        per_view_blocks=_PerViewNormalBlockDiagnostics(
            setup_gradient_by_view=_rows_to_tuple(setup_gradient_by_view_arr),
            pose_gradient_by_view=_rows_to_tuple(pose_gradient_by_view_arr),
            setup_hessian_diag_by_view=_rows_to_tuple(setup_hessian_diag_by_view_arr),
            pose_hessian_diag_by_view=_rows_to_tuple(pose_hessian_diag_by_view_arr),
            setup_pose_coupling_norm_by_view=tuple(
                float(value) for value in setup_pose_coupling_norm_by_view_arr
            ),
        ),
    )


def _small_stack_setup_normal_equations_for_geometry(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
    weights: jax.Array,
    n_setup: int,
    damping: float,
    setup_prior_strength: float,
    pose_prior_strength: float,
    prior_reference: jax.Array,
) -> _NormalEquations:
    n_params = n_setup
    step = jnp.asarray(cfg.finite_difference_step, dtype=jnp.float32)
    directions = jnp.eye(n_params, dtype=jnp.float32)
    residual = (
        _residual_for_params(
            volume,
            observed,
            geometry,
            params,
            mask=mask,
            cfg=cfg,
            sigma=cfg.sigma,
            fit_gain_offset=cfg.fit_gain_offset,
            fit_background_offset=cfg.fit_background_offset,
        )
        * weights
    ).reshape(-1)

    def residual_for_direction(direction: jax.Array, sign: jax.Array) -> jax.Array:
        filtered = _residual_for_params(
            volume,
            observed,
            geometry,
            params + sign * step * direction,
            mask=mask,
            cfg=cfg,
            sigma=cfg.sigma,
            fit_gain_offset=cfg.fit_gain_offset,
            fit_background_offset=cfg.fit_background_offset,
        )
        return (filtered * weights).reshape(-1)

    def plus_residual(direction: jax.Array) -> jax.Array:
        return residual_for_direction(direction, jnp.float32(1.0))

    def minus_residual(direction: jax.Array) -> jax.Array:
        return residual_for_direction(direction, jnp.float32(-1.0))

    plus = jax.vmap(plus_residual)(directions)
    minus = jax.vmap(minus_residual)(directions)
    jac = ((plus - minus) / (jnp.float32(2.0) * step)).T
    prior_delta = params - jnp.asarray(prior_reference, dtype=jnp.float32)
    prior_diag = jnp.concatenate(
        [
            jnp.full((n_setup,), float(setup_prior_strength), dtype=jnp.float32),
            jnp.full((n_params - n_setup,), float(pose_prior_strength), dtype=jnp.float32),
        ],
        axis=0,
    )
    hessian = jac.T @ jac + jnp.diag(prior_diag + jnp.asarray(damping, dtype=jnp.float32))
    gradient = jac.T @ residual + prior_diag * prior_delta
    return _NormalEquations(
        hessian=hessian,
        gradient=gradient,
        data_rows=int(observed.size),
        per_view_blocks=_per_view_normal_block_diagnostics(
            jac,
            residual,
            n_setup=n_setup,
            n_views=geometry.pose.n_views,
            pose_dim=0,
        ),
    )


def _rows_to_tuple(values: jax.Array) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in values)


def _dynamic_view_weights_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    view: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
) -> jax.Array:
    residual = _weighted_residual_dynamic_view_for_params(
        volume,
        observed,
        geometry,
        params,
        view,
        mask=mask,
        cfg=cfg,
        weights=jnp.ones(
            (1, int(observed.shape[1]), int(observed.shape[2])),
            dtype=jnp.float32,
        ),
    )
    return jnp.sqrt(_residual_weights(residual, cfg=cfg))


def _weighted_residual_dynamic_view_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    view: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
    weights: jax.Array,
) -> jax.Array:
    predicted = _predicted_dynamic_view_for_params(
        volume,
        observed,
        geometry,
        params,
        view,
        config=cfg,
    )
    view_i = jnp.asarray(view, dtype=jnp.int32)
    obs_view = jax.lax.dynamic_slice(
        observed,
        (view_i, 0, 0),
        (1, int(observed.shape[1]), int(observed.shape[2])),
    )
    mask_view = (
        None
        if mask is None
        else jax.lax.dynamic_slice(
            mask,
            (view_i, 0, 0),
            (1, int(observed.shape[1]), int(observed.shape[2])),
        )
    )
    predicted = _with_fitted_nuisance(
        predicted,
        obs_view,
        mask=mask_view,
        fit_gain_offset=cfg.fit_gain_offset,
        fit_background_offset=cfg.fit_background_offset,
    )
    residual = (predicted - obs_view) / jnp.asarray(cfg.sigma, dtype=jnp.float32)
    filtered = apply_residual_filter_schedule(
        residual,
        cfg.residual_filters,
        mask=mask_view,
    ).residual
    weight_view = jax.lax.dynamic_slice(
        weights,
        (view_i, 0, 0),
        (1, int(observed.shape[1]), int(observed.shape[2])),
    )
    return filtered * weight_view


def _residual_weights(residual: jax.Array, *, cfg: JointSchurLMConfig) -> jax.Array:
    if cfg.loss_mode == "l2":
        return jnp.ones_like(residual, dtype=jnp.float32)
    return pseudo_huber_weights(residual, delta=cfg.delta)


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


def _canonicalize_step_gauge(
    setup_step: jax.Array,
    pose_step: jax.Array,
    *,
    n_setup: int,
    n_views: int,
    pose_dim: int,
    active_setup_parameters: tuple[SetupSchurParameter, ...] | None,
    active_pose_dofs: tuple[PoseSchurDof, ...] | None,
) -> tuple[jax.Array, jax.Array]:
    if pose_dim == 0:
        return setup_step, pose_step
    dofs = active_pose_dofs or _POSE_DOF_ORDER[:pose_dim]
    setup_parameters = active_setup_parameters or _SETUP_PARAMETER_ORDER[:n_setup]
    setup_index = {parameter: index for index, parameter in enumerate(setup_parameters)}
    pose_matrix = pose_step.reshape((n_views, pose_dim))
    setup = setup_step
    pose = pose_matrix
    for index, dof in enumerate(dofs):
        mean_step = jnp.mean(pose[:, index])
        pose = pose.at[:, index].add(-mean_step)
        target = _setup_gauge_target(dof)
        target_index = None if target is None else setup_index.get(target)
        if target_index is not None:
            setup = setup.at[target_index].add(mean_step)
    return setup, pose.reshape(-1)


def _setup_gauge_target(dof: PoseSchurDof) -> SetupSchurParameter | None:
    if dof == "phi_residual_rad":
        return "theta_offset_rad"
    if dof == "dx_px":
        return "det_u_px"
    if dof == "dz_px":
        return "det_v_px"
    return None


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


def _active_setup_parameters(
    geometry: GeometryState,
    config: JointSchurLMConfig | None = None,
) -> tuple[SetupSchurParameter, ...]:
    if config is not None and config.active_setup_parameters is not None:
        _validate_active_setup_parameters(config.active_setup_parameters, geometry=geometry)
        return config.active_setup_parameters
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


def _frozen_parameters(
    geometry: GeometryState,
    config: JointSchurLMConfig | None = None,
) -> tuple[str, ...]:
    frozen: list[str] = []
    if not geometry.setup.det_v_px.active:
        frozen.append("det_v_px")
    active_setup = _active_setup_parameters(geometry, config)
    frozen.extend(
        parameter
        for parameter in _SETUP_PARAMETER_ORDER
        if parameter not in active_setup and parameter not in frozen
    )
    active_pose = _POSE_DOF_ORDER if config is None else config.active_pose_dofs
    frozen.extend(dof for dof in _POSE_DOF_ORDER if dof not in active_pose)
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


def _validate_active_setup_parameters(
    active_setup_parameters: tuple[SetupSchurParameter, ...],
    *,
    geometry: GeometryState,
) -> None:
    if len(set(active_setup_parameters)) != len(active_setup_parameters):
        raise ValueError("JointSchurLMConfig.active_setup_parameters must not contain duplicates")
    unknown = tuple(
        parameter
        for parameter in active_setup_parameters
        if parameter not in _SETUP_PARAMETER_ORDER
    )
    if unknown:
        raise ValueError(f"unknown active setup parameters {unknown!r}")
    if "det_v_px" in active_setup_parameters and not geometry.setup.det_v_px.active:
        raise ValueError("det_v_px cannot be active when geometry det_v_px is inactive")


def _n_setup_params(
    geometry: GeometryState,
    config: JointSchurLMConfig | None = None,
) -> int:
    return len(_active_setup_parameters(geometry, config))


def _pack_joint(geometry: GeometryState, config: JointSchurLMConfig | None = None) -> jax.Array:
    setup_values = [
        _setup_parameter_values(geometry)[parameter]
        for parameter in _active_setup_parameters(geometry, config)
    ]
    if config is not None and _pose_dim(config) == 0:
        return jnp.asarray(setup_values, dtype=jnp.float32)
    active_pose = _active_pose_arrays(
        geometry,
        config.active_pose_dofs if config else _POSE_DOF_ORDER,
    )
    pose_values = np.stack(active_pose, axis=1).reshape(-1)
    return jnp.asarray([*setup_values, *pose_values], dtype=jnp.float32)


def _setup_parameter_values(geometry: GeometryState) -> dict[SetupSchurParameter, float]:
    return {
        "theta_offset_rad": geometry.setup.theta_offset_rad.value,
        "det_u_px": geometry.setup.det_u_px.value,
        "det_v_px": geometry.setup.det_v_px.value if geometry.setup.det_v_px.active else 0.0,
        "detector_roll_rad": geometry.setup.detector_roll_rad.value,
        "axis_rot_x_rad": geometry.setup.axis_rot_x_rad.value,
        "axis_rot_y_rad": geometry.setup.axis_rot_y_rad.value,
        "theta_scale": geometry.setup.theta_scale.value,
    }


def _active_pose_arrays(
    geometry: GeometryState,
    active_pose_dofs: tuple[PoseSchurDof, ...],
) -> tuple[np.ndarray, ...]:
    arrays = {
        "alpha_rad": geometry.pose.alpha_rad,
        "beta_rad": geometry.pose.beta_rad,
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
    active_setup = _active_setup_parameters(geometry, config)
    n_setup = len(active_setup)
    setup_values = _setup_parameter_values(geometry)
    for index, parameter in enumerate(active_setup):
        setup_values[parameter] = float(params[index])
    setup = geometry.setup
    setup = setup.replace_parameter(
        "theta_offset_rad",
        geometry.setup.theta_offset_rad.with_value(setup_values["theta_offset_rad"]),
    )
    setup = setup.replace_parameter(
        "det_u_px",
        geometry.setup.det_u_px.with_value(setup_values["det_u_px"]),
    )
    if geometry.setup.det_v_px.active:
        setup = setup.replace_parameter(
            "det_v_px",
            geometry.setup.det_v_px.with_value(setup_values["det_v_px"]),
        )
    setup = setup.replace_parameter(
        "detector_roll_rad",
        geometry.setup.detector_roll_rad.with_value(setup_values["detector_roll_rad"]),
    )
    setup = setup.replace_parameter(
        "axis_rot_x_rad",
        geometry.setup.axis_rot_x_rad.with_value(setup_values["axis_rot_x_rad"]),
    )
    setup = setup.replace_parameter(
        "axis_rot_y_rad",
        geometry.setup.axis_rot_y_rad.with_value(setup_values["axis_rot_y_rad"]),
    )
    setup = setup.replace_parameter(
        "theta_scale",
        geometry.setup.theta_scale.with_value(setup_values["theta_scale"]),
    )
    active_pose_dofs = config.active_pose_dofs if config is not None else _POSE_DOF_ORDER
    if not active_pose_dofs:
        return GeometryState(setup=setup, pose=geometry.pose, acquisition=geometry.acquisition)
    pose_matrix = np.asarray(params[n_setup:], dtype=np.float64).reshape(
        geometry.pose.n_views,
        len(active_pose_dofs),
    )
    pose_updates = {dof: pose_matrix[:, index] for index, dof in enumerate(active_pose_dofs)}
    return GeometryState(
        setup=setup,
        pose=geometry.pose.with_updates(
            alpha_rad=pose_updates.get("alpha_rad"),
            beta_rad=pose_updates.get("beta_rad"),
            phi_residual_rad=pose_updates.get("phi_residual_rad"),
            dx_px=pose_updates.get("dx_px"),
            dz_px=pose_updates.get("dz_px"),
        ),
        acquisition=geometry.acquisition,
    )


def _split_joint(
    geometry: GeometryState,
    params: jax.Array,
    config: JointSchurLMConfig | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    active_setup = _active_setup_parameters(geometry, config)
    n_setup = len(active_setup)
    setup_values = {
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
    for index, parameter in enumerate(active_setup):
        setup_values[parameter] = params[index]
    active_pose_dofs = config.active_pose_dofs if config is not None else _POSE_DOF_ORDER
    pose_values = {
        "alpha_rad": jnp.asarray(geometry.pose.alpha_rad, dtype=jnp.float32),
        "beta_rad": jnp.asarray(geometry.pose.beta_rad, dtype=jnp.float32),
        "phi_residual_rad": jnp.asarray(geometry.pose.phi_residual_rad, dtype=jnp.float32),
        "dx_px": jnp.asarray(geometry.pose.dx_px, dtype=jnp.float32),
        "dz_px": jnp.asarray(geometry.pose.dz_px, dtype=jnp.float32),
    }
    if active_pose_dofs:
        pose = params[n_setup:].reshape((geometry.pose.n_views, len(active_pose_dofs)))
        for index, dof in enumerate(active_pose_dofs):
            pose_values[dof] = pose[:, index]
    return (
        setup_values["theta_offset_rad"],
        setup_values["det_u_px"],
        setup_values["det_v_px"],
        setup_values["detector_roll_rad"],
        setup_values["axis_rot_x_rad"],
        setup_values["axis_rot_y_rad"],
        setup_values["theta_scale"],
        pose_values["alpha_rad"],
        pose_values["beta_rad"],
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
    if fit_gain_offset or fit_background_offset:
        predicted = _predicted_for_params(volume, geometry, params, config=cfg)
        predicted = _with_fitted_nuisance(
            predicted,
            observed,
            mask=mask,
            fit_gain_offset=fit_gain_offset,
            fit_background_offset=fit_background_offset,
        )
        residual = (predicted - observed) / jnp.asarray(sigma, dtype=jnp.float32)
        return apply_residual_filter_schedule(
            residual,
            cfg.residual_filters,
            mask=mask,
        ).residual

    n_views = int(geometry.pose.n_views)
    obs = jnp.asarray(observed, dtype=jnp.float32)

    def body(out: jax.Array, view: jax.Array) -> tuple[jax.Array, None]:
        view_i = jnp.asarray(view, dtype=jnp.int32)
        predicted = _predicted_dynamic_view_for_params(
            volume,
            observed,
            geometry,
            params,
            view_i,
            config=cfg,
        )
        obs_view = jax.lax.dynamic_slice(
            obs,
            (view_i, 0, 0),
            (1, int(observed.shape[1]), int(observed.shape[2])),
        )
        mask_view = (
            None
            if mask is None
            else jax.lax.dynamic_slice(
                mask,
                (view_i, 0, 0),
                (1, int(observed.shape[1]), int(observed.shape[2])),
            )
        )
        residual = (predicted - obs_view) / jnp.asarray(sigma, dtype=jnp.float32)
        filtered = apply_residual_filter_schedule(
            residual,
            cfg.residual_filters,
            mask=mask_view,
        ).residual
        return out.at[view_i].set(filtered[0]), None

    init = jnp.zeros_like(obs, dtype=jnp.float32)
    residual, _ = jax.lax.scan(body, init, jnp.arange(n_views, dtype=jnp.int32))
    return residual


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
    (
        theta_offset,
        det_u,
        det_v,
        detector_roll,
        axis_x,
        axis_y,
        theta_scale,
        alpha_pose,
        beta_pose,
        phi_pose,
        dx_pose,
        dz_pose,
    ) = _split_joint(
        geometry,
        params,
        config,
    )
    return project_parallel_reference_arrays(
        volume,
        theta_rad=(
            theta_scale * jnp.asarray(geometry.pose.theta_nominal_rad, dtype=jnp.float32)
            + theta_offset
            + phi_pose
        ),
        dx_px=det_u + dx_pose,
        dz_px=det_v + dz_pose,
        alpha_rad=alpha_pose,
        beta_rad=beta_pose,
        detector_roll_rad=detector_roll,
        axis_rot_x_rad=axis_x,
        axis_rot_y_rad=axis_y,
        nominal_axis_unit=nominal_axis_unit_from_geometry(geometry),
        acquisition_model=geometry.acquisition.model,
        laminography_tilt_rad=geometry.acquisition.laminography_tilt_rad,
        laminography_tilt_about=geometry.acquisition.laminography_tilt_about,
    )


def _predicted_dynamic_view_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    view: jax.Array,
    *,
    config: JointSchurLMConfig | None = None,
) -> jax.Array:
    (
        theta_offset,
        det_u,
        det_v,
        detector_roll,
        axis_x,
        axis_y,
        theta_scale,
        alpha_pose,
        beta_pose,
        phi_pose,
        dx_pose,
        dz_pose,
    ) = _split_joint(
        geometry,
        params,
        config,
    )
    view_i = jnp.asarray(view, dtype=jnp.int32)
    theta_nominal = jax.lax.dynamic_slice(
        jnp.asarray(geometry.pose.theta_nominal_rad, dtype=jnp.float32),
        (view_i,),
        (1,),
    )
    return project_parallel_reference_arrays(
        volume,
        theta_rad=theta_scale * theta_nominal
        + theta_offset
        + jax.lax.dynamic_slice(phi_pose, (view_i,), (1,)),
        dx_px=det_u + jax.lax.dynamic_slice(dx_pose, (view_i,), (1,)),
        dz_px=det_v + jax.lax.dynamic_slice(dz_pose, (view_i,), (1,)),
        alpha_rad=jax.lax.dynamic_slice(alpha_pose, (view_i,), (1,)),
        beta_rad=jax.lax.dynamic_slice(beta_pose, (view_i,), (1,)),
        detector_roll_rad=detector_roll,
        axis_rot_x_rad=axis_x,
        axis_rot_y_rad=axis_y,
        nominal_axis_unit=nominal_axis_unit_from_geometry(geometry),
        acquisition_model=geometry.acquisition.model,
        laminography_tilt_rad=geometry.acquisition.laminography_tilt_rad,
        laminography_tilt_about=geometry.acquisition.laminography_tilt_about,
        detector_shape=(int(observed.shape[1]), int(observed.shape[2])),
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
    if not cfg.fit_gain_offset and not cfg.fit_background_offset:
        return {
            "gain_offset_model": None,
            "background_offset_model": None,
        }
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
    if cfg.fit_gain_offset or cfg.fit_background_offset:
        predicted = _predicted_for_params(volume, geometry, params, config=cfg)
        predicted = _with_fitted_nuisance(
            predicted,
            observed,
            mask=mask,
            fit_gain_offset=cfg.fit_gain_offset,
            fit_background_offset=cfg.fit_background_offset,
        )
        residual = apply_residual_filter_schedule(
            (predicted - observed) / jnp.asarray(cfg.sigma, dtype=jnp.float32),
            cfg.residual_filters,
            mask=mask,
        ).residual
        data_loss = residual_loss(
            residual,
            jnp.zeros_like(residual),
            mask=None,
            sigma=1.0,
            delta=cfg.delta,
            mode="l2" if cfg.loss_mode == "l2" else "pseudo_huber",
        ).loss
    else:
        data_loss = _loss_for_params_streamed(
            volume,
            observed,
            geometry,
            params,
            mask=mask,
            cfg=cfg,
        )
    if prior_reference is None:
        return data_loss
    prior_loss = _parameter_prior_loss(
        params,
        prior_reference=prior_reference,
        cfg=cfg,
        n_setup=_n_setup_params(geometry, cfg),
    )
    return data_loss + prior_loss


def _loss_for_params_streamed(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
) -> jax.Array:
    n_views = int(geometry.pose.n_views)
    obs = jnp.asarray(observed, dtype=jnp.float32)
    normalizer = jnp.asarray(obs.size, dtype=jnp.float32)

    def body(loss_acc: jax.Array, view: jax.Array) -> tuple[jax.Array, None]:
        view_i = jnp.asarray(view, dtype=jnp.int32)
        predicted = _predicted_dynamic_view_for_params(
            volume,
            observed,
            geometry,
            params,
            view_i,
            config=cfg,
        )
        obs_view = jax.lax.dynamic_slice(
            obs,
            (view_i, 0, 0),
            (1, int(observed.shape[1]), int(observed.shape[2])),
        )
        mask_view = (
            None
            if mask is None
            else jax.lax.dynamic_slice(
                mask,
                (view_i, 0, 0),
                (1, int(observed.shape[1]), int(observed.shape[2])),
            )
        )
        filtered = apply_residual_filter_schedule(
            (predicted - obs_view) / jnp.asarray(cfg.sigma, dtype=jnp.float32),
            cfg.residual_filters,
            mask=mask_view,
        ).residual
        loss_map = (
            jnp.asarray(0.5, dtype=jnp.float32) * filtered * filtered
            if cfg.loss_mode == "l2"
            else pseudo_huber_loss(filtered, delta=cfg.delta)
        )
        return loss_acc + jnp.sum(loss_map) / jnp.maximum(normalizer, 1.0), None

    loss, _ = jax.lax.scan(
        body,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.arange(n_views, dtype=jnp.int32),
    )
    return loss


def _loss_by_view_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    cfg: JointSchurLMConfig,
) -> tuple[float, ...]:
    losses: list[float] = []
    for view in range(geometry.pose.n_views):
        if cfg.fit_gain_offset or cfg.fit_background_offset:
            predicted = _predicted_for_params(volume, geometry, params, config=cfg)
            predicted = _with_fitted_nuisance(
                predicted,
                observed,
                mask=mask,
                fit_gain_offset=cfg.fit_gain_offset,
                fit_background_offset=cfg.fit_background_offset,
            )
            filtered = apply_residual_filter_schedule(
                (predicted - observed) / jnp.asarray(cfg.sigma, dtype=jnp.float32),
                cfg.residual_filters,
                mask=mask,
            ).residual
            filtered_view = filtered[view]
        else:
            view_i = jnp.asarray(view, dtype=jnp.int32)
            predicted_view = _predicted_dynamic_view_for_params(
                volume,
                observed,
                geometry,
                params,
                view_i,
                config=cfg,
            )
            obs_view = jax.lax.dynamic_slice(
                jnp.asarray(observed, dtype=jnp.float32),
                (view_i, 0, 0),
                (1, int(observed.shape[1]), int(observed.shape[2])),
            )
            mask_view = (
                None
                if mask is None
                else jax.lax.dynamic_slice(
                    mask,
                    (view_i, 0, 0),
                    (1, int(observed.shape[1]), int(observed.shape[2])),
                )
            )
            filtered_view = apply_residual_filter_schedule(
                (predicted_view - obs_view) / jnp.asarray(cfg.sigma, dtype=jnp.float32),
                cfg.residual_filters,
                mask=mask_view,
            ).residual[0]
        loss = residual_loss(
            filtered_view,
            jnp.zeros_like(filtered_view),
            mask=None,
            sigma=1.0,
            delta=cfg.delta,
            mode="l2" if cfg.loss_mode == "l2" else "pseudo_huber",
        ).loss
        losses.append(float(loss))
    return tuple(losses)
