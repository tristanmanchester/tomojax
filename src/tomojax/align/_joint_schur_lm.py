"""Joint setup+pose Schur LM reference solver for supported v2 DOFs."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._lm_numerics import finite_difference_jacobian
from tomojax.forward import project_parallel_reference_arrays, pseudo_huber_weights, residual_loss
from tomojax.geometry import CanonicalizedGeometry, GeometryState, canonicalize_geometry_gauges

if TYPE_CHECKING:
    from pathlib import Path

_POSE_DIM = 3


@dataclass(frozen=True)
class JointSchurLMConfig:
    max_iterations: int = 6
    damping: float = 1e-2
    sigma: float = 1.0
    delta: float = 1.0
    finite_difference_step: float = 1e-3


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
    params = _pack_joint(geometry)
    n_setup = _n_setup_params(geometry)
    initial_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
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
    )
    iterations = 0

    for _ in range(max(0, int(cfg.max_iterations))):
        residual = _residual_for_params(vol, obs, geometry, params, mask=mask, sigma=cfg.sigma)
        weights = jnp.sqrt(pseudo_huber_weights(residual, delta=cfg.delta)).reshape(-1)

        def weighted_residual(
            candidate: jax.Array,
            weights_current: jax.Array = weights,
        ) -> jax.Array:
            raw = _residual_for_params(vol, obs, geometry, candidate, mask=mask, sigma=cfg.sigma)
            return raw.reshape(-1) * weights_current

        r = weighted_residual(params)
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
            pose_dim=_POSE_DIM,
            damping=cfg.damping,
        )
        candidate = params + schur.step
        candidate_loss = _loss_for_params(vol, obs, geometry, candidate, mask=mask, cfg=cfg)
        current_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
        accepted = bool(candidate_loss <= current_loss)
        params = candidate if accepted else params
        iterations += 1
        diagnostics = schur.diagnostics
        if float(jnp.linalg.norm(schur.step)) < 1e-5:
            break

    solved = _geometry_with_params(geometry, params)
    canonicalized = canonicalize_geometry_gauges(solved)
    final_loss = _loss_for_params(vol, obs, geometry, params, mask=mask, cfg=cfg)
    return JointSchurLMResult(
        geometry=solved,
        canonicalized_geometry=canonicalized,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        iterations=iterations,
        active_setup_parameters=_active_setup_parameters(geometry),
        active_pose_dofs=("phi_residual_rad", "dx_px", "dz_px"),
        frozen_parameters=_frozen_parameters(geometry),
        diagnostics=diagnostics,
    )


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


def schur_step_from_jacobian(
    jacobian: jax.Array,
    residual: jax.Array,
    *,
    n_setup: int,
    n_views: int,
    pose_dim: int,
    damping: float,
) -> SchurStep:
    """Solve a damped joint normal equation by per-view Schur complement."""
    jac = jnp.asarray(jacobian, dtype=jnp.float32)
    r = jnp.asarray(residual, dtype=jnp.float32)
    n_params = n_setup + n_views * pose_dim
    hessian = jac.T @ jac + jnp.eye(n_params, dtype=jnp.float32) * jnp.asarray(
        damping,
        dtype=jnp.float32,
    )
    gradient = jac.T @ r
    dense_step = jnp.linalg.solve(hessian, -gradient)

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
        pose_blocks.append((h_pp, h_pg, g_p))
        pose_block_conditions.append(float(jnp.linalg.cond(h_pp)))

    setup_step = jnp.linalg.solve(schur_matrix, schur_rhs)
    pose_steps: list[jax.Array] = []
    for h_pp, h_pg, g_p in pose_blocks:
        pose_steps.append(jnp.linalg.solve(h_pp, -g_p - h_pg @ setup_step))
    pose_step = (
        jnp.concatenate(pose_steps, axis=0) if pose_steps else jnp.asarray([], dtype=jnp.float32)
    )
    step = jnp.concatenate([setup_step, pose_step], axis=0)
    schur_eigenvalues = _eigvalsh_tuple(schur_matrix)
    diagnostics = JointSchurDiagnostics(
        schur_condition=float(jnp.linalg.cond(schur_matrix)),
        setup_update_norm=float(jnp.linalg.norm(setup_step)),
        pose_update_norm=float(jnp.linalg.norm(pose_step)),
        dense_step_difference_norm=float(jnp.linalg.norm(step - dense_step)),
        global_eigenvalues=_eigvalsh_tuple(hessian),
        schur_eigenvalues=schur_eigenvalues,
        pose_block_conditions=tuple(pose_block_conditions),
        setup_correlation_matrix=_correlation_matrix_tuple(schur_matrix),
        weak_mode_labels=_weak_mode_labels(schur_eigenvalues),
    )
    return SchurStep(step=step, dense_step=dense_step, diagnostics=diagnostics)


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


def _frozen_parameters(geometry: GeometryState) -> tuple[str, ...]:
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
    return tuple(frozen)


def _n_setup_params(geometry: GeometryState) -> int:
    return 3 if geometry.setup.det_v_px.active else 2


def _pack_joint(geometry: GeometryState) -> jax.Array:
    setup_values = [geometry.setup.theta_offset_rad.value, geometry.setup.det_u_px.value]
    if geometry.setup.det_v_px.active:
        setup_values.append(geometry.setup.det_v_px.value)
    pose_values = np.stack(
        [geometry.pose.phi_residual_rad, geometry.pose.dx_px, geometry.pose.dz_px],
        axis=1,
    ).reshape(-1)
    return jnp.asarray([*setup_values, *pose_values], dtype=jnp.float32)


def _geometry_with_params(geometry: GeometryState, params: jax.Array) -> GeometryState:
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
    pose_matrix = np.asarray(params[n_setup:], dtype=np.float64).reshape(
        geometry.pose.n_views,
        _POSE_DIM,
    )
    return GeometryState(
        setup=setup,
        pose=geometry.pose.with_updates(
            phi_residual_rad=pose_matrix[:, 0],
            dx_px=pose_matrix[:, 1],
            dz_px=pose_matrix[:, 2],
        ),
    )


def _split_joint(
    geometry: GeometryState,
    params: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    n_setup = _n_setup_params(geometry)
    theta_offset = params[0]
    det_u = params[1]
    det_v = params[2] if geometry.setup.det_v_px.active else jnp.asarray(0.0, dtype=jnp.float32)
    pose = params[n_setup:].reshape((geometry.pose.n_views, _POSE_DIM))
    return theta_offset, det_u, det_v, pose[:, 0], pose[:, 1], pose[:, 2]


def _residual_for_params(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    params: jax.Array,
    *,
    mask: jax.Array | None,
    sigma: float,
) -> jax.Array:
    theta_offset, det_u, det_v, phi_pose, dx_pose, dz_pose = _split_joint(geometry, params)
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta_offset + phi_pose,
        dx_px=det_u + dx_pose,
        dz_px=det_v + dz_pose,
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
    cfg: JointSchurLMConfig,
) -> jax.Array:
    theta_offset, det_u, det_v, phi_pose, dx_pose, dz_pose = _split_joint(geometry, params)
    predicted = project_parallel_reference_arrays(
        volume,
        theta_rad=theta_offset + phi_pose,
        dx_px=det_u + dx_pose,
        dz_px=det_v + dz_pose,
    )
    return residual_loss(predicted, observed, mask=mask, sigma=cfg.sigma, delta=cfg.delta).loss
