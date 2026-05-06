from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
from dataclasses import replace
import json
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from tomojax.align import (
    JointSchurLMConfig,
    adapt_joint_schur_damping,
    adapt_joint_schur_trust_radius,
    joint_schur_normal_eq_summary,
    schur_step_from_jacobian,
    solve_joint_schur_lm,
    write_joint_schur_normal_eq_summary,
)
from tomojax.forward import project_parallel_reference
from tomojax.geometry import GeometryState

if TYPE_CHECKING:
    from pathlib import Path


def test_joint_schur_damping_policy_adapts_and_clamps() -> None:
    config = JointSchurLMConfig(
        damping_decrease_factor=0.25,
        damping_increase_factor=4.0,
        min_damping=0.1,
        max_damping=10.0,
    )

    assert adapt_joint_schur_damping(1.0, accepted=True, config=config) == 0.25
    assert adapt_joint_schur_damping(0.2, accepted=True, config=config) == 0.1
    assert adapt_joint_schur_damping(2.0, accepted=False, config=config) == 8.0
    assert adapt_joint_schur_damping(3.0, accepted=False, config=config) == 10.0

    fixed = JointSchurLMConfig(adapt_damping=False)
    assert adapt_joint_schur_damping(1.0, accepted=True, config=fixed) == 1.0
    assert adapt_joint_schur_damping(1.0, accepted=False, config=fixed) == 1.0


def test_joint_schur_trust_radius_policy_adapts_and_clamps() -> None:
    config = JointSchurLMConfig(
        trust_shrink_ratio=0.25,
        trust_expand_ratio=0.75,
        trust_shrink_factor=0.5,
        trust_expand_factor=3.0,
        min_trust_radius=0.1,
        max_trust_radius=5.0,
    )

    assert (
        adapt_joint_schur_trust_radius(
            2.0,
            accepted=True,
            reduction_ratio=0.1,
            clipped=False,
            config=config,
        )
        == 1.0
    )
    assert (
        adapt_joint_schur_trust_radius(
            0.15,
            accepted=False,
            reduction_ratio=0.5,
            clipped=False,
            config=config,
        )
        == 0.1
    )
    assert (
        adapt_joint_schur_trust_radius(
            2.0,
            accepted=True,
            reduction_ratio=0.9,
            clipped=True,
            config=config,
        )
        == 5.0
    )
    assert (
        adapt_joint_schur_trust_radius(
            2.0,
            accepted=True,
            reduction_ratio=0.9,
            clipped=False,
            config=config,
        )
        == 2.0
    )
    assert (
        adapt_joint_schur_trust_radius(
            None,
            accepted=False,
            reduction_ratio=None,
            clipped=True,
            config=config,
        )
        is None
    )
    fixed = JointSchurLMConfig(adapt_trust_radii=False)
    assert (
        adapt_joint_schur_trust_radius(
            2.0,
            accepted=False,
            reduction_ratio=None,
            clipped=True,
            config=fixed,
        )
        == 2.0
    )


def test_schur_step_matches_dense_normal_solve() -> None:
    jacobian = jnp.asarray(
        [
            [1.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.0],
            [0.2, -0.1, 0.0, 0.0, 0.0, 0.6, 0.2, 0.1],
            [0.1, 0.3, 0.0, 0.0, 0.0, 0.2, 0.5, 0.3],
            [0.0, 0.4, 0.2, 0.3, 0.5, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.4, 0.1, 0.7],
        ],
        dtype=jnp.float32,
    )
    residual = jnp.asarray([0.4, -0.2, 0.1, -0.3, 0.2, -0.1], dtype=jnp.float32)

    step = schur_step_from_jacobian(
        jacobian,
        residual,
        n_setup=2,
        n_views=2,
        pose_dim=3,
        damping=1e-2,
    )
    hessian = jacobian.T @ jacobian + jnp.eye(8, dtype=jnp.float32) * 1e-2
    gradient = jacobian.T @ residual
    predicted = -float(
        jnp.vdot(gradient, step.step).real + 0.5 * jnp.vdot(step.step, hessian @ step.step).real
    )

    np.testing.assert_allclose(np.asarray(step.step), np.asarray(step.dense_step), atol=5e-6)
    assert step.diagnostics.dense_step_difference_norm < 7e-6
    np.testing.assert_allclose(step.diagnostics.predicted_reduction, predicted, atol=1e-7)
    assert step.diagnostics.predicted_reduction > 0.0
    assert np.isfinite(step.diagnostics.schur_condition)
    assert len(step.diagnostics.global_eigenvalues) == 8
    assert len(step.diagnostics.schur_eigenvalues) == 2
    assert len(step.diagnostics.pose_block_conditions) == 2
    assert len(step.diagnostics.setup_gradient_by_view) == 2
    assert len(step.diagnostics.pose_gradient_by_view) == 2
    assert len(step.diagnostics.setup_hessian_diag_by_view) == 2
    assert len(step.diagnostics.pose_hessian_diag_by_view) == 2
    assert len(step.diagnostics.setup_pose_coupling_norm_by_view) == 2
    np.testing.assert_allclose(
        np.asarray(step.diagnostics.setup_gradient_by_view[0]),
        np.asarray(jacobian[:3, :2].T @ residual[:3]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(step.diagnostics.pose_gradient_by_view[1]),
        np.asarray(jacobian[3:, 5:8].T @ residual[3:]),
        atol=1e-6,
    )
    with_prior_rows = schur_step_from_jacobian(
        jnp.concatenate([jacobian, jnp.eye(8, dtype=jnp.float32) * 0.1], axis=0),
        jnp.concatenate([residual, jnp.zeros(8, dtype=jnp.float32)], axis=0),
        n_setup=2,
        n_views=2,
        pose_dim=3,
        damping=1e-2,
        data_rows=int(residual.size),
    )
    np.testing.assert_allclose(
        np.asarray(with_prior_rows.diagnostics.setup_gradient_by_view[0]),
        np.asarray(step.diagnostics.setup_gradient_by_view[0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(with_prior_rows.diagnostics.pose_gradient_by_view[1]),
        np.asarray(step.diagnostics.pose_gradient_by_view[1]),
        atol=1e-6,
    )
    assert len(step.diagnostics.setup_correlation_matrix) == 2
    assert step.diagnostics.trust_scale == 1.0
    assert step.diagnostics.trust_clipped is False
    assert len(step.diagnostics.setup_update_by_parameter) == 2
    assert len(step.diagnostics.pose_update_max_by_dof) == 3
    np.testing.assert_allclose(
        np.diag(np.asarray(step.diagnostics.setup_correlation_matrix)),
        np.ones(2),
        atol=1e-6,
    )


def test_schur_step_applies_setup_trust_radius() -> None:
    jacobian = jnp.asarray(
        [
            [1.0, 0.2, 0.3, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.4, 0.1],
            [0.0, 0.4, 0.2, 0.3, 0.5],
            [0.3, 0.0, 0.4, 0.1, 0.7],
        ],
        dtype=jnp.float32,
    )
    residual = jnp.asarray([0.4, -0.2, 0.2, -0.1], dtype=jnp.float32)

    unrestricted = schur_step_from_jacobian(
        jacobian,
        residual,
        n_setup=2,
        n_views=1,
        pose_dim=3,
        damping=1e-2,
    )
    clipped = schur_step_from_jacobian(
        jacobian,
        residual,
        n_setup=2,
        n_views=1,
        pose_dim=3,
        damping=1e-2,
        setup_trust_radius=0.05,
    )

    assert clipped.diagnostics.trust_clipped is True
    assert 0.0 < clipped.diagnostics.trust_scale < 1.0
    assert clipped.diagnostics.setup_update_norm <= 0.050001
    np.testing.assert_allclose(
        np.asarray(clipped.step),
        np.asarray(unrestricted.step) * clipped.diagnostics.trust_scale,
        atol=1e-6,
    )


def test_joint_schur_lm_recovers_realized_supported_geometry() -> None:
    volume = _theta_asymmetric_volume()
    nominal = GeometryState.zeros(2)
    setup = nominal.setup.replace_parameter(
        "det_v_px",
        replace(nominal.setup.det_v_px, active=True),
    )
    nominal = GeometryState(setup=setup, pose=nominal.pose)
    truth_setup = setup.replace_parameter(
        "theta_offset_rad", setup.theta_offset_rad.with_value(0.04)
    )
    truth_setup = truth_setup.replace_parameter("det_u_px", setup.det_u_px.with_value(0.18))
    truth_setup = truth_setup.replace_parameter("det_v_px", setup.det_v_px.with_value(-0.12))
    truth_pose = nominal.pose.with_updates(
        phi_residual_rad=np.asarray([0.03, -0.02], dtype=np.float64),
        dx_px=np.asarray([0.08, -0.06], dtype=np.float64),
        dz_px=np.asarray([-0.05, 0.07], dtype=np.float64),
    )
    truth = GeometryState(setup=truth_setup, pose=truth_pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_joint_schur_lm(
        volume,
        observed,
        nominal,
        config=JointSchurLMConfig(
            max_iterations=8,
            damping=1e-3,
            delta=1.0,
            parameter_prior_strength=1.0e-3,
        ),
    )

    assert result.final_loss < result.initial_loss
    assert len(result.iteration_diagnostics) == result.iterations
    assert result.iteration_diagnostics[-1] == result.diagnostics
    assert result.active_setup_parameters == ("theta_offset_rad", "det_u_px", "det_v_px")
    assert result.active_pose_dofs == ("phi_residual_rad", "dx_px", "dz_px")
    assert result.diagnostics.dense_step_difference_norm < 2e-4
    assert result.diagnostics.parameter_prior_strength == 1.0e-3
    accepted_diagnostics = [
        diagnostics for diagnostics in result.iteration_diagnostics if diagnostics.accepted
    ]
    assert accepted_diagnostics
    assert accepted_diagnostics[0].next_damping <= accepted_diagnostics[0].damping
    assert accepted_diagnostics[0].predicted_reduction > 0.0
    assert accepted_diagnostics[0].actual_reduction >= 0.0
    accepted_reduction_ratio = accepted_diagnostics[0].reduction_ratio
    if accepted_reduction_ratio is None:
        assert accepted_diagnostics[0].predicted_reduction <= 1e-12
    else:
        assert accepted_reduction_ratio >= 0.0
    assert result.diagnostics.next_setup_trust_radius is None
    assert result.diagnostics.next_pose_trust_radius is None
    assert len(result.diagnostics.current_loss_by_view) == 2
    assert len(result.diagnostics.candidate_loss_by_view) == 2
    assert len(result.diagnostics.actual_reduction_by_view) == 2
    canonical = result.canonicalized_geometry.state
    np.testing.assert_allclose(
        canonical.setup.theta_offset_rad.value + canonical.pose.phi_residual_rad,
        truth.setup.theta_offset_rad.value + truth.pose.phi_residual_rad,
        atol=0.05,
    )
    np.testing.assert_allclose(
        canonical.setup.det_u_px.value + canonical.pose.dx_px,
        truth.setup.det_u_px.value + truth.pose.dx_px,
        atol=0.08,
    )
    np.testing.assert_allclose(
        canonical.setup.det_v_px.value + canonical.pose.dz_px,
        truth.setup.det_v_px.value + truth.pose.dz_px,
        atol=0.08,
    )


def test_joint_schur_writes_normal_eq_summary_artifact(tmp_path: Path) -> None:
    volume = _theta_asymmetric_volume()
    nominal = GeometryState.zeros(1)
    truth_setup = nominal.setup.replace_parameter(
        "theta_offset_rad",
        nominal.setup.theta_offset_rad.with_value(0.04),
    )
    truth = GeometryState(setup=truth_setup, pose=nominal.pose)
    observed = project_parallel_reference(volume, truth)
    result = solve_joint_schur_lm(
        volume,
        observed,
        nominal,
        config=JointSchurLMConfig(
            max_iterations=4,
            damping=1e-3,
            setup_trust_radius=0.05,
            pose_trust_radius=0.05,
            parameter_prior_strength=2.0e-3,
        ),
    )

    summary = joint_schur_normal_eq_summary(result)
    path = write_joint_schur_normal_eq_summary(result, tmp_path / "normal_eq_summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "normal_eq_summary.json"
    assert payload == summary
    assert payload["solver"] == "joint_schur_lm_reference"
    assert len(payload["iteration_diagnostics"]) == result.iterations
    assert payload["iteration_diagnostics"][-1] == payload["diagnostics"]
    assert payload["active_setup_parameters"] == ["theta_offset_rad", "det_u_px"]
    assert payload["active_pose_dofs"] == ["phi_residual_rad", "dx_px", "dz_px"]
    for field in (
        "schur_condition",
        "global_eigenvalues",
        "schur_eigenvalues",
        "pose_block_conditions",
        "setup_correlation_matrix",
        "weak_mode_labels",
        "trust_scale",
        "trust_clipped",
        "setup_update_by_parameter",
        "pose_update_max_by_dof",
        "damping",
        "next_damping",
        "accepted",
        "current_loss",
        "candidate_loss",
        "predicted_reduction",
        "actual_reduction",
        "reduction_ratio",
        "next_setup_trust_radius",
        "next_pose_trust_radius",
        "current_loss_by_view",
        "candidate_loss_by_view",
        "actual_reduction_by_view",
        "setup_gradient_by_view",
        "pose_gradient_by_view",
        "setup_hessian_diag_by_view",
        "pose_hessian_diag_by_view",
        "setup_pose_coupling_norm_by_view",
        "parameter_prior_strength",
    ):
        assert field in payload["diagnostics"]
    assert payload["diagnostics"]["parameter_prior_strength"] == 2.0e-3
    assert payload["diagnostics"]["next_setup_trust_radius"] is not None
    assert payload["diagnostics"]["next_pose_trust_radius"] is not None
    assert len(payload["diagnostics"]["current_loss_by_view"]) == 1
    assert len(payload["diagnostics"]["candidate_loss_by_view"]) == 1
    assert len(payload["diagnostics"]["actual_reduction_by_view"]) == 1
    assert len(payload["diagnostics"]["setup_gradient_by_view"]) == 1
    assert len(payload["diagnostics"]["pose_gradient_by_view"]) == 1
    assert len(payload["diagnostics"]["setup_pose_coupling_norm_by_view"]) == 1
    assert len(payload["diagnostics"]["pose_block_conditions"]) == 1
    assert len(payload["diagnostics"]["setup_correlation_matrix"]) == 2


def _theta_asymmetric_volume() -> jnp.ndarray:
    volume = jnp.zeros((7, 7, 7), dtype=jnp.float32)
    volume = volume.at[1, 2, 1].set(1.0)
    volume = volume.at[5, 1, 4].set(0.7)
    volume = volume.at[2, 5, 6].set(0.4)
    return volume.at[4, 4, 2].set(0.2)
