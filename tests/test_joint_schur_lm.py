from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from tomojax.align import JointSchurLMConfig, schur_step_from_jacobian, solve_joint_schur_lm
from tomojax.forward import project_parallel_reference
from tomojax.geometry import GeometryState


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

    np.testing.assert_allclose(np.asarray(step.step), np.asarray(step.dense_step), atol=5e-6)
    assert step.diagnostics.dense_step_difference_norm < 7e-6
    assert np.isfinite(step.diagnostics.schur_condition)


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
        config=JointSchurLMConfig(max_iterations=8, damping=1e-3, delta=1.0),
    )

    assert result.final_loss < result.initial_loss
    assert result.active_setup_parameters == ("theta_offset_rad", "det_u_px", "det_v_px")
    assert result.active_pose_dofs == ("phi_residual_rad", "dx_px", "dz_px")
    assert result.diagnostics.dense_step_difference_norm < 2e-4
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


def _theta_asymmetric_volume() -> jnp.ndarray:
    volume = jnp.zeros((7, 7, 7), dtype=jnp.float32)
    volume = volume.at[1, 2, 1].set(1.0)
    volume = volume.at[5, 1, 4].set(0.7)
    volume = volume.at[2, 5, 6].set(0.4)
    return volume.at[4, 4, 2].set(0.2)
