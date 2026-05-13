from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from tomojax.align.api import PoseOnlyLMConfig, solve_pose_only_lm
from tomojax.forward import project_parallel_reference
from tomojax.geometry import AcquisitionParameters, GeometryState


def test_pose_only_lm_recovers_detector_shift_pose_components() -> None:
    volume = _asymmetric_volume()
    nominal = GeometryState.zeros(2)
    setup = nominal.setup.replace_parameter(
        "det_v_px",
        replace(nominal.setup.det_v_px, active=True),
    )
    nominal = GeometryState(setup=setup, pose=nominal.pose)
    true_pose = nominal.pose.with_updates(
        dx_px=np.array([0.35, -0.25], dtype=np.float64),
        dz_px=np.array([-0.20, 0.30], dtype=np.float64),
    )
    truth = GeometryState(setup=setup, pose=true_pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_pose_only_lm(
        volume,
        observed,
        nominal,
        config=PoseOnlyLMConfig(max_iterations=8, damping=1e-3, delta=1.0),
    )

    assert result.final_loss < result.initial_loss
    assert result.active_dofs == ("phi_residual_rad", "dx_px", "dz_px")
    assert result.frozen_dofs == ("alpha_rad", "beta_rad")
    np.testing.assert_allclose(result.geometry.pose.dx_px, true_pose.dx_px, atol=0.08)
    np.testing.assert_allclose(result.geometry.pose.dz_px, true_pose.dz_px, atol=0.08)


def test_pose_only_lm_residual_preserves_laminography_acquisition() -> None:
    volume = _asymmetric_volume()
    nominal = GeometryState.zeros(2)
    nominal = GeometryState(
        setup=nominal.setup,
        pose=nominal.pose.with_updates(
            theta_nominal_rad=np.array([0.0, np.pi / 3.0], dtype=np.float64),
        ),
        acquisition=AcquisitionParameters.parallel_laminography(
            tilt_rad=float(np.deg2rad(30.0)),
        ),
    )
    observed = project_parallel_reference(volume, nominal)

    result = solve_pose_only_lm(
        volume,
        observed,
        nominal,
        config=PoseOnlyLMConfig(max_iterations=0),
    )

    assert result.initial_loss < 1.0e-10


def test_pose_only_lm_canonicalizes_solved_gauges() -> None:
    volume = _asymmetric_volume()
    nominal = GeometryState.zeros(2)
    true_pose = nominal.pose.with_updates(dx_px=np.array([0.2, 0.4], dtype=np.float64))
    truth = GeometryState(setup=nominal.setup, pose=true_pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_pose_only_lm(
        volume,
        observed,
        nominal,
        config=PoseOnlyLMConfig(max_iterations=8, damping=1e-3),
    )

    np.testing.assert_allclose(
        np.mean(result.canonicalized_geometry.state.pose.dx_px),
        0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.canonicalized_geometry.state.setup.det_u_px.value
        + result.canonicalized_geometry.state.pose.dx_px,
        result.geometry.setup.det_u_px.value + result.geometry.pose.dx_px,
        atol=1e-8,
    )


def test_pose_only_lm_recovers_phi_residual_pose_component() -> None:
    volume = _theta_asymmetric_volume()
    nominal = GeometryState.zeros(2)
    true_pose = nominal.pose.with_updates(
        phi_residual_rad=np.array([0.08, -0.05], dtype=np.float64),
    )
    truth = GeometryState(setup=nominal.setup, pose=true_pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_pose_only_lm(
        volume,
        observed,
        nominal,
        config=PoseOnlyLMConfig(max_iterations=10, damping=1e-3, delta=1.0),
    )

    assert result.final_loss < result.initial_loss
    np.testing.assert_allclose(
        result.geometry.pose.phi_residual_rad,
        true_pose.phi_residual_rad,
        atol=0.025,
    )


def test_pose_only_lm_recovers_alpha_beta_pose_components() -> None:
    volume = _roll_asymmetric_volume()
    nominal = GeometryState.zeros(2)
    nominal = GeometryState(
        setup=nominal.setup,
        pose=nominal.pose.with_updates(
            theta_nominal_rad=np.array([0.0, np.pi / 2.0], dtype=np.float64),
        ),
    )
    true_pose = nominal.pose.with_updates(
        alpha_rad=np.array([0.08, -0.06], dtype=np.float64),
        beta_rad=np.array([-0.05, 0.07], dtype=np.float64),
    )
    truth = GeometryState(setup=nominal.setup, pose=true_pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_pose_only_lm(
        volume,
        observed,
        nominal,
        config=PoseOnlyLMConfig(
            max_iterations=12,
            damping=1e-3,
            delta=1.0,
            finite_difference_step=1e-2,
            active_pose_dofs=("alpha_rad", "beta_rad"),
        ),
    )

    assert result.final_loss < result.initial_loss
    assert result.active_dofs == ("alpha_rad", "beta_rad")
    assert result.frozen_dofs == ("phi_residual_rad", "dx_px", "dz_px")
    np.testing.assert_allclose(result.geometry.pose.alpha_rad, true_pose.alpha_rad, atol=0.025)
    np.testing.assert_allclose(result.geometry.pose.beta_rad, true_pose.beta_rad, atol=0.025)


def test_pose_only_lm_canonicalizes_solved_phi_gauge() -> None:
    volume = _theta_asymmetric_volume()
    nominal = GeometryState.zeros(2)
    true_pose = nominal.pose.with_updates(
        phi_residual_rad=np.array([0.04, 0.08], dtype=np.float64),
    )
    truth = GeometryState(setup=nominal.setup, pose=true_pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_pose_only_lm(
        volume,
        observed,
        nominal,
        config=PoseOnlyLMConfig(max_iterations=10, damping=1e-3),
    )

    np.testing.assert_allclose(
        np.mean(result.canonicalized_geometry.state.pose.phi_residual_rad),
        0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.canonicalized_geometry.state.theta_total_rad(),
        result.geometry.theta_total_rad(),
        atol=1e-8,
    )


def _asymmetric_volume() -> jnp.ndarray:
    volume = jnp.zeros((6, 6, 6), dtype=jnp.float32)
    volume = volume.at[1, :, 2].set(1.0)
    volume = volume.at[4, :, 4].set(0.6)
    return volume.at[2, :, 1].set(0.3)


def _theta_asymmetric_volume() -> jnp.ndarray:
    volume = jnp.zeros((7, 7, 7), dtype=jnp.float32)
    volume = volume.at[1, 2, 1].set(1.0)
    volume = volume.at[5, 1, 4].set(0.7)
    volume = volume.at[2, 5, 6].set(0.4)
    return volume.at[4, 4, 2].set(0.2)


def _roll_asymmetric_volume() -> jnp.ndarray:
    volume = jnp.zeros((9, 9, 9), dtype=jnp.float32)
    volume = volume.at[2, :, 6].set(1.0)
    volume = volume.at[6, :, 2].set(0.7)
    volume = volume.at[1, :, 1].set(0.3)
    return volume.at[7, :, 5].set(0.2)
