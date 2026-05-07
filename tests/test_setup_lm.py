from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from tomojax.align import SetupOnlyLMConfig, solve_setup_only_lm
from tomojax.forward import project_parallel_reference
from tomojax.geometry import GeometryState


def test_setup_only_lm_recovers_active_detector_shift_components() -> None:
    volume = _asymmetric_volume()
    nominal = GeometryState.zeros(2)
    setup = nominal.setup.replace_parameter(
        "det_v_px",
        replace(nominal.setup.det_v_px, active=True),
    )
    nominal = GeometryState(setup=setup, pose=nominal.pose)
    truth_setup = setup.replace_parameter("det_u_px", setup.det_u_px.with_value(0.30))
    truth_setup = truth_setup.replace_parameter("det_v_px", setup.det_v_px.with_value(-0.20))
    truth = GeometryState(setup=truth_setup, pose=nominal.pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_setup_only_lm(
        volume,
        observed,
        nominal,
        config=SetupOnlyLMConfig(max_iterations=8, damping=1e-3, delta=1.0),
    )

    assert result.final_loss < result.initial_loss
    assert result.active_parameters == (
        "theta_offset_rad",
        "det_u_px",
        "det_v_px",
        "detector_roll_rad",
    )
    assert result.frozen_parameters == (
        "axis_rot_x_rad",
        "axis_rot_y_rad",
        "theta_scale",
    )
    np.testing.assert_allclose(result.geometry.setup.det_u_px.value, 0.30, atol=0.08)
    np.testing.assert_allclose(result.geometry.setup.det_v_px.value, -0.20, atol=0.08)


def test_setup_only_lm_keeps_inactive_detector_v_frozen() -> None:
    volume = _asymmetric_volume()
    nominal = GeometryState.zeros(1)
    truth_setup = nominal.setup.replace_parameter(
        "det_u_px",
        nominal.setup.det_u_px.with_value(-0.25),
    )
    truth = GeometryState(setup=truth_setup, pose=nominal.pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_setup_only_lm(
        volume,
        observed,
        nominal,
        config=SetupOnlyLMConfig(max_iterations=8, damping=1e-3, delta=1.0),
    )

    assert result.final_loss < result.initial_loss
    assert result.active_parameters == (
        "theta_offset_rad",
        "det_u_px",
        "detector_roll_rad",
    )
    assert result.geometry.setup.det_v_px.active is False
    assert result.geometry.setup.det_v_px.value == 0.0
    assert "det_v_px" in result.frozen_parameters
    np.testing.assert_allclose(result.geometry.setup.det_u_px.value, -0.25, atol=0.08)


def test_setup_only_lm_recovers_detector_roll() -> None:
    volume = _roll_asymmetric_volume()
    nominal = GeometryState.zeros(3)
    truth_setup = nominal.setup.replace_parameter(
        "detector_roll_rad",
        nominal.setup.detector_roll_rad.with_value(0.05),
    )
    truth = GeometryState(setup=truth_setup, pose=nominal.pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_setup_only_lm(
        volume,
        observed,
        nominal,
        config=SetupOnlyLMConfig(
            max_iterations=10,
            damping=1e-3,
            delta=1.0,
            active_parameters=("detector_roll_rad",),
        ),
    )

    assert result.final_loss < result.initial_loss
    assert "detector_roll_rad" in result.active_parameters
    assert "detector_roll_rad" not in result.frozen_parameters
    np.testing.assert_allclose(result.geometry.setup.detector_roll_rad.value, 0.05, atol=0.02)


def test_setup_only_lm_recovers_axis_tilt() -> None:
    volume = _roll_asymmetric_volume()
    nominal = GeometryState.zeros(3)
    truth_setup = nominal.setup.replace_parameter(
        "axis_rot_x_rad",
        nominal.setup.axis_rot_x_rad.with_value(-0.20),
    )
    truth_setup = truth_setup.replace_parameter(
        "axis_rot_y_rad",
        nominal.setup.axis_rot_y_rad.with_value(-0.10),
    )
    truth = GeometryState(setup=truth_setup, pose=nominal.pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_setup_only_lm(
        volume,
        observed,
        nominal,
        config=SetupOnlyLMConfig(
            max_iterations=10,
            damping=1e-3,
            delta=1.0,
            finite_difference_step=1.0e-2,
            active_parameters=("axis_rot_x_rad", "axis_rot_y_rad"),
        ),
    )

    assert result.final_loss < result.initial_loss
    assert result.active_parameters == ("axis_rot_x_rad", "axis_rot_y_rad")
    np.testing.assert_allclose(result.geometry.setup.axis_rot_x_rad.value, -0.20, atol=0.025)
    np.testing.assert_allclose(result.geometry.setup.axis_rot_y_rad.value, -0.10, atol=0.025)


def test_setup_only_lm_recovers_theta_offset() -> None:
    volume = _theta_asymmetric_volume()
    nominal = GeometryState.zeros(2)
    truth_setup = nominal.setup.replace_parameter(
        "theta_offset_rad",
        nominal.setup.theta_offset_rad.with_value(0.07),
    )
    truth = GeometryState(setup=truth_setup, pose=nominal.pose)
    observed = project_parallel_reference(volume, truth)

    result = solve_setup_only_lm(
        volume,
        observed,
        nominal,
        config=SetupOnlyLMConfig(max_iterations=10, damping=1e-3, delta=1.0),
    )

    assert result.final_loss < result.initial_loss
    assert "theta_offset_rad" in result.active_parameters
    assert "theta_offset_rad" not in result.frozen_parameters
    np.testing.assert_allclose(result.geometry.setup.theta_offset_rad.value, 0.07, atol=0.025)


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
