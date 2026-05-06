from __future__ import annotations

from dataclasses import replace

import numpy as np

from tomojax.geometry import (
    GeometryState,
    PoseParameters,
    SetupParameters,
    canonicalize_geometry_gauges,
)


def test_canonicalize_gauges_zero_centres_dx_and_phi_residuals() -> None:
    setup = SetupParameters.defaults()
    pose = PoseParameters(
        alpha_rad=np.zeros(4, dtype=np.float64),
        beta_rad=np.zeros(4, dtype=np.float64),
        phi_residual_rad=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        dx_px=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        dz_px=np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64),
    )
    original_dx = setup.det_u_px.value + pose.dx_px
    original_phi = setup.theta_offset_rad.value + pose.phi_residual_rad

    result = canonicalize_geometry_gauges(GeometryState(setup=setup, pose=pose))

    assert result.state.setup.det_u_px.value == 2.5
    assert result.state.setup.theta_offset_rad.value == 0.25
    np.testing.assert_allclose(np.mean(result.state.pose.dx_px), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.mean(result.state.pose.phi_residual_rad), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.state.setup.det_u_px.value + result.state.pose.dx_px,
        original_dx,
    )
    np.testing.assert_allclose(
        result.state.setup.theta_offset_rad.value + result.state.pose.phi_residual_rad,
        original_phi,
    )
    assert [transfer.applied for transfer in result.report.transfers] == [True, True, False]


def test_canonicalize_gauges_transfers_dz_only_when_det_v_active() -> None:
    inactive = GeometryState.zeros(3)
    inactive_pose = inactive.pose.with_updates(dz_px=np.array([3.0, 4.0, 5.0], dtype=np.float64))
    inactive_result = canonicalize_geometry_gauges(
        GeometryState(setup=inactive.setup, pose=inactive_pose)
    )

    assert inactive_result.state.setup.det_v_px.value == 0.0
    np.testing.assert_allclose(inactive_result.state.pose.dz_px, [3.0, 4.0, 5.0])
    assert inactive_result.report.transfers[2].reason == "det_v_px inactive"

    active_setup = inactive.setup.replace_parameter(
        "det_v_px",
        inactive.setup.det_v_px.with_value(1.0),
    )
    active_setup = active_setup.replace_parameter(
        "det_v_px",
        replace(active_setup.det_v_px, active=True),
    )
    original_dz = active_setup.det_v_px.value + inactive_pose.dz_px

    active_result = canonicalize_geometry_gauges(
        GeometryState(setup=active_setup, pose=inactive_pose)
    )

    assert active_result.state.setup.det_v_px.value == 5.0
    np.testing.assert_allclose(np.mean(active_result.state.pose.dz_px), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        active_result.state.setup.det_v_px.value + active_result.state.pose.dz_px,
        original_dz,
    )
    assert active_result.report.transfers[2].applied is True


def test_pose_parameters_validate_shapes() -> None:
    with np.testing.assert_raises(ValueError):
        _ = PoseParameters(
            alpha_rad=np.zeros(2, dtype=np.float64),
            beta_rad=np.zeros(3, dtype=np.float64),
            phi_residual_rad=np.zeros(2, dtype=np.float64),
            dx_px=np.zeros(2, dtype=np.float64),
            dz_px=np.zeros(2, dtype=np.float64),
        )
