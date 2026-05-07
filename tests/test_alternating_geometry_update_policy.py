from __future__ import annotations

# pyright: reportPrivateUsage=false
import numpy as np

# check-public-imports: allow-private
from tomojax.align._alternating_geometry_update import (
    _active_pose_dofs,
    _active_setup_parameters,
)
from tomojax.geometry import AcquisitionParameters, GeometryState


def test_alternating_setup_policy_freezes_theta_scale() -> None:
    active = _active_setup_parameters(
        (
            "theta_offset_rad",
            "theta_scale",
            "det_u_px",
        )
    )

    assert active == ("theta_offset_rad", "det_u_px")


def test_alternating_pose_policy_keeps_requested_pose_dofs_outside_global_setup() -> None:
    geometry = GeometryState.zeros(4)

    active = _active_pose_dofs(
        (
            "alpha_rad",
            "beta_rad",
            "phi_residual_rad",
            "dx_px",
            "dz_px",
        ),
        geometry,
        active_setup_parameters=("theta_offset_rad", "det_u_px"),
    )

    assert active == (
        "alpha_rad",
        "beta_rad",
        "phi_residual_rad",
        "dx_px",
        "dz_px",
    )


def test_alternating_pose_policy_keeps_nonzero_tilt_pose_dofs() -> None:
    geometry = GeometryState.zeros(4)
    geometry = GeometryState(
        setup=geometry.setup,
        pose=geometry.pose.with_updates(
            alpha_rad=np.asarray([0.0, 1.0e-3, 0.0, 0.0], dtype=np.float64),
            beta_rad=np.asarray([0.0, 0.0, -2.0e-3, 0.0], dtype=np.float64),
        ),
        acquisition=geometry.acquisition,
    )

    active = _active_pose_dofs(
        (
            "alpha_rad",
            "beta_rad",
            "phi_residual_rad",
        ),
        geometry,
        active_setup_parameters=("theta_offset_rad", "det_u_px"),
    )

    assert active == ("alpha_rad", "beta_rad", "phi_residual_rad")


def test_alternating_pose_policy_freezes_zero_pose_for_global_setup_block() -> None:
    geometry = GeometryState.zeros(4)

    active = _active_pose_dofs(
        (
            "alpha_rad",
            "beta_rad",
            "phi_residual_rad",
            "dx_px",
            "dz_px",
        ),
        geometry,
        active_setup_parameters=(
            "theta_offset_rad",
            "det_u_px",
            "detector_roll_rad",
            "axis_rot_x_rad",
            "axis_rot_y_rad",
        ),
    )

    assert active == ()


def test_alternating_pose_policy_keeps_laminography_pose_for_global_setup_block() -> None:
    geometry = GeometryState.zeros(4)
    geometry = GeometryState(
        setup=geometry.setup,
        pose=geometry.pose,
        acquisition=AcquisitionParameters.parallel_laminography(tilt_rad=1.0),
    )

    active = _active_pose_dofs(
        (
            "alpha_rad",
            "beta_rad",
            "phi_residual_rad",
            "dx_px",
            "dz_px",
        ),
        geometry,
        active_setup_parameters=(
            "theta_offset_rad",
            "det_u_px",
            "detector_roll_rad",
            "axis_rot_x_rad",
            "axis_rot_y_rad",
        ),
    )

    assert active == (
        "alpha_rad",
        "beta_rad",
        "phi_residual_rad",
        "dx_px",
        "dz_px",
    )
