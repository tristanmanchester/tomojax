from __future__ import annotations

# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import jax

# check-public-imports: allow-private
from tomojax.align._alternating_geometry_update import (
    _active_pose_dofs,
    _active_setup_parameters,
    _anchored_geometry_update_volume,
    _det_u_recentering_shift_px,
)
from tomojax.align.api import reference_continuation_schedule
from tomojax.geometry import AcquisitionParameters, GeometryState


def _jax_array(value: object) -> jax.Array:
    return cast("jax.Array", jnp.asarray(value, dtype=jnp.float32))


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


def test_det_u_recentering_shift_uses_projection_centroid() -> None:
    observed = np.zeros((2, 3, 7), dtype=np.float32)
    observed[:, :, 1] = 1.0
    mask = np.ones_like(observed)

    assert _det_u_recentering_shift_px(_jax_array(observed), _jax_array(mask)) == -2


def test_coarse_setup_global_anchoring_recenters_stopped_volume() -> None:
    stopped = np.arange(3 * 5 * 4, dtype=np.float32).reshape((3, 5, 4))
    observed = np.zeros((2, 3, 7), dtype=np.float32)
    observed[:, :, 1] = 1.0
    mask = np.ones_like(observed)
    level = reference_continuation_schedule("reference").levels[0]

    anchored = _anchored_geometry_update_volume(
        _jax_array(stopped),
        _jax_array(observed),
        _jax_array(mask),
        level=level,
        source="stopped_reconstruction",
        active_setup_parameters=(
            "theta_offset_rad",
            "det_u_px",
            "detector_roll_rad",
            "axis_rot_x_rad",
            "axis_rot_y_rad",
        ),
    )

    np.testing.assert_array_equal(np.asarray(anchored), np.roll(stopped, -2, axis=1))


def test_anchoring_releases_outside_coarse_setup_global() -> None:
    stopped = np.arange(3 * 5 * 4, dtype=np.float32).reshape((3, 5, 4))
    observed = np.zeros((2, 3, 7), dtype=np.float32)
    observed[:, :, 1] = 1.0
    mask = np.ones_like(observed)
    fine_level = reference_continuation_schedule("reference").levels[-1]

    anchored = _anchored_geometry_update_volume(
        _jax_array(stopped),
        _jax_array(observed),
        _jax_array(mask),
        level=fine_level,
        source="stopped_reconstruction",
        active_setup_parameters=(
            "theta_offset_rad",
            "det_u_px",
            "detector_roll_rad",
            "axis_rot_x_rad",
            "axis_rot_y_rad",
        ),
    )

    np.testing.assert_array_equal(np.asarray(anchored), np.asarray(stopped))
