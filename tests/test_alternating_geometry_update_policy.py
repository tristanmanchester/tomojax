from __future__ import annotations

# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import jax
    import pytest

# check-public-imports: allow-private
from tomojax.align._alternating_geometry_update import (
    _active_pose_dofs,
    _active_setup_parameters,
    _anchored_geometry_update_volume,
    _det_u_recentering_shift_px,
    _pose_trust_radius,
    _run_geometry_updates,
)

# check-public-imports: allow-private
import tomojax.align._alternating_orchestration as alternating_orchestration

# check-public-imports: allow-private
from tomojax.align._alternating_orchestration import (
    _alignment_train_masks,
    _apply_candidate_refresh_acceptance,
    _candidate_refresh_initial_volume,
    _maybe_run_final_pose_polish,
    _run_phi_polish,
    _run_polish_stage,
    _uses_geometry_first_det_u_bootstrap,
)

# check-public-imports: allow-private
from tomojax.align._alternating_verification import _geometry_recovery_payload
from tomojax.align.api import (
    JointSchurLMConfig,
    reference_continuation_schedule,
    solve_joint_schur_lm,
)
from tomojax.bench import AlternatingSmokeConfig
from tomojax.forward import project_parallel_reference
from tomojax.geometry import AcquisitionParameters, GaugeReport, GeometryState


def _jax_array(value: object) -> jax.Array:
    return cast("jax.Array", jnp.asarray(value, dtype=jnp.float32))


def _tiny_asymmetric_volume() -> jax.Array:
    volume = jnp.zeros((8, 8, 8), dtype=jnp.float32)
    volume = volume.at[2:5, 3:6, 1:4].set(1.0)
    return volume.at[5, 2, 6].set(0.5)


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

    expected = np.zeros_like(stopped)
    expected[:1, :, :] = stopped[2:, :, :]
    np.testing.assert_array_equal(np.asarray(anchored), expected)


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


def test_phi_polish_runs_phi_only_geometry_update() -> None:
    volume = _tiny_asymmetric_volume()
    nominal = GeometryState.zeros(2)
    true_geometry = GeometryState(
        setup=nominal.setup,
        pose=nominal.pose.with_updates(
            phi_residual_rad=np.asarray([0.04, -0.05], dtype=np.float64),
        ),
        acquisition=nominal.acquisition,
    )
    observed = project_parallel_reference(volume, true_geometry)

    summary, geometry, _report, result = _run_phi_polish(
        AlternatingSmokeConfig(
            geometry_update_volume_source="fixed_synthetic_truth",
            geometry_update_phi_polish_updates=2,
        ),
        reference_continuation_schedule("reference").levels[-1],
        truth_volume=volume,
        stopped_volume=volume,
        observed=observed,
        train_mask=jnp.ones_like(observed),
        full_mask=jnp.ones_like(observed),
        heldout_mask=None,
        geometry=nominal,
    )

    assert summary.role == "polish"
    assert summary.executed_geometry_updates == 2
    assert result.active_setup_parameters == ()
    assert result.active_pose_dofs == ("phi_residual_rad",)
    assert result.final_loss <= result.initial_loss
    assert float(jnp.max(jnp.abs(jnp.asarray(geometry.pose.phi_residual_rad)))) > 0.0


def test_final_pose_polish_can_open_det_u_with_all_pose_dofs() -> None:
    volume = _tiny_asymmetric_volume()
    nominal = GeometryState.zeros(2)
    true_setup = nominal.setup.replace_parameter(
        "det_u_px",
        nominal.setup.det_u_px.with_value(0.25),
    )
    true_geometry = GeometryState(
        setup=true_setup,
        pose=nominal.pose.with_updates(
            alpha_rad=np.asarray([0.01, -0.015], dtype=np.float64),
            beta_rad=np.asarray([-0.02, 0.01], dtype=np.float64),
            phi_residual_rad=np.asarray([0.04, -0.05], dtype=np.float64),
            dx_px=np.asarray([0.3, -0.2], dtype=np.float64),
            dz_px=np.asarray([-0.1, 0.15], dtype=np.float64),
        ),
        acquisition=nominal.acquisition,
    )
    observed = project_parallel_reference(volume, true_geometry)

    summary, _geometry, _report, result = _run_polish_stage(
        AlternatingSmokeConfig(
            geometry_update_volume_source="fixed_synthetic_truth",
            geometry_update_final_pose_polish_updates=2,
        ),
        reference_continuation_schedule("reference").levels[-1],
        truth_volume=volume,
        stopped_volume=volume,
        observed=observed,
        train_mask=jnp.ones_like(observed),
        full_mask=jnp.ones_like(observed),
        heldout_mask=None,
        geometry=nominal,
        role="final_pose_polish",
        updates=2,
        active_setup_parameters=("det_u_px",),
        active_pose_dofs=(
            "alpha_rad",
            "beta_rad",
            "phi_residual_rad",
            "dx_px",
            "dz_px",
        ),
    )

    assert summary.role == "final_pose_polish"
    assert result.active_setup_parameters == ("det_u_px",)
    assert result.active_pose_dofs == (
        "alpha_rad",
        "beta_rad",
        "phi_residual_rad",
        "dx_px",
        "dz_px",
    )
    assert result.final_loss <= result.initial_loss


def test_final_pose_polish_respects_configured_setup_parameters() -> None:
    volume = _tiny_asymmetric_volume()
    nominal = GeometryState.zeros(2)
    true_geometry = GeometryState(
        setup=nominal.setup,
        pose=nominal.pose.with_updates(
            alpha_rad=np.asarray([0.01, -0.015], dtype=np.float64),
            beta_rad=np.asarray([-0.02, 0.01], dtype=np.float64),
            phi_residual_rad=np.asarray([0.04, -0.05], dtype=np.float64),
            dx_px=np.asarray([0.3, -0.2], dtype=np.float64),
            dz_px=np.asarray([-0.1, 0.15], dtype=np.float64),
        ),
        acquisition=nominal.acquisition,
    )
    observed = project_parallel_reference(volume, true_geometry)

    _geometry, _report, result = _maybe_run_final_pose_polish(
        AlternatingSmokeConfig(
            geometry_update_volume_source="fixed_synthetic_truth",
            geometry_update_final_pose_polish_updates=1,
            geometry_update_active_setup_parameters=(),
        ),
        reference_continuation_schedule("reference").levels[-1],
        summaries=[],
        truth_volume=volume,
        stopped_volume=volume,
        observed=observed,
        train_mask=jnp.ones_like(observed),
        full_mask=jnp.ones_like(observed),
        heldout_mask=None,
        geometry=nominal,
        gauge_report=GaugeReport(()),
        last_schur_result=None,
        role="final_pose_polish",
        updates=1,
    )

    assert result is not None
    assert result.active_setup_parameters == ()
    assert result.active_pose_dofs == (
        "alpha_rad",
        "beta_rad",
        "phi_residual_rad",
        "dx_px",
        "dz_px",
    )


def test_geometry_recovery_can_exclude_flagged_bad_view() -> None:
    true_geometry = GeometryState.zeros(3)
    true_geometry = GeometryState(
        setup=true_geometry.setup,
        pose=true_geometry.pose.with_updates(
            dx_px=np.asarray([0.0, 0.1, 20.0], dtype=np.float64),
            dz_px=np.asarray([0.0, -0.1, -20.0], dtype=np.float64),
        ),
        acquisition=true_geometry.acquisition,
    )
    final_geometry = GeometryState(
        setup=true_geometry.setup,
        pose=true_geometry.pose.with_updates(
            dx_px=np.asarray([0.0, 0.1, 0.0], dtype=np.float64),
            dz_px=np.asarray([0.0, -0.1, 0.0], dtype=np.float64),
        ),
        acquisition=true_geometry.acquisition,
    )

    full = _geometry_recovery_payload(true_geometry, GeometryState.zeros(3), final_geometry)
    excluded = _geometry_recovery_payload(
        true_geometry,
        GeometryState.zeros(3),
        final_geometry,
        excluded_view_indices=(2,),
    )

    assert full["det_u_realized_rmse_px_passed"] is False
    assert excluded["det_u_realized_rmse_px_passed"] is True
    assert excluded["det_u_realized_rmse_px_all_views"] == full["det_u_realized_rmse_px"]
    assert excluded["excluded_bad_view_indices"] == [2]


def test_stopped_preview_policy_constrains_first_preview_only() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import (
        _effective_preview_initialization,
        _effective_preview_residual_filter_mode,
        _effective_preview_volume_support,
        _preview_reconstruction_iterations,
    )

    schedule = reference_continuation_schedule("reference")
    config = AlternatingSmokeConfig(
        stopped_preview_policy="constant_cylindrical_first_level",
        preview_initialization="backprojection",
        preview_volume_support="none",
        preview_residual_filter_mode="continuation",
    )

    coarse = schedule.levels[0]
    fine = schedule.levels[-1]

    assert _effective_preview_initialization(config, coarse) == "constant"
    assert _effective_preview_volume_support(config, coarse) == "cylindrical"
    assert _effective_preview_residual_filter_mode(config, coarse) == "raw"
    assert _preview_reconstruction_iterations(config, coarse) == coarse.reconstruction_iterations
    assert _effective_preview_initialization(config, fine) == "backprojection"
    assert _effective_preview_volume_support(config, fine) == "none"
    assert _effective_preview_residual_filter_mode(config, fine) == "continuation"
    assert _preview_reconstruction_iterations(config, fine) == fine.reconstruction_iterations


def test_stopped_preview_no_fista_policy_skips_first_preview_reconstruction_only() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import (
        _effective_preview_initialization,
        _effective_preview_residual_filter_mode,
        _effective_preview_volume_support,
        _preview_reconstruction_iterations,
    )

    schedule = reference_continuation_schedule("reference")
    config = AlternatingSmokeConfig(
        stopped_preview_policy="constant_cylindrical_first_level_no_fista",
        geometry_update_volume_source="stopped_reconstruction",
    )

    coarse = schedule.levels[0]
    fine = schedule.levels[-1]

    assert _effective_preview_initialization(config, coarse) == "constant"
    assert _effective_preview_volume_support(config, coarse) == "cylindrical"
    assert _effective_preview_residual_filter_mode(config, coarse) == "raw"
    assert _preview_reconstruction_iterations(config, coarse) == 0
    assert _preview_reconstruction_iterations(config, fine) == fine.reconstruction_iterations


def test_preview_reconstruction_uses_valid_mask_even_when_train_views_requested() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _preview_reconstruction_mask

    mask = jnp.ones((3, 2, 2), dtype=jnp.float32)
    train_mask = mask.at[-1, :, :].set(0.0)

    selected = _preview_reconstruction_mask(
        AlternatingSmokeConfig(preview_reconstruction_mask_source="train_views"),
        mask=mask,
        train_mask=train_mask,
    )

    np.testing.assert_array_equal(np.asarray(selected), np.asarray(mask))


def test_preview_reconstruction_mask_source_defaults_to_all_views() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _preview_reconstruction_mask

    mask = jnp.ones((3, 2, 2), dtype=jnp.float32)
    train_mask = mask.at[-1, :, :].set(0.0)

    selected = _preview_reconstruction_mask(
        AlternatingSmokeConfig(),
        mask=mask,
        train_mask=train_mask,
    )

    np.testing.assert_array_equal(np.asarray(selected), np.asarray(mask))


def test_train_view_reconstruction_option_no_longer_disables_coarse_early_exit() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _allows_coarse_early_exit

    assert _allows_coarse_early_exit(AlternatingSmokeConfig()) is True
    assert (
        _allows_coarse_early_exit(
            AlternatingSmokeConfig(preview_reconstruction_mask_source="train_views")
        )
        is True
    )


def test_fixed_truth_geometry_updates_use_level_residual_sigma() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _effective_residual_sigma

    level = reference_continuation_schedule("reference").levels[0]

    assert (
        _effective_residual_sigma(
            AlternatingSmokeConfig(geometry_update_volume_source="fixed_synthetic_truth"),
            level=level,
            estimated=500.0,
        )
        == level.residual_sigma
    )


def test_pose_trust_radius_uses_level_default_when_unset() -> None:
    assert _pose_trust_radius(2.0, configured=None) == 2.0


def test_pose_trust_radius_negative_sentinel_disables_clipping() -> None:
    assert _pose_trust_radius(2.0, configured=-1.0) is None


def test_pose_trust_radius_can_override_level_radius() -> None:
    assert _pose_trust_radius(2.0, configured=10.0) == 10.0


def test_stopped_geometry_updates_keep_estimated_residual_sigma_floor() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _effective_residual_sigma

    level = reference_continuation_schedule("reference").levels[0]

    assert (
        _effective_residual_sigma(
            AlternatingSmokeConfig(geometry_update_volume_source="stopped_reconstruction"),
            level=level,
            estimated=500.0,
        )
        == 500.0
    )


def test_theta_activation_policy_freezes_theta_until_configured_level() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _active_setup_parameters_for_level

    config = AlternatingSmokeConfig(
        geometry_update_active_setup_parameters=(
            "theta_offset_rad",
            "det_u_px",
            "detector_roll_rad",
        ),
        geometry_update_theta_activate_at_level_factor=1,
    )

    assert _active_setup_parameters_for_level(config, 4) == (
        "det_u_px",
        "detector_roll_rad",
    )
    assert _active_setup_parameters_for_level(config, 1) == (
        "theta_offset_rad",
        "det_u_px",
        "detector_roll_rad",
    )


def test_alpha_beta_activation_policy_freezes_angular_pose_until_configured_level() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _active_pose_dofs_for_level

    config = AlternatingSmokeConfig(
        geometry_update_active_pose_dofs=(
            "alpha_rad",
            "beta_rad",
            "phi_residual_rad",
            "dx_px",
            "dz_px",
        ),
        geometry_update_alpha_beta_activate_at_level_factor=1,
    )

    assert _active_pose_dofs_for_level(config, 4) == (
        "phi_residual_rad",
        "dx_px",
        "dz_px",
    )
    assert _active_pose_dofs_for_level(config, 1) == (
        "alpha_rad",
        "beta_rad",
        "phi_residual_rad",
        "dx_px",
        "dz_px",
    )


def test_stopped_preview_policy_is_inactive_for_fixed_truth() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import (
        _effective_preview_initialization,
        _effective_preview_residual_filter_mode,
        _effective_preview_volume_support,
    )

    level = reference_continuation_schedule("reference").levels[0]
    config = AlternatingSmokeConfig(
        geometry_update_volume_source="fixed_synthetic_truth",
        stopped_preview_policy="constant_cylindrical_first_level",
        preview_initialization="backprojection",
        preview_volume_support="none",
        preview_residual_filter_mode="continuation",
    )

    assert _effective_preview_initialization(config, level) == "backprojection"
    assert _effective_preview_volume_support(config, level) == "none"
    assert _effective_preview_residual_filter_mode(config, level) == "continuation"


def test_fixed_truth_geometry_updates_use_full_alignment_mask() -> None:
    mask = jnp.ones((4, 3, 2), dtype=jnp.float32)
    config = AlternatingSmokeConfig(
        geometry_update_volume_source="fixed_synthetic_truth",
        heldout_view_index=-1,
    )

    train_mask, heldout_mask = _alignment_train_masks(config, mask)

    np.testing.assert_allclose(np.asarray(train_mask), np.asarray(mask))
    assert heldout_mask is None


def test_stopped_reconstruction_geometry_updates_keep_heldout_mask() -> None:
    mask = jnp.ones((4, 3, 2), dtype=jnp.float32)
    config = AlternatingSmokeConfig(
        geometry_update_volume_source="stopped_reconstruction",
        heldout_view_index=-1,
    )

    train_mask, heldout_mask = _alignment_train_masks(config, mask)

    assert heldout_mask is not None
    assert float(train_mask[-1].sum()) == 0.0
    assert float(heldout_mask[-1].sum()) == float(mask[-1].sum())


def test_setup_only_geometry_update_solver_recovers_setup_without_pose() -> None:
    volume = _tiny_asymmetric_volume()
    nominal = GeometryState.zeros(2)
    truth_setup = nominal.setup.replace_parameter(
        "det_u_px",
        nominal.setup.det_u_px.with_value(0.25),
    )
    truth = GeometryState(setup=truth_setup, pose=nominal.pose)
    observed = project_parallel_reference(volume, truth)
    level = reference_continuation_schedule("reference").levels[0]

    geometry, _report, result = _run_geometry_updates(
        volume,
        observed,
        nominal,
        jnp.ones_like(observed),
        level,
        6,
        sigma=1.0,
        setup_prior_strength=None,
        pose_prior_strength=None,
        pose_trust_radius=None,
        active_setup_parameters=("det_u_px",),
        solver="setup_only_lm",
        pose_frozen=True,
        active_pose_dofs=(),
        fit_gain_offset_nuisance=False,
        fit_background_nuisance=False,
    )

    assert result.active_setup_parameters == ("det_u_px",)
    assert result.active_pose_dofs == ()
    assert result.diagnostics.accepted is True
    assert result.final_loss < result.initial_loss
    np.testing.assert_allclose(geometry.setup.det_u_px.value, 0.25, atol=0.08)


def test_setup_only_geometry_update_solver_requires_frozen_pose() -> None:
    volume = _tiny_asymmetric_volume()
    geometry = GeometryState.zeros(2)
    observed = project_parallel_reference(volume, geometry)
    level = reference_continuation_schedule("reference").levels[0]

    with np.testing.assert_raises_regex(
        ValueError,
        "setup_only_lm geometry updates require frozen pose DOFs",
    ):
        _ = _run_geometry_updates(
            volume,
            observed,
            geometry,
            jnp.ones_like(observed),
            level,
            1,
            sigma=1.0,
            setup_prior_strength=None,
            pose_prior_strength=None,
            pose_trust_radius=None,
            active_setup_parameters=("det_u_px",),
            solver="setup_only_lm",
            pose_frozen=False,
            active_pose_dofs=(),
            fit_gain_offset_nuisance=False,
            fit_background_nuisance=False,
        )


def test_stopped_preview_policy_reuses_first_preview_for_later_geometry_updates() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _stopped_geometry_update_volume

    schedule = reference_continuation_schedule("reference")
    config = AlternatingSmokeConfig(
        stopped_preview_policy="constant_cylindrical_first_level",
        geometry_update_volume_source="stopped_reconstruction",
    )
    first = _jax_array(np.full((2, 2, 2), 1.0, dtype=np.float32))
    current = _jax_array(np.full((2, 2, 2), 2.0, dtype=np.float32))

    reused = _stopped_geometry_update_volume(
        config,
        schedule.levels[-1],
        current_volume=current,
        constrained_first_preview_volume=first,
    )

    np.testing.assert_array_equal(np.asarray(reused), np.asarray(first))


def test_stopped_preview_policy_uses_current_volume_for_first_geometry_update() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _stopped_geometry_update_volume

    level = reference_continuation_schedule("reference").levels[0]
    config = AlternatingSmokeConfig(
        stopped_preview_policy="constant_cylindrical_first_level",
        geometry_update_volume_source="stopped_reconstruction",
    )
    first = _jax_array(np.full((2, 2, 2), 1.0, dtype=np.float32))
    current = _jax_array(np.full((2, 2, 2), 2.0, dtype=np.float32))

    selected = _stopped_geometry_update_volume(
        config,
        level,
        current_volume=current,
        constrained_first_preview_volume=first,
    )

    np.testing.assert_array_equal(np.asarray(selected), np.asarray(current))


def test_heldout_acceptance_rejects_stopped_geometry_that_worsens_validation() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _apply_heldout_geometry_acceptance

    volume = _tiny_asymmetric_volume()
    before = GeometryState.zeros(2)
    observed = project_parallel_reference(volume, before)
    candidate_setup = before.setup.replace_parameter(
        "det_u_px",
        before.setup.det_u_px.with_value(2.0),
    )
    candidate = GeometryState(setup=candidate_setup, pose=before.pose)
    result = solve_joint_schur_lm(
        volume,
        observed,
        before,
        config=JointSchurLMConfig(max_iterations=0, active_setup_parameters=("det_u_px",)),
    )
    level = reference_continuation_schedule("reference").levels[0]
    heldout_mask = jnp.ones_like(observed, dtype=jnp.float32).at[0, :, :].set(0.0)

    geometry, _report, gated = _apply_heldout_geometry_acceptance(
        AlternatingSmokeConfig(),
        volume,
        observed,
        before_geometry=before,
        candidate_geometry=candidate,
        heldout_mask=heldout_mask,
        level=level,
        sigma=1.0,
        update_report=result.canonicalized_geometry.report,
        schur_result=result,
    )

    assert gated.diagnostics.accepted is False
    assert gated.final_loss == gated.initial_loss
    assert geometry.setup.det_u_px.value == before.setup.det_u_px.value


def test_heldout_acceptance_does_not_gate_fixed_truth_oracle() -> None:
    # check-public-imports: allow-private
    from tomojax.align._alternating_orchestration import _apply_heldout_geometry_acceptance

    volume = _tiny_asymmetric_volume()
    before = GeometryState.zeros(2)
    observed = project_parallel_reference(volume, before)
    candidate_setup = before.setup.replace_parameter(
        "det_u_px",
        before.setup.det_u_px.with_value(2.0),
    )
    candidate = GeometryState(setup=candidate_setup, pose=before.pose)
    result = solve_joint_schur_lm(
        volume,
        observed,
        before,
        config=JointSchurLMConfig(max_iterations=0, active_setup_parameters=("det_u_px",)),
    )
    level = reference_continuation_schedule("reference").levels[0]

    geometry, _report, _gated = _apply_heldout_geometry_acceptance(
        AlternatingSmokeConfig(geometry_update_volume_source="fixed_synthetic_truth"),
        volume,
        observed,
        before_geometry=before,
        candidate_geometry=candidate,
        heldout_mask=jnp.ones_like(observed, dtype=jnp.float32),
        level=level,
        sigma=1.0,
        update_report=result.canonicalized_geometry.report,
        schur_result=result,
    )

    assert geometry.setup.det_u_px.value == candidate.setup.det_u_px.value


def test_candidate_refresh_acceptance_carries_candidate_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    level = reference_continuation_schedule("reference").levels[0]
    true_geometry = GeometryState.zeros(4)
    before = GeometryState(
        setup=true_geometry.setup.replace_parameter(
            "det_u_px",
            true_geometry.setup.det_u_px.with_value(2.0),
        ),
        pose=true_geometry.pose,
    )
    volume = _tiny_asymmetric_volume()
    observed = project_parallel_reference(volume, true_geometry)
    full_mask = jnp.ones_like(observed, dtype=jnp.float32)
    train_mask = full_mask.at[-1, :, :].set(0.0)
    heldout_mask = jnp.zeros_like(full_mask).at[-1, :, :].set(full_mask[-1])
    schur_result = solve_joint_schur_lm(
        volume,
        observed,
        before,
        mask=train_mask,
        config=JointSchurLMConfig(
            max_iterations=1,
            active_setup_parameters=("det_u_px",),
            active_pose_dofs=(),
            parameter_prior_strength=0.0,
        ),
    )
    initial_seeds: list[jax.Array] = []

    def fake_refresh(
        config: AlternatingSmokeConfig,
        observed: jax.Array,
        geometry: GeometryState,
        *,
        initial_volume: jax.Array,
        reconstruction_mask: jax.Array,
        stage: str,
        mask_provenance: list[object],
        level: object,
    ) -> jax.Array:
        del config, observed, reconstruction_mask, stage, mask_provenance, level
        initial_seeds.append(initial_volume)
        return jnp.full_like(initial_volume, geometry.setup.det_u_px.value)

    def fake_loss(
        volume: jax.Array,
        observed: jax.Array,
        geometry: GeometryState,
        mask: jax.Array,
        level: object,
        *,
        sigma: float,
        loss_mode: str,
    ) -> float:
        del observed, geometry, mask, level, sigma, loss_mode
        return float(jnp.mean(volume))

    monkeypatch.setattr(alternating_orchestration, "_candidate_refresh_volume", fake_refresh)
    monkeypatch.setattr(alternating_orchestration, "_projection_loss", fake_loss)

    geometry, _report, accepted, refreshed = _apply_candidate_refresh_acceptance(
        AlternatingSmokeConfig(size=8, geometry_update_volume_source="stopped_reconstruction"),
        volume,
        observed,
        before_geometry=before,
        candidate_geometry=true_geometry,
        reconstruction_mask=full_mask,
        train_mask=train_mask,
        heldout_mask=heldout_mask,
        level=level,
        sigma=1.0,
        update_report=GaugeReport(()),
        schur_result=schur_result,
        mask_provenance=[],
    )

    assert accepted.diagnostics.accepted is True
    assert geometry.setup.det_u_px.value == true_geometry.setup.det_u_px.value
    assert refreshed.shape == volume.shape
    assert len(initial_seeds) == 2
    assert np.allclose(np.asarray(initial_seeds[0]), np.asarray(initial_seeds[1]))
    assert not np.allclose(np.asarray(initial_seeds[0]), np.asarray(volume))


def test_candidate_refresh_acceptance_rejects_worse_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    level = reference_continuation_schedule("reference").levels[0]
    true_geometry = GeometryState.zeros(4)
    candidate = GeometryState(
        setup=true_geometry.setup.replace_parameter(
            "det_u_px",
            true_geometry.setup.det_u_px.with_value(2.0),
        ),
        pose=true_geometry.pose,
    )
    volume = _tiny_asymmetric_volume()
    observed = project_parallel_reference(volume, true_geometry)
    full_mask = jnp.ones_like(observed, dtype=jnp.float32)
    train_mask = full_mask.at[-1, :, :].set(0.0)
    heldout_mask = jnp.zeros_like(full_mask).at[-1, :, :].set(full_mask[-1])
    schur_result = solve_joint_schur_lm(
        volume,
        observed,
        true_geometry,
        mask=train_mask,
        config=JointSchurLMConfig(
            max_iterations=1,
            active_setup_parameters=("det_u_px",),
            active_pose_dofs=(),
            parameter_prior_strength=0.0,
        ),
    )

    def fake_refresh(
        config: AlternatingSmokeConfig,
        observed: jax.Array,
        geometry: GeometryState,
        *,
        initial_volume: jax.Array,
        reconstruction_mask: jax.Array,
        stage: str,
        mask_provenance: list[object],
        level: object,
    ) -> jax.Array:
        del config, observed, reconstruction_mask, stage, mask_provenance, level
        return jnp.full_like(initial_volume, geometry.setup.det_u_px.value)

    def fake_loss(
        volume: jax.Array,
        observed: jax.Array,
        geometry: GeometryState,
        mask: jax.Array,
        level: object,
        *,
        sigma: float,
        loss_mode: str,
    ) -> float:
        del observed, geometry, mask, level, sigma, loss_mode
        return float(jnp.mean(volume))

    monkeypatch.setattr(alternating_orchestration, "_candidate_refresh_volume", fake_refresh)
    monkeypatch.setattr(alternating_orchestration, "_projection_loss", fake_loss)

    geometry, _report, accepted, refreshed = _apply_candidate_refresh_acceptance(
        AlternatingSmokeConfig(size=8, geometry_update_volume_source="stopped_reconstruction"),
        volume,
        observed,
        before_geometry=true_geometry,
        candidate_geometry=candidate,
        reconstruction_mask=full_mask,
        train_mask=train_mask,
        heldout_mask=heldout_mask,
        level=level,
        sigma=1.0,
        update_report=GaugeReport(()),
        schur_result=schur_result,
        mask_provenance=[],
    )

    assert accepted.diagnostics.accepted is False
    assert geometry.setup.det_u_px.value == true_geometry.setup.det_u_px.value
    assert refreshed.shape == volume.shape


def test_candidate_refresh_initial_volume_is_neutral_shared_seed() -> None:
    level = reference_continuation_schedule("reference").levels[0]
    absorbed_volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8))
    observed = jnp.full((4, 8, 8), 16.0, dtype=jnp.float32)
    config = AlternatingSmokeConfig(
        size=8,
        preview_volume_support="cylindrical",
        geometry_update_volume_source="stopped_reconstruction",
    )

    initial = _candidate_refresh_initial_volume(
        config,
        observed,
        current_volume=absorbed_volume,
        level=level,
    )

    assert initial.shape == absorbed_volume.shape
    assert not np.allclose(np.asarray(initial), np.asarray(absorbed_volume))
    assert np.isclose(float(initial[4, 4, 4]), 2.0)
    assert float(initial[0, 0, 4]) == 0.0


def test_geometry_first_bootstrap_is_limited_to_stopped_detu_gate() -> None:
    level = reference_continuation_schedule("balanced").levels[0]

    assert _uses_geometry_first_det_u_bootstrap(
        AlternatingSmokeConfig(
            size=64,
            geometry_update_volume_source="stopped_reconstruction",
            geometry_update_pose_frozen=True,
            geometry_update_active_setup_parameters=("det_u_px",),
        ),
        level,
    )
    assert not _uses_geometry_first_det_u_bootstrap(
        AlternatingSmokeConfig(
            size=64,
            geometry_update_volume_source="fixed_synthetic_truth",
            geometry_update_pose_frozen=True,
            geometry_update_active_setup_parameters=("det_u_px",),
        ),
        level,
    )
    assert not _uses_geometry_first_det_u_bootstrap(
        AlternatingSmokeConfig(
            size=64,
            geometry_update_volume_source="stopped_reconstruction",
            geometry_update_pose_frozen=True,
            geometry_update_active_setup_parameters=("det_u_px", "theta_offset_rad"),
        ),
        level,
    )
