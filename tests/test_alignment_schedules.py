from __future__ import annotations

import pytest

from tomojax.align.schedules import (
    AlignmentSchedule,
    AlignmentStage,
    resolve_alignment_schedule,
    schedule_preset,
)


def test_pose_only_schedule_uses_fixed_volume_pose_stage():
    schedule = schedule_preset("pose_only")

    assert schedule.stages[0].objective_kind == "fixed_volume"
    assert schedule.stages[0].optimizer == "gn"
    assert schedule.stages[0].active_dofs == ("alpha", "beta", "phi", "dx", "dz")


def test_pose_parity_stage_presets_use_fixed_volume_gn():
    phi_schedule = schedule_preset("pose_phi_only")
    dx_dz_schedule = schedule_preset("pose_dx_dz_after_phi")

    assert phi_schedule.stages[0].active_dofs == ("phi",)
    assert phi_schedule.stages[0].objective_kind == "fixed_volume"
    assert phi_schedule.stages[0].optimizer == "gn"
    assert dx_dz_schedule.stages[0].active_dofs == ("dx", "dz")
    assert dx_dz_schedule.stages[0].objective_kind == "fixed_volume"
    assert dx_dz_schedule.stages[0].optimizer == "gn"


def test_cor_schedule_activates_only_detector_u_and_bilevel_cv():
    schedule = schedule_preset("cor")
    stage = schedule.stages[0]

    assert stage.active_dofs == ("det_u_px",)
    assert stage.objective_kind == "bilevel_cv"
    assert stage.optimizer == "validation_lm"


def test_setup_safe_stages_setup_then_pose_polish():
    schedule = schedule_preset("setup_safe")

    assert [stage.name for stage in schedule.stages] == [
        "cor",
        "detector_roll",
        "axis_direction",
        "pose_polish",
    ]
    assert schedule.stages[-1].objective_kind == "fixed_volume"
    assert schedule.stages[-1].optimizer == "gn"
    assert all(stage.optimizer == "validation_lm" for stage in schedule.stages[:-1])


def test_resolved_setup_safe_executes_as_separate_stages():
    resolved = resolve_alignment_schedule(schedule="setup_safe", outer_iters=3)

    assert [stage.name for stage in resolved.stages] == [
        "cor",
        "detector_roll",
        "axis_direction",
        "pose_polish",
    ]
    assert resolved.stages[0].active_geometry_dofs == ("det_u_px",)
    assert resolved.stages[1].active_geometry_dofs == ("detector_roll_deg",)
    assert resolved.stages[2].active_geometry_dofs == (
        "axis_rot_x_deg",
        "axis_rot_y_deg",
    )
    assert resolved.stages[3].active_pose_dofs == ("alpha", "beta", "phi", "dx", "dz")
    assert all(stage.maxiter == 3 for stage in resolved.stages)
    assert resolved.to_dict()["stages"][0]["gauge_decision"]["status"] == "ok"


def test_lamino_tilt_schedule_uses_observable_axis_dof():
    schedule = schedule_preset("lamino_tilt")

    assert schedule.stages[0].active_dofs == ("axis_rot_x_deg",)
    assert schedule.stages[0].optimizer == "validation_lm"


def test_detector_center_2d_is_not_public_schedule_preset():
    with pytest.raises(ValueError, match="detector_center_2d"):
        schedule_preset("detector_center_2d")


def test_direct_mixed_setup_pose_requires_expert_gauge_policy():
    with pytest.raises(ValueError, match="expert gauge_policy"):
        resolve_alignment_schedule(optimise_dofs=("det_u_px", "dx"))

    resolved = resolve_alignment_schedule(
        optimise_dofs=("det_u_px", "dx"),
        gauge_policy="anchor_mean",
    )

    assert [stage.name for stage in resolved.stages] == ["direct_setup", "direct_pose"]
    assert resolved.gauge_decision.status == "allowed_with_gauge_policy"


def test_bilevel_cv_stages_do_not_default_to_gn():
    with pytest.raises(ValueError, match="must use validation_lm"):
        AlignmentSchedule(
            name="bad",
            stages=(
                AlignmentStage(
                    name="bad",
                    active_dofs=("det_u_px",),
                    objective_kind="bilevel_cv",
                    optimizer="gn",
                ),
            ),
        ).validate()


def test_stage_with_no_active_dofs_fails_validation():
    with pytest.raises(ValueError, match="no active DOFs"):
        AlignmentSchedule(
            name="bad",
            stages=(
                AlignmentStage(
                    name="bad",
                    active_dofs=(),
                    objective_kind="fixed_volume",
                    optimizer="lbfgs",
                ),
            ),
        ).validate()


def test_expert_coupled_requires_explicit_active_dofs():
    with pytest.raises(ValueError, match="requires explicit active_dofs"):
        schedule_preset("expert_coupled")

    schedule = schedule_preset(
        "expert_coupled",
        active_dofs=("det_u_px", "dx"),
        gauge_policy="prior_required",
    )

    assert schedule.stages[0].gauge_policy == "prior_required"
    assert schedule.stages[0].active_dofs == ("det_u_px", "dx")
