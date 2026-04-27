from __future__ import annotations

import pytest

from tomojax.align.schedules import AlignmentSchedule, AlignmentStage, schedule_preset


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


def test_lamino_tilt_schedule_uses_observable_axis_dof():
    schedule = schedule_preset("lamino_tilt")

    assert schedule.stages[0].active_dofs == ("axis_rot_x_deg",)
    assert schedule.stages[0].optimizer == "validation_lm"


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
