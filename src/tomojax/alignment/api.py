"""Public alignment API for configuration, schedules, losses, and execution."""

from __future__ import annotations

from tomojax.alignment._config import coupled_pose_config, resolved_schedule_for_config
from tomojax.alignment._consistency import OrbitHeights, orbit_heights
from tomojax.alignment._gauge import least_motion_estimate
from tomojax.alignment._geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state
from tomojax.alignment._geometry.geometry_blocks import normalize_geometry_dofs
from tomojax.alignment._geometry.parametrizations import (
    PoseTranslationFrame,
    apply_pose_update,
    apply_pose_updates,
    pad_pose_params,
    se3_from_pose_params,
)
from tomojax.alignment._model.dof_specs import DofSpec, dof_spec
from tomojax.alignment._model.dofs import (
    POSE_WIDTH,
    DofBounds,
    normalize_alignment_dofs,
    normalize_bounds,
)
from tomojax.alignment._model.schedules import (
    PUBLIC_SCHEDULE_PRESETS,
    AlignmentSchedule,
    AlignmentStage,
    GaugePolicy,
    GaugePolicyError,
    ResolvedAlignmentSchedule,
    resolve_alignment_schedule,
)
from tomojax.alignment._model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.alignment._modes import (
    MODES,
    AlignmentMode,
    AlignmentPlan,
    ConeSetup,
    alignment_plan,
    cone_setup,
)
from tomojax.alignment._objectives.loss_specs import (
    AlignmentLossConfig,
    AlignmentLossSchedule,
    AlignmentLossSpec,
    L2LossSpec,
    L2OtsuLossSpec,
    LossScheduleEntry,
    PWLSLossSpec,
    loss_spec_name,
    loss_spec_params,
    parse_loss_schedule,
    parse_loss_spec,
    resolve_loss_for_level,
    validate_loss_schedule_levels,
)
from tomojax.alignment._prealign import implied_detector_offset
from tomojax.alignment._profiles import (
    AlignmentProfilePolicy,
    QualityTier,
    profile_policy_from_config,
    resolve_profiled_cli_defaults,
)
from tomojax.alignment.io.checkpoint import (
    AlignmentCheckpoint,
    CheckpointError,
    CheckpointMetadata,
    load_alignment_checkpoint,
    save_alignment_checkpoint,
    validate_alignment_checkpoint,
)
from tomojax.alignment.io.params_export import (
    alignment_params_payload,
    save_alignment_params_csv,
    save_alignment_params_json,
)
from tomojax.alignment.io.resume import (
    AlignmentRun,
    alignment_checkpoint_metadata,
    alignment_checkpointing,
    resume_state_from_checkpoint,
    write_alignment_checkpoint,
)
from tomojax.alignment.pipeline import (
    AlignConfig,
    AlignInfo,
    AlignMultiresInfo,
    AlignMultiresResumeState,
    AlignResumeState,
    OuterStat,
    align,
    align_multires,
)

__all__ = [
    "MODES",
    "POSE_WIDTH",
    "PUBLIC_SCHEDULE_PRESETS",
    "AlignConfig",
    "AlignInfo",
    "AlignMultiresInfo",
    "AlignMultiresResumeState",
    "AlignResumeState",
    "AlignmentCheckpoint",
    "AlignmentLossConfig",
    "AlignmentLossSchedule",
    "AlignmentLossSpec",
    "AlignmentMode",
    "AlignmentPlan",
    "AlignmentProfilePolicy",
    "AlignmentRun",
    "AlignmentSchedule",
    "AlignmentStage",
    "AlignmentState",
    "BaseGeometryArrays",
    "CheckpointError",
    "CheckpointMetadata",
    "ConeSetup",
    "DofBounds",
    "DofSpec",
    "GaugePolicy",
    "GaugePolicyError",
    "L2LossSpec",
    "L2OtsuLossSpec",
    "LossScheduleEntry",
    "OrbitHeights",
    "OuterStat",
    "PWLSLossSpec",
    "PoseState",
    "PoseTranslationFrame",
    "QualityTier",
    "ResolvedAlignmentSchedule",
    "SetupGeometryState",
    "align",
    "align_multires",
    "alignment_checkpoint_metadata",
    "alignment_checkpointing",
    "alignment_params_payload",
    "alignment_plan",
    "apply_alignment_state",
    "apply_pose_update",
    "apply_pose_updates",
    "cone_setup",
    "coupled_pose_config",
    "dof_spec",
    "implied_detector_offset",
    "least_motion_estimate",
    "load_alignment_checkpoint",
    "loss_spec_name",
    "loss_spec_params",
    "normalize_alignment_dofs",
    "normalize_bounds",
    "normalize_geometry_dofs",
    "orbit_heights",
    "pad_pose_params",
    "parse_loss_schedule",
    "parse_loss_spec",
    "profile_policy_from_config",
    "resolve_alignment_schedule",
    "resolve_loss_for_level",
    "resolve_profiled_cli_defaults",
    "resolved_schedule_for_config",
    "resume_state_from_checkpoint",
    "save_alignment_checkpoint",
    "save_alignment_params_csv",
    "save_alignment_params_json",
    "se3_from_pose_params",
    "validate_alignment_checkpoint",
    "validate_loss_schedule_levels",
    "write_alignment_checkpoint",
]
