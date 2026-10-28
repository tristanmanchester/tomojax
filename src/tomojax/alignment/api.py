"""Public alignment configuration, execution, schedules and geometry helpers."""

from __future__ import annotations

from tomojax.alignment._config import coupled_pose_config
from tomojax.alignment._consistency import orbit_heights
from tomojax.alignment._gauge import least_motion_estimate
from tomojax.alignment._geometry.detector_center import (
    CenterEstimate,
    CenterPair,
    estimate_center,
    estimate_center_report,
)
from tomojax.alignment._geometry.parametrizations import (
    PoseTranslationFrame,
    apply_pose_updates,
    se3_from_pose_params,
)
from tomojax.alignment._model.schedules import AlignmentSchedule, AlignmentStage
from tomojax.alignment._modes import MODES, AlignmentMode, AlignmentPlan, alignment_plan
from tomojax.alignment._objectives.loss_specs import (
    AlignmentLossSpec,
    HuberLossSpec,
    L2LossSpec,
    L2OtsuLossSpec,
)
from tomojax.alignment._prealign import implied_detector_offset
from tomojax.alignment._profiles import QualityTier
from tomojax.alignment._projection_matching import ProjectionMatchingConfig, match_projections
from tomojax.alignment.io.params_export import save_alignment_params_csv, save_alignment_params_json
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
    "AlignConfig",
    "AlignInfo",
    "AlignMultiresInfo",
    "AlignMultiresResumeState",
    "AlignResumeState",
    "AlignmentLossSpec",
    "AlignmentMode",
    "AlignmentPlan",
    "AlignmentSchedule",
    "AlignmentStage",
    "CenterEstimate",
    "CenterPair",
    "HuberLossSpec",
    "L2LossSpec",
    "L2OtsuLossSpec",
    "OuterStat",
    "PoseTranslationFrame",
    "ProjectionMatchingConfig",
    "QualityTier",
    "align",
    "align_multires",
    "alignment_plan",
    "apply_pose_updates",
    "coupled_pose_config",
    "estimate_center",
    "estimate_center_report",
    "implied_detector_offset",
    "least_motion_estimate",
    "match_projections",
    "orbit_heights",
    "save_alignment_params_csv",
    "save_alignment_params_json",
    "se3_from_pose_params",
]
