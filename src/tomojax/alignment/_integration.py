"""Private orchestration shared only by the scan workflow and alignment CLI.

User-facing settings and execution belong in ``alignment.api``. This module
collects checkpoint and configuration plumbing, not a second public facade.
"""

from __future__ import annotations

from tomojax.alignment._config import resolved_schedule_for_config
from tomojax.alignment._geometry.parametrizations import pad_pose_params
from tomojax.alignment._modes import cone_setup
from tomojax.alignment._objectives.loss_specs import (
    loss_spec_name,
    parse_loss_schedule,
    parse_loss_spec,
)
from tomojax.alignment.io.resume import AlignmentRun, alignment_checkpointing

__all__ = [
    "AlignmentRun",
    "alignment_checkpointing",
    "cone_setup",
    "loss_spec_name",
    "pad_pose_params",
    "parse_loss_schedule",
    "parse_loss_spec",
    "resolved_schedule_for_config",
]
