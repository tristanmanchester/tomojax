"""Pallas stack projection, adjoint, and loss facade."""

from __future__ import annotations

from ._pallas_adjoint import backproject_view_T_pallas, sum_backproject_views_T_pallas
from ._pallas_loss import (
    forward_project_loss_and_grad_T_pallas,
    forward_project_parallel_z_views_pallas,
    forward_project_residual_sse_T_pallas,
)
from ._pallas_stack_bind import (
    BoundForwardProjectResidualSseTPallas,
    BoundForwardProjectViewsTPallas,
    bind_forward_project_residual_sse_T_pallas,
    bind_forward_project_views_T_pallas,
    forward_project_views_T_pallas,
)
from ._pallas_stack_call import (
    forward_project_residual_sse_T_pallas_with_state,
    forward_project_views_T_pallas_with_state,
)
from ._pallas_stack_state import (
    block_forward_project_views_T_pallas_state,
    prepare_forward_project_views_T_pallas_state,
)

__all__ = [
    "BoundForwardProjectResidualSseTPallas",
    "BoundForwardProjectViewsTPallas",
    "backproject_view_T_pallas",
    "bind_forward_project_residual_sse_T_pallas",
    "bind_forward_project_views_T_pallas",
    "block_forward_project_views_T_pallas_state",
    "forward_project_loss_and_grad_T_pallas",
    "forward_project_parallel_z_views_pallas",
    "forward_project_residual_sse_T_pallas",
    "forward_project_residual_sse_T_pallas_with_state",
    "forward_project_views_T_pallas",
    "forward_project_views_T_pallas_with_state",
    "prepare_forward_project_views_T_pallas_state",
    "sum_backproject_views_T_pallas",
]
