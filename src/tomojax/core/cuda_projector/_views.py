"""CUDA stack projection, adjoint, and loss facade."""

from __future__ import annotations

from ._adjoint import backproject_view_T_cuda, sum_backproject_views_T_cuda
from ._loss import (
    forward_project_loss_and_grad_T_cuda,
    forward_project_parallel_z_views_cuda,
    forward_project_residual_sse_T_cuda,
)
from ._stack_bind import (
    BoundForwardProjectResidualSseTCuda,
    BoundForwardProjectViewsTCuda,
    bind_forward_project_residual_sse_T_cuda,
    bind_forward_project_views_T_cuda,
    forward_project_view_T_cuda,
    forward_project_views_T_cuda,
)
from ._stack_call import (
    forward_project_residual_sse_T_cuda_with_state,
    forward_project_views_T_cuda_with_state,
)
from ._stack_state import (
    block_forward_project_views_T_cuda_state,
    prepare_forward_project_views_T_cuda_state,
)

__all__ = [
    "BoundForwardProjectResidualSseTCuda",
    "BoundForwardProjectViewsTCuda",
    "backproject_view_T_cuda",
    "bind_forward_project_residual_sse_T_cuda",
    "bind_forward_project_views_T_cuda",
    "block_forward_project_views_T_cuda_state",
    "forward_project_loss_and_grad_T_cuda",
    "forward_project_parallel_z_views_cuda",
    "forward_project_residual_sse_T_cuda",
    "forward_project_residual_sse_T_cuda_with_state",
    "forward_project_view_T_cuda",
    "forward_project_views_T_cuda",
    "forward_project_views_T_cuda_with_state",
    "prepare_forward_project_views_T_cuda_state",
    "sum_backproject_views_T_cuda",
]
