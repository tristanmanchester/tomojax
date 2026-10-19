"""Public API for backend selection and memory heuristics."""

from tomojax.backends._memory import (
    ViewsPerBatchEstimate,
    default_gather_dtype,
    device_free_memory_bytes,
    estimate_views_per_batch,
    estimate_views_per_batch_info,
    is_oom_error,
)
from tomojax.backends._subprocesses import (
    check_output_resolved_command,
    run_resolved_command,
)
from tomojax.backends.cuda_projector import (
    CudaModuleCapability,
    resolve_cuda_callable,
    resolve_cuda_module,
)

__all__ = [
    "CudaModuleCapability",
    "ViewsPerBatchEstimate",
    "check_output_resolved_command",
    "default_gather_dtype",
    "device_free_memory_bytes",
    "estimate_views_per_batch",
    "estimate_views_per_batch_info",
    "is_oom_error",
    "resolve_cuda_callable",
    "resolve_cuda_module",
    "run_resolved_command",
]
