"""Public API for backend selection and memory heuristics."""

from tomojax.backends._memory import (
    ViewsPerBatchEstimate,
    default_gather_dtype,
    device_free_memory_bytes,
    estimate_views_per_batch,
    estimate_views_per_batch_info,
)
from tomojax.backends._subprocesses import check_output_command, run_command
from tomojax.backends.pallas_projector import (
    PallasModuleCapability,
    resolve_pallas_callable,
    resolve_pallas_module,
)

__all__ = [
    "PallasModuleCapability",
    "ViewsPerBatchEstimate",
    "check_output_command",
    "default_gather_dtype",
    "device_free_memory_bytes",
    "estimate_views_per_batch",
    "estimate_views_per_batch_info",
    "resolve_pallas_callable",
    "resolve_pallas_module",
    "run_command",
]
