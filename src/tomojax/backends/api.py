"""Public API for backend selection and memory heuristics."""

from tomojax.backends._memory import (
    ViewsPerBatchEstimate,
    default_gather_dtype,
    device_free_memory_bytes,
    estimate_views_per_batch,
    estimate_views_per_batch_info,
)

__all__ = [
    "ViewsPerBatchEstimate",
    "default_gather_dtype",
    "device_free_memory_bytes",
    "estimate_views_per_batch",
    "estimate_views_per_batch_info",
]
