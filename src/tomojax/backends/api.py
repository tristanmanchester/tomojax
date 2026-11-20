"""Public API for backend selection and memory heuristics."""

from tomojax.backends._memory import (
    device_block_available,
    device_free_memory_bytes,
    host_available_memory_bytes,
    is_oom_error,
)
from tomojax.backends._subprocesses import (
    check_output_resolved_command,
    run_resolved_command,
)

__all__ = [
    "check_output_resolved_command",
    "device_block_available",
    "device_free_memory_bytes",
    "host_available_memory_bytes",
    "is_oom_error",
    "run_resolved_command",
]
