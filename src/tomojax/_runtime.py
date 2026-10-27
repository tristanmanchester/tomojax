"""Process-level allocator defaults, applied before JAX creates a backend."""

from __future__ import annotations

import os


def configure_jax_allocator_defaults(
    *, allocator: str | None = None, memory_fraction: float | None = None
) -> None:
    """Leave GPU memory available to CUDA libraries outside JAX's pool.

    JAX eagerly reserves 75% of device memory by default, leaving too little
    for cuSolver initialization and TomoJAX's CuPy workspaces on small GPUs.
    Growing the pool on demand avoids that reservation while retaining reuse.
    Python retains JAX's memory ceiling; only the CLI requests a larger pool.
    Neither spelling of an explicitly configured memory fraction is overwritten.
    Explicit environment settings take precedence. An existing JAX backend
    cannot be reconfigured; these defaults must run before its first operation.
    """
    _ = os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if memory_fraction is not None and "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        _ = os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", str(memory_fraction))
    if allocator is not None:
        _ = os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", allocator)
