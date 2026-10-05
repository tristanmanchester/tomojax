"""JAX allocator defaults for TomoJAX command-line entry points."""

from __future__ import annotations

import os


def configure_jax_allocator_defaults(*, allocator: str | None = None) -> None:
    """Avoid JAX reserving most GPU memory before TomoJAX can chunk work.

    JAX caps its allocator at 75% of device memory by default; iterative
    solves of large scans need more of the device than that.
    """
    _ = os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _ = os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    if allocator is not None:
        _ = os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", allocator)
