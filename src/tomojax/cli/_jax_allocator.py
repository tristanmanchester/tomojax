"""JAX allocator defaults for TomoJAX command-line entry points."""

from __future__ import annotations

import os


def configure_jax_allocator_defaults() -> None:
    """Avoid JAX reserving most GPU memory before TomoJAX can chunk work."""
    _ = os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
