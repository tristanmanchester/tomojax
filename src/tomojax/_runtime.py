"""Process-level allocator defaults, applied before JAX creates a backend."""

from __future__ import annotations

import os

# The pool's ceiling, as a fraction of device memory: one policy for Python and the shell.
MEMORY_FRACTION = 0.9


def configure_jax_allocator_defaults() -> None:
    """Grow JAX's pool on demand up to 90% of device memory, for Python and the shell alike.

    JAX eagerly reserves 75% of device memory by default, leaving too little for
    cuSolver initialization and TomoJAX's CuPy workspaces on small GPUs; growing
    the pool on demand avoids that reservation while retaining reuse. The ceiling
    is 90%: the alignment places its stacks by the room under it (``_stacks``), so
    a lower ceiling in Python than in the shell gave the same call two answers,
    and the shell's align had used the releasing ``platform`` allocator instead
    (14% slower on K11 54013). Neither spelling of an explicitly configured memory
    fraction, nor an explicit preallocation or allocator, is overwritten. An
    existing JAX backend cannot be reconfigured: these defaults apply at import,
    before its first operation.
    """
    _ = os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        _ = os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", str(MEMORY_FRACTION))
