"""Process-level allocator defaults, applied before JAX creates a backend."""

from __future__ import annotations

import os


def configure_jax_allocator_defaults() -> None:
    """Grow JAX's pool on demand, to JAX's own ceiling, for Python and the shell alike.

    JAX eagerly reserves 75% of device memory by default, leaving too little for
    cuSolver initialization and TomoJAX's CuPy workspaces on small GPUs; growing
    the pool on demand avoids that reservation while retaining reuse. The ceiling
    stays JAX's 75%. The pool never hands memory back, so what lives outside it
    (the CUDA context, kernel modules, cuSolver, CuPy) has only what the ceiling
    leaves: at 90% that was 0.3 GiB on an 8 GB card once a level had grown the
    pool, and K11 67208's refinement could not load a kernel. One setting for
    every caller, at import; an explicit ``XLA_CLIENT_MEM_FRACTION`` (or JAX's
    legacy ``XLA_PYTHON_CLIENT_MEM_FRACTION``), preallocation or allocator still
    wins. An existing JAX backend cannot be reconfigured: these defaults apply at
    import, before its first operation.
    """
    _ = os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
