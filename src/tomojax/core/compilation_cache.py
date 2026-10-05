"""Persistent JAX compilation cache, enabled by default for TomoJAX programs.

Compiled solver and projector programs are reused across Python processes, so a
repeated script or CLI run skips XLA compilation for shapes it has seen before.
CuPy and Triton cache their kernels the same way.

- A cache directory configured through JAX (``JAX_COMPILATION_CACHE_DIR`` or
  ``jax.config``) is left untouched.
- ``TOMOJAX_JAX_CACHE=off`` disables the default.
- ``TOMOJAX_JAX_CACHE_DIR`` overrides ``${XDG_CACHE_HOME:-~/.cache}/tomojax/jax_cache``.
"""

from __future__ import annotations

from functools import cache
import logging
import os
from pathlib import Path

import jax


def default_cache_dir() -> Path:
    """Return the directory TomoJAX uses when the user has not chosen one."""
    override = os.environ.get("TOMOJAX_JAX_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return base / "tomojax" / "jax_cache"


@cache
def enable_persistent_compilation_cache() -> None:
    """Enable the cache once per process; failures leave JAX's settings unchanged."""
    if os.environ.get("TOMOJAX_JAX_CACHE", "").strip().lower() in {"0", "off", "false", "no"}:
        return
    if jax.config.jax_compilation_cache_dir:
        return
    cache_dir = default_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logging.getLogger(__name__).debug("JAX compilation cache disabled: %s", error)
        return
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    # Cache every program: small helpers also pay noticeable compile latency.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
    )
