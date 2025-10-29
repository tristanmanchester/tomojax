from __future__ import annotations

from typing import Optional
import os
import math
from functools import lru_cache
import subprocess


def _bytes_per(dtype: str) -> int:
    d = dtype.lower()
    if d in ("fp16", "float16", "half"):  # no int here; only used for gather
        return 2
    if d in ("bf16", "bfloat16"):
        return 2
    return 4  # default fp32


def device_free_memory_bytes() -> Optional[int]:
    """Best-effort query of free device memory (bytes).

    - Prefer JAX's device API when available: `jax.device_get_memory_info` (>=0.4.14).
    - Fall back to host available memory via psutil.
    - Returns None if nothing is available.
    """
    try:  # JAX GPU/TPU or CPU
        import jax  # type: ignore

        if hasattr(jax, "device_get_memory_info"):
            devs = jax.devices("gpu")
            if devs:
                free, total = jax.device_get_memory_info(devs[0])  # type: ignore[attr-defined]
                # On CPU backends, this may reflect host RAM; still usable as a bound
                return int(free)
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def estimate_views_per_batch(
    *,
    n_views: int,
    grid_nxyz: tuple[int, int, int],
    det_nuv: tuple[int, int],
    gather_dtype: str = "fp32",
    checkpoint_projector: bool = True,
    algo: str = "fbp",
    safety_frac: float = 0.75,
    free_bytes_override: Optional[int] = None,
) -> int:
    """Estimate a safe views_per_batch for FBP/FISTA based on memory.

    Heuristic upper bound that accounts for per-view projection storage (nv*nu) and
    a transient per-view volume contribution (nx*ny*nz) that may appear in batched VJP.

    Returns at least 1 and at most n_views. Falls back to a conservative default (8 or all)
    if free memory cannot be determined.
    """
    nx, ny, nz = map(int, grid_nxyz)
    nv, nu = map(int, det_nuv)
    rays = nv * nu
    vox = nx * ny * nz

    # Base dtypes: projections and volumes are fp32; gather buffer can be reduced
    proj_bytes = 4
    vol_bytes = 4
    gather_bytes = _bytes_per(gather_dtype)

    # Per-view footprint (rough upper bound)
    # - input view (nv*nu*fp32)
    # - VJP intermediate returning (nx*ny*nz*fp32) before reduction across batch
    # - gather buffer is not explicitly materialized across the full batch in Python,
    #   but account a term to keep the bound conservative.
    per_view = proj_bytes * rays + vol_bytes * vox + gather_bytes * rays

    # Static accumulator and small constants
    # FISTA holds extra TV dual variables (≈3 volumes) during proximal steps.
    if algo.lower() == "fbp":
        static_bytes = vol_bytes * vox
    else:
        static_bytes = vol_bytes * vox * 4  # x + (p1,p2,p3)

    # Algorithm factor: FISTA uses both fwd and VJP per batch; FBP uses only VJP
    algo_factor = 1.5 if algo.lower() == "fbp" else 2.0

    # Empirical overhead fudge to cover extra buffers, remat, etc.
    fudge = 2.0 if algo.lower() == "fbp" else 4.0

    free_bytes = free_bytes_override
    if free_bytes is None:
        free_bytes = device_free_memory_bytes()

    if not free_bytes or free_bytes <= 0:
        # Fallback heuristics: pick a small-but-reasonable batch
        return max(1, min(n_views, 8))

    budget = int(max(1, safety_frac) * free_bytes)
    if budget <= static_bytes:
        return 1

    per_batch = lambda b: static_bytes + int(algo_factor * fudge * per_view * b)

    # Largest b such that per_batch(b) <= budget
    b_est = (budget - static_bytes) / float(algo_factor * fudge * per_view)
    b = int(max(1, math.floor(b_est)))
    # Apply a conservative soft cap to avoid oversized vectorization; override via env
    # Default clamp to keep auto-batching conservative on diverse GPUs
    cap_default = 8
    try:
        cap_env = int(os.getenv("TOMOJAX_MAX_VIEWS_PER_BATCH", str(cap_default)))
    except Exception:
        cap_env = cap_default
    # Additional dynamic caps for very large volumes
    if vox >= 512 ** 3:
        cap_env = 1
    elif vox >= 256 ** 3:
        cap_env = min(cap_env, 2)
    cap = max(1, cap_env)
    return max(1, min(int(n_views), cap, b))


def default_gather_dtype() -> str:
    """Choose a default gather dtype based on the active JAX backend.

    Returns "bf16" on GPU/TPU (mixed-precision gather with fp32 accumulation)
    and "fp32" on CPU or if backend detection fails.
    """
    backend = _current_backend()
    if isinstance(backend, str) and backend.lower() in ("gpu", "tpu"):
        if backend.lower() == "tpu":
            return "bf16"
        cc = _gpu_compute_capability()
        if cc and cc < (8, 0):
            if _device_supports_dtype("float16"):
                return "fp16"
            return "fp32"
        # GPU: prefer bf16 if the device JIT supports it, else fp16/fp32.
        if _device_supports_dtype("bfloat16"):
            return "bf16"
        if _device_supports_dtype("float16"):
            return "fp16"
    return "fp32"


@lru_cache(maxsize=None)
def _device_supports_dtype(dtype_name: str) -> bool:
    """Heuristic check whether the active accelerator can JIT kernels with `dtype`."""
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
    except Exception:
        return False
    try:
        devs = jax.devices("gpu")
    except Exception:
        return False
    if not devs:
        return False
    if devs[0].platform != "gpu":
        return False
    try:
        # Simple kernel to trigger compilation in the requested dtype.
        fn = jax.jit(lambda x: x + x)
        dtype = getattr(jnp, dtype_name)
        arr = jnp.ones((1,), dtype=dtype)
        fn(arr).block_until_ready()
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def _gpu_compute_capability() -> Optional[tuple[int, int]]:
    """Return (major, minor) compute capability for the first CUDA device, if available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        first_line = output.strip().splitlines()[0]
        parts = first_line.replace(" ", "").split(".")
        if len(parts) >= 2:
            major, minor = int(parts[0]), int(parts[1])
            return major, minor
    except Exception:
        pass
    return None


def _current_backend() -> Optional[str]:
    """Best-effort helper to query the active JAX backend name."""
    try:
        import jax  # type: ignore

        backend = getattr(jax, "default_backend", lambda: None)()
        if backend:
            return backend
        return os.environ.get("JAX_PLATFORM_NAME")
    except Exception:
        return None
