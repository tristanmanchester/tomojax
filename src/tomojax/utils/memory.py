from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional
import os
import math
from functools import lru_cache

from .subprocesses import check_output_command


@dataclass(frozen=True)
class ViewsPerBatchEstimate:
    """Memory-estimator result for callers that need fallback diagnostics."""

    views_per_batch: int
    free_bytes: Optional[int]
    fallback_used: bool
    fallback_reason: str | None = None


def _bytes_per(dtype: str) -> int:
    d = str(dtype).lower()
    if d in ("fp16", "float16", "half"):  # no int here; only used for gather
        return 2
    if d in ("bf16", "bfloat16"):
        return 2
    if d in ("fp64", "float64", "double"):
        return 8
    return 4  # default fp32


def _normalized_safety_fraction(safety_frac: float) -> float:
    """Clamp memory budgeting to an honest (0, 1] fraction."""
    try:
        frac = float(safety_frac)
    except (TypeError, ValueError):
        frac = 0.75
    if not math.isfinite(frac):
        frac = 0.75
    return min(max(frac, 1e-6), 1.0)


def _host_available_memory_bytes() -> Optional[int]:
    """Best-effort host available-memory query using stdlib facilities only."""
    try:
        names = getattr(os, "sysconf_names", {})
        if "SC_AVPHYS_PAGES" not in names or "SC_PAGE_SIZE" not in names:
            return None
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if avail_pages <= 0 or page_size <= 0:
            return None
        return avail_pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _free_bytes_from_memory_info(info: object) -> Optional[int]:
    """Extract free bytes from JAX/XLA memory-info shapes."""
    if isinstance(info, Mapping):
        for key in ("free", "free_bytes", "bytes_free", "available", "bytes_available"):
            if key in info:
                try:
                    return int(info[key])
                except (TypeError, ValueError):
                    return None
        if "bytes_limit" in info and "bytes_in_use" in info:
            try:
                return max(0, int(info["bytes_limit"]) - int(info["bytes_in_use"]))
            except (TypeError, ValueError):
                return None
        return None

    if isinstance(info, tuple | list) and len(info) >= 1:
        try:
            return int(info[0])
        except (TypeError, ValueError):
            return None
    return None


def device_free_memory_bytes() -> Optional[int]:
    """Best-effort query of free device memory (bytes).

    - Prefer JAX's device API when available: `jax.device_get_memory_info` (>=0.4.14).
    - Also support current `Device.memory_stats()` reports when exposed.
    - Fall back to host available memory via `os.sysconf` when supported.
    - Returns None if nothing is available.
    """
    try:  # JAX GPU/TPU or CPU
        import jax  # type: ignore

        devs = jax.devices("gpu")
        if not devs:
            return _host_available_memory_bytes()
        if hasattr(jax, "device_get_memory_info"):
            free = _free_bytes_from_memory_info(
                jax.device_get_memory_info(devs[0])  # type: ignore[attr-defined]
            )
            if free is not None:
                return free
        memory_stats = getattr(devs[0], "memory_stats", None)
        if callable(memory_stats):
            free = _free_bytes_from_memory_info(memory_stats())
            if free is not None:
                return free
    except Exception:
        pass
    return _host_available_memory_bytes()


def estimate_views_per_batch(
    *,
    n_views: int,
    grid_nxyz: tuple[int, int, int],
    det_nuv: tuple[int, int],
    gather_dtype: str = "fp32",
    projection_dtype: str = "fp32",
    volume_dtype: str = "fp32",
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
    estimate = estimate_views_per_batch_info(
        n_views=n_views,
        grid_nxyz=grid_nxyz,
        det_nuv=det_nuv,
        gather_dtype=gather_dtype,
        projection_dtype=projection_dtype,
        volume_dtype=volume_dtype,
        checkpoint_projector=checkpoint_projector,
        algo=algo,
        safety_frac=safety_frac,
        free_bytes_override=free_bytes_override,
        fallback_batch=8,
    )
    return estimate.views_per_batch


def estimate_views_per_batch_info(
    *,
    n_views: int,
    grid_nxyz: tuple[int, int, int],
    det_nuv: tuple[int, int],
    gather_dtype: str = "fp32",
    projection_dtype: str = "fp32",
    volume_dtype: str = "fp32",
    checkpoint_projector: bool = True,
    algo: str = "fbp",
    safety_frac: float = 0.75,
    free_bytes_override: Optional[int] = None,
    fallback_batch: int = 8,
) -> ViewsPerBatchEstimate:
    """Estimate views-per-batch and report whether a fallback was required.

    ``fallback_batch`` lets user-facing CLIs choose a stricter fallback while the
    legacy integer API keeps its previous small-batch fallback behavior.
    """
    n_views_i = max(1, int(n_views))
    nx, ny, nz = map(int, grid_nxyz)
    nv, nu = map(int, det_nuv)
    rays = nv * nu
    vox = nx * ny * nz

    # Base dtypes: projections and volumes default to fp32; gather can be reduced.
    proj_bytes = _bytes_per(projection_dtype)
    vol_bytes = _bytes_per(volume_dtype)
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
    if not checkpoint_projector:
        fudge *= 1.25

    free_bytes = free_bytes_override
    if free_bytes is None:
        free_bytes = device_free_memory_bytes()

    if not free_bytes or free_bytes <= 0:
        # Fallback heuristics: pick a small-but-reasonable batch
        fallback = max(1, min(n_views_i, int(fallback_batch)))
        return ViewsPerBatchEstimate(
            views_per_batch=fallback,
            free_bytes=free_bytes,
            fallback_used=True,
            fallback_reason="available memory could not be determined",
        )

    budget = int(_normalized_safety_fraction(safety_frac) * free_bytes)
    if budget <= static_bytes:
        return ViewsPerBatchEstimate(
            views_per_batch=1,
            free_bytes=int(free_bytes),
            fallback_used=False,
            fallback_reason=None,
        )

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
    if vox >= 512**3:
        cap_env = 1
    elif vox >= 256**3:
        cap_env = min(cap_env, 2)
    cap = max(1, cap_env)
    batch = max(1, min(n_views_i, cap, b))
    return ViewsPerBatchEstimate(
        views_per_batch=batch,
        free_bytes=int(free_bytes),
        fallback_used=False,
        fallback_reason=None,
    )


def default_gather_dtype() -> str:
    """Choose a default gather dtype based on the active JAX backend.

    Returns "bf16" when the active accelerator supports bfloat16 gathers,
    falls back to "fp16" on older GPUs that support float16 but not efficient
    bfloat16, and otherwise returns "fp32". CPU or backend-detection failures
    also fall back to "fp32".
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
        output = check_output_command(  # nosec B603,B607
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=-3,
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
