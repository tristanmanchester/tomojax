"""Contrast conversion utilities.

Provide helpers to move between transmission (intensity) and absorption (line
integral) domains, matching the Beer–Lambert model used throughout TomoJAX.
"""

from __future__ import annotations

from typing import Any

import numpy as _np

try:  # pragma: no cover - optional JAX dependency
    import jax  # type: ignore
    import jax.numpy as _jnp  # type: ignore

    if hasattr(jax, "Array"):
        _JAX_ARRAY_TYPES = (jax.Array,)  # type: ignore[attr-defined]
    else:  # Fallback for older JAX versions
        from jax.interpreters.xla import DeviceArray as _DeviceArray  # type: ignore

        _JAX_ARRAY_TYPES = (_DeviceArray,)  # type: ignore[assignment]
except Exception:  # pragma: no cover - no JAX installed
    _jnp = None  # type: ignore
    _JAX_ARRAY_TYPES: tuple[type, ...] = ()

ArrayLike = Any


def _using_jax(x: ArrayLike) -> bool:
    return bool(_JAX_ARRAY_TYPES) and isinstance(x, _JAX_ARRAY_TYPES)


def _ensure_array(x: ArrayLike, *, backend: Any | None = None) -> Any:
    if backend is None:
        backend = _jnp if _using_jax(x) else _np
    if backend is _jnp:
        arr = _jnp.asarray(x)
        dtype = _np.dtype(arr.dtype)
        if not _np.issubdtype(dtype, _np.floating):
            arr = _jnp.asarray(arr, dtype=_jnp.float32)
        return arr
    arr = _np.asarray(x)
    if not _np.issubdtype(arr.dtype, _np.floating):
        arr = arr.astype(_np.float32, copy=False)
    return arr


def transmission_to_absorption(
    transmission: ArrayLike,
    *,
    min_intensity: float = 1e-6,
) -> Any:
    """Convert transmission (intensity) data into absorption (-log I)."""
    backend = _jnp if _using_jax(transmission) else _np
    arr = _ensure_array(transmission, backend=backend)
    if min_intensity is not None:
        arr = backend.maximum(arr, backend.asarray(min_intensity, dtype=arr.dtype))
    return -backend.log(arr)


def absorption_to_transmission(
    absorption: ArrayLike,
    *,
    max_absorption: float | None = None,
) -> Any:
    """Convert absorption data back to transmission (exp(-absorption))."""
    backend = _jnp if _using_jax(absorption) else _np
    arr = _ensure_array(absorption, backend=backend)
    if max_absorption is not None:
        arr = backend.minimum(arr, backend.asarray(max_absorption, dtype=arr.dtype))
    return backend.exp(-arr)


def flat_dark_to_absorption(
    projections: ArrayLike,
    flats: ArrayLike,
    darks: ArrayLike | None = None,
    *,
    min_intensity: float = 1e-6,
) -> Any:
    """Apply flat/dark-field correction and convert to absorption values."""
    backend = _jnp if _using_jax(projections) else _np
    proj = _ensure_array(projections, backend=backend)
    flats_arr = _ensure_array(flats, backend=backend)
    darks_arr = (
        backend.zeros_like(flats_arr)
        if darks is None
        else _ensure_array(darks, backend=backend)
    )

    flat_avg = backend.mean(flats_arr, axis=0)
    dark_avg = backend.mean(darks_arr, axis=0)

    denom = backend.maximum(
        flat_avg - dark_avg, backend.asarray(min_intensity, dtype=proj.dtype)
    )
    norm = (proj - dark_avg) / denom
    norm = backend.maximum(norm, backend.asarray(min_intensity, dtype=proj.dtype))
    return transmission_to_absorption(norm, min_intensity=min_intensity)


__all__ = [
    "transmission_to_absorption",
    "absorption_to_transmission",
    "flat_dark_to_absorption",
]
