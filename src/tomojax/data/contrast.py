"""Contrast conversion utilities.

Provide helpers to move between transmission (intensity) and absorption (line
integral) domains, matching the Beer-Lambert model used throughout TomoJAX.
"""

from __future__ import annotations

from typing import Any

import numpy as _np

try:  # pragma: no cover - optional JAX dependency
    import jax  # type: ignore
    import jax.numpy as _jnp  # type: ignore

    _JAX_ARRAY_TYPES = (jax.Array,)  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - no JAX installed
    _jnp = None  # type: ignore
    _JAX_ARRAY_TYPES: tuple[type, ...] = ()

ArrayLike = Any


def _using_jax(x: ArrayLike) -> bool:
    return bool(_JAX_ARRAY_TYPES) and isinstance(x, _JAX_ARRAY_TYPES)


def _array_backend(x: ArrayLike) -> Any:
    if _using_jax(x) and _jnp is not None:
        return _jnp
    return _np


def _ensure_array(x: ArrayLike, *, backend: Any | None = None) -> Any:
    if backend is None:
        backend = _array_backend(x)
    if backend is _jnp:
        assert _jnp is not None
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
    backend = _array_backend(transmission)
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
    backend = _array_backend(absorption)
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
    transmission_min: float | None = None,
) -> Any:
    """Apply flat/dark-field correction and convert to absorption values."""
    norm = flat_dark_to_transmission(
        projections,
        flats,
        darks,
        denominator_min=min_intensity,
        clip_min=transmission_min if transmission_min is not None else min_intensity,
    )
    return transmission_to_absorption(norm, min_intensity=transmission_min or min_intensity)


def flat_dark_to_transmission(
    projections: ArrayLike,
    flats: ArrayLike,
    darks: ArrayLike | None = None,
    *,
    denominator_min: float = 1e-6,
    clip_min: float | None = None,
) -> Any:
    """Apply flat/dark-field correction and return normalized transmission."""
    backend = _array_backend(projections)
    proj = _ensure_array(projections, backend=backend)
    flats_arr = _ensure_array(flats, backend=backend)
    darks_arr = None if darks is None else _ensure_array(darks, backend=backend)

    flat_avg = backend.mean(flats_arr, axis=0) if getattr(flats_arr, "ndim", 0) >= 3 else flats_arr
    if darks_arr is None:
        dark_avg = backend.zeros_like(flat_avg)
    else:
        dark_avg = (
            backend.mean(darks_arr, axis=0) if getattr(darks_arr, "ndim", 0) >= 3 else darks_arr
        )

    denom = backend.maximum(
        flat_avg - dark_avg,
        backend.asarray(denominator_min, dtype=proj.dtype),
    )
    norm = (proj - dark_avg) / denom
    if clip_min is not None:
        norm = backend.maximum(norm, backend.asarray(clip_min, dtype=proj.dtype))
    return norm


__all__ = [
    "absorption_to_transmission",
    "flat_dark_to_absorption",
    "flat_dark_to_transmission",
    "transmission_to_absorption",
]
