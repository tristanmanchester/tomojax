"""Frequency-domain filters for filtered backprojection."""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np


def _normalize_filter_name(name: str | None) -> str:
    """Normalize filter names and preserve the default ramp fallback."""
    normalized = str(name or "").strip().lower()
    return normalized or "ramp"


def _ramp_filter_np(n: int, du: float) -> np.ndarray:
    # Frequency coordinates (cycles per unit)
    freqs = np.fft.fftfreq(n, d=du).astype(np.float32, copy=False)
    return np.abs(freqs, dtype=np.float32)


def _shepp_logan_filter_np(n: int, du: float) -> np.ndarray:
    f = _ramp_filter_np(n, du)
    freqs = np.fft.fftfreq(n, d=du).astype(np.float32, copy=False)
    fmax = float(np.max(np.abs(freqs))) + 1e-12
    x = freqs / (2.0 * fmax)
    # np.sinc implements sin(pi x)/(pi x) without emitting divide-by-zero warnings at x=0.
    sinc = np.sinc(x).astype(np.float32, copy=False)
    return (f * sinc).astype(np.float32, copy=False)


def _hann_filter_np(n: int, du: float) -> np.ndarray:
    f = _ramp_filter_np(n, du)
    # Create Hann window in frequency with cutoff at Nyquist
    freqs = np.fft.fftfreq(n, d=du).astype(np.float32, copy=False)
    fmax = float(np.max(np.abs(freqs))) + 1e-12
    w = 0.5 + 0.5 * np.cos(np.pi * (freqs / fmax), dtype=np.float32)
    return (f * w.astype(np.float32, copy=False)).astype(np.float32, copy=False)


@functools.lru_cache(maxsize=16)
def _build_filter_np(
    name: str,
    n: int,
    du: float,
    *,
    one_sided: bool,
    dtype_name: str,
) -> np.ndarray:
    filter_name = _normalize_filter_name(name)
    if filter_name == "ramp":
        H = _ramp_filter_np(n, du)
    elif filter_name == "shepp-logan":
        H = _shepp_logan_filter_np(n, du)
    elif filter_name == "hann":
        H = _hann_filter_np(n, du)
    else:
        raise ValueError(f"Unknown filter {name}")
    if one_sided:
        H = H[: int(n) // 2 + 1]
    dtype = np.float64 if dtype_name in {"float64", "complex128"} else np.float32
    out = np.asarray(H, dtype=dtype)
    out.setflags(write=False)
    return out


def get_filter_np(name: str, n: int, du: float) -> np.ndarray:
    """Host-side filter lookup with a small explicit LRU cache.

    Returns a read-only np.float32 array of length ``n``. Callers that
    need to mutate filter coefficients must copy the returned array first.
    """
    return _build_filter_np(
        _normalize_filter_name(name),
        int(n),
        float(du),
        one_sided=False,
        dtype_name="float32",
    )


def get_rfft_filter_np(name: str, n: int, du: float, dtype_name: str) -> np.ndarray:
    """Return the immutable one-sided RFFT filter for row filtering."""
    return _build_filter_np(
        _normalize_filter_name(name),
        int(n),
        float(du),
        one_sided=True,
        dtype_name=str(dtype_name),
    )


def get_filter(name: str, n: int, du: float) -> jnp.ndarray:
    """JAX array wrapper for cached, host-computed filters."""
    H_np = get_filter_np(name, n, du)
    return jnp.asarray(H_np, dtype=jnp.float32)


def clear_filter_caches() -> None:
    """Clear cached host filter coefficients."""
    _build_filter_np.cache_clear()
