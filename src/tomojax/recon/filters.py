from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import numpy as np
import jax.numpy as jnp


_FILTER_CACHE: "OrderedDict[Tuple[str, int, float], np.ndarray]" = OrderedDict()
_FILTER_CACHE_CAP = 8


def _normalize_filter_name(name: str | None) -> str:
    """Canonicalize filter aliases and preserve the default ramp fallback."""
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


def get_filter_np(name: str, n: int, du: float) -> np.ndarray:
    """Host-side filter lookup with a small LRU cache.

    Returns an np.float32 array of length ``n``.
    """
    filter_name = _normalize_filter_name(name)
    key = (filter_name, int(n), float(du))
    if key in _FILTER_CACHE:
        _FILTER_CACHE.move_to_end(key)
        return _FILTER_CACHE[key]
    if filter_name in ("ramp", "ram-lak", "ramlak"):
        H = _ramp_filter_np(n, du)
    elif filter_name in ("shepp", "shepp-logan", "shepplogan"):
        H = _shepp_logan_filter_np(n, du)
    elif filter_name in ("hann", "hanning"):
        H = _hann_filter_np(n, du)
    else:
        raise ValueError(f"Unknown filter {name}")
    _FILTER_CACHE[key] = H
    if len(_FILTER_CACHE) > _FILTER_CACHE_CAP:
        _FILTER_CACHE.popitem(last=False)
    return H


def get_filter(name: str, n: int, du: float) -> jnp.ndarray:
    """JAX array wrapper for cached, host-computed filters."""
    H_np = get_filter_np(name, n, du)
    return jnp.asarray(H_np, dtype=jnp.float32)
