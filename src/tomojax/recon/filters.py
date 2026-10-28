"""Frequency-domain filters for filtered backprojection."""

from __future__ import annotations

import functools
import math
import operator

import jax.numpy as jnp
import numpy as np


def _normalize_filter_name(name: str | None) -> str:
    """Normalize filter names and preserve the default ramp fallback."""
    normalized = str(name or "").strip().lower()
    return normalized or "ramp"


@functools.lru_cache(maxsize=16)
def get_fbp_filter_np(name: str, nu: int, du: float, dtype_name: str) -> np.ndarray:
    """Return a padded discrete ramp filter for a finite detector row.

    The spatial ramp is h[0] = 1/(4*du), h[k] = -1/(pi**2*k**2*du)
    for odd k, and zero for nonzero even k. Keeping its finite DC coefficient
    avoids the bias introduced by sampling abs(f) with an exactly zero DC bin.
    Padding to at least twice the detector width prevents circular wraparound
    in the retained output pixels. The caller must crop the inverse FFT to nu.
    """
    if isinstance(nu, bool) or operator.index(nu) <= 0:
        raise ValueError("nu must be a positive integer")
    if not math.isfinite(du) or du <= 0:
        raise ValueError("du must be finite and > 0")
    n = max(64, 1 << (2 * int(nu) - 1).bit_length())
    k = np.arange(n, dtype=np.int64)
    k = np.minimum(k, n - k)
    impulse = np.zeros(n, dtype=np.float64)
    impulse[0] = 0.25 / du
    odd = (k % 2) != 0
    impulse[odd] = -1.0 / (np.pi**2 * k[odd] ** 2 * du)
    spectrum = np.fft.rfft(impulse).real
    normalized_frequency = 2.0 * np.fft.rfftfreq(n)
    filter_name = _normalize_filter_name(name)
    if filter_name == "shepp-logan":
        spectrum *= np.sinc(normalized_frequency / 2.0)
    elif filter_name == "hann":
        spectrum *= 0.5 + 0.5 * np.cos(np.pi * normalized_frequency)
    elif filter_name != "ramp":
        raise ValueError(f"Unknown filter {name}")
    dtype = np.float64 if dtype_name in {"float64", "complex128"} else np.float32
    out = np.asarray(spectrum, dtype=dtype)
    out.setflags(write=False)
    return out


def rfft_filter_array(filter: str, nu: int, du: float, dtype: jnp.dtype) -> jnp.ndarray:
    """Return the padded discrete FBP filter without doubling RFFT bins."""
    Hr_np = get_fbp_filter_np(filter, int(nu), float(du), str(dtype))
    return jnp.asarray(Hr_np, dtype=dtype)


def fft_filter_rows(rows: jnp.ndarray, rfft_filter: jnp.ndarray) -> jnp.ndarray:
    """Zero-pad detector rows, filter, and crop to prevent circular wraparound."""
    nu = int(rows.shape[-1])
    n_fft = 2 * (int(rfft_filter.shape[0]) - 1)
    F = jnp.fft.rfft(rows, n=n_fft, axis=-1)
    return jnp.fft.irfft(F * rfft_filter, n=n_fft, axis=-1)[..., :nu]


def clear_filter_caches() -> None:
    """Clear cached host filter coefficients."""
    get_fbp_filter_np.cache_clear()
