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


def fft_filter_rows(
    rows: jnp.ndarray, rfft_filter: jnp.ndarray, *, edge_width: int | None = None
) -> jnp.ndarray:
    """Filter zero-extended rows, optionally continuing measured endpoints to infinity.

    ``edge_width`` is the measured width of symmetrically zero-padded ``rows``.
    Its endpoints supply constant exterior line integrals. Boundary-tail sums
    compensate the zero-extended convolution without extra detector/FFT storage.
    The ramp has zero infinite DC: symmetry fixes each half-sum to ``-h[0]/2``.
    The same completion is used for the smoothed finite discrete filters.
    """
    nu = int(rows.shape[-1])
    n_fft = 2 * (int(rfft_filter.shape[0]) - 1)
    pad = 0 if edge_width is None else (nu - edge_width) // 2
    index = jnp.arange(nu) - pad
    if edge_width is not None:
        # The infinite ramp annihilates a constant. Remove the left endpoint
        # before the FFT, avoiding cancellation between large finite-row and
        # boundary responses (and preserving constant rows exactly).
        rows = jnp.where((index >= 0) & (index < edge_width), rows - rows[..., pad, None], 0)
    F = jnp.fft.rfft(rows, n=n_fft, axis=-1)
    filtered = jnp.fft.irfft(F * rfft_filter, n=n_fft, axis=-1)[..., :nu]
    if edge_width is None:
        return filtered
    impulse = jnp.fft.irfft(rfft_filter, n=n_fft)
    cumulative = jnp.concatenate([jnp.zeros((1,), impulse.dtype), jnp.cumsum(impulse[1:nu])])
    tails = -0.5 * impulse[0] - cumulative

    def tail_at(index: jnp.ndarray) -> jnp.ndarray:
        # sum_{k > i} h[k], with h even and its infinite sum zero.
        distance = jnp.where(index < 0, -index - 1, index)
        return jnp.where(index < 0, -1.0, 1.0) * tails[distance]

    right = rows[..., nu - pad - 1, None] * tail_at(edge_width - 1 - index)
    return filtered + right


def clear_filter_caches() -> None:
    """Clear cached host filter coefficients."""
    _build_filter_np.cache_clear()
    get_fbp_filter_np.cache_clear()
