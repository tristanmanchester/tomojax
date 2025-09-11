from __future__ import annotations

import jax.numpy as jnp


def ramp_filter(n: int, du: float) -> jnp.ndarray:
    # Frequency coordinates (cycles per unit)
    freqs = jnp.fft.fftfreq(n, d=du)
    return jnp.abs(freqs)


def shepp_logan_filter(n: int, du: float) -> jnp.ndarray:
    f = ramp_filter(n, du)
    freqs = jnp.fft.fftfreq(n, d=du)
    fmax = jnp.max(jnp.abs(freqs)) + 1e-12
    x = freqs / (2.0 * fmax)
    # normalized sinc: sin(pi x)/(pi x), set sinc(0)=1
    sinc = jnp.where(x == 0, 1.0, jnp.sin(jnp.pi * x) / (jnp.pi * x))
    return f * sinc


def hann_filter(n: int, du: float) -> jnp.ndarray:
    f = ramp_filter(n, du)
    # Create Hann window in frequency with cutoff at Nyquist
    freqs = jnp.fft.fftfreq(n, d=du)
    fmax = jnp.max(jnp.abs(freqs)) + 1e-12
    w = 0.5 + 0.5 * jnp.cos(jnp.pi * freqs / fmax)
    return f * w


def get_filter(name: str, n: int, du: float) -> jnp.ndarray:
    name = (name or "ramp").lower()
    if name in ("ramp", "ram-lak", "ramlak"):
        return ramp_filter(n, du)
    if name in ("shepp", "shepp-logan", "shepplogan"):
        return shepp_logan_filter(n, du)
    if name in ("hann", "hanning"):
        return hann_filter(n, du)
    raise ValueError(f"Unknown filter {name}")
