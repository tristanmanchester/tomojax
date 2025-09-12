from __future__ import annotations

import jax
import jax.numpy as jnp


def _wrap_shift(idx: int, n: int) -> int:
    # Convert argmax index to signed shift in [-n/2, n/2)
    return idx if idx <= n // 2 else idx - n


def phase_corr_shift(ref: jnp.ndarray, tgt: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate (du, dv) translation from ref -> tgt via phase correlation.

    Inputs are 2D arrays (nv, nu). Returns shifts in pixel units (u, v).
    Uses integer-precision peak (robust and cheap); refine later if needed.
    """
    ref = ref.astype(jnp.float32)
    tgt = tgt.astype(jnp.float32)
    F1 = jnp.fft.rfft2(ref)
    F2 = jnp.fft.rfft2(tgt)
    G = F1 * jnp.conj(F2)
    denom = jnp.maximum(jnp.abs(G), 1e-6)
    R = G / denom
    corr = jnp.fft.irfft2(R, s=ref.shape)
    # Locate peak
    flat_idx = jnp.argmax(corr)
    nv, nu = ref.shape
    v_idx = (flat_idx // nu).astype(jnp.int32)
    u_idx = (flat_idx % nu).astype(jnp.int32)
    du = jnp.float32(_wrap_shift(int(u_idx), int(nu)))
    dv = jnp.float32(_wrap_shift(int(v_idx), int(nv)))
    return du, dv

