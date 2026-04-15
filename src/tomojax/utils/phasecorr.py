from __future__ import annotations

import jax.numpy as jnp


def _wrap_shift(idx: jnp.ndarray | int, n: int) -> jnp.ndarray:
    """Convert argmax indices to signed shifts on the half-open wrapped interval.

    For even ``n`` this is ``[-n/2, n/2)``; for odd ``n`` the central positive
    index is preserved and the interval becomes ``[-floor(n/2), ceil(n/2))``.
    """
    threshold = (int(n) + 1) // 2
    idx_arr = jnp.asarray(idx)
    return jnp.where(idx_arr < threshold, idx_arr, idx_arr - int(n))


def phase_corr_shift(ref: jnp.ndarray, tgt: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate (du, dv) translation from ref -> tgt via phase correlation.

    Inputs are 2D arrays (nv, nu). Returns integer-pixel shifts in (u, v).
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
    du = _wrap_shift(u_idx, nu).astype(jnp.float32)
    dv = _wrap_shift(v_idx, nv).astype(jnp.float32)
    return du, dv
