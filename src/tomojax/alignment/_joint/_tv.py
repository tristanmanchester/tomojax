"""Anisotropic physical Huber TV on a scalar field, for lagged-diffusivity steps.

``scales`` are (h/hz, h/hy, h/hx) with h = (hx hy hz)**(1/3), so equal pitches
give the ordinary isotropic TV. Boundary faces are zero-Neumann: a forward
difference off the grid is zero, never wrapped. Differences are FP32 and local
norms and sums FP64.
"""

import jax
import jax.numpy as jnp

F32 = jnp.float32
F64 = jnp.float64


def _behind(x, axis):
    pads = [(0, 0)] * 3
    pads[axis] = (1, 0)
    return jnp.pad(jax.lax.slice_in_dim(x, 0, x.shape[axis] - 1, axis=axis), pads)


def _forward_diff(x, axis):
    pads = [(0, 0)] * 3
    pads[axis] = (0, 1)
    return jnp.pad(jnp.diff(x, axis=axis), pads)


def _magnitude(x, scales):
    q = jnp.zeros_like(x, dtype=F64)
    for a in range(3):
        diff = _forward_diff(x, a).astype(F64) * jnp.asarray(scales[a], F64)
        q = q + diff * diff
    return jnp.sqrt(q)


def conductance(x, eps, scales):
    """Lagged diffusivity 1 / max(eps, |grad x|)."""
    return (1 / jnp.maximum(jnp.asarray(eps, F64), _magnitude(x, scales))).astype(F32)


def energy(x, eps, scales):
    """Huber TV with width ``eps``."""
    t = _magnitude(x, scales)
    eps = jnp.asarray(eps, F64)
    return jnp.sum(jnp.where(t <= eps, t * t / (2 * eps), t - eps / 2), dtype=F64)


def diffusion(x, d, scales):
    """The lagged TV operator applied to ``x`` with conductance ``d``."""
    out = jnp.zeros_like(x, dtype=F64)
    for a in range(3):
        positive = (d * _forward_diff(x, a)) * jnp.asarray(scales[a] ** 2, F32)
        out = out - positive.astype(F64) + _behind(positive, a).astype(F64)
    return out.astype(F32)


def inverse_diagonal(d, damp, data_diag, lam, lm, scales):
    """Jacobi inverse of data diagonal + lm * damping + lam * TV diagonal."""
    tv = jnp.zeros_like(d, dtype=F64)
    for a in range(3):
        last = jnp.arange(d.shape[a]).reshape([-1 if i == a else 1 for i in range(3)])
        positive = jnp.where(last < d.shape[a] - 1, d, 0) * jnp.asarray(scales[a] ** 2, F32)
        tv = tv + positive.astype(F64) + _behind(positive, a).astype(F64)
    v = data_diag + F32(lm) * damp + F32(lam) * tv.astype(F32)
    return 1 / jnp.maximum(v, F32(1e-8))


def dot(a, b):
    """FP32 product with FP64 accumulation."""
    return jnp.sum((a * b).astype(F64), dtype=F64)
