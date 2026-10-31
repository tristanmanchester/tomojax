"""Storage and vector operations for one joint CG implementation.

Host vectors are owned C-order FP32 arrays. Only those owned vectors may be
updated in place; right-hand sides, diagonals and operator outputs are borrowed.
Diagonal CG regenerates its preconditioned residual in bounded chunks.
Device vectors use the same combinations functionally inside a compiled loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import fft as host_fft

type BlockVector = tuple[jax.Array | np.ndarray, ...]
type Scalar = jax.Array | np.ndarray | np.floating | np.integer | float | int
type Predicate = jax.Array | np.ndarray | np.bool_ | bool

_CHUNK = 2**20


def _flat(array):
    # A noncontiguous borrowed product must not require a full-volume copy.
    return array.reshape(-1) if array.flags.c_contiguous else array.flat


def host_dot(left, right) -> float:
    total = 0.0
    for a, b in zip(left, right, strict=True):
        av, bv = _flat(a), _flat(b)
        for start in range(0, a.size, _CHUNK):
            chunk = slice(start, start + _CHUNK)
            total += float(np.einsum("i,i->", av[chunk], bv[chunk], dtype=np.float64))
    return total


@dataclass(frozen=True)
class BlockPlacement:
    """Host or device storage and execution for block-vector operations."""

    host: bool = False

    @property
    def array(self):
        return np if self.host else jnp

    @property
    def fft(self):
        return host_fft if self.host else jnp.fft

    def zeros(self, block) -> BlockVector:
        if self.host:
            return tuple(np.zeros(a.shape, np.float32) for a in block)
        return tuple(jnp.zeros_like(a) for a in block)

    def own(self, block) -> BlockVector:
        if self.host:
            return tuple(np.array(a, dtype=np.float32, order="C", copy=True) for a in block)
        return block

    def reset(self, owned, borrowed) -> BlockVector:
        if self.host:
            for target, source in zip(owned, borrowed, strict=True):
                np.copyto(target, source)
            return owned
        return borrowed

    def multiply(self, owned, factor):
        if self.host:
            np.multiply(owned, factor, out=owned)
            return owned
        return owned * factor

    def map_owned(self, owned, function):
        if not self.host:
            return function(owned)
        flat = _flat(owned)
        for start in range(0, owned.size, _CHUNK):
            flat[start : start + _CHUNK] = function(flat[start : start + _CHUNK])
        return owned

    def combine(self, owned, other, *, scale: Scalar = 1, other_scale: Scalar = 1) -> BlockVector:
        if not self.host:
            return tuple(scale * a + other_scale * b for a, b in zip(owned, other, strict=True))
        for target, source in zip(owned, other, strict=True):
            a, b = _flat(target), _flat(source)
            for start in range(0, target.size, _CHUNK):
                end = start + _CHUNK
                if scale != 1:
                    a[start:end] *= np.float32(float(scale))
                a[start:end] += np.float32(float(other_scale)) * b[start:end]
        return owned

    def dot(self, left, right) -> Scalar:
        if self.host:
            return host_dot(left, right)
        return sum(
            jnp.vdot(a, b, precision=jax.lax.Precision.HIGHEST).real
            for a, b in zip(left, right, strict=True)
        )

    def diagonal_direction(self, direction, residual, inverse, gamma, dot):
        """Update a CG direction without a host-sized preconditioned residual."""
        if not self.host:
            z = tuple(r * m for r, m in zip(residual, inverse, strict=True))
            next_gamma = dot(residual, z)
            beta = next_gamma / jnp.where(gamma > 0, gamma, 1)
            return self.combine(direction, z, scale=beta), next_gamma
        dtype = np.result_type(*(a.dtype for a in (*residual, *inverse)))
        scratch = np.empty(min(_CHUNK, max(a.size for a in residual)), dtype)
        next_gamma = 0.0
        # Round each product before the FP64 dot, in host_dot's chunk order.
        for r, m in zip(residual, inverse, strict=True):
            rf, mf = _flat(r), _flat(m)
            for start in range(0, r.size, _CHUNK):
                stop = min(start + _CHUNK, r.size)
                z = scratch[: stop - start]
                np.multiply(rf[start:stop], mf[start:stop], out=z)
                next_gamma += float(np.einsum("i,i->", rf[start:stop], z, dtype=np.float64))
        beta = np.float32(float(next_gamma / np.where(gamma > 0, gamma, 1)))
        for p, r, m in zip(direction, residual, inverse, strict=True):
            pf, rf, mf = _flat(p), _flat(r), _flat(m)
            for start in range(0, r.size, _CHUNK):
                stop = min(start + _CHUNK, r.size)
                z = scratch[: stop - start]
                np.multiply(rf[start:stop], mf[start:stop], out=z)
                pf[start:stop] *= beta
                pf[start:stop] += z
        return direction, next_gamma

    def while_loop(self, condition, body, state):
        if self.host:
            while condition(state):
                state = body(state)
            return state
        return jax.lax.while_loop(condition, body, state)


DEVICE = BlockPlacement()
HOST = BlockPlacement(host=True)
