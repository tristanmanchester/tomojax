"""Storage and vector operations for one joint CG implementation.

Host vectors are owned C-order FP32 arrays. Only those owned vectors may be
updated in place; right-hand sides, diagonals and operator outputs are borrowed.
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

    def while_loop(self, condition, body, state):
        if self.host:
            while condition(state):
                state = body(state)
            return state
        return jax.lax.while_loop(condition, body, state)


DEVICE = BlockPlacement()
HOST = BlockPlacement(host=True)
