"""Block floating-point storage for the three components of the SPDHG TV dual."""

from __future__ import annotations

import jax
import jax.numpy as jnp

_GROUP = 8
type Shape = tuple[int, int, int]


def _buffers(shape: Shape, *, enabled: bool) -> tuple[jax.ShapeDtypeStruct, ...]:
    planes, rows, columns = shape
    area = rows * columns
    tail = (planes, area + (area + _GROUP - 1) // _GROUP)
    return tuple(
        jax.ShapeDtypeStruct(s if enabled else (0,), dtype)
        for s, dtype in ((shape, jnp.uint32), (shape, jnp.uint32), (tail, jnp.uint8))
    )


def flux_bytes(shape: Shape, *, enabled: bool) -> int:
    """Bytes carried between updates, including the final short group of each plane."""
    return sum(buf.size * buf.dtype.itemsize for buf in _buffers(shape, enabled=enabled))


def empty_flux(shape: Shape, *, enabled: bool) -> tuple[jax.Array, ...]:
    """Zero duals in the same representation as subsequent iterations."""
    return tuple(jnp.zeros(buf.shape, buf.dtype) for buf in _buffers(shape, enabled=enabled))


def pack_flux(p: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    """Pack three signed 24-bit coefficients per voxel, sharing a binary exponent locally.

    At normal signal scales the quantum is twice the FP32 spacing at the group maximum. Rounding
    costs at most one ulp of that maximum, including at the signed positive limit.
    The two words hold bits 0..63 of the concatenated coefficients. The tail
    holds bits 64..71 followed by one exponent byte per group.
    """
    peak = jnp.maximum(jnp.maximum(jnp.abs(p[0]), jnp.abs(p[1])), jnp.abs(p[2]))
    planes, ny, nx = peak.shape
    area = ny * nx
    blocks = (area + _GROUP - 1) // _GROUP
    padded = jnp.pad(peak.reshape(planes, area), ((0, 0), (0, _GROUP * blocks - area)))
    largest = padded.reshape(planes, blocks, _GROUP).max(axis=-1)
    exponent = jax.lax.bitcast_convert_type(largest, jnp.uint32) >> 23
    unit = _flux_unit(exponent, peak.shape)
    a, b, c = (
        jnp.minimum(jnp.rint(v / unit), (1 << 23) - 1).astype(jnp.int32).astype(jnp.uint32)
        for v in p
    )
    return (
        (a & 0xFFFFFF) | (b << 24),
        ((b >> 8) & 0xFFFF) | (c << 16),
        jnp.concatenate(((c >> 16).reshape(planes, area), exponent), axis=-1).astype(jnp.uint8),
    )


def _flux_unit(exponent: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    # 22 magnitude fraction bits. Keep the unit normal for an all-zero group.
    unit = jax.lax.bitcast_convert_type((jnp.maximum(exponent, 23) - 22) << 23, jnp.float32)
    return jnp.repeat(unit, _GROUP, axis=1)[:, : shape[1] * shape[2]].reshape(shape)


def unpack_flux(p: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    """Sign-extend the three coefficients and restore the physical scale."""
    a, b, tail = p
    area = a.shape[1] * a.shape[2]
    c = tail[:, :area].astype(jnp.uint32).reshape(a.shape)
    unit = _flux_unit(tail[:, area:].astype(jnp.uint32), a.shape)
    words = (a, (b << 8) | (a >> 24), (c << 16) | (b >> 16))
    return tuple(((word << 8).astype(jnp.int32) >> 8).astype(jnp.float32) * unit for word in words)
