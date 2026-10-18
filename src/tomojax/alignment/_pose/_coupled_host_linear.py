"""PCG state on the host, with normal products supplied by bounded device passes.

Only the direction and one normal-product accumulator need to visit the GPU.
Vector updates use bounded host chunks, so expression temporaries never add
another volume to the working set. The normal callback may return read-only
host views; its input remains owned by this solver and must not be modified.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, cast

import numpy as np

type HostBlock = tuple[np.ndarray, np.ndarray]
_CHUNK = 2**20


class HostLinearResult(NamedTuple):
    increment: HostBlock
    iterations: int
    relative_residual: float
    finite: bool


def host_dot(a: HostBlock, b: HostBlock) -> float:
    total = 0.0
    for pair in zip(a, b, strict=True):
        left, right = (value.reshape(-1) for value in pair)
        for start in range(0, left.size, _CHUNK):
            end = start + _CHUNK
            # Accumulate small block reductions in double precision, without
            # making double-precision volume copies.
            total += float(np.einsum("i,i->", left[start:end], right[start:end], dtype=np.float64))
    return total


def _axpy(out: HostBlock, scale: float, vector: HostBlock) -> None:
    for pair in zip(out, vector, strict=True):
        target, source = (value.reshape(-1) for value in pair)
        for start in range(0, target.size, _CHUNK):
            end = start + _CHUNK
            target[start:end] += np.float32(scale) * source[start:end]


def _direction(out: HostBlock, residual: HostBlock, inverse: HostBlock, beta: float) -> float:
    """Set p = M^-1 r + beta p and return r.T M^-1 r, with one chunk of scratch."""
    total = 0.0
    for triple in zip(out, residual, inverse, strict=True):
        p, r, diagonal = (value.reshape(-1) for value in triple)
        for start in range(0, p.size, _CHUNK):
            end = start + _CHUNK
            z = r[start:end] * diagonal[start:end]
            total += float(np.einsum("i,i->", r[start:end], z, dtype=np.float64))
            p[start:end] *= np.float32(beta)
            p[start:end] += z
    return total


def solve_host_normal(
    normal: Callable[[HostBlock], HostBlock],
    rhs: HostBlock,
    inverse_diagonal: HostBlock,
    *,
    max_iters: int,
    rtol: float,
) -> HostLinearResult:
    """The same preconditioned block CG as ``solve_coupled_normal``, staged on host.

    The solver owns three FP32 arrays per block: solution, residual, direction.
    ``normal`` owns its returned product until the next call. It can stage GPU
    work independently of these host arrays. The explicit final residual uses
    the original RHS, and an exhausted iteration budget is not convergence.
    """
    # Chunked ravel updates must be views, even when the caller owns Fortran arrays.
    solution = cast("HostBlock", tuple(np.zeros(a.shape, np.float32) for a in rhs))
    residual = cast(
        "HostBlock", tuple(np.array(a, dtype=np.float32, order="C", copy=True) for a in rhs)
    )
    direction = cast("HostBlock", tuple(np.zeros_like(a) for a in solution))
    gamma = _direction(direction, residual, inverse_diagonal, 0.0)
    rhs_squared = host_dot(rhs, rhs)
    threshold = rtol**2 * rhs_squared
    failed, count = False, 0
    while (
        count < max_iters and host_dot(residual, residual) > threshold and gamma > 0 and not failed
    ):
        product = normal(direction)
        denominator = host_dot(direction, product)
        valid = np.isfinite(denominator) and denominator > 0 and np.isfinite(gamma)
        alpha = gamma / denominator if valid else 0.0
        _axpy(solution, alpha, direction)
        _axpy(residual, -alpha, product)
        del product
        # Computing gamma first avoids a full preconditioned residual array.
        next_gamma = 0.0
        for pair in zip(residual, inverse_diagonal, strict=True):
            r, inv = (value.reshape(-1) for value in pair)
            for start in range(0, r.size, _CHUNK):
                end = start + _CHUNK
                z = r[start:end] * inv[start:end]
                next_gamma += float(np.einsum("i,i->", r[start:end], z, dtype=np.float64))
        _direction(direction, residual, inverse_diagonal, next_gamma / gamma if gamma > 0 else 0)
        gamma = next_gamma
        failed = not valid or not np.isfinite(gamma)
        count += 1
    del residual, direction
    product = normal(solution)
    # Reuse a solver-owned buffer for the true residual; do not mutate the RHS
    # or the callback's potentially read-only result.
    relative_squared = 0.0
    for pair in zip(rhs, product, strict=True):
        b, ax = (value.reshape(-1) for value in pair)
        for start in range(0, b.size, _CHUNK):
            difference = b[start : start + _CHUNK] - ax[start : start + _CHUNK]
            relative_squared += float(np.einsum("i,i->", difference, difference, dtype=np.float64))
    relative = np.sqrt(relative_squared / max(rhs_squared, 1e-30))
    return HostLinearResult(
        solution, count, float(relative), bool(not failed and np.isfinite(relative))
    )
