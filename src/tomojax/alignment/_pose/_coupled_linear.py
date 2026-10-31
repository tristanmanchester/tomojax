"""Matrix-free normal-equation solves for coupled volume and pose increments."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax

from ._coupled_vectors import DEVICE, BlockPlacement, BlockVector, Predicate, Scalar


class CoupledLinearResult(NamedTuple):
    increment: BlockVector
    iterations: Scalar
    relative_residual: Scalar
    finite: Predicate


def _relative_residual(normal, solution, rhs, squared, scratch, placement, dot):
    residual = placement.combine(placement.reset(scratch, rhs), normal(solution), other_scale=-1)
    return placement.array.sqrt(dot(residual, residual) / placement.array.maximum(squared, 1e-30))


def solve_coupled_normal(
    normal: Callable[[BlockVector], BlockVector],
    rhs: BlockVector,
    inverse_diagonal: BlockVector,
    *,
    max_iters: int,
    rtol: float | jax.Array,
    dot: Callable[[BlockVector, BlockVector], Scalar] | None = None,
    placement: BlockPlacement = DEVICE,
    volume_ramp: float | None = None,
) -> CoupledLinearResult:
    """Solve a positive (semi)definite block normal system by preconditioned CG.

    The two arrays may have different shapes: no dense volume matrix, joint
    Jacobian, or iteration history is materialised. ``inverse_diagonal`` must
    be positive on the active subspace. Inactive coordinates have zero RHS and
    zero operator coupling. A final *explicit* normal residual is returned;
    exhausting the budget never implies convergence. The nonlinear caller must
    independently check its actual constrained objective before accepting a step.
    ``dot`` supplies the device reduction, for example summing across devices
    when each holds part of a block. Host vectors use the chunked Euclidean
    inner product of their placement.
    """
    xp = placement.array
    dot = placement.dot if dot is None else dot
    zero = placement.zeros(rhs)
    residual = placement.own(rhs)

    def precondition(r: BlockVector) -> BlockVector:
        if volume_ramp is None:
            return r[0] * inverse_diagonal[0], r[1] * inverse_diagonal[1]
        volume, pose = r
        fft = placement.fft
        scale = xp.sqrt(inverse_diagonal[0])
        nx, ny, _ = volume.shape
        fx = fft.fftfreq(nx).astype(volume.dtype)[:, None]
        fy = fft.rfftfreq(ny).astype(volume.dtype)[None, :]
        ramp = xp.sqrt(fx * fx + fy * fy + 1 / (nx * ny))
        ramp = ramp / (1 + volume_ramp * (nx * ny) ** 0.5 * ramp)
        spectrum = fft.rfft2(scale * volume, axes=(0, 1))
        filtered = xp.asarray(fft.irfft2(spectrum * ramp[:, :, None], s=(nx, ny), axes=(0, 1)))
        return scale * filtered, pose * inverse_diagonal[1]

    z0 = precondition(residual)
    gamma0 = dot(rhs, z0)
    rhs_squared = dot(rhs, rhs)
    threshold = xp.square(xp.asarray(rtol, xp.asarray(rhs_squared).dtype)) * rhs_squared

    def condition(state):
        iteration, _, residual, _, gamma, failed = state
        return (
            (iteration < max_iters) & (dot(residual, residual) > threshold) & (gamma > 0) & ~failed
        )

    def body(state):
        iteration, x, residual, direction, gamma, failed = state
        normal_direction = normal(direction)
        denominator = dot(direction, normal_direction)
        valid = xp.isfinite(denominator) & (denominator > 0) & xp.isfinite(gamma)
        alpha = xp.where(valid, gamma / xp.where(valid, denominator, 1), 0)
        updated = placement.combine(x, direction, other_scale=alpha)
        next_residual = placement.combine(residual, normal_direction, other_scale=-alpha)
        # Release the borrowed normal output before applying the preconditioner.
        del normal_direction
        if volume_ramp is None:
            next_direction, next_gamma = placement.diagonal_direction(
                direction, next_residual, inverse_diagonal, gamma, dot
            )
        else:
            z = precondition(next_residual)
            next_gamma = dot(next_residual, z)
            beta = next_gamma / xp.where(gamma > 0, gamma, 1)
            next_direction = placement.combine(direction, z, scale=beta)
        failed = failed | ~valid | ~xp.isfinite(next_gamma)
        return iteration + 1, updated, next_residual, next_direction, next_gamma, failed

    count, solution, residual, direction, _, failed = placement.while_loop(
        condition, body, (xp.int32(0), zero, residual, z0, gamma0, xp.bool_(False))
    )
    del direction, z0
    relative = _relative_residual(normal, solution, rhs, rhs_squared, residual, placement, dot)
    finite = ~failed & xp.isfinite(relative)
    return CoupledLinearResult(solution, count, relative, finite)


def solve_schur_normal(
    normal,
    schur,
    rhs,
    volume_inverse,
    cross,
    pose_rows,
    solve_pose,
    *,
    free,
    movable,
    max_iters,
    rtol,
    placement=DEVICE,
    dot=None,
    volume_ramp=None,
):
    """Eliminate pose blocks, retaining the full system's stopping scale.

    ``cross`` returns a fresh owned volume; it becomes the reduced RHS and
    finally residual scratch. Normal products remain borrowed and read-only.
    """
    xp = placement.array
    dot = placement.dot if dot is None else dot
    empty = xp.zeros((0,), rhs[0].dtype)
    reduced_rhs = placement.multiply(cross(solve_pose(rhs[1])), free)
    reduced_rhs, _ = placement.combine((reduced_rhs, empty), (rhs[0], empty), scale=-1)
    squared = dot(rhs, rhs)
    reduced_squared = dot((reduced_rhs, empty), (reduced_rhs, empty))
    # Back-substitution makes the full residual the volume Schur residual.
    # Match the stacked solve's absolute threshold, not the reduced RHS norm.
    reduced = solve_coupled_normal(
        schur,
        (reduced_rhs, empty),
        (volume_inverse, empty),
        max_iters=max_iters,
        rtol=rtol * xp.sqrt(squared / xp.maximum(reduced_squared, 1e-30)),
        placement=placement,
        dot=dot,
        volume_ramp=volume_ramp,
    )
    dx = placement.multiply(reduced.increment[0], free)
    dp = movable * solve_pose(rhs[1] - pose_rows(dx))
    scratch = (reduced_rhs, placement.own((rhs[1],))[0])
    relative = _relative_residual(normal, (dx, dp), rhs, squared, scratch, placement, dot)
    return CoupledLinearResult(
        (dx, dp), reduced.iterations, relative, reduced.finite & xp.isfinite(relative)
    )
