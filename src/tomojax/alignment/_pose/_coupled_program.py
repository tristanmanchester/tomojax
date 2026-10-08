"""Reusable compiled joint objectives with scan arrays kept out of cache keys."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp

from tomojax.alignment._geometry.parametrizations import apply_pose_updates
from tomojax.core.cone import cone_backproject, cone_project, frame_coefficients
from tomojax.core.joseph import (
    forward_project_planes,
    plane_coefficients,
    sum_backproject_planes,
)
from tomojax.core.projector import forward_project_view_T, sum_backproject_views_T
from tomojax.core.trilinear import exact_adjoint, exact_forward
from tomojax.recon.fista_tv_core import FistaCoreConfig, regulariser_value_arrays

from ._coupled_linear import CoupledLinearResult, solve_coupled_normal
from ._pose_block import pose_block_solver
from ._pose_jacobian import PoseJacobianOptions, pose_prediction_and_columns

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Grid
    from tomojax.recon.types import Regulariser


# Views projected together in the update's data-space passes.
_VIEWS_PER_BATCH = 32


class _BatchOps(NamedTuple):
    """One pose stack's operators over a batch of views starting at ``start``."""

    cached: jax.Array | None  # every view's pose columns, when they fit
    view_columns: Callable  # view i's weighted pose columns
    batch_columns: Callable  # the batch's pose columns
    project: Callable  # (start, dx) -> W A dx over the batch
    transpose: Callable  # (start, valid, y, out) -> out + A^T W y over its new views
    pose_forward: Callable  # (start, cols, dp) -> C dp over the batch
    pose_adjoint: Callable  # (start, cols, y, out) -> out with the batch's rows C^T y


@dataclass(frozen=True)
class CoupledSpec:
    """Only choices that change the compiled program, never measurement arrays."""

    grid: Grid
    detector: Detector
    backend: Literal["jax", "pallas"]
    jacobian: PoseJacobianOptions
    cache_columns: bool
    regulariser: Regulariser
    huber_delta: float
    lambda_tv: float
    recon_positivity: bool
    gn_volume_damping: float
    gn_damping: float
    gn_joint_solver: Literal["stacked", "pose_eliminated"]
    gn_joint_rtol: float
    gn_joint_iters: int
    has_smoothness: bool


class CoupledArrays(NamedTuple):
    """Per-scan numerical inputs; changing values can reuse a compiled program."""

    poses: jax.Array
    projections: jax.Array
    # Per-pixel least-squares weights, or a scalar when every pixel weighs the same.
    weights: jax.Array
    mask: jax.Array
    active: jax.Array
    smoothness: jax.Array
    det_grid: tuple[jax.Array, jax.Array]
    frames: jax.Array | None = None


def _build_program(
    spec: CoupledSpec, arrays: CoupledArrays
) -> tuple[
    Callable[[jax.Array, jax.Array], CoupledLinearResult],
    Callable[[jax.Array, jax.Array], jax.Array],
]:
    # This factory executes during tracing, behind stable module-level jit
    # functions. Its closures contain tracers rather than embedded scan arrays.
    cfg, backend = spec, spec.backend
    det_grid = None if backend == "pallas" else arrays.det_grid
    mask, weights, active = arrays.mask, arrays.weights, arrays.active
    # A scalar mask means none: skip multiplying volumes by it, which would copy them.
    unmasked = mask.ndim == 0

    def masked(v):
        return v if unmasked else mask * v

    n_views = arrays.poses.shape[0]
    cache_columns = spec.cache_columns
    predict_columns = partial(
        pose_prediction_and_columns,
        grid=spec.grid,
        detector=spec.detector,
        det_grid=arrays.det_grid,
        options=spec.jacobian,
    )
    reg_cfg = FistaCoreConfig(regulariser=cfg.regulariser, huber_delta=cfg.huber_delta)

    def regularisation(x):
        if cfg.lambda_tv == 0:
            return jnp.float32(0)
        return cfg.lambda_tv * regulariser_value_arrays(x, reg_cfg)

    def smoothness(p):
        d2 = p[:-2] - 2 * p[1:-1] + p[2:]
        return jnp.sum(jnp.square(d2 * arrays.smoothness))

    def poses(p):
        return apply_pose_updates(
            arrays.poses, p, translation_frame=spec.jacobian.translation_frame
        )

    joseph = {"joseph": "linear", "joseph_cubic": "cubic"}.get(spec.jacobian.integrator)

    cone_backend = "cuda" if backend == "pallas" else "jax"

    def forward(t, x, frames=None):
        if spec.jacobian.cone_beam:  # each view in its lab frame
            assert frames is not None
            coeff = frame_coefficients(t, frames, spec.grid, spec.detector)
            return cone_project(masked(x), coeff, spec.grid, spec.detector, backend=cone_backend)
        if joseph is not None:
            return forward_project_planes(
                plane_coefficients(t, spec.grid, spec.detector, arrays.det_grid),
                masked(x),
                spec.grid,
                spec.detector,
                backend=backend,
                interpolation=joseph,
            )
        if spec.jacobian.integrator == "exact":
            return exact_forward(
                t, spec.grid, spec.detector, masked(x), backend=backend, det_grid=det_grid
            )
        return jax.lax.map(
            lambda ti: forward_project_view_T(
                ti,
                spec.grid,
                spec.detector,
                masked(x),
                gather_dtype="fp32",
                det_grid=det_grid,
                unroll=spec.jacobian.unroll,
                use_checkpoint=spec.jacobian.checkpoint,
            ),
            t,
        )

    def adjoint(t, y, frames=None):
        if spec.jacobian.cone_beam:
            assert frames is not None
            coeff = frame_coefficients(t, frames, spec.grid, spec.detector)
            return masked(
                cone_backproject(y, coeff, spec.grid, spec.detector, backend=cone_backend)
            )
        if joseph is not None:
            return masked(
                sum_backproject_planes(
                    plane_coefficients(t, spec.grid, spec.detector, arrays.det_grid),
                    y,
                    spec.grid,
                    spec.detector,
                    backend=backend,
                    interpolation=joseph,
                )
            )
        if spec.jacobian.integrator == "exact":
            return masked(
                exact_adjoint(t, spec.grid, spec.detector, y, backend=backend, det_grid=det_grid)
            )

        # Accumulate one volume, never a volume stack per view.
        def add(x, pair):
            ti, yi = pair
            return x + sum_backproject_views_T(
                ti[None],
                spec.grid,
                spec.detector,
                yi[None],
                gather_dtype="fp32",
                det_grid=det_grid,
                unroll=spec.jacobian.unroll,
            ), None

        return masked(
            jax.lax.scan(
                add, jnp.zeros((spec.grid.nx, spec.grid.ny, spec.grid.nz), jnp.float32), (t, y)
            )[0]
        )

    # Data-space work runs over batches of views: projecting, comparing and
    # transposing a batch before the next, so no projection-sized array is
    # stored but the cached pose columns (POSE_WIDTH of them, when they fit)
    # and one in the pose-eliminated solve. The last batch is shifted back to
    # end at the last view; ``valid`` marks its views not seen before.
    size = min(n_views, _VIEWS_PER_BATCH)
    count = -(-n_views // size)
    volume_zeros = jnp.zeros((spec.grid.nx, spec.grid.ny, spec.grid.nz), jnp.float32)

    def take(a, start):
        return jax.lax.dynamic_slice_in_dim(a, start, size)

    def over_batches(body, init):
        def step(i, carry):
            start = jnp.minimum(i * size, n_views - size)
            valid = (start + jnp.arange(size) >= i * size).astype(jnp.float32)
            return body(start, valid, carry)

        return jax.lax.fori_loop(0, count, step, init)

    def batch_weights(start):
        return weights if weights.ndim == 0 else take(weights, start)

    def batch_frames(start):
        return None if arrays.frames is None else take(arrays.frames, start)

    def loss(p, x):
        t = poses(p)

        def body(start, valid, total):
            residual = batch_weights(start) * (
                forward(take(t, start), x, batch_frames(start)) - take(arrays.projections, start)
            )
            squares = jnp.sum(jnp.square(residual), axis=(1, 2))
            return total + 0.5 * jnp.sum(valid * squares)

        return over_batches(body, jnp.float32(0)) + regularisation(x) + smoothness(p)

    highest = jax.lax.Precision.HIGHEST

    def batch_ops(t, p, x):  # the batch operators of one pose stack; see _BatchOps
        def columns(i):
            return (
                predict_columns(
                    p[i],
                    arrays.poses[i],
                    None if arrays.frames is None else arrays.frames[i],
                    masked(x),
                    arrays.projections[i],
                    weights if weights.ndim == 0 else weights[i],
                )[1]
                * active[:, None]
            )

        cached = jax.lax.map(columns, jnp.arange(n_views)) if cache_columns else None

        def view_columns(i):
            return cached[i] if cached is not None else columns(i)

        def batch_columns(start):
            if cached is not None:
                return take(cached, start)
            return jax.lax.map(columns, start + jnp.arange(size))

        def project(start, dx):
            return batch_weights(start) * forward(take(t, start), dx, batch_frames(start))

        def transpose(start, valid, y, out):
            y = batch_weights(start) * y * valid[:, None, None]
            return out + adjoint(take(t, start), y, batch_frames(start))

        def pose_forward(start, cols, dp):
            return jnp.einsum("nk,nkp->np", take(dp, start), cols, precision=highest).reshape(
                (size, spec.detector.nv, spec.detector.nu)
            )

        def pose_adjoint(start, cols, y, out):
            rows = jnp.einsum("nkp,np->nk", cols, y.reshape(size, -1), precision=highest)
            return jax.lax.dynamic_update_slice_in_dim(out, rows, start, axis=0)

        return _BatchOps(
            cached, view_columns, batch_columns, project, transpose, pose_forward, pose_adjoint
        )

    def dot(a, b):
        return jnp.vdot(a, b, precision=jax.lax.Precision.HIGHEST).real

    def solve_pose_eliminated(ops, p, x, free, rhs, normal, reg_hessian, volume_inverse):
        """Eliminate the per-view pose blocks exactly and solve for the volume by CG."""
        if ops.cached is not None:
            gram = jnp.einsum("nkp,nlp->nkl", ops.cached, ops.cached, precision=highest)
        else:

            def view_gram(i):
                columns = ops.view_columns(i)
                return jnp.matmul(columns, columns.T, precision=highest)

            gram = jax.lax.map(view_gram, jnp.arange(n_views))
        solve_pose = pose_block_solver(
            gram + cfg.gn_damping * jnp.eye(p.shape[1], dtype=p.dtype),
            arrays.smoothness * active,
            has_smoothness=spec.has_smoothness,
        )
        dp_rhs = solve_pose(rhs[1])

        def pose_only(start, valid, out):
            return ops.transpose(
                start, valid, ops.pose_forward(start, ops.batch_columns(start), dp_rhs), out
            )

        reduced_rhs = rhs[0] - free * over_batches(pose_only, volume_zeros)

        def pose_rows(dx):  # C^T W A dx, and the batches' W A dx
            def body(start, _valid, carry):
                rows, ys = carry
                y = ops.project(start, dx)
                rows = ops.pose_adjoint(start, ops.batch_columns(start), y, rows)
                return rows, jax.lax.dynamic_update_slice_in_dim(ys, y, start, axis=0)

            stored = jnp.zeros((n_views, spec.detector.nv, spec.detector.nu), jnp.float32)
            return over_batches(body, (jnp.zeros_like(p), stored))

        def reduced_normal(increment):
            dx = free * increment[0]
            rows, ys = pose_rows(dx)
            dp = solve_pose(rows)

            def body(start, valid, out):
                y = take(ys, start) - ops.pose_forward(start, ops.batch_columns(start), dp)
                return ops.transpose(start, valid, y, out)

            ax = over_batches(body, volume_zeros)
            return (
                free * (ax + reg_hessian(dx) + cfg.gn_volume_damping * dx),
                increment[1],
            )

        full_squared = dot(rhs[0], rhs[0]) + dot(rhs[1], rhs[1])
        # After exact pose back-substitution, the full residual is the
        # volume Schur residual. Use the same absolute stopping threshold
        # as stacked CG rather than rescaling it by the eliminated RHS.
        reduced_rtol = cfg.gn_joint_rtol * jnp.sqrt(
            full_squared / jnp.maximum(dot(reduced_rhs, reduced_rhs), 1e-30)
        )
        empty = jnp.zeros((0,), x.dtype)
        reduced = solve_coupled_normal(
            reduced_normal,
            (reduced_rhs, empty),
            (volume_inverse, empty),
            max_iters=cfg.gn_joint_iters,
            rtol=reduced_rtol,
        )
        dx = free * reduced.increment[0]
        dp = active * solve_pose(rhs[1] - pose_rows(dx)[0])
        actual = normal((dx, dp))
        rx, rp = rhs[0] - actual[0], rhs[1] - actual[1]
        relative = jnp.sqrt((dot(rx, rx) + dot(rp, rp)) / jnp.maximum(full_squared, 1e-30))
        finite = reduced.finite & jnp.isfinite(relative)
        return CoupledLinearResult((dx, dp), reduced.iterations, relative, finite)

    def update(p, x):
        ops = batch_ops(poses(p), p, x)
        if cfg.lambda_tv:
            reg_grad, reg_hessian = jax.linearize(jax.grad(regularisation), x)
        else:  # no regulariser: no zero volumes to carry
            reg_grad, reg_hessian = jnp.float32(0), lambda _dx: jnp.float32(0)
        smooth_grad, smooth_hessian = jax.linearize(jax.grad(smoothness), p)

        def gradient_body(start, valid, carry):
            gx, gp = carry
            measured = batch_weights(start) * take(arrays.projections, start)
            residual = ops.project(start, x) - measured
            cols = ops.batch_columns(start)
            return (
                ops.transpose(start, valid, residual, gx),
                ops.pose_adjoint(start, cols, residual, gp),
            )

        gx, gp = over_batches(gradient_body, (volume_zeros, jnp.zeros_like(p)))
        gx = gx + reg_grad
        gp = (gp + smooth_grad) * active
        # KKT-active zero voxels cannot move below zero. Release them when
        # the local gradient points into the feasible region.
        if cfg.recon_positivity:
            free = ((x > 0) | (gx < 0)).astype(x.dtype) * (mask != 0)
        else:  # every voxel free: a scalar, not a volume of ones
            free = jnp.float32(1) if unmasked else (mask != 0).astype(x.dtype)

        def normal(increment):  # y = W A dx + C dp, then (A^T W y, C^T y), batch by batch
            dx, dp = free * increment[0], active * increment[1]

            def body(start, valid, carry):
                ax, ap = carry
                cols = ops.batch_columns(start)
                y = ops.project(start, dx) + ops.pose_forward(start, cols, dp)
                return ops.transpose(start, valid, y, ax), ops.pose_adjoint(start, cols, y, ap)

            ax, ap = over_batches(body, (volume_zeros, jnp.zeros_like(p)))
            return (
                free * (ax + reg_hessian(dx) + cfg.gn_volume_damping * dx),
                active * (ap + smooth_hessian(dp) + cfg.gn_damping * dp),
            )

        def ones_body(start, valid, out):
            return ops.transpose(start, valid, ops.project(start, jnp.ones_like(x)), out)

        row_bound = over_batches(ones_body, volume_zeros)
        reg_bound = 12 * cfg.lambda_tv / cfg.huber_delta if cfg.lambda_tv else 0
        volume_inverse = 1 / jnp.maximum(
            row_bound + reg_bound + cfg.gn_volume_damping, cfg.gn_volume_damping
        )
        rhs = (-free * gx, -gp)
        if cfg.gn_joint_solver == "pose_eliminated":
            return solve_pose_eliminated(ops, p, x, free, rhs, normal, reg_hessian, volume_inverse)
        pose_diagonal = (
            jnp.sum(ops.cached**2, axis=2)
            if ops.cached is not None
            else jax.lax.map(
                lambda i: jnp.sum(ops.view_columns(i) ** 2, axis=1), jnp.arange(n_views)
            )
        )
        pose_inverse = 1 / (pose_diagonal + 12 * arrays.smoothness**2 + cfg.gn_damping)
        return solve_coupled_normal(
            normal,
            rhs,
            (volume_inverse, pose_inverse),
            max_iters=cfg.gn_joint_iters,
            rtol=cfg.gn_joint_rtol,
        )

    return update, loss


@partial(jax.jit, static_argnames=("spec",))
def run_update(
    arrays: CoupledArrays, p: jax.Array, x: jax.Array, *, spec: CoupledSpec
) -> CoupledLinearResult:
    update, _ = _build_program(spec, arrays)
    return update(p, x)


@partial(jax.jit, static_argnames=("spec",))
def run_loss(arrays: CoupledArrays, p: jax.Array, x: jax.Array, *, spec: CoupledSpec) -> jax.Array:
    _, loss = _build_program(spec, arrays)
    return loss(p, x)
