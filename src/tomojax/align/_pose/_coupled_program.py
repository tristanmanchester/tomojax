"""Reusable compiled joint objectives with scan arrays kept out of cache keys."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp

from tomojax.align._geometry.parametrizations import apply_pose_updates
from tomojax.core.cone import cone_backproject, cone_coefficients, cone_project
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
    from tomojax.core.geometry.cone import ConeBeam
    from tomojax.geometry import Detector, Grid
    from tomojax.recon.types import Regulariser


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
    beam: ConeBeam | None = None


class CoupledArrays(NamedTuple):
    """Per-scan numerical inputs; changing values can reuse a compiled program."""

    poses: jax.Array
    projections: jax.Array
    weights: jax.Array
    mask: jax.Array
    active: jax.Array
    smoothness: jax.Array
    det_grid: tuple[jax.Array, jax.Array]


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

    def forward(t, x):
        if spec.beam is not None:
            coeff = cone_coefficients(t, spec.grid, spec.detector, spec.beam)
            return cone_project(mask * x, coeff, spec.grid, spec.detector, backend=cone_backend)
        if joseph is not None:
            return forward_project_planes(
                plane_coefficients(t, spec.grid, spec.detector, arrays.det_grid),
                mask * x,
                spec.grid,
                spec.detector,
                backend=backend,
                interpolation=joseph,
            )
        if spec.jacobian.integrator == "exact":
            return exact_forward(
                t, spec.grid, spec.detector, mask * x, backend=backend, det_grid=det_grid
            )
        return jax.lax.map(
            lambda ti: forward_project_view_T(
                ti,
                spec.grid,
                spec.detector,
                mask * x,
                gather_dtype="fp32",
                det_grid=det_grid,
                unroll=spec.jacobian.unroll,
                use_checkpoint=spec.jacobian.checkpoint,
            ),
            t,
        )

    def adjoint(t, y):
        if spec.beam is not None:
            coeff = cone_coefficients(t, spec.grid, spec.detector, spec.beam)
            return mask * cone_backproject(y, coeff, spec.grid, spec.detector, backend=cone_backend)
        if joseph is not None:
            return mask * sum_backproject_planes(
                plane_coefficients(t, spec.grid, spec.detector, arrays.det_grid),
                y,
                spec.grid,
                spec.detector,
                backend=backend,
                interpolation=joseph,
            )
        if spec.jacobian.integrator == "exact":
            return mask * exact_adjoint(
                t, spec.grid, spec.detector, y, backend=backend, det_grid=det_grid
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

        return (
            mask
            * jax.lax.scan(
                add, jnp.zeros((spec.grid.nx, spec.grid.ny, spec.grid.nz), jnp.float32), (t, y)
            )[0]
        )

    def loss(p, x):
        residual = weights * (forward(poses(p), x) - arrays.projections)
        return (
            0.5 * jnp.vdot(residual, residual, precision=jax.lax.Precision.HIGHEST).real
            + regularisation(x)
            + smoothness(p)
        )

    def update(p, x):
        t = poses(p)
        residual = weights * (forward(t, x) - arrays.projections)

        def columns(i):
            return (
                predict_columns(p[i], arrays.poses[i], mask * x, arrays.projections[i], weights[i])[
                    1
                ]
                * active[:, None]
            )

        indices = jnp.arange(n_views)
        cached = jax.lax.map(columns, indices) if cache_columns else None

        def get_columns(i):
            return cached[i] if cached is not None else columns(i)

        highest = jax.lax.Precision.HIGHEST

        # Cached columns apply to all views in one contraction; otherwise each
        # view's columns are recomputed inside a sequential map.
        def pose_forward(dp):
            if cached is not None:
                return jnp.einsum("nk,nkp->np", dp, cached, precision=highest).reshape(
                    residual.shape
                )
            return jax.lax.map(
                lambda i: jnp.matmul(dp[i], get_columns(i), precision=highest), indices
            ).reshape(residual.shape)

        def pose_adjoint(y):
            if cached is not None:
                return jnp.einsum("nkp,np->nk", cached, y.reshape(n_views, -1), precision=highest)
            return jax.lax.map(
                lambda i: jnp.matmul(get_columns(i), y[i].ravel(), precision=highest), indices
            )

        reg_grad, reg_hessian = jax.linearize(jax.grad(regularisation), x)
        smooth_grad, smooth_hessian = jax.linearize(jax.grad(smoothness), p)
        gx = adjoint(t, weights * residual) + reg_grad
        gp = (pose_adjoint(residual) + smooth_grad) * active
        # KKT-active zero voxels cannot move below zero. Release them when
        # the local gradient points into the feasible region.
        free = ((x > 0) | (gx < 0)).astype(x.dtype) if cfg.recon_positivity else jnp.ones_like(x)
        free = free * (mask != 0)

        def normal(increment):
            dx, dp = increment
            dx, dp = free * dx, active * dp
            y = weights * forward(t, dx) + pose_forward(dp)
            return (
                free * (adjoint(t, weights * y) + reg_hessian(dx) + cfg.gn_volume_damping * dx),
                active * (pose_adjoint(y) + smooth_hessian(dp) + cfg.gn_damping * dp),
            )

        row_bound = adjoint(t, weights**2 * forward(t, jnp.ones_like(x)))
        reg_bound = 12 * cfg.lambda_tv / cfg.huber_delta if cfg.lambda_tv else 0
        volume_inverse = 1 / jnp.maximum(
            row_bound + reg_bound + cfg.gn_volume_damping, cfg.gn_volume_damping
        )
        rhs = (-free * gx, -gp)
        if cfg.gn_joint_solver == "pose_eliminated":
            if cached is not None:
                gram = jnp.einsum("nkp,nlp->nkl", cached, cached, precision=highest)
            else:
                gram = jax.lax.map(
                    lambda i: jnp.matmul(get_columns(i), get_columns(i).T, precision=highest),
                    indices,
                )
            solve_pose = pose_block_solver(
                gram + cfg.gn_damping * jnp.eye(p.shape[1], dtype=p.dtype),
                arrays.smoothness * active,
                has_smoothness=spec.has_smoothness,
            )
            reduced_rhs = rhs[0] - free * adjoint(t, weights * pose_forward(solve_pose(rhs[1])))

            def reduced_normal(increment):
                dx = free * increment[0]
                y = weights * forward(t, dx)
                y = y - pose_forward(solve_pose(pose_adjoint(y)))
                return (
                    free * (adjoint(t, weights * y) + reg_hessian(dx) + cfg.gn_volume_damping * dx),
                    increment[1],
                )

            def dot(a, b):
                return jnp.vdot(a, b, precision=jax.lax.Precision.HIGHEST).real

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
            dp = active * solve_pose(rhs[1] - pose_adjoint(weights * forward(t, dx)))
            actual = normal((dx, dp))
            rx, rp = rhs[0] - actual[0], rhs[1] - actual[1]
            relative = jnp.sqrt((dot(rx, rx) + dot(rp, rp)) / jnp.maximum(full_squared, 1e-30))
            finite = reduced.finite & jnp.isfinite(relative)
            return CoupledLinearResult((dx, dp), reduced.iterations, relative, finite)

        pose_diagonal = (
            jnp.sum(cached**2, axis=2)
            if cached is not None
            else jax.lax.map(lambda i: jnp.sum(get_columns(i) ** 2, axis=1), indices)
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
