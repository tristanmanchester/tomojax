"""Joint free-voxel/pose Gauss-Newton with bounded pose-column storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align._geometry.parametrizations import apply_pose_updates
from tomojax.core.projector import (
    forward_project_view_T,
    get_detector_grid_device,
    sum_backproject_views_T,
)
from tomojax.core.trilinear import exact_adjoint, exact_forward
from tomojax.recon.fista_tv_core import FistaCoreConfig, regulariser_value_arrays

from ._coupled_linear import CoupledLinearResult, solve_coupled_normal
from ._pose_block import pose_block_solver
from ._pose_context import _PoseObjectiveContext
from ._pose_jacobian import build_pose_prediction_and_columns

# Large acquisitions recompute per-view columns instead of retaining five
# complete sinograms. This caps the optional cache, not the acquisition size.
_POSE_CACHE_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class CoupledObjective:
    update: Callable[[jax.Array, jax.Array], CoupledLinearResult]
    loss: Callable[[jax.Array, jax.Array], jax.Array]
    projector_backend: str
    pose_columns_cached: bool


def build_coupled_objective(ctx: _PoseObjectiveContext) -> CoupledObjective:
    cfg = ctx.cfg
    if not ctx.loss_adapter.supports_gauss_newton:
        raise ValueError("joint GN requires a least-squares alignment loss")
    if cfg.gather_dtype not in {"fp32", "float32"}:
        raise ValueError("joint GN requires gather_dtype='fp32' for matched linear operators")
    # Pallas exact integration accepts dynamic rigid poses. The sampled
    # reference keeps its matched explicit transpose and bounded view loop.
    backend = (
        "pallas" if cfg.projector_backend == "pallas" and jax.default_backend() == "gpu" else "jax"
    )
    if cfg.ray_integrator != "exact":
        backend = "jax"
    canonical = get_detector_grid_device(ctx.detector)
    if backend == "pallas" and not all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(ctx.det_grid, canonical, strict=True)
    ):
        backend = "jax"
    det_grid = None if backend == "pallas" else ctx.det_grid
    mask = ctx.volume_mask if ctx.volume_mask is not None else jnp.float32(1)
    weights = ctx.loss_adapter.gauss_newton_weights(
        ctx.projections, ctx.loss_mask if ctx.has_loss_mask else None
    )
    active = ctx.active_mask.astype(jnp.float32)
    predict_columns = build_pose_prediction_and_columns(ctx)
    cache_columns = ctx.n_views * ctx.nv * ctx.nu * 5 * 4 <= _POSE_CACHE_BYTES
    reg_cfg = FistaCoreConfig(regulariser=cfg.regulariser, huber_delta=cfg.huber_delta)

    def regularisation(x):
        if cfg.lambda_tv == 0:
            return jnp.float32(0)
        return cfg.lambda_tv * regulariser_value_arrays(x, reg_cfg)

    def smoothness(p):
        d2 = p[:-2] - 2 * p[1:-1] + p[2:]
        return jnp.sum(jnp.square(d2 * ctx.smoothness_weights))

    def poses(p):
        return apply_pose_updates(ctx.pose_stack, p, translation_frame=cfg.pose_translation_frame)

    def forward(t, x):
        if cfg.ray_integrator == "exact":
            return exact_forward(
                t, ctx.grid, ctx.detector, mask * x, backend=backend, det_grid=det_grid
            )
        return jax.lax.map(
            lambda ti: forward_project_view_T(
                ti,
                ctx.grid,
                ctx.detector,
                mask * x,
                gather_dtype="fp32",
                det_grid=det_grid,
                unroll=cfg.projector_unroll,
                use_checkpoint=cfg.checkpoint_projector,
            ),
            t,
        )

    def adjoint(t, y):
        if cfg.ray_integrator == "exact":
            return mask * exact_adjoint(
                t, ctx.grid, ctx.detector, y, backend=backend, det_grid=det_grid
            )

        # Accumulate one volume, never a volume stack per view.
        def add(x, pair):
            ti, yi = pair
            return x + sum_backproject_views_T(
                ti[None],
                ctx.grid,
                ctx.detector,
                yi[None],
                gather_dtype="fp32",
                det_grid=det_grid,
                unroll=cfg.projector_unroll,
            ), None

        return (
            mask
            * jax.lax.scan(
                add, jnp.zeros((ctx.grid.nx, ctx.grid.ny, ctx.grid.nz), jnp.float32), (t, y)
            )[0]
        )

    def loss(p, x):
        residual = weights * (forward(poses(p), x) - ctx.projections)
        return (
            0.5 * jnp.vdot(residual, residual, precision=jax.lax.Precision.HIGHEST).real
            + regularisation(x)
            + smoothness(p)
        )

    def update(p, x):
        t = poses(p)
        residual = weights * (forward(t, x) - ctx.projections)

        def columns(i):
            return (
                predict_columns(p[i], ctx.pose_stack[i], mask * x, ctx.projections[i], weights[i])[
                    1
                ]
                * active[:, None]
            )

        indices = jnp.arange(ctx.n_views)
        cached = jax.lax.map(columns, indices) if cache_columns else None

        def get_columns(i):
            return cached[i] if cached is not None else columns(i)

        def pose_forward(dp):
            return jax.lax.map(
                lambda i: jnp.matmul(dp[i], get_columns(i), precision=jax.lax.Precision.HIGHEST),
                indices,
            ).reshape(residual.shape)

        def pose_adjoint(y):
            return jax.lax.map(
                lambda i: jnp.matmul(
                    get_columns(i), y[i].ravel(), precision=jax.lax.Precision.HIGHEST
                ),
                indices,
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
            gram = jax.lax.map(
                lambda i: jnp.matmul(
                    get_columns(i), get_columns(i).T, precision=jax.lax.Precision.HIGHEST
                ),
                indices,
            )
            solve_pose = pose_block_solver(
                gram + cfg.gn_damping * jnp.eye(p.shape[1], dtype=p.dtype),
                ctx.smoothness_weights * active,
                has_smoothness=bool(cfg.w_rot or cfg.w_trans),
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

        pose_diagonal = jax.lax.map(lambda i: jnp.sum(get_columns(i) ** 2, axis=1), indices)
        pose_inverse = 1 / (pose_diagonal + 12 * ctx.smoothness_weights**2 + cfg.gn_damping)
        return solve_coupled_normal(
            normal,
            rhs,
            (volume_inverse, pose_inverse),
            max_iters=cfg.gn_joint_iters,
            rtol=cfg.gn_joint_rtol,
        )

    return CoupledObjective(jax.jit(update), jax.jit(loss), backend, cache_columns)
