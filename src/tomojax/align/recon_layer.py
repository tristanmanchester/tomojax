from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from tomojax.core.geometry import Detector, Grid
from tomojax.core.projector import forward_project_view_T
from tomojax.recon._tv_ops import huber_tv_value, isotropic_tv_value
from tomojax.recon.fista_tv_core import (
    FistaCoreConfig,
    FistaCoreResult,
    _projection_loss,
    fista_tv_core_arrays,
)

from .geometry_applier import BaseGeometryArrays, apply_alignment_state
from .state import AlignmentState


ReconDifferentiationMode = Literal["unrolled", "implicit"]


@dataclass(frozen=True, slots=True)
class ReconLayerConfig:
    iters: int = 5
    lambda_tv: float = 0.005
    regulariser: Literal["huber_tv", "tv"] = "huber_tv"
    huber_delta: float = 1e-2
    L: float = 100.0
    positivity: bool = False
    differentiation_mode: ReconDifferentiationMode = "unrolled"
    checkpoint_projector: bool = True
    projector_unroll: int = 1
    gather_dtype: str = "fp32"
    views_per_batch: int = 1
    implicit_cg_iters: int = 32
    implicit_cg_tol: float = 1e-3
    implicit_damping: float = 1e-4


@dataclass(frozen=True, slots=True)
class ReconLayerResult:
    x: jnp.ndarray
    info: dict[str, object]


@dataclass(frozen=True, slots=True)
class ReconLayer:
    """Bilevel reconstruction layer over effective geometry arrays."""

    base: BaseGeometryArrays
    grid: Grid
    detector: Detector
    config: ReconLayerConfig

    def reconstruct(
        self,
        *,
        state: AlignmentState,
        projections: jnp.ndarray,
        init_x: jnp.ndarray | None = None,
        view_weights: jnp.ndarray | None = None,
    ) -> ReconLayerResult:
        if self.config.regulariser != "huber_tv" and self.config.lambda_tv != 0.0:
            raise ValueError(
                "ReconLayer differentiable modes require huber_tv or lambda_tv=0"
            )
        effective = apply_alignment_state(self.base, state)
        x0 = (
            jnp.zeros((self.grid.nx, self.grid.ny, self.grid.nz), dtype=jnp.float32)
            if init_x is None
            else jnp.asarray(init_x, dtype=jnp.float32)
        )
        core_cfg = FistaCoreConfig(
            iters=int(self.config.iters),
            lambda_tv=float(self.config.lambda_tv),
            regulariser=self.config.regulariser,
            huber_delta=float(self.config.huber_delta),
            L=float(self.config.L),
            positivity=bool(self.config.positivity),
            checkpoint_projector=bool(self.config.checkpoint_projector),
            projector_unroll=int(self.config.projector_unroll),
            gather_dtype=str(self.config.gather_dtype),
            views_per_batch=max(1, int(self.config.views_per_batch)),
        )
        y = jnp.asarray(projections, dtype=jnp.float32)
        if self.config.differentiation_mode == "implicit":
            x = _implicit_reconstruct_arrays(
                x0=x0,
                T_all=effective.pose_stack,
                det_grid=effective.det_grid,
                projections=y,
                grid=self.grid,
                detector=self.detector,
                cfg=core_cfg,
                view_weights=view_weights,
                cg_iters=int(self.config.implicit_cg_iters),
                cg_tol=float(self.config.implicit_cg_tol),
                damping=float(self.config.implicit_damping),
            )
            result = fista_tv_core_arrays(
                x0=x0,
                T_all=effective.pose_stack,
                det_grid=effective.det_grid,
                projections=y,
                grid=self.grid,
                detector=self.detector,
                cfg=core_cfg,
                view_weights=view_weights,
            )
            return ReconLayerResult(
                x=x,
                info={
                    **result.info(),
                    "differentiation_mode": self.config.differentiation_mode,
                    "inner_regulariser": self.config.regulariser,
                    "implicit_gradient_status": "cg_adjoint",
                    "implicit_cg_iters": int(self.config.implicit_cg_iters),
                    "implicit_cg_tol": float(self.config.implicit_cg_tol),
                    "implicit_damping": float(self.config.implicit_damping),
                },
            )
        result = fista_tv_core_arrays(
            x0=x0,
            T_all=effective.pose_stack,
            det_grid=effective.det_grid,
            projections=y,
            grid=self.grid,
            detector=self.detector,
            cfg=core_cfg,
            view_weights=view_weights,
        )
        return ReconLayerResult(
            x=result.x,
            info={
                **result.info(),
                "differentiation_mode": self.config.differentiation_mode,
                "inner_regulariser": self.config.regulariser,
            },
        )


def core_result_to_info(result: FistaCoreResult) -> dict[str, object]:
    return result.info()


def _implicit_reconstruct_arrays(
    *,
    x0: jnp.ndarray,
    T_all: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    projections: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    cfg: FistaCoreConfig,
    view_weights: jnp.ndarray | None,
    cg_iters: int,
    cg_tol: float,
    damping: float,
) -> jnp.ndarray:
    det_u, det_v = det_grid
    weights = (
        jnp.ones((int(projections.shape[0]),), dtype=jnp.float32)
        if view_weights is None
        else jnp.asarray(view_weights, dtype=jnp.float32).reshape((int(projections.shape[0]),))
    )
    weights = jnp.sqrt(jnp.maximum(weights, jnp.float32(0.0)))[:, None, None]

    def solve_primal(
        T: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        y: jnp.ndarray,
        x_init: jnp.ndarray,
    ) -> jnp.ndarray:
        return fista_tv_core_arrays(
            x0=x_init,
            T_all=T,
            det_grid=(u, v),
            projections=y,
            grid=grid,
            detector=detector,
            cfg=cfg,
            view_weights=view_weights,
        ).x

    @jax.custom_vjp
    def solve(T: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, y: jnp.ndarray, x_init: jnp.ndarray):
        return solve_primal(T, u, v, y, x_init)

    def solve_fwd(
        T: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        y: jnp.ndarray,
        x_init: jnp.ndarray,
    ):
        x_star = solve_primal(T, u, v, y, x_init)
        return x_star, (x_star, T, u, v, y)

    def solve_bwd(res, x_bar: jnp.ndarray):
        x_star, T, u, v, y = res

        def objective_x(x: jnp.ndarray, T_arg: jnp.ndarray, u_arg: jnp.ndarray, v_arg: jnp.ndarray):
            data = _projection_loss(
                T_all=T_arg,
                grid=grid,
                detector=detector,
                volume=x,
                det_grid=(u_arg, v_arg),
                projections=y,
                weights=weights,
                checkpoint_projector=bool(cfg.checkpoint_projector),
                projector_unroll=int(cfg.projector_unroll),
                gather_dtype=str(cfg.gather_dtype),
                views_per_batch=int(cfg.views_per_batch),
            )
            if cfg.lambda_tv == 0.0:
                return data
            reg = (
                huber_tv_value(x, float(cfg.huber_delta))
                if cfg.regulariser == "huber_tv"
                else isotropic_tv_value(x)
            )
            return data + jnp.asarray(cfg.lambda_tv, dtype=jnp.float32) * reg

        grad_x = jax.grad(objective_x, argnums=0)

        def hvp(direction: jnp.ndarray) -> jnp.ndarray:
            return jax.jvp(
                lambda x: grad_x(x, T, u, v),
                (x_star,),
                (direction,),
            )[1] + jnp.asarray(damping, dtype=jnp.float32) * direction

        adjoint, _ = jsp.sparse.linalg.cg(
            hvp,
            x_bar,
            tol=float(cg_tol),
            maxiter=int(cg_iters),
        )

        def stationarity(T_arg: jnp.ndarray, u_arg: jnp.ndarray, v_arg: jnp.ndarray):
            return grad_x(x_star, T_arg, u_arg, v_arg)

        _, pullback = jax.vjp(stationarity, T, u, v)
        grad_T, grad_u, grad_v = pullback(adjoint)
        return (
            -grad_T,
            -grad_u,
            -grad_v,
            jnp.zeros_like(y),
            jnp.zeros_like(x0),
        )

    solve.defvjp(solve_fwd, solve_bwd)
    return solve(T_all, det_u, det_v, projections, x0)
