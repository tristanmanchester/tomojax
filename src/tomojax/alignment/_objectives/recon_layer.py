"""Differentiable reconstruction layer used by bilevel alignment objectives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from tomojax.alignment._geometry.geometry_applier import BaseGeometryArrays, apply_alignment_state
from tomojax.alignment._geometry.parametrizations import (
    PoseTranslationFrame,
    apply_pose_update,
    apply_pose_updates,
)
from tomojax.core.geometry.cone import ConeBeam, beam_of
from tomojax.recon.fista_tv_core import (
    FistaCoreConfig,
    FistaCoreResult,
    fista_objective_arrays,
    fista_tv_core_arrays,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.alignment._model.state import AlignmentState
    from tomojax.core.geometry import Detector, Geometry, Grid

ReconDifferentiationMode = Literal["unrolled", "implicit"]

_apply_updates = jax.jit(apply_pose_updates, static_argnames=("translation_frame",))


@dataclass(frozen=True, slots=True)
class PoseAdjustedGeometry:
    """Geometry adapter that applies per-view pose offsets during reconstruction."""

    geometry: Geometry
    pose_params: jnp.ndarray
    translation_frame: PoseTranslationFrame = "object"

    def pose_for_view(self, i: int) -> tuple[tuple[jnp.ndarray, ...], ...]:
        """Return the nominal pose composed with the aligned pose update."""
        T_nom = jnp.asarray(self.geometry.pose_for_view(i), dtype=jnp.float32)
        return tuple(
            map(
                tuple,
                apply_pose_update(
                    T_nom, self.pose_params[i], translation_frame=self.translation_frame
                ),
            )
        )

    def stack_poses(self, n_views: int, dtype: jnp.dtype) -> jnp.ndarray:
        """Return the first ``n_views`` poses, updated in one vectorised step."""
        from tomojax.core.geometry.views import stack_view_poses

        nominal = stack_view_poses(self.geometry, n_views, dtype=jnp.float32)
        poses = _apply_updates(
            nominal,
            jnp.asarray(self.pose_params[:n_views]),
            translation_frame=self.translation_frame,
        )
        return poses.astype(dtype)

    def rays_for_view(self, i: int) -> object:
        """Return detector rays for the wrapped geometry view."""
        return self.geometry.rays_for_view(i)

    @property
    def beam(self) -> ConeBeam | None:
        """Return the wrapped geometry's cone beam, if any."""
        return beam_of(self.geometry)


@dataclass(frozen=True, slots=True)
class ReconLayerConfig:
    """Configuration for differentiable preview reconstruction.

    Implicit differentiation uses a damped Hessian at the returned iterate,
    restricted to free voxels when nonnegativity is enabled. Pinned voxels
    have zero local derivative; this does not certify primal convergence.
    """

    iterations: int = 5
    tv_weight: float = 0.005
    regulariser: Literal["huber_tv", "tv"] = "huber_tv"
    huber_delta: float = 1e-2
    lipschitz: float = 100.0
    nonnegative: bool = False
    differentiation_mode: ReconDifferentiationMode = "unrolled"
    checkpoint_projector: bool = True
    projector_unroll: int = 1
    gather_dtype: str = "fp32"
    views_per_batch: int = 1
    implicit_cg_iterations: int = 32
    implicit_cg_tol: float = 1e-3
    implicit_damping: float = 1e-4
    ray_integrator: str = "sampled"


@dataclass(frozen=True, slots=True)
class ReconLayerResult:
    """Reconstruction result and diagnostic info."""

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
        """Run a reconstruction layer for an alignment state."""
        effective = apply_alignment_state(self.base, state)
        x0 = (
            jnp.zeros((self.grid.nz, self.grid.ny, self.grid.nx), dtype=jnp.float32)
            if init_x is None
            else jnp.asarray(init_x, dtype=jnp.float32)
        )
        core_cfg = FistaCoreConfig(
            iterations=int(self.config.iterations),
            tv_weight=float(self.config.tv_weight),
            regulariser=self.config.regulariser,
            huber_delta=float(self.config.huber_delta),
            lipschitz=float(self.config.lipschitz),
            nonnegative=bool(self.config.nonnegative),
            checkpoint_projector=bool(self.config.checkpoint_projector),
            projector_unroll=int(self.config.projector_unroll),
            gather_dtype=str(self.config.gather_dtype),
            views_per_batch=max(1, int(self.config.views_per_batch)),
            ray_integrator=self.config.ray_integrator,
        )
        y = jnp.asarray(projections, dtype=jnp.float32)
        if self.config.differentiation_mode == "implicit":
            result = _implicit_reconstruct_arrays(
                x0=x0,
                T_all=effective.pose_stack,
                det_grid=effective.det_grid,
                projections=y,
                grid=self.grid,
                detector=self.detector,
                cfg=core_cfg,
                view_weights=view_weights,
                cg_iters=int(self.config.implicit_cg_iterations),
                cg_tol=float(self.config.implicit_cg_tol),
                damping=float(self.config.implicit_damping),
            )
            return ReconLayerResult(
                x=result.x,
                info={
                    **result.info(),
                    "differentiation_mode": self.config.differentiation_mode,
                    "inner_regulariser": self.config.regulariser,
                    "implicit_gradient_status": "cg_adjoint",
                    "implicit_cg_iterations": int(self.config.implicit_cg_iterations),
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


type _ImplicitOperands = tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None
]


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
) -> FistaCoreResult[jnp.ndarray]:
    # Solve once. Attach the implicit derivative to the resulting volume while
    # preserving the diagnostics from that same solve.
    result = fista_tv_core_arrays(
        x0=x0,
        T_all=T_all,
        det_grid=det_grid,
        projections=projections,
        grid=grid,
        detector=detector,
        cfg=cfg,
        view_weights=view_weights,
    )
    det_u, det_v = det_grid

    @jax.custom_vjp
    def attach_derivative(operands: _ImplicitOperands) -> jnp.ndarray:
        return operands[0]

    def solve_fwd(operands: _ImplicitOperands) -> tuple[jnp.ndarray, _ImplicitOperands]:
        return operands[0], operands

    def solve_bwd(
        res: _ImplicitOperands,
        x_bar: jnp.ndarray,
    ) -> tuple[_ImplicitOperands]:
        x_star, T, u, v, y, weights = res

        def objective_x(
            x: jnp.ndarray,
            T_arg: jnp.ndarray,
            u_arg: jnp.ndarray,
            v_arg: jnp.ndarray,
            y_arg: jnp.ndarray,
            weights_arg: jnp.ndarray | None,
        ) -> jnp.ndarray:
            return fista_objective_arrays(
                T_all=T_arg,
                grid=grid,
                detector=detector,
                volume=x,
                det_grid=(u_arg, v_arg),
                projections=y_arg,
                cfg=cfg,
                view_weights=weights_arg,
            )

        grad_x = jax.grad(objective_x, argnums=0)

        adjoint = _implicit_adjoint(
            lambda x: grad_x(x, T, u, v, y, weights),
            x_star,
            x_bar,
            cfg=cfg,
            damping=damping,
            cg_iters=cg_iters,
            cg_tol=cg_tol,
        )

        def stationarity(
            T_arg: jnp.ndarray,
            u_arg: jnp.ndarray,
            v_arg: jnp.ndarray,
            y_arg: jnp.ndarray,
            weights_arg: jnp.ndarray | None,
        ) -> jnp.ndarray:
            return grad_x(x_star, T_arg, u_arg, v_arg, y_arg, weights_arg)

        _, pullback = jax.vjp(stationarity, T, u, v, y, weights)
        grad_T, grad_u, grad_v, grad_y, grad_weights = pullback(adjoint)
        return (
            (
                jnp.zeros_like(x_star),
                -grad_T,
                -grad_u,
                -grad_v,
                -grad_y,
                None if grad_weights is None else -grad_weights,
            ),
        )

    attach_derivative.defvjp(solve_fwd, solve_bwd)
    return replace(
        result, x=attach_derivative((result.x, T_all, det_u, det_v, projections, view_weights))
    )


def _implicit_adjoint(
    gradient: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    cotangent: jnp.ndarray,
    *,
    cfg: FistaCoreConfig,
    damping: float,
    cg_iters: int,
    cg_tol: float,
) -> jnp.ndarray:
    """Solve the damped Hessian only on the current free-variable subspace.

    At a stable active set, bound-pinned voxels have zero derivative. Identity
    rows on those coordinates keep CG nonsingular without coupling them back
    into the free Hessian. At a bound transition we select the pinned derivative.
    """
    lower = cfg.lower_bound
    if cfg.nonnegative:
        lower = 0.0 if lower is None else max(0.0, lower)
    free = jnp.ones_like(x, dtype=jnp.bool_)
    if lower is not None:
        free = free & (x > lower)
    if cfg.upper_bound is not None:
        free = free & (x < cfg.upper_bound)

    def hvp(direction: jnp.ndarray) -> jnp.ndarray:
        tangent = jnp.where(free, direction, 0.0)
        product = jax.jvp(gradient, (x,), (tangent,))[1] + damping * tangent
        return jnp.where(free, product, direction)

    adjoint, _ = jsp.sparse.linalg.cg(
        hvp, jnp.where(free, cotangent, 0.0), tol=float(cg_tol), maxiter=int(cg_iters)
    )
    return jnp.where(free, adjoint, 0.0)
