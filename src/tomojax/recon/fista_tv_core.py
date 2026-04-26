from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from tomojax.core.geometry import Detector, Grid
from tomojax.core.projector import forward_project_view_T
from tomojax.recon._tv_ops import huber_tv_grad, huber_tv_value, isotropic_tv_value


CoreRegulariser = Literal["tv", "huber_tv"]


@dataclass(frozen=True, slots=True)
class FistaCoreConfig:
    iters: int = 10
    lambda_tv: float = 0.005
    regulariser: CoreRegulariser = "huber_tv"
    huber_delta: float = 1e-2
    L: float = 100.0
    positivity: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    checkpoint_projector: bool = True
    projector_unroll: int = 1
    gather_dtype: str = "fp32"
    support: jnp.ndarray | None = None


@dataclass(frozen=True, slots=True)
class FistaCoreResult:
    x: jnp.ndarray
    loss: jnp.ndarray
    data_loss: jnp.ndarray
    regulariser_value: jnp.ndarray
    effective_iters: jnp.ndarray
    status: str

    def info(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "data_loss": self.data_loss,
            "regulariser_value": self.regulariser_value,
            "effective_iters": self.effective_iters,
            "status": self.status,
        }

    def python_info(self) -> dict[str, object]:
        return {
            "loss": [float(v) for v in list(self.loss)],
            "data_loss": float(self.data_loss),
            "regulariser_value": float(self.regulariser_value),
            "effective_iters": int(self.effective_iters),
            "status": self.status,
        }


def fista_tv_core_arrays(
    *,
    x0: jnp.ndarray,
    T_all: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    projections: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    cfg: FistaCoreConfig,
    view_weights: jnp.ndarray | None = None,
) -> FistaCoreResult:
    """Array-level unrolled FISTA/HUBER-TV core.

    This is intentionally stricter than the public `fista_tv` API: arrays in,
    arrays out, no Python `Geometry` object construction, and a static iteration
    count suitable for differentiating tiny bilevel reference problems.
    """
    x_init = _project_constraints(jnp.asarray(x0, dtype=jnp.float32), cfg)
    z_init = x_init
    t_init = jnp.float32(1.0)
    L = jnp.maximum(jnp.asarray(cfg.L, dtype=jnp.float32), jnp.float32(1e-6))
    lam = jnp.asarray(cfg.lambda_tv, dtype=jnp.float32)
    weights = (
        jnp.ones((int(projections.shape[0]),), dtype=jnp.float32)
        if view_weights is None
        else jnp.asarray(view_weights, dtype=jnp.float32).reshape((int(projections.shape[0]),))
    )
    weights = jnp.sqrt(jnp.maximum(weights, jnp.float32(0.0)))[:, None, None]

    def data_loss_fn(x: jnp.ndarray) -> jnp.ndarray:
        masked = _apply_support(x, cfg.support)
        pred = _project_stack(
            T_all=T_all,
            grid=grid,
            detector=detector,
            volume=masked,
            det_grid=det_grid,
            checkpoint_projector=cfg.checkpoint_projector,
            projector_unroll=cfg.projector_unroll,
            gather_dtype=cfg.gather_dtype,
        )
        resid = (pred - projections).astype(jnp.float32) * weights
        return jnp.float32(0.5) * jnp.vdot(resid, resid).real

    data_value_and_grad = jax.value_and_grad(data_loss_fn)

    def regulariser_value(x: jnp.ndarray) -> jnp.ndarray:
        if cfg.regulariser == "huber_tv":
            return huber_tv_value(x, float(cfg.huber_delta))
        return isotropic_tv_value(x)

    def objective_value(x: jnp.ndarray) -> jnp.ndarray:
        return data_loss_fn(x) + lam * regulariser_value(x)

    def body(carry, k):
        x_prev, z_prev, t_prev, loss_arr = carry
        _, grad = data_value_and_grad(z_prev)
        if cfg.regulariser == "huber_tv" and float(cfg.lambda_tv) != 0.0:
            grad = grad + lam * huber_tv_grad(z_prev, float(cfg.huber_delta))
        step = z_prev - grad / L
        # The differentiable reference path intentionally uses gradient FISTA for
        # smoothed regularisers. Exact TV belongs to the public reconstruction
        # adapter or a future nonsmooth implicit path, not the bilevel hot path.
        x_next = _project_constraints(step, cfg)
        t_next = jnp.float32(0.5) * (jnp.float32(1.0) + jnp.sqrt(jnp.float32(1.0) + 4.0 * t_prev * t_prev))
        z_next = x_next + ((t_prev - jnp.float32(1.0)) / t_next) * (x_next - x_prev)
        z_next = _project_constraints(z_next, cfg)
        loss_arr = loss_arr.at[k].set(objective_value(x_next).astype(jnp.float32))
        return (x_next, z_next, t_next, loss_arr), None

    n_iters = int(cfg.iters)
    loss0 = jnp.zeros((n_iters,), dtype=jnp.float32)
    (x_final, _, _, loss), _ = jax.lax.scan(
        body,
        (x_init, z_init, t_init, loss0),
        jnp.arange(n_iters, dtype=jnp.int32),
    )
    data_final = data_loss_fn(x_final)
    reg_final = regulariser_value(x_final)
    return FistaCoreResult(
        x=x_final,
        loss=loss,
        data_loss=data_final,
        regulariser_value=reg_final,
        effective_iters=jnp.asarray(n_iters, dtype=jnp.int32),
        status="ok",
    )


def _project_stack(
    *,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    checkpoint_projector: bool,
    projector_unroll: int,
    gather_dtype: str,
) -> jnp.ndarray:
    return jax.vmap(
        lambda T: forward_project_view_T(
            T,
            grid,
            detector,
            volume,
            use_checkpoint=checkpoint_projector,
            unroll=int(projector_unroll),
            gather_dtype=gather_dtype,
            det_grid=det_grid,
        )
    )(T_all)


def _apply_support(x: jnp.ndarray, support: jnp.ndarray | None) -> jnp.ndarray:
    if support is None:
        return x
    return x * jnp.asarray(support, dtype=x.dtype)


def _project_constraints(x: jnp.ndarray, cfg: FistaCoreConfig) -> jnp.ndarray:
    out = x
    lower = cfg.lower_bound
    if cfg.positivity:
        lower = 0.0 if lower is None else max(0.0, float(lower))
    if lower is not None:
        out = jnp.maximum(out, jnp.asarray(lower, dtype=out.dtype))
    if cfg.upper_bound is not None:
        out = jnp.minimum(out, jnp.asarray(cfg.upper_bound, dtype=out.dtype))
    return out
