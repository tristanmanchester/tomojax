from __future__ import annotations

import math
from typing import Dict

import jax
import jax.numpy as jnp

from .geometry.base import Grid, Detector, Geometry
from .projector import forward_project_view


def view_loss(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    measured: jnp.ndarray,
    view_index: int,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
) -> jnp.ndarray:
    pred = forward_project_view(
        geometry=geometry,
        grid=grid,
        detector=detector,
        volume=volume,
        view_index=view_index,
        step_size=step_size,
        n_steps=n_steps,
        use_checkpoint=True,
    )
    resid = (pred - measured).astype(jnp.float32)
    return 0.5 * jnp.vdot(resid, resid).real


view_loss_value_and_grad = jax.jit(
    jax.value_and_grad(view_loss, argnums=3),  # grad wrt volume
    static_argnames=("geometry", "grid", "detector", "view_index", "step_size", "n_steps"),
)


def adjoint_test_once(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    y_like: jnp.ndarray,
    view_index: int,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
) -> float:
    """Check <A x, y> vs <x, A^T y> for one view using autodiff VJP.

    Use tiny sizes for sanity checks; memory grows with volume size.
    """

    def fwd_wrt_vol(vol):
        return forward_project_view(
            geometry=geometry,
            grid=grid,
            detector=detector,
            volume=vol,
            view_index=view_index,
            step_size=step_size,
            n_steps=n_steps,
            use_checkpoint=True,
        ).ravel()

    Ax = fwd_wrt_vol(volume)
    lhs = jnp.vdot(Ax, y_like.ravel())
    _, vjp = jax.vjp(fwd_wrt_vol, volume)
    ATy = vjp(y_like.ravel().astype(jnp.float32))[0]
    rhs = jnp.vdot(volume, ATy)
    rel = float(jnp.abs(lhs - rhs) / (jnp.abs(lhs) + 1e-12))
    return rel

