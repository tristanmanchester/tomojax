from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry.base import Grid, Detector, Geometry
from .projector import backproject_view, forward_project_view


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
    gather_dtype: str = "fp32",
) -> float:
    """Check <A x, y> vs <x, A^T y> for one view using the explicit adjoint.

    Use tiny sizes for sanity checks; memory grows with volume size.
    """
    Ax = forward_project_view(
        geometry=geometry,
        grid=grid,
        detector=detector,
        volume=volume,
        view_index=view_index,
        step_size=step_size,
        n_steps=n_steps,
        use_checkpoint=True,
        gather_dtype=gather_dtype,
    ).ravel()
    lhs = jnp.vdot(Ax, y_like.ravel())
    ATy = backproject_view(
        geometry=geometry,
        grid=grid,
        detector=detector,
        image=y_like.astype(jnp.float32),
        view_index=view_index,
        step_size=step_size,
        n_steps=n_steps,
        gather_dtype=gather_dtype,
    )
    rhs = jnp.vdot(volume, ATy)
    rel = float(jnp.abs(lhs - rhs) / (jnp.abs(lhs) + 1e-12))
    return rel
