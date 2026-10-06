"""Small differentiable operators built on the reference projector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .projector import forward_project_view

if TYPE_CHECKING:
    from .geometry.base import Detector, Geometry, Grid


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
    gather_dtype: str = "fp32",
) -> jnp.ndarray:
    """Return half squared residual loss for one measured detector view."""
    pred = forward_project_view(
        geometry=geometry,
        grid=grid,
        detector=detector,
        volume=volume,
        view_index=view_index,
        step_size=step_size,
        n_steps=n_steps,
        use_checkpoint=True,
        gather_dtype=gather_dtype,
    )
    resid = (pred - measured).astype(jnp.float32)
    return 0.5 * jnp.vdot(resid, resid).real


view_loss_value_and_grad = jax.jit(
    jax.value_and_grad(view_loss, argnums=3),  # grad wrt volume
    static_argnames=(
        "geometry",
        "grid",
        "detector",
        "view_index",
        "step_size",
        "n_steps",
        "gather_dtype",
    ),
)
