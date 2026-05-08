"""Held-out residual checks for alternating smoke runs."""
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.forward import (
    apply_residual_filter_schedule,
    project_parallel_reference,
    residual_loss,
    robust_residual_scale,
)

if TYPE_CHECKING:
    from tomojax.align._continuation import ContinuationLevel
    from tomojax.geometry import GeometryState


def _heldout_masks(
    mask: jax.Array,
    heldout_view_index: int | None,
) -> tuple[jax.Array, jax.Array | None]:
    if heldout_view_index is None:
        return mask, None
    n_views = int(mask.shape[0])
    view_index = int(heldout_view_index)
    if view_index < 0:
        view_index += n_views
    if view_index < 0 or view_index >= n_views:
        raise ValueError("heldout_view_index must select an existing view")
    heldout = jnp.zeros_like(mask, dtype=jnp.float32).at[view_index, :, :].set(mask[view_index])
    train = jnp.asarray(mask, dtype=jnp.float32).at[view_index, :, :].set(0.0)
    return train, heldout


def _projection_loss(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    mask: jax.Array,
    level: ContinuationLevel,
    *,
    sigma: float,
    loss_mode: str = "pseudo_huber",
) -> float:
    predicted = project_parallel_reference(volume, geometry)
    filtered = apply_residual_filter_schedule(
        predicted - observed, level.residual_filters, mask=mask
    )
    result = residual_loss(
        filtered.residual,
        jnp.zeros_like(filtered.residual),
        mask=None,
        sigma=sigma,
        delta=level.residual_delta,
        mode="l2" if loss_mode == "l2" else "pseudo_huber",
    )
    return float(result.loss)


def _level_residual_sigma(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    mask: jax.Array,
    level: ContinuationLevel,
) -> float:
    predicted = project_parallel_reference(volume, geometry)
    filtered = apply_residual_filter_schedule(
        predicted - observed,
        level.residual_filters,
        mask=mask,
    )
    return float(robust_residual_scale(filtered.residual))


def _heldout_projection_loss(
    volume: jax.Array,
    observed: jax.Array,
    geometry: GeometryState,
    heldout_mask: jax.Array | None,
    level: ContinuationLevel,
    *,
    sigma: float,
    loss_mode: str = "pseudo_huber",
) -> float | None:
    if heldout_mask is None:
        return None
    return _projection_loss(
        volume,
        observed,
        geometry,
        heldout_mask,
        level,
        sigma=sigma,
        loss_mode=loss_mode,
    )


def _heldout_residual_check(
    volume: jax.Array,
    observed: jax.Array,
    *,
    before_geometry: GeometryState,
    after_geometry: GeometryState,
    heldout_mask: jax.Array | None,
    level: ContinuationLevel,
    sigma: float,
    tolerance: float,
    loss_mode: str = "pseudo_huber",
) -> tuple[float | None, float | None, bool | None]:
    before = _heldout_projection_loss(
        volume,
        observed,
        before_geometry,
        heldout_mask,
        level,
        sigma=sigma,
        loss_mode=loss_mode,
    )
    after = _heldout_projection_loss(
        volume,
        observed,
        after_geometry,
        heldout_mask,
        level,
        sigma=sigma,
        loss_mode=loss_mode,
    )
    passed = _heldout_residual_passed(before=before, after=after, tolerance=tolerance)
    return before, after, passed


def _heldout_residual_passed(
    *,
    before: float | None,
    after: float | None,
    tolerance: float,
) -> bool | None:
    if before is None or after is None:
        return None
    return bool(after <= before + tolerance)
