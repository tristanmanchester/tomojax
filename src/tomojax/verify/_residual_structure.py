"""Residual-structure checks for failure classification."""
# pyright: reportAny=false, reportUnknownMemberType=false

from __future__ import annotations

import jax.numpy as jnp


def residual_structure_summary(residual: object, mask: object) -> dict[str, object]:
    """Summarise residual structure that can indicate unmodelled nuisance terms."""
    values = jnp.asarray(residual, dtype=jnp.float32)
    mask_values = jnp.asarray(mask, dtype=bool)
    masked = jnp.where(mask_values, values, 0.0)
    valid_count = jnp.maximum(jnp.sum(mask_values), jnp.asarray(1, dtype=jnp.int32))
    global_rms = jnp.sqrt(jnp.sum(masked * masked) / valid_count)
    view_mean = jnp.mean(masked, axis=(1, 2))
    column_mean = jnp.mean(masked, axis=(0, 1))
    view_structure_ratio = float(
        jnp.sqrt(jnp.mean(view_mean * view_mean)) / jnp.maximum(global_rms, 1.0e-12)
    )
    column_structure_ratio = float(
        jnp.sqrt(jnp.mean(column_mean * column_mean)) / jnp.maximum(global_rms, 1.0e-12)
    )
    view_threshold = 0.35
    column_threshold = 0.35
    passed = view_structure_ratio <= view_threshold and column_structure_ratio <= column_threshold
    return {
        "view_mean_rms_ratio": view_structure_ratio,
        "column_mean_rms_ratio": column_structure_ratio,
        "view_mean_rms_ratio_threshold": view_threshold,
        "column_mean_rms_ratio_threshold": column_threshold,
        "passed": passed,
        "reason": "checks per-view gain/offset and detector-column residual structure",
    }


__all__ = ["residual_structure_summary"]
