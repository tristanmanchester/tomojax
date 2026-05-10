"""Data-derived soft support and low-frequency anchor for alignment volumes."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.recon._reference import reconstruct_backprojection_reference
from tomojax.recon._support import centered_volume_support

if TYPE_CHECKING:
    from tomojax.geometry import GeometryState


@dataclass(frozen=True)
class ScoutSupportResult:
    """Frozen scout support and anchor derived without truth data."""

    scout_volume: jax.Array
    support_probability: jax.Array
    low_frequency_anchor: jax.Array
    provenance: dict[str, object]


def build_scout_support(
    projections: jax.Array,
    geometry: GeometryState,
    *,
    projection_valid_mask: jax.Array | None,
    volume_shape: tuple[int, int, int],
    smoothing_radius: int = 2,
    dilation_radius: int = 2,
    threshold_fraction: float = 0.15,
) -> ScoutSupportResult:
    """Build a conservative frozen object-frame support from initial geometry."""
    observed = jnp.asarray(projections, dtype=jnp.float32)
    if tuple(int(dim) for dim in volume_shape) != (
        int(volume_shape[0]),
        int(volume_shape[1]),
        int(volume_shape[2]),
    ):
        raise ValueError("volume_shape must be a concrete 3D shape")
    weighted = observed
    if projection_valid_mask is not None:
        weighted = weighted * jnp.asarray(projection_valid_mask, dtype=jnp.float32)
    scout = reconstruct_backprojection_reference(
        weighted,
        geometry,
        depth=int(volume_shape[0]),
    )
    scout = jnp.maximum(jnp.asarray(scout, dtype=jnp.float32), 0.0)
    broad = centered_volume_support(volume_shape, kind="cylindrical").astype(jnp.float32)
    scout = scout * broad
    smoothed = _box_blur3d(scout, radius=max(1, int(smoothing_radius)))
    positive_max = jnp.maximum(jnp.max(smoothed), jnp.asarray(1.0e-6, dtype=jnp.float32))
    threshold = positive_max * jnp.asarray(float(threshold_fraction), dtype=jnp.float32)
    scale = jnp.maximum(threshold * 0.25, jnp.asarray(1.0e-6, dtype=jnp.float32))
    soft = jax.nn.sigmoid((smoothed - threshold) / scale) * broad
    dilated = _max_pool3d(soft, radius=max(0, int(dilation_radius))) * broad
    anchor = _box_blur3d(scout, radius=max(1, int(smoothing_radius) * 2)) * broad
    provenance: dict[str, object] = {
        "schema": "tomojax.scout_support_provenance.v1",
        "geometry_source": "initial_metadata",
        "mask_source": "projection_valid_mask",
        "uses_truth": False,
        "smoothing_radius": int(smoothing_radius),
        "dilation_radius": int(dilation_radius),
        "threshold_fraction": float(threshold_fraction),
        "threshold": float(threshold),
        "support_mass_fraction": float(jnp.mean(dilated)),
        "freeze_policy": "per_level_before_alignment",
    }
    return ScoutSupportResult(
        scout_volume=scout.astype(jnp.float32),
        support_probability=dilated.astype(jnp.float32),
        low_frequency_anchor=anchor.astype(jnp.float32),
        provenance=provenance,
    )


def _box_blur3d(volume: jax.Array, *, radius: int) -> jax.Array:
    if int(radius) <= 0:
        return jnp.asarray(volume, dtype=jnp.float32)
    vol = jnp.asarray(volume, dtype=jnp.float32)
    padded = jnp.pad(vol, ((radius, radius), (radius, radius), (radius, radius)), mode="edge")
    acc = jnp.zeros_like(vol)
    count = 0
    for dx in range(2 * radius + 1):
        for dy in range(2 * radius + 1):
            for dz in range(2 * radius + 1):
                acc = acc + padded[
                    dx : dx + vol.shape[0],
                    dy : dy + vol.shape[1],
                    dz : dz + vol.shape[2],
                ]
                count += 1
    return acc / jnp.asarray(count, dtype=jnp.float32)


def _max_pool3d(volume: jax.Array, *, radius: int) -> jax.Array:
    if int(radius) <= 0:
        return jnp.asarray(volume, dtype=jnp.float32)
    vol = jnp.asarray(volume, dtype=jnp.float32)
    padded = jnp.pad(
        vol,
        ((radius, radius), (radius, radius), (radius, radius)),
        mode="constant",
        constant_values=0.0,
    )
    windows = []
    for dx in range(2 * radius + 1):
        for dy in range(2 * radius + 1):
            windows.extend(
                padded[
                    dx : dx + vol.shape[0],
                    dy : dy + vol.shape[1],
                    dz : dz + vol.shape[2],
                ]
                for dz in range(2 * radius + 1)
            )
    return jnp.max(jnp.stack(windows, axis=0), axis=0)
