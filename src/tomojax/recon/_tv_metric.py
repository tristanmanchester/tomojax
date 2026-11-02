"""Global, dimensionless regularization in physical voxel coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import jax
from jax.core import Tracer
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.cone import cone_parts


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TVMetric:
    """Resolved once on all measured views, before batching or slab decomposition."""

    gradient_scale_xyz: tuple[float, float, float] = field(metadata={"static": True})
    weight: Any
    delta: Any
    weight_over_contrast: Any
    weight_over_delta: Any
    length_scale: float = field(metadata={"static": True})
    contrast_scale: Any
    weight_sum: Any
    observed_count: Any


def _contrast_sums(y, w, xp):
    if w is None:
        centered = y - xp.mean(y, axis=(-2, -1), keepdims=True)
        return xp.asarray(y.size, dtype=y.dtype), xp.sum(centered * centered), y.size
    active = w > 0
    y = xp.where(active, y, 0)
    total = xp.sum(w, axis=(-2, -1), keepdims=True)
    mean = xp.sum(w * y, axis=(-2, -1), keepdims=True) / xp.where(total > 0, total, 1)
    centered = xp.where(active, y - mean, 0)
    return xp.sum(total), xp.sum(w * centered * centered), xp.sum(active)


def resolve_tv_metric(
    geometry, grid, detector, projections, *, weights=None, tv_weight, huber_delta
) -> TVMetric:
    """Convert dimensionless TV/Huber weights to the summed data objective.

    The contrast scale is weighted per-view-centered RMS even for absolute-data
    fitting. The length scale is the geometric mean detector width in the object
    plane. Physical TV is an integral, not a mean over the reconstruction box.
    With voxel volume dV and geometric voxel pitch h, the effective weight is
    alpha * s * W * dV / (L * h). Host input is read in bounded batches; device
    input retains its derivatives.
    """
    n, nv, nu = projections.shape
    batch = min(n, 16)
    area = detector.nu * detector.du * detector.nv * detector.dv
    parts = cone_parts(geometry, detector)
    length = (
        math.sqrt(area)
        if parts is None
        else math.sqrt(
            area
            * sum(
                (n if count is None else count) / beam.magnification**2 for count, beam, _ in parts
            )
            / n
        )
    )
    dv = grid.vx * grid.vy * grid.vz
    h = dv ** (1 / 3)
    scales = (h / grid.vx, h / grid.vy, h / grid.vz)
    if isinstance(projections, jax.Array | Tracer):
        y = jnp.asarray(projections, dtype=jnp.float32)
        w = None if weights is None else jnp.asarray(weights, dtype=jnp.float32)

        def body(i, state):
            valid = jnp.minimum(batch, n - i * batch)
            start = jnp.minimum(i * batch, n - batch)
            values = jax.lax.dynamic_slice_in_dim(y, start, batch)
            mask = (jnp.arange(batch) >= batch - valid)[:, None, None]
            if w is None:
                centered = values - jnp.mean(values, axis=(-2, -1), keepdims=True)
                count = wt = valid.astype(y.dtype) * nv * nu
                energy = jnp.sum(jnp.where(mask, centered * centered, 0))
            else:
                wb = jax.lax.dynamic_slice_in_dim(w, start, batch) * mask
                wt, energy, count = _contrast_sums(values, wb, jnp)
            return state[0] + wt, state[1] + energy, state[2] + count

        zero = jnp.asarray(0.0, y.dtype)
        total, energy, count = jax.lax.fori_loop(
            0, (n + batch - 1) // batch, body, (zero, zero, zero)
        )
        variance = energy / jnp.where(total > 0, total, 1)
        # The zero subgradient of RMS at a constant view keeps flat-data AD finite.
        scale = jnp.where(variance > 0, jnp.sqrt(jnp.where(variance > 0, variance, 1)), 0)
    else:
        total = energy = count = 0.0
        for first in range(0, n, batch):
            values = np.asarray(projections[first : first + batch], dtype=np.float64)
            wb = None if weights is None else np.asarray(weights[first : first + batch], np.float64)
            wt, e, observed = _contrast_sums(values, wb, np)
            total += float(wt)
            energy += float(e)
            count += float(observed)
        scale = math.sqrt(energy / total) if total else 0.0
    factor = tv_weight * total * dv / (length * h)
    return TVMetric(
        scales,
        factor * scale,
        huber_delta * scale * h / length**2,
        factor,
        factor * length**2 / (huber_delta * h),
        length,
        scale,
        total,
        count,
    )


def raise_tv_weight(metric: TVMetric, noise_gradient_rms) -> TVMetric:
    """Raise the empirical TV minimum to the noise's contrast-gradient RMS.

    The supplied RMS is sigma * RMS(A.T @ (w * C_w(epsilon))) for one iid
    unit-variance probe. It uses per-view contrast even for absolute fitting,
    as a regularization scale, not the absolute data gradient's covariance.
    The selected dimensionless strength is held fixed under differentiation;
    the physical metric's intensity/unit scaling retains its derivatives.
    """
    ratio = jax.lax.stop_gradient(
        jnp.maximum(1, noise_gradient_rms / jnp.where(metric.weight > 0, metric.weight, 1))
    )
    return replace(
        metric,
        weight=metric.weight * ratio,
        weight_over_contrast=metric.weight_over_contrast * ratio,
        weight_over_delta=metric.weight_over_delta * ratio,
    )
