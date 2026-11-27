"""Global, dimensionless regularization in physical voxel coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING, overload

import jax
from jax.core import Tracer
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.cone import cone_parts

if TYPE_CHECKING:
    from types import ModuleType

    from tomojax.geometry import Detector, Geometry, Grid


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TVMetric:
    """Resolved once on all measured views, before batching or slab decomposition."""

    gradient_scale_xyz: tuple[float, float, float] = field(metadata={"static": True})
    weight: float | jax.Array
    delta: float | jax.Array
    weight_over_contrast: float | jax.Array
    length_scale: float = field(metadata={"static": True})
    contrast_scale: float | jax.Array
    weight_sum: float | jax.Array
    observed_count: float | jax.Array
    noise_score_rms: float | jax.Array | None = None


if TYPE_CHECKING:

    @overload
    def _contrast_sums(
        y: jax.Array, w: jax.Array | None, xp: ModuleType
    ) -> tuple[jax.Array, jax.Array, jax.Array | int]: ...

    @overload
    def _contrast_sums(
        y: np.ndarray, w: np.ndarray | None, xp: ModuleType
    ) -> tuple[np.ndarray | np.floating, np.floating, np.floating | np.integer | int]: ...


def _contrast_sums(
    y: jax.Array | np.ndarray,
    w: jax.Array | np.ndarray | None,
    xp: ModuleType,
) -> tuple[
    jax.Array | np.ndarray | np.floating,
    jax.Array | np.floating,
    jax.Array | np.floating | np.integer | int,
]:
    if w is None:
        centered = y - xp.mean(y, axis=(-2, -1), keepdims=True)
        return xp.asarray(y.size, dtype=y.dtype), xp.sum(centered * centered), y.size
    active = w > 0
    y = xp.where(active, y, 0)
    total = xp.sum(w, axis=(-2, -1), keepdims=True)
    mean = xp.sum(w * y, axis=(-2, -1), keepdims=True) / xp.where(total > 0, total, 1)
    centered = xp.where(active, y - mean, 0)
    return xp.sum(total), xp.sum(w * centered * centered), xp.sum(active)


def _device_contrast_sums(
    projections: jax.Array, weights: jax.Array | np.ndarray | None, batch: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Accumulate total weight, centered energy and observed count in view batches."""
    n, nv, nu = projections.shape
    y = jnp.asarray(projections, dtype=jnp.float32)
    w = None if weights is None else jnp.asarray(weights, dtype=jnp.float32)

    def body(
        i: jax.Array, state: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
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
    return jax.lax.fori_loop(0, (n + batch - 1) // batch, body, (zero, zero, zero))


def resolve_tv_metric(
    geometry: Geometry | None,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    weights: jax.Array | np.ndarray | None = None,
    tv_weight: float | jax.Array,
    huber_delta: float | jax.Array,
) -> TVMetric:
    """Convert dimensionless TV/Huber weights to the summed data objective.

    The contrast scale is weighted per-view-centered RMS even for absolute-data
    fitting. The length scale is the geometric mean detector width in the object
    plane. Physical TV is an integral, not a mean over the reconstruction box.
    With voxel volume dV and geometric voxel pitch h, the effective weight is
    alpha * s * W * dV / (L * h). Host input is read in bounded batches; device
    input retains its derivatives.
    """
    n = projections.shape[0]
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
        total, energy, count = _device_contrast_sums(projections, weights, batch)
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
        length,
        scale,
        total,
        count,
    )


def voxel_noise_weight(
    metric: TVMetric,
    noise_gradient_rms: float | jax.Array,
    shape: tuple[int, int, int],
    support: jax.Array | None = None,
) -> TVMetric:
    """Calibrate the linear TV penalty against isolated noise excursions.

    On a flat interior a unit impulse contributes its outgoing difference norm
    and one incoming difference per active axis. For N equal-variance Gaussian
    scores, sqrt(2 log(2N)) SD bounds the expected largest absolute score,
    even when the scores are correlated. The probe estimates a spatially
    averaged SD, so the calibration is approximate for nonuniform coverage.
    Shape is in storage order, reversed relative to the metric's scales.
    """
    scales = [c for c, n in zip(metric.gradient_scale_xyz, reversed(shape), strict=True) if n > 1]
    cost = sum(scales) + math.hypot(*scales)
    voxels = math.prod(shape)
    if support is None:
        threshold = math.sqrt(2 * math.log(2 * voxels))
    else:
        active = jnp.maximum(jnp.count_nonzero(jnp.broadcast_to(support, shape)), 1)
        noise_gradient_rms = noise_gradient_rms * jnp.sqrt(voxels / active)
        threshold = jnp.sqrt(2 * jnp.log(2.0 * active))
    weight = noise_gradient_rms * threshold / cost if cost else 0
    ratio = jax.lax.stop_gradient(weight / jnp.where(metric.weight > 0, metric.weight, 1))
    return replace(
        metric,
        weight=metric.weight * ratio,
        weight_over_contrast=metric.weight_over_contrast * ratio,
        noise_score_rms=noise_gradient_rms,
    )
