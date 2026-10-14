"""Detector-coordinate pullback for a calibrated radial lens distortion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np

from ._steps import Domain, Step


@dataclass(frozen=True, kw_only=True)
class RadialDistortion(Step):
    """Unwarp transmission using a calibrated backward radial polynomial.

    ``center`` is ``(column, row)`` in zero-based detector pixel coordinates,
    not the rotation axis or the detector's physical centre. ``coefficients``
    are ``(c0, c1, ...)`` of the backward map: an undistorted output pixel at
    radius ``r`` samples the measured image at radius ``r * sum(c_k * r**k)``.
    This is the polynomial convention used by Algotom and Discorpy.

    Bilinear interpolation runs after flat/dark correction and before the log.
    Pixels mapping outside the acquired detector repeat its nearest edge.
    The calibrated output has the same detector geometry and shape; this is
    an intensity pullback, with no Jacobian factor. ``(1.0,)`` is identity.

    Use a calibration in the coordinates of the supplied frames. For a crop
    subtract its first column/row from ``center``. For mean binning by ``f``,
    use ``(center - (f - 1)/2)/f`` and replace ``c_k`` by ``c_k * f**k``.
    Correct before cropping when the map needs source pixels outside the crop.
    """

    domain: ClassVar[Domain] = "transmission"
    center: tuple[float, float]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        coefficients = tuple(float(value) for value in self.coefficients)
        if len(center) != 2 or not np.isfinite(center).all():
            raise ValueError("RadialDistortion center must be a finite (column, row) pair")
        if not coefficients or not np.isfinite(coefficients).all():
            raise ValueError("RadialDistortion coefficients must be nonempty and finite")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "coefficients", coefficients)

    def workspace_pixels(self, rows: int, cols: int) -> int:
        """Reserve double-precision coordinate workspace, with no work for identity."""
        return (
            0 if self.coefficients[0] == 1 and not any(self.coefficients[1:]) else 2 * rows * cols
        )

    def apply(self, batch: jax.Array) -> jax.Array:
        if self.coefficients[0] == 1 and not any(self.coefficients[1:]):
            return batch
        return pullback_rows(
            self,
            batch,
            input_first=0,
            output_first=0,
            output_rows=batch.shape[1],
        )


def _coordinates(step: RadialDistortion, first: int, rows: int, cols: int, xp):
    """The calibrated map in GLOBAL camera coordinates; FP32 only at the stencil."""
    u = xp.arange(cols, dtype=xp.float64)[None, :] - step.center[0]
    v = (xp.arange(rows, dtype=xp.float64)[:, None] + first) - step.center[1]
    radius = xp.sqrt(u * u + v * v)
    scale = xp.zeros_like(radius)
    for coefficient in reversed(step.coefficients):
        scale = scale * radius + coefficient
    return (
        (step.center[1] + scale * v).astype(xp.float32),
        (step.center[0] + scale * u).astype(xp.float32),
    )


def source_rows(step: RadialDistortion, first: int, stop: int, rows: int, cols: int):
    """Bound every bilinear neighbour in this owned output band, including camera edges."""
    if step.coefficients[0] == 1 and not any(step.coefficients[1:]):
        return first, stop
    y, _ = _coordinates(step, first, stop - first, cols, np)
    y = np.clip(y, 0, rows - 1)
    # A spare row also covers host/device FP64 arithmetic rounding near an integer.
    return max(0, int(np.floor(y.min())) - 1), min(rows, int(np.floor(y.max())) + 3)


def pullback_rows(
    step: RadialDistortion,
    batch: jax.Array,
    *,
    input_first: int,
    output_first: int,
    output_rows: int,
) -> jax.Array:
    """Sample an owned band using a measured halo, not the halo's artificial edges."""
    if step.coefficients[0] == 1 and not any(step.coefficients[1:]):
        return jax.lax.dynamic_slice(
            batch,
            (0, output_first - input_first, 0),
            (batch.shape[0], output_rows, batch.shape[2]),
        )
    # Compute global geometry in FP64 even with caller x64 disabled. Casting
    # AFTER subtracting the halo origin would change the bilinear weights.
    with jax.enable_x64(True):
        y, x = _coordinates(step, output_first, output_rows, batch.shape[2], jnp)
    coordinates = (y - jnp.float32(input_first), x)
    return jax.vmap(lambda image: map_coordinates(image, coordinates, order=1, mode="nearest"))(
        batch
    )
