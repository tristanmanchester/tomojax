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
        rows, cols = batch.shape[-2:]
        # Host maps make XLA bake every stencil index/weight into the executable.
        # Compute only the geometry in FP64, even with the caller's x64 disabled;
        # FP32 loses subpixel calibration accuracy on large/cropped detectors.
        with jax.enable_x64(True):
            u = jnp.arange(cols, dtype=jnp.float64)[None, :] - self.center[0]
            v = jnp.arange(rows, dtype=jnp.float64)[:, None] - self.center[1]
            radius = jnp.sqrt(u * u + v * v)
            scale = jnp.zeros_like(radius)
            for coefficient in reversed(self.coefficients):
                scale = scale * radius + coefficient
            coordinates = (
                (self.center[1] + scale * v).astype(jnp.float32),
                (self.center[0] + scale * u).astype(jnp.float32),
            )
        return jax.vmap(lambda image: map_coordinates(image, coordinates, order=1, mode="nearest"))(
            batch
        )
