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
        """Reserve the resampled image; bilinear neighbours fuse into its gather."""
        return rows * cols

    def apply(self, batch: jax.Array) -> jax.Array:
        rows, cols = batch.shape[-2:]
        # Compute the calibration in host float64 even when JAX x64 is disabled;
        # a full detector's large pixel coordinates amplify polynomial roundoff.
        u, v = np.meshgrid(np.arange(cols) - self.center[0], np.arange(rows) - self.center[1])
        radius = np.hypot(u, v)
        scale = np.polynomial.polynomial.polyval(radius, self.coefficients)
        coordinates = (
            jnp.asarray(self.center[1] + scale * v, jnp.float32),
            jnp.asarray(self.center[0] + scale * u, jnp.float32),
        )
        return jax.vmap(lambda image: map_coordinates(image, coordinates, order=1, mode="nearest"))(
            batch
        )
