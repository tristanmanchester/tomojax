"""Exterior profiles for a Fourier slab, with two shared radial transforms on CUDA."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ._continuation import radius_views, row_fits
from ._fourier_grid import radial_coefficients

if TYPE_CHECKING:
    from tomojax.core.cuda_types import CuPyValue
    from tomojax.geometry import Detector


class FourierExterior:
    """Fit each slice's two exterior profiles in the scan's original view ordering."""

    def __init__(
        self,
        xp: CuPyValue,
        detector: Detector,
        views: int,
        order: np.ndarray | None,
        flipped: CuPyValue,
        nfft: int,
    ) -> None:
        self.xp, self.flipped, self.nfft = xp, flipped, nfft
        self.fits = row_fits(detector)
        self.sharing = xp.asarray(radius_views(views, order))
        if xp is not np:
            phase, taper, _ = radial_coefficients(3 * detector.nu, 3 * nfft, detector.du)
            self.tail_phase = xp.asarray(phase, dtype=xp.complex64)
            self.tail_taper = xp.asarray(taper, dtype=xp.float32)

    def _angular_mean(
        self, low: np.ndarray | CuPyValue, high: np.ndarray | CuPyValue
    ) -> tuple[np.ndarray | CuPyValue, np.ndarray | CuPyValue]:
        """Average each edge's continuation amplitudes over a sixteenth of the half turn.

        The Cartesian samples near zero frequency each take two views of the
        many that cross them, so a continuation whose amplitude scatters from
        view to view would reach the image as low-frequency error. Averaging
        over neighbouring views costs the image nothing it resolves in a term
        this smooth. The +u edge of the sorted half turn runs on into the -u
        edge at its start, and a view sorted from the opposing half turn has
        its edges exchanged.
        """
        xp: CuPyValue = self.xp
        views = int(low.shape[-2])
        reach = views // 32
        if reach == 0:
            return low, high
        flipped: CuPyValue = xp.not_equal(self.flipped, 0)[:, None]
        ring: CuPyValue = xp.concatenate(
            [xp.where(flipped, low, high), xp.where(flipped, high, low)], axis=-2
        ).astype(xp.float64)
        ring = xp.concatenate([ring[..., -reach - 1 :, :], ring, ring[..., :reach, :]], axis=-2)
        ring = ring.cumsum(axis=-2)
        ring = ring[..., 2 * reach + 1 :, :] - ring[..., : -2 * reach - 1, :]
        ring = xp.multiply(ring, 1 / (2 * reach + 1)).astype(low.dtype)
        plus, minus = ring[..., :views, :], ring[..., views:, :]
        return xp.where(flipped, plus, minus), xp.where(flipped, minus, plus)

    def transform(self, raw: CuPyValue) -> tuple[CuPyValue, tuple[CuPyValue, ...], CuPyValue]:
        """Form the continued NumPy rows or the two CUDA spectra and their view amplitudes."""
        xp = self.xp
        fits, first, last = (
            self.fits,
            raw[..., : self.fits.window],
            raw[..., -self.fits.window :],
        )
        place = partial(xp.asarray, dtype=raw.dtype)
        radius2 = fits.radius(place, first[:, self.sharing], last[:, self.sharing])
        low, high = self._angular_mean(*fits.amplitudes(place, first, last, radius2))
        if xp is not np:
            unit = xp.ones_like(radius2)
            left, right = fits.tails(place, unit, unit, radius2)
            nu = fits.nu
            exterior_mass = (
                left.sum(axis=-1, dtype=xp.float64) * low[..., 0]
                + right.sum(axis=-1, dtype=xp.float64) * high[..., 0]
            )
            # Both profiles use the original, three-width centered coordinate.
            pair = xp.concatenate(
                [
                    xp.pad(left, ((0, 0), (0, 0), (0, 2 * nu))),
                    xp.pad(right, ((0, 0), (0, 0), (2 * nu, 0))),
                ],
                axis=1,
            )
            basis = xp.fft.rfft(pair * self.tail_taper, n=3 * self.nfft) * self.tail_phase
            return raw, (basis, xp.concatenate([low, high], axis=-1)), exterior_mass
        before, after = fits.tails(place, low, high, radius2)
        return xp.concatenate([before, raw, after], axis=-1), (), None
