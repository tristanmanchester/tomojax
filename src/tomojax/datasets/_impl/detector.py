"""Count-domain detector response and finite flat-field calibration noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, kw_only=True)
class DetectorSimulation:
    """Detector measurements with their calibration references and conditional truth.

    Parameters
    ----------
    projections, expected, ideal : numpy.ndarray
        Corrected line integrals, shape (view, row, column). ``expected`` applies
        the sampled flat/dark references to noiseless sample counts; it is not
        the expectation of a logarithm of noisy counts. ``ideal`` uses noiseless
        references but includes the imposed gain and offset changes.
    counts : numpy.ndarray
        Noisy sample counts, shape (view, row, column).
    flat, dark : numpy.ndarray
        Sampled calibration means in counts, shape (row, column).
    seed : int
        Random seed used for all measurements.
    model : DetectorModel
        Count model used to generate the measurements.
    """

    projections: NDArray[np.float64]
    expected: NDArray[np.float64]
    ideal: NDArray[np.float64]
    counts: NDArray[np.float64]
    flat: NDArray[np.float64]
    dark: NDArray[np.float64]
    seed: int
    model: DetectorModel


@dataclass(frozen=True, kw_only=True)
class DetectorModel:
    """Independent-pixel Poisson/read-noise detector in recorded count units.

    Static sensitivity is shared by sample and flat measurements and cancels
    under exact calibration. Finite reference noise persists across all views.
    Spatially correlated noise and time-dependent illumination are not modelled.

    Parameters
    ----------
    signal, dark : array_like
        Expected flat signal above dark, and dark offset, in counts; both have
        shape (row, column). ``signal`` must be positive.
    photon_gain : float
        Recorded counts per detected photon, nonnegative. Zero disables shot
        noise, useful for testing deterministic response changes.
    read_noise : float
        Per-frame dark/read-noise standard deviation in counts, nonnegative.
    """

    signal: ArrayLike
    dark: ArrayLike
    photon_gain: float
    read_noise: float

    def __post_init__(self) -> None:
        if self.photon_gain < 0:
            raise ValueError("photon_gain must be nonnegative counts per detected photon.")

    @classmethod
    def from_flats(
        cls, flats: ArrayLike, darks: ArrayLike, *, read_noise: float | None = None
    ) -> DetectorModel:
        """Estimate a count model from repeated flat and dark images.

        Adjacent-frame differences remove static sensitivity. Flat images are
        normalized to their common mean flux before estimating temporal noise.
        This single-exposure estimate assumes independent pixels and frames.

        Parameters
        ----------
        flats, darks : array_like
            Repeated images in counts, shape (frame, row, column). At least two
            flats are required, and two darks unless ``read_noise`` is supplied.
        read_noise : float, optional
            Known per-frame read-noise standard deviation in counts. Supplying
            zero explicitly assumes negligible dark noise, not measured zero.

        Returns
        -------
        DetectorModel
            Mean response maps and scalar effective photon gain/read noise.

        Raises
        ------
        ValueError
            Too few reference images to estimate the requested noise terms.
        """
        f, d = np.asarray(flats, dtype=float), np.asarray(darks, dtype=float)
        if len(f) < 2 or (len(d) < 2 and read_noise is None):
            raise ValueError("Supply two flats and two darks, or an explicit read_noise.")
        dark = d.mean(axis=0)
        signal = f.mean(axis=0) - dark
        if read_noise is None:
            n = len(d) // 2
            read_noise = float(np.sqrt(np.mean((d[: 2 * n : 2] - d[1 : 2 * n : 2]) ** 2) / 2))
        normalized = f - dark
        flux = normalized.mean(axis=(1, 2), keepdims=True)
        normalized = normalized * flux.mean() / flux
        n = len(f) // 2
        variance = np.mean((normalized[: 2 * n : 2] - normalized[1 : 2 * n : 2]) ** 2) / 2
        gain = max(0.0, float((variance - read_noise**2) / signal.mean()))
        return cls(signal=signal, dark=dark, photon_gain=gain, read_noise=read_noise)

    def simulate(
        self,
        line_integrals: ArrayLike,
        *,
        flat_frames: int,
        dark_frames: int,
        seed: int,
        gain_change: ArrayLike = 1.0,
        offset_change: ArrayLike = 0.0,
    ) -> DetectorSimulation:
        """Simulate counts and apply finite flat/dark calibration references.

        Parameters
        ----------
        line_integrals : array_like
            Noise-free line integrals, shape (view, row, column). Projection
            discretisation is the caller's choice, independent of this model.
        flat_frames, dark_frames : int
            Positive numbers of frames averaged into the two reference maps.
        seed : int
            Reproducible seed for independent sample, flat and dark draws.
        gain_change : array_like, optional
            Positive sample/calibration photon-rate ratio (optical throughput or
            quantum efficiency), broadcastable to (view, row, column). One
            means unchanged response. Counts per detected photon stays fixed;
            this is not a change in electronic conversion gain.
        offset_change : array_like, optional
            Sample minus calibration dark offset in counts, broadcastable to
            (view, row, column). Zero means unchanged offset.

        Returns
        -------
        DetectorSimulation
            Counts, calibration references, corrected data, and conditional
            truth. Nonpositive corrected transmission produces NaN or infinity;
            it is not clipped into apparently valid line integrals.

        Raises
        ------
        ValueError
            Reference frame counts are not positive.
        """
        if flat_frames < 1 or dark_frames < 1:
            raise ValueError("flat_frames and dark_frames must be positive.")
        q = np.asarray(line_integrals, dtype=float)
        signal, dark = np.asarray(self.signal), np.asarray(self.dark)
        rng_sample, rng_flat, rng_dark = (
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
        )
        flat = self._draw(signal, rng_flat, flat_frames) + dark
        dark_reference = dark + rng_dark.normal(
            scale=self.read_noise / np.sqrt(dark_frames), size=dark.shape
        )
        attenuated = signal * np.exp(-q) * np.asarray(gain_change)
        mean = attenuated + dark + np.asarray(offset_change)
        counts = self._draw(attenuated, rng_sample, 1) + dark + np.asarray(offset_change)
        denominator = flat - dark_reference
        return DetectorSimulation(
            projections=-np.log((counts - dark_reference) / denominator),
            expected=-np.log((mean - dark_reference) / denominator),
            ideal=-np.log((mean - dark) / signal),
            counts=counts,
            flat=flat,
            dark=dark_reference,
            seed=seed,
            model=self,
        )

    def _draw(
        self, signal: NDArray[np.float64], rng: np.random.Generator, frames: int
    ) -> NDArray[np.float64]:
        shot = (
            rng.poisson(signal * frames / self.photon_gain) * self.photon_gain / frames
            if self.photon_gain > 0
            else signal
        )
        return shot + rng.normal(scale=self.read_noise / np.sqrt(frames), size=signal.shape)
