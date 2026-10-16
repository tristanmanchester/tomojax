"""Shared known-motion challenge on measured complex nanotomography projections.

Generate once from the FULL acquisition-order angles, before view selection or
binning. Tune on seed 1 only; seeds 2 and 3 are held-out checks. Fourier-shift
full complex detector frames BEFORE cropping, unwrapping or ramp removal; do not
shift wrapped phase or a cropped ROI. Keep equal-angle groups in one FSC half,
even when copied observations acquire different injected shifts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft

# The released PSI reference's native crop (MATLAB half-probe/32-pixel rule).
# Score a fixed central axial region for every seed AND the zero-motion control:
# object material crosses the crop's top/bottom, so shifted detector coverage
# cannot support its first/last rows in every view. This is not a fitted mask.
PSI_OBJECT_CROP = (slice(212, 628), slice(197, 1509))
PSI_FSC_REGION = (slice(None), slice(None), slice(32, -32))


@dataclass(frozen=True)
class InjectedMotion:
    """Detector-pixel displacements in ``(u,v)`` order, with their components.

    Positive u/v moves image features toward increasing column/row index. These
    are also the truth detector-frame dx/dz poses for the shifted data; applying
    the negative image shift restores the original complex frame. In physical
    units multiply by detector ``(du,dv)``. Remove only a common object-translation
    gauge when comparing shifts-only estimates, using ``shift_error``.
    """

    drift_uv: np.ndarray
    offset_uv: np.ndarray
    jitter_uv: np.ndarray
    sub_tomogram: np.ndarray

    @property
    def shifts_uv(self) -> np.ndarray:
        """The total image displacement and detector-frame pose truth, in pixels."""
        return self.drift_uv + self.offset_uv + self.jitter_uv


def injected_motion(angles: np.ndarray, *, seed: int) -> InjectedMotion:
    """The lab's fixed 8 px drift / ±3 px offset / 0.5 px jitter model.

    Ascending acquisition runs define interleaved sub-tomograms: a decrease in
    recorded angle starts a new run, including short calibration/repeated runs.
    No sorting, modulo reduction or duplicate removal is performed. The released
    PSI file has 57 such runs. A single ``default_rng(seed)`` draws u, then v;
    for each axis it draws five periods uniformly in [0.2, 0.5] scan lengths,
    five phases uniformly in [0, 2π), one offset per run uniformly in [-3, 3],
    then independent Gaussian jitter with standard deviation 0.5 pixels.
    The sum of sinusoids is scaled to maximum absolute displacement 8 pixels.
    """
    angles = np.asarray(angles).ravel()
    count = len(angles)
    groups = np.cumsum(np.r_[0, np.diff(angles) < 0])
    rng = np.random.default_rng(seed)
    time = np.arange(count, dtype=float) / count
    drift, offset, jitter = (np.empty((count, 2)) for _ in range(3))
    for axis in range(2):
        periods = rng.uniform(0.2, 0.5, 5)
        phases = rng.uniform(0, 2 * np.pi, 5)
        wave = np.sin(2 * np.pi * time[:, None] / periods + phases).sum(axis=1)
        drift[:, axis] = 8 * wave / np.max(np.abs(wave))
        offset[:, axis] = rng.uniform(-3, 3, int(groups[-1]) + 1)[groups]
        jitter[:, axis] = rng.normal(0, 0.5, count)
    return InjectedMotion(drift, offset, jitter, groups)


def shift_complex_objects(objects: np.ndarray, shifts_uv: np.ndarray) -> np.ndarray:
    """Apply periodic subpixel Fourier translations to a batch of full frames.

    ``objects`` is complex ``(view,v,u)`` and ``shifts_uv`` is ``(view,2)``.
    Returns the same complex dtype without changing the input. Call in small
    view batches to bound FFT workspace; all batches use the shifts generated
    from the full acquisition, not independently generated motion. The reliable
    interior crop must exclude the periodic boundaries and the probe margin.
    """
    objects, shifts_uv = np.asarray(objects), np.asarray(shifts_uv)
    if objects.ndim != 3 or not np.iscomplexobj(objects):
        raise ValueError("motion injection requires complex (view,v,u) detector frames")
    if shifts_uv.shape != (len(objects), 2):
        raise ValueError("shifts_uv must have shape (view,2)")
    spectrum = fft.fft2(objects, axes=(-2, -1))
    fv = fft.fftfreq(objects.shape[1])[None, :, None]
    fu = fft.fftfreq(objects.shape[2])[None, None, :]
    phase = np.exp(
        -2j * np.pi * (fv * shifts_uv[:, 1, None, None] + fu * shifts_uv[:, 0, None, None])
    ).astype(spectrum.dtype)
    spectrum *= phase
    return fft.ifft2(spectrum, axes=(-2, -1))
