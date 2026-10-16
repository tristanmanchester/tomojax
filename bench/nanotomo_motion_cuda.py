"""Optional CUDA executor for the shared complex-projection motion challenge.

This does not generate motion. Pass the shifts from ``nanotomo_motion`` in full
acquisition order, and shift full detector frames before any crop or phase step.
"""

from __future__ import annotations

import numpy as np


def shift_complex_objects_cuda(objects: np.ndarray, shifts_uv: np.ndarray) -> np.ndarray:
    """Periodically translate complex ``(view,v,u)`` frames by pixel ``(u,v)``.

    This executes the same Fourier shift as the canonical CPU helper. Positive
    shifts move features toward increasing column/row indices. NumPy input and
    output retain their complex dtype; CuPy stays optional and is imported only
    when called. It uses the current CUDA device, without changing it. Small
    view batches bound FFT workspace, including prime-sized detector frames.
    Floating-point FFT roundoff may differ from the CPU implementation.
    """
    import cupy as cp

    objects, shifts_uv = np.asarray(objects), np.asarray(shifts_uv)
    if objects.ndim != 3 or not np.iscomplexobj(objects):
        raise ValueError("motion injection requires complex (view,v,u) detector frames")
    if shifts_uv.shape != (len(objects), 2):
        raise ValueError("shifts_uv must have shape (view,2)")
    source, shifts = cp.asarray(objects), cp.asarray(shifts_uv)
    spectrum = cp.fft.fft2(source, axes=(-2, -1))
    fv = cp.fft.fftfreq(source.shape[1])[None, :, None]
    fu = cp.fft.fftfreq(source.shape[2])[None, None, :]
    phase = cp.exp(
        -2j * cp.pi * (fv * shifts[:, 1, None, None] + fu * shifts[:, 0, None, None])
    ).astype(spectrum.dtype)
    spectrum *= phase
    return cp.asnumpy(cp.fft.ifft2(spectrum, axes=(-2, -1)))
