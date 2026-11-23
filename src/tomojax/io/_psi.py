"""PSI cSAXS ptychographic tomography: a MATLAB file of aligned phase projections.

The cSAXS tomography codes (Odstrcil et al., Opt. Express 2019) save, in a MATLAB
7.3 (HDF5) file, ``aligned_phase_sinogram``: the unwrapped, aligned phase of each
ptychographic projection, MATLAB-ordered [rows, columns, views]; ``rotation_angle``
in degrees; ``pixel_size`` and ``lambda`` in metres. The phase is already a line
integral, of 2π/λ times the refractive index decrement δ, so the scan's volume is
that phase per unit length. Projections are parallel. The file records no rotation
centre: the alignment leaves the axis at the middle column.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from tomojax import progress
from tomojax.core.host_volume import discard_failed_outputs
from tomojax.io._hdf5_array import read_hdf5_array

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
# What the cSAXS codes save besides the phase: complex, unaligned projections.
_COMPLEX = "stack_object"


def read_psi_mat(path: Path) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    """Its phase line integrals (view, v, u), angles, pixel size and wavelength (mm).

    Raises
    ------
    ValueError
        If the file holds no ``aligned_phase_sinogram`` (a MATLAB 5 file, or only the
        complex unaligned projections), naming what it holds.
    """
    with path.open("rb") as handle:
        handle.seek(512)  # a MATLAB 7.3 file is HDF5 behind a 512-byte header
        hdf5 = handle.read(8) == _HDF5_SIGNATURE
    if not hdf5:
        from scipy.io import whosmat

        names = [name for name, _, _ in whosmat(path)]
        raise ValueError(_missing(path, names))
    with discard_failed_outputs() as outputs, h5py.File(path, "r") as file:
        if "aligned_phase_sinogram" not in file:
            raise ValueError(_missing(path, list(file)))
        stack = file["aligned_phase_sinogram"]
        if not isinstance(stack, h5py.Dataset) or stack.ndim != 3:
            raise ValueError(f"{path}: aligned_phase_sinogram is not a 3D array")
        views, columns, rows = stack.shape  # MATLAB's [rows, columns, views], reversed
        task = progress.for_unit("views")
        progress.remaining("views", views)
        projections = outputs.allocate((views, rows, columns))
        # Whole chunks of views at a time: each compressed chunk spans several views.
        step = (stack.chunks or (1,))[0]
        for first in range(0, views, step):
            progress.check_cancelled()
            block = read_hdf5_array(stack, (slice(first, first + step), slice(None), slice(None)))
            projections[first : first + step] = np.swapaxes(block, 1, 2)
            if task is not None:
                task.advance(min(views, first + step), {})
        angles = np.asarray(file["rotation_angle"], np.float64).ravel()
        metres = 1e3  # the file's lengths in mm, every reader's unit
        pixel = float(np.asarray(file["pixel_size"]).item()) * metres
        wavelength = float(np.asarray(file["lambda"]).item()) * metres
        if len(angles) != views:
            raise ValueError(f"{path}: {len(angles)} rotation angles for {views} projections")
        return projections, angles, pixel, wavelength, path.stem


def _missing(path: Path, names: list[str]) -> str:
    held = ", ".join(sorted(names)) or "nothing"
    phase = (
        f"; {_COMPLEX} holds complex projections not yet unwrapped or aligned"
        if _COMPLEX in names
        else ""
    )
    return (
        f"{path} holds {held}{phase}: tj.load reads the phase line integrals a PSI "
        "cSAXS tomography file saves as aligned_phase_sinogram (MATLAB 7.3)"
    )


__all__ = ["read_psi_mat"]
