"""Read the PSI nanotomography MATLAB fixtures without copying whole HDF5 stacks.

MATLAB v5 arrays are loaded by SciPy; v7.3 stores axes reversed in HDF5 and may
encode complex arrays as real/imag fields. Readers return TomoJAX (view,v,u)
blocks. No unwrapping, phase sign change, cropping or geometric alignment is
performed implicitly. The small synthetic fixture has no shift ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from scipy.io import loadmat

if TYPE_CHECKING:
    from types import TracebackType


_ALL = slice(None)


def complex_values(value: np.ndarray) -> np.ndarray:
    """View MATLAB's packed real/imag pairs without another full complex stack."""
    if value.dtype.names and {"real", "imag"} <= set(value.dtype.names):
        real, real_offset = value.dtype.fields["real"][:2]
        imag, imag_offset = value.dtype.fields["imag"][:2]
        if (
            real == imag
            and real.kind == "f"
            and real.itemsize in (4, 8)
            and real_offset == 0
            and imag_offset == real.itemsize
            and value.dtype.itemsize == 2 * real.itemsize
        ):
            kind = np.dtype(f"c{value.dtype.itemsize}").newbyteorder(real.byteorder)
            return value.view(kind)
        return value["real"] + 1j * value["imag"]
    return value


class PtychographyData:
    """An open PSI file, with explicit metadata and view-block access.

    Use as a context manager. The HDF5 complex object and aligned reference stay
    on disk until requested, so scans larger than device memory can be prepared
    in bounded batches. Beware that PSI's raw chunks span every view: read all
    views of the chosen spatial ROI once, then process that array in view
    batches, rather than decompressing the whole scan on each view read.
    Physical metadata is returned as stored, without guessed
    defaults for synthetic files that do not contain wavelength or pixel size.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.hdf5 = h5py.is_hdf5(self.path)
        self.file: Any = h5py.File(self.path, "r") if self.hdf5 else loadmat(self.path)
        key = "rotation_angle" if "rotation_angle" in self.file else "theta"
        self.angles = np.asarray(self.file[key], dtype=np.float64).ravel()
        self.shape = self._shape("stack_object")
        self.reference_shape = (
            self._shape("aligned_phase_sinogram") if "aligned_phase_sinogram" in self.file else None
        )
        self.metadata = {
            name: np.asarray(self.file[name]).squeeze().tolist()
            for name in ("lambda", "pixel_size", "probe_size")
            if name in self.file
        }

    def _shape(self, name: str) -> tuple[int, int, int]:
        shape = self.file[name].shape
        order = (0, 2, 1) if self.hdf5 else (2, 0, 1)
        return tuple(int(shape[axis]) for axis in order)

    def __enter__(self) -> PtychographyData:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.hdf5:
            self.file.close()

    def read(
        self,
        views: slice = _ALL,
        *,
        rows: slice = _ALL,
        cols: slice = _ALL,
        reference: bool = False,
    ) -> np.ndarray:
        """Read complex objects or PSI's aligned phase as ``(view,v,u)``.

        View/row/column slices refer to the returned axis order. The reference
        has its own recorded shape and is not implicitly resized to the raw
        objects; a different reference crop must be dealt with explicitly.
        """
        name = "aligned_phase_sinogram" if reference else "stack_object"
        selection = (views, cols, rows) if self.hdf5 else (rows, cols, views)
        if self.hdf5:
            from tomojax.io.api import read_hdf5_array

            value = read_hdf5_array(self.file[name], selection)
        else:
            value = np.asarray(self.file[name][selection])
        value = complex_values(value)
        return value.transpose(0, 2, 1) if self.hdf5 else value.transpose(2, 0, 1)
