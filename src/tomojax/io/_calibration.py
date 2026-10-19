"""Geometry-independent lazy flat and dark image loading."""

from pathlib import Path

from tomojax._data import FrameStack, locate_frames

from ._tiff import read_tiff_frames
from ._zeiss import read_zeiss_images


def read_calibration_frames(path: str | Path) -> FrameStack:
    """Read a lazy calibration stack from HDF5, TIFF or Zeiss Xradia images.

    No sample-view geometry is required. Every image contributes to the flat
    or dark mean.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 stack, TIFF file or folder, or Zeiss image container. Zeiss content
        is recognised independently of the filename.

    Returns
    -------
    FrameStack
        Lazy images in (frame, row, column) order, with their stored dtype.

    Raises
    ------
    ValueError
        If image dimensions, type or format are unsupported.
    OSError
        If the input cannot be read.
    """
    file = Path(path)
    zeiss = read_zeiss_images(file)
    if zeiss is not None:
        return zeiss
    if file.suffix.lower() in {".nxs", ".h5", ".hdf", ".hdf5"}:
        return locate_frames(str(file)).stack
    return read_tiff_frames(file, lazy=True)
