"""Content-discovered raw acquisitions, independent of Scan/Frames orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np

    from ._datasets import ProjectionDataset


@dataclass(frozen=True)
class RawAcquisition:
    """A reader's geometry record and raw stacks, without holding files open.

    ``record.projections`` may be a shape-only broadcast placeholder when
    ``counts`` is lazy. Calibration positions have the existing Frames meaning:
    number of sample views acquired before each flat; view positions default
    to view index plus one half. No corrections have been applied yet.
    """

    record: ProjectionDataset
    counts: Any
    flats: Any = None
    darks: Any = None
    flat_positions: np.ndarray | None = None
    white_level: float | None = None
    view_positions: np.ndarray | None = None


def metadata_candidates(path: Path) -> list[Path]:
    """Directory sidecars, without reopening known TIFF image entries.

    A directly supplied file is always content-probed, regardless of suffix.
    Folder image names identify pixel storage; other names remain eligible for
    content discovery, including renamed parameter files.
    """
    if not path.is_dir():
        return [path]
    from ._tiff import TIFF_SUFFIXES

    return [
        file
        for file in path.iterdir()
        if file.suffix.lower() not in TIFF_SUFFIXES and file.is_file()
    ]


def read_acquisition(
    path: str | PathLike[str],
    *,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
) -> RawAcquisition | None:
    """Recognize a vendor acquisition by content and read its raw metadata/stacks.

    None means this is not a recognized acquisition, so the caller can try
    ordinary datasets or TIFF stacks. Recognized but incomplete or ambiguous
    inputs raise rather than silently becoming an unrelated format. Readers
    own format-specific discovery; this function owns their dispatch order.

    Parameters
    ----------
    path : str or os.PathLike
        Acquisition file or folder.
    data_path : str or None, optional
        Explicit dataset path for formats that hold several candidate stacks.
    image_key_path : str or None, optional
        Explicit frame-label dataset for external HDF5 acquisitions.
    angles_path : str or None, optional
        Explicit rotation-angle dataset for external HDF5 acquisitions.

    Returns
    -------
    RawAcquisition or None
        The reader's geometry record and raw stacks, or None if no reader
        recognizes the input.

    Raises
    ------
    ValueError
        If a recognized acquisition is incomplete or ambiguous.
    """
    from ._esrf import read_esrf
    from ._hdf5_acquisition import read_hdf5_acquisition
    from ._nikon import read_nikon
    from ._skyscan import read_skyscan
    from ._zeiss import read_zeiss

    target = Path(path)
    for reader in (read_nikon, read_zeiss, read_esrf, read_skyscan):
        acquisition = reader(target, data_path=data_path)
        if acquisition is not None:
            return acquisition
    return read_hdf5_acquisition(
        target, data_path=data_path, image_key_path=image_key_path, angles_path=angles_path
    )
