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


def read_acquisition(
    path: str | PathLike[str], *, data_path: str | None = None
) -> RawAcquisition | None:
    """Recognize a vendor acquisition by content and read its raw metadata/stacks.

    None means this is not a recognized acquisition, so the caller can try
    ordinary datasets or TIFF stacks. Recognized but incomplete or ambiguous
    inputs raise rather than silently becoming an unrelated format. Readers
    own format-specific discovery; this function owns their dispatch order.
    """
    from ._nikon import read_nikon

    return read_nikon(Path(path), data_path=data_path)
