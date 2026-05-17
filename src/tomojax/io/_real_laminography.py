from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

type PathLike = str | Path

_PROJECTIONS_PATH = "entry/imaging/data"
_THETAS_PATH = "entry/imaging_sum/smaract_zrot"


@dataclass(frozen=True, slots=True)
class RealLaminographyInput:
    """Loaded real-laminography projections and stage angles."""

    projections: np.ndarray
    thetas_deg: np.ndarray


def load_real_laminography_input(
    path: PathLike,
    *,
    flip_u: bool = False,
    flip_v: bool = False,
    transpose_detector: bool = False,
) -> RealLaminographyInput:
    """Load the measured real-laminography NX/HDF5 input.

    The HDF5 layout is the beamline layout currently used by the
    real-laminography regression scripts:

    - `entry/imaging/data`: projection stack as `(n_views, nv, nu)`
    - `entry/imaging_sum/smaract_zrot`: rotation angles in degrees

    Detector transforms are applied after loading in the historical runner
    order: transpose detector axes, then flip v, then flip u.
    """
    input_path = Path(path)
    with h5py.File(input_path, "r") as handle:
        projections = np.asarray(handle[_PROJECTIONS_PATH], dtype=np.float32)
        thetas = np.asarray(handle[_THETAS_PATH], dtype=np.float32)

    if projections.ndim != 3:
        raise ValueError(
            f"{_PROJECTIONS_PATH} must be a 3D projection stack; got shape {projections.shape}"
        )
    if thetas.ndim != 1:
        raise ValueError(f"{_THETAS_PATH} must be a 1D angle array; got shape {thetas.shape}")
    if int(projections.shape[0]) != int(thetas.shape[0]):
        raise ValueError(
            f"projection count {projections.shape[0]} does not match angle count {thetas.shape[0]}"
        )

    if transpose_detector:
        projections = np.transpose(projections, (0, 2, 1))
    if flip_v:
        projections = projections[:, ::-1, :]
    if flip_u:
        projections = projections[:, :, ::-1]

    return RealLaminographyInput(
        projections=np.asarray(projections, dtype=np.float32),
        thetas_deg=np.asarray(thetas, dtype=np.float32),
    )
