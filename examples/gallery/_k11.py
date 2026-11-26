"""Diamond K11 laminography scans at the detector's 2.16 micrometer pitch.

The raw NeXus files record the rotation angles but not the tilt, so each
script passes the tilt it was scanned at. Alignment runs on a 6.48 micrometer
grid; reconstructions for the figures are single planes at the detector pitch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np

import tomojax as tj

if TYPE_CHECKING:
    from pathlib import Path

PITCH_UM = 2.16
VOXEL_UM = 6.48


def load(path: Path, tilt: float) -> tj.Scan:
    """The scan as loaded: nominal geometry on a 704 x 704 x 234 grid of 6.48 um voxels."""
    with h5py.File(path) as source:
        sample = np.asarray(source["entry/instrument/EtherCAT/image_key"]) == 0
        angles = np.asarray(source["entry/imaging_sum/smaract_zrot"], np.float64)[sample]
    geometry = tj.LaminographyGeometry(
        tj.Grid(704, 704, 234, VOXEL_UM, VOXEL_UM, VOXEL_UM),
        tj.Detector(640, 540, PITCH_UM, PITCH_UM),
        angles,
        tilt=tilt,
    )
    return tj.load_frames(path, geometry=geometry).corrected()


def plane(
    scan: tj.Scan,
    center: tuple[float, float, float],
    shape: tuple[int, int, int],
    method: str = "fbp",
    **options: float,
) -> np.ndarray:
    """One plane of 2.16 um voxels about ``center`` (x, y, z in um), as (rows, columns).

    ``shape`` is (nx, ny, nz) with exactly one entry 1: an xy plane has rows along y,
    an xz plane rows along z, a yz plane rows along z.
    """
    grid = tj.Grid(*shape, PITCH_UM, PITCH_UM, PITCH_UM, vol_center=list(center))
    volume = np.asarray(tj.reconstruct(scan, method, grid=grid, **options).volume, np.float32)
    flat = shape.index(1)
    return volume[(slice(None),) * (2 - flat) + (0,)]
