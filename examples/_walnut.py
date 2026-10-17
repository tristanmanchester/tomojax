"""Read the FIPS walnut TIFF convention and recorded ASTRA geometry.

Shared by the import and multi-orbit alignment examples; ASTRA is not needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import imageio.v3 as iio
import numpy as np

import tomojax as tj

if TYPE_CHECKING:
    from pathlib import Path

ROWS, COLUMNS, VIEWS = 972, 768, 1200


def read(path: Path) -> np.ndarray:
    """One TIFF as the scanner's portrait read-out, turned the way ASTRA expects."""
    return np.transpose(np.flipud(np.asarray(iio.imread(path), np.float32)))


def load_orbit(walnut: Path, orbit: int, every: int) -> tuple[np.ndarray, np.ndarray]:
    """Line integrals ``(rows, views, columns)`` and ASTRA ``cone_vec`` rows of one orbit.

    Projections are dark- and flat-corrected and log-transformed, and read in
    reverse order to match the geometry file, as the authors' scripts do.
    """
    folder = walnut / "Projections" / f"tubeV{orbit}"
    vectors = np.loadtxt(folder / "scan_geom_original.geom")[0:VIEWS:every]
    dark = read(folder / "di000000.tif")
    flat = np.mean([read(folder / f"io00000{k}.tif") for k in (0, 1)], axis=0) - dark
    indices = range(VIEWS, 0, -every)
    data = np.empty((ROWS, len(indices), COLUMNS), np.float32)
    for view, index in enumerate(indices):
        image = (read(folder / f"scan_{index:06d}.tif") - dark) / flat
        data[:, view, :] = -np.log(np.clip(image, 1e-6, None))
    return data, vectors


def load_scan(
    walnut: Path, *, orbits: tuple[int, ...], every: int, binning: int, size: int
) -> tj.Scan:
    """Build a scan on the authors' 50.1 mm cubic field of view."""
    loaded = [load_orbit(walnut, orbit, every) for orbit in orbits]
    data = np.concatenate([d for d, _ in loaded], axis=1)
    vectors = np.concatenate([v for _, v in loaded])
    window = {f"WindowMin{a}": -25.05 for a in "XYZ"} | {f"WindowMax{a}": 25.05 for a in "XYZ"}
    scan = tj.Scan.from_astra(
        data,
        {"type": "cone_vec", "DetectorRowCount": ROWS, "DetectorColCount": COLUMNS,
         "Vectors": vectors},
        {"GridColCount": size, "GridRowCount": size, "GridSliceCount": size, "option": window},
    )  # fmt: skip
    return scan.binned(binning)
