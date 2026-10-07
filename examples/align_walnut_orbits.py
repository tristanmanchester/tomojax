"""Bring a real three-orbit lab CT scan into register with ``tomojax.align``.

The FIPS walnut collection (Der Sarkissian et al., Scientific Data 6, 215,
2019; CC BY 4.0; Zenodo record 2686726) scans each walnut on a FleX-ray
scanner in three circular orbits with the source at three heights. The
scanner's recorded geometry (``scan_geom_original.geom``) places orbits 2 and
3 a fraction of a millimetre off, so a reconstruction from all three blurs
and doubles edges. This loads that uncorrected record, lets ``tj.align``
find each orbit's true height, and reconstructs with the result.

    python examples/align_walnut_orbits.py ~/data/walnuts/Walnut1 -o walnut.nxs

It needs a CUDA GPU (8 GB is enough) and about six minutes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np

import tomojax as tj

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("walnut", type=Path, help="A WalnutN directory of the collection")
    parser.add_argument("-o", "--output", type=Path, help="Save the reconstruction (.nxs)")
    parser.add_argument("--every", type=int, default=4, help="Use every Nth view")
    args = parser.parse_args()
    logging.getLogger("tifffile").setLevel(logging.ERROR)  # a malformed tag, skipped

    orbits = [load_orbit(args.walnut, orbit, args.every) for orbit in (1, 2, 3)]
    data = np.concatenate([d for d, _ in orbits], axis=1)
    vectors = np.concatenate([v for _, v in orbits])
    # The reference reconstructions' volume: a 50.1 mm cube of 0.1 mm voxels.
    window = {f"WindowMin{a}": -25.05 for a in "XYZ"} | {f"WindowMax{a}": 25.05 for a in "XYZ"}
    scan = tj.Scan.from_astra(
        data,
        {"type": "cone_vec", "DetectorRowCount": ROWS, "DetectorColCount": COLUMNS,
         "Vectors": vectors},
        {"GridColCount": 501, "GridRowCount": 501, "GridSliceCount": 501, "option": window},
    )  # fmt: skip
    # The pixels sample the axis twice as finely as the voxels: bin them 2 x 2.
    scan = scan.binned(2)
    print(f"{len(scan.angles)} views of {scan.detector.nv} x {scan.detector.nu} pixels")

    # Per-view pose alignment of all three orbits together; levels 4 and 2 are
    # coarse-to-fine binnings of the grid and detector. The full-resolution
    # level needs more memory than an 8 GB GPU has (tj.align would stop before
    # it on its own), and these two already register the orbits.
    result = tj.align(scan, levels=(4, 2))

    # Each view's vertical correction (the dz pose column), averaged per orbit.
    shift = (result.scan.poses - scan.poses)[:, 4].reshape(3, -1).mean(axis=1)
    relative = shift - shift[0]
    print("orbit heights relative to orbit 1 (mm):", [round(float(h), 3) for h in relative])

    # The data fix the orbits' heights relative to one another, but not the
    # walnut's absolute height: moving the whole object, and every view with it,
    # predicts the same projections. tj.align reports the estimate that moves
    # the views least, so the volume can sit a fraction of a millimetre from
    # where the scanner's record would place it.
    reconstruction = tj.reconstruct(
        result.scan, "fista", iterations=20, tv_weight=0.0, nonnegative=True
    )
    if args.output is not None:
        tj.save(args.output, reconstruction)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
