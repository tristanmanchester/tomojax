"""2DeteCT slices as fan-beam frames, on the grid of the published segmentation.

2DeteCT (Kiss et al., Sci. Data 10, 576, 2023; CC BY 4.0) scans each slice
three times on a FleX-ray scanner: mode 1 at 90 kV with a Thoraeus filter and
3 W (low dose), mode 2 at 90 kV, Thoraeus filter and 90 W (high fidelity, the
source of the reference segmentation), and mode 3 at 60 kV unfiltered (beam
hardening). The out-of-distribution slices (5521 onwards) are held out; slice
5920 belongs to the noise study, whose modes are permuted, and is not used.

Counts are binned 2x along the detector, as for the published reconstructions,
and each view is normalised so that the open beam beside the sample tube has
unit transmission: the recorded flats sit 2-8 % below the open beam of the
sample frames, and in mode 2 they drift within each batch of ten slices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path("~/data/2detect").expanduser()
SOURCE_TO_AXIS = 431.019989
SOURCE_TO_DETECTOR = 529.000488
PIXEL = 0.0748
BINNING = 2
COLUMNS = 1912 // BINNING
SIZE = 1024
VOXEL = COLUMNS * PIXEL * BINNING * SOURCE_TO_AXIS / SOURCE_TO_DETECTOR / SIZE
# Raw columns that see the open beam beside the sample tube in every view.
OPEN = np.r_[0:64, 1848:1912]
# The scanner had two detectors; their centres differ (sharpest FBP of modes 2 and 3).
FIRST_DETECTOR = (range(1, 2831), range(5521, 5871))
# Segmentation labels.
BACKGROUND, WALL, FILLER, OBJECTS = 1, 2, 3, 4


def slices(split: str) -> list[int]:
    """Slice numbers present locally: ``dev`` (in distribution) or ``heldout`` (OOD)."""
    found = sorted(int(p.name[5:]) for p in (DATA / "raw").glob("slice*"))
    if split == "dev":
        return [s for s in found if s <= 5000]
    if split == "heldout":
        return [s for s in found if s >= 5521 and s != 5920]
    raise ValueError(f"split must be 'dev' or 'heldout', not {split!r}")


def _read(path: Path) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(path), dtype=np.float64)


def frames(slice_id: int, mode: int) -> dict[str, np.ndarray]:
    """Binned counts ``(views, 1, 956)``, flat and dark ``(1, 1, 956)``, open-beam normalised.

    The last of the 3601 views repeats the first and is dropped. The returned
    counts are rescaled about the dark so that ``(counts - dark) / (flat -
    dark)`` averages 1 over the open-beam columns of every view.
    """
    folder = DATA / "raw" / f"slice{slice_id:05d}" / f"mode{mode}"
    counts = _read(folder / "sinogram.tif")[:-1]
    dark = _read(folder / "dark.tif")[0]
    flat = 0.5 * (_read(folder / "flat1.tif")[0] + _read(folder / "flat2.tif")[0])
    open_beam = ((counts[:, OPEN] - dark[OPEN]) / (flat[OPEN] - dark[OPEN])).mean(axis=1)
    counts = dark + (counts - dark) / open_beam[:, None]

    def binned(a: np.ndarray) -> np.ndarray:
        return a.reshape(*a.shape[:-1], COLUMNS, BINNING).sum(axis=-1)

    return {
        "counts": binned(counts)[:, None, :].astype(np.float32),
        "flat": binned(flat)[None, None, :].astype(np.float32),
        "dark": binned(dark)[None, None, :].astype(np.float32),
        "open_beam": open_beam.astype(np.float32),
    }


def geometry(slice_id: int, *, center_shift: float | None = None) -> object:
    """TomoJAX fan-beam geometry on the 1024 x 1024 grid of the published images.

    ``center_shift`` is the detector centre in binned pixels; by default the
    value that focuses each of the two detectors.
    """
    import tomojax as tj

    if center_shift is None:
        center_shift = -1.2 if any(slice_id in r for r in FIRST_DETECTOR) else -0.45
    shift = center_shift
    du = PIXEL * BINNING
    return tj.ConeGeometry(
        grid=tj.Grid(nx=SIZE, ny=SIZE, nz=1, vx=VOXEL, vy=VOXEL, vz=VOXEL),
        detector=tj.Detector(nu=COLUMNS, nv=1, du=du, dv=du, center=(shift * du, 0.0)),
        angles=-np.arange(3600) * 0.1,
        beam=tj.ConeBeam(SOURCE_TO_AXIS, SOURCE_TO_DETECTOR),
    )


def orient(image: np.ndarray) -> np.ndarray:
    """Map a published ``(row, column)`` image to TomoJAX ``(y, x)`` indexing."""
    return np.asarray(image)[::-1]


def segmentation(slice_id: int) -> np.ndarray:
    """Published four-class segmentation of the mode 2 image, in ``(y, x)`` indexing."""
    import tifffile

    path = DATA / "reference" / f"slice{slice_id:05d}" / "mode2" / "segmentation.tif"
    return orient(tifffile.imread(path))


def published(slice_id: int, mode: int) -> np.ndarray:
    """Published reconstruction in ``(y, x)`` indexing, in units of mm^-1."""
    import tifffile

    path = DATA / "reference" / f"slice{slice_id:05d}" / f"mode{mode}" / "reconstruction.tif"
    return orient(tifffile.imread(path)) / VOXEL
