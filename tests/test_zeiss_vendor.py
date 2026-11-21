"""The Zeiss centre convention against the vendor's own reconstruction of a real scan.

The scan and its reconstruction are not public; the test runs where they are held
and is skipped elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tomojax as tj

SCAN = Path.home() / "data/zeiss-cuo/copper-oxide-spl1/Tristan Manchester_Copper_oxyde_spl1.txrm"
VENDOR = SCAN.with_name(f"{SCAN.stem}_recon.txm")

pytestmark = pytest.mark.skipif(
    not (SCAN.exists() and VENDOR.exists()), reason="needs the CuO-spl1 scan and its reconstruction"
)


def _vendor_middle_slice(binning: int) -> np.ndarray:
    """The vendor reconstruction's middle slice, binned as the scan will be."""
    import olefile

    with olefile.OleFileIO(str(VENDOR)) as source:

        def number(name: str) -> int:
            return int(np.frombuffer(source.openstream(name).read()[:4], "<u4")[0])

        width, height = number("ImageInfo/ImageWidth"), number("ImageInfo/ImageHeight")
        index = number("ImageInfo/NoOfImages") // 2 + 1
        raw = source.openstream(f"ImageData{(index - 1) // 100 + 1}/Image{index}").read()
    image = np.frombuffer(raw, "<u2").reshape(height, width).astype(np.float64)
    rows, columns = height // binning * binning, width // binning * binning
    return image[:rows, :columns].reshape(rows // binning, binning, -1, binning).mean((1, 3))


def _high_pass(image: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return image - gaussian_filter(image, 4)


def _peak_correlation(ours: np.ndarray, theirs: np.ndarray) -> float:
    """Largest normalised cross-correlation over all relative shifts."""
    rows, columns = max(ours.shape[0], theirs.shape[0]), max(ours.shape[1], theirs.shape[1])
    a, b = np.zeros((rows, columns)), np.zeros((rows, columns))
    a[: ours.shape[0], : ours.shape[1]] = (ours - ours.mean()) / ours.std()
    b[: theirs.shape[0], : theirs.shape[1]] = (theirs - theirs.mean()) / theirs.std()
    product = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    return float(np.fft.ifft2(product).real.max() / a.size)


def test_the_recorded_centre_shift_reproduces_the_vendor_reconstruction():
    # spl1 records AutoRecon/CenterShift = +2.55 px. Applied with the wrong sign the
    # centre is 5.1 px out, every small feature becomes a ring and the fine structure
    # no longer matches the vendor's slice (correlation about 0.70 against 0.90).
    frames = tj.load_frames(SCAN)
    rows = frames.counts.shape[1]
    scan = frames.cropped(slice(rows // 2 - 32, rows // 2 + 32), slice(None)).corrected(binning=4)
    grid = scan.geometry.grid
    plane = tj.Grid(grid.nx, grid.ny, 1, grid.vx, grid.vy, grid.vz)
    ours = np.asarray(tj.reconstruct(scan, "fbp", grid=plane).volume, np.float64)[0]
    # The vendor stores its slices turned half a turn and mirrored left to right.
    ours = np.rot90(np.flip(ours, 1), 2)
    correlation = _peak_correlation(_high_pass(ours), _high_pass(_vendor_middle_slice(4)))
    assert correlation > 0.85
