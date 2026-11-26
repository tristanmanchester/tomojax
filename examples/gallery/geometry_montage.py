"""One real slice for each geometry TomoJAX reconstructs.

    uv run python examples/gallery/geometry_montage.py DATA geometry-montage.png

``DATA`` holds ``full_rows1048-1111.nxs`` (Diamond I12, Zenodo 3339629, parallel
beam), ``k11-54286.nxs`` (Diamond K11, laminography),
``CuO_2_tomo-B.txrm`` (Zeiss Xradia, cone beam) and ``Walnut1`` (the FIPS
walnut, cone beam on three orbits). Each slice is at the detector's own
resolution and cropped to the largest bright region:

- the Diamond I12 cut of Vo et al.: 1801 views over a half turn,
  ``corrected(*RINGS)`` for its rings, the axis at column 1417.5 (the dataset notes),
  FBP of one detector row;
- K11 54286: 1441 views at 35°, aligned by ``tj.align(scan)``, FBP in-plane
  through the glass-sphere layer at 2.16 µm;
- CuO_2 tomo-B: 3201 views with the center and per-view shifts the file records,
  FDK of the central plane at 1.13 µm;
- Walnut 1: all 3600 views of its three orbits, the default ``tj.reconstruct``
  (CGLS for several cone arrangements) on the authors' 501³ grid of 0.1 mm.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _common import save, style
from _k11 import load as load_k11, plane
from lab_cone_slices import central_slice
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax.corrections import Stripes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _walnut import load_scan

RINGS = (Stripes(9), Stripes(51, varying=True))  # the I12 scan's ring correction


def object_crop(image: np.ndarray, margin: float = 0.06) -> np.ndarray:
    """The smallest centered square holding the largest bright region, with a margin."""
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(np.asarray(image, np.float64), 4)
    low, high = np.percentile(smooth, (5, 99.5))
    labels, count = ndimage.label(smooth > (low + high) / 2)
    largest = np.argmax(ndimage.sum(np.ones_like(smooth), labels, range(1, count + 1))) + 1
    rows, columns = np.nonzero(labels == largest)
    center = np.array(((rows.min() + rows.max()) / 2, (columns.min() + columns.max()) / 2))
    half = (1 + margin) * max(rows.max() - rows.min(), columns.max() - columns.min()) / 2
    half = min(half, *center, *(np.array(image.shape) - center))
    r0, c0 = (center - half).astype(int)
    side = int(2 * half)
    return image[r0 : r0 + side, c0 : c0 + side]


def slices(data: Path) -> dict[str, np.ndarray]:
    """The four slices, keyed by their panel titles."""
    frames = tj.load_frames(
        data / "full_rows1048-1111.nxs", detector=tj.Detector(2560, 64, 1, 1, center=(-138, 0))
    )
    scan = frames.cropped(slice(30, 33), slice(None)).corrected(*RINGS)
    i12 = np.asarray(tj.reconstruct(scan, "fbp").volume, np.float32)[1]
    k11 = tj.align(load_k11(data / "k11-54286.nxs", 35.0)).scan
    walnut = load_scan(data / "Walnut1", orbits=(1, 2, 3), every=1, binning=1, size=501)
    return {
        "Parallel beam · Diamond I12, Vo et al.": i12,
        "Laminography · Diamond K11 54286": plane(k11, (0, 0, 12.96), (704, 704, 1)),
        "Cone beam · Zeiss Xradia CuO_2": central_slice(tj.load_frames(data / "CuO_2_tomo-B.txrm"))[
            0
        ],
        "Three cone orbits · FIPS walnut 1": np.asarray(tj.reconstruct(walnut).volume)[250],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    style()
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.4))
    for ax, (title, image) in zip(axes.ravel(), slices(args.data).items(), strict=True):
        crop = object_crop(image)
        low, high = np.percentile(crop, (0.5, 99.7))
        ax.imshow(crop, origin="lower", vmin=low, vmax=high)
        ax.set_title(f"{title}\n{crop.shape[0]} pixels across")
        ax.set_axis_off()
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
