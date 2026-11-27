r"""Ring removal on three real scans: fine rings, wide rings and a full turn.

    uv run python examples/gallery/ring_removal.py k11-52567-row1080.nxs \
        k11-52510-row1080.nxs 159950-row1035.nxs ring-removal.png

Each input is one detector row of raw counts with its flats and darks, written
as NXtomo (``image_key`` and angles in degrees) so ``tj.load_frames`` reads it:

- Diamond K11 (DIAD) scan 52567 of visit mg40414-1, detector row 1080: 8000
  views over a half turn on 2560 pixels of 0.54 µm, with fine rings;
- K11 scan 52510 of the same visit, row 1080: a setup scan, 8000 views over a
  half turn, with wide rings;
- Diamond I13 scan 159950 of visit mg31912-1, row 1035: 3001 views over a
  full turn on 2560 pixels.

Each row is corrected to line integrals, its rotation center is measured with
``tj.align(scan, fit="center")``, and the slice is reconstructed by the default
``tj.reconstruct`` without and with ``scan.corrected(Rings())``. Rings are
centered on the rotation axis, so each panel shows the 380 pixels around it.
The two slices of a row share one gray range; the third panel is their
difference, what ``Rings()`` removed, on a gray range symmetric about zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import save, shrunk, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax.corrections import Rings

HALF = 190  # half the side of the region shown around the rotation axis, in pixels

ROWS = (
    "Half turn, fine rings\nK11 52567",
    "Half turn, wide rings\nK11 52510",
    "Full turn\nI13 159950",
)


def slices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The slice without and with ``Rings()``, 380 pixels around the axis."""
    plain = tj.align(tj.load_frames(path).corrected(), fit="center").scan
    detector = plain.detector
    print(f"{path.name}: center {detector.center[0] / detector.du:+.2f} px")
    images = []
    for scan in (plain, plain.corrected(Rings())):
        image = np.asarray(tj.reconstruct(scan).volume, np.float32)[0]
        middle = image.shape[0] // 2
        images.append(image[middle - HALF : middle + HALF, middle - HALF : middle + HALF])
    return images[0], images[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("fine", type=Path, help="a half turn with fine rings")
    parser.add_argument("wide", type=Path, help="a half turn with wide rings")
    parser.add_argument("full", type=Path, help="a full turn")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    style()
    fig, axes = plt.subplots(3, 3, figsize=(8.9, 9.1))
    for label, path, row in zip(ROWS, (args.fine, args.wide, args.full), axes, strict=True):
        without, with_rings = slices(path)
        removed = without - with_rings
        low, high = np.percentile(with_rings, (0.5, 99.5))
        limit = np.percentile(np.abs(removed), 99.5)
        for ax, image, (vmin, vmax) in zip(
            row,
            (without, with_rings, removed),
            ((low, high), (low, high), (-limit, limit)),
            strict=True,
        ):
            ax.imshow(shrunk(image, 420), origin="lower", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(label, fontsize=9)
    for ax, title in zip(
        axes[0], ("Without a ring step", "With Rings()", "Removed by Rings()"), strict=True
    ):
        ax.set_title(title)
    fig.suptitle(f"One detector row of each scan · {2 * HALF} pixels around the rotation axis")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, args.output)


if __name__ == "__main__":
    main()
