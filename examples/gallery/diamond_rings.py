"""A real synchrotron scan from raw counts, with and without ring removal.

Diamond I12 (Vo et al., Zenodo 3339629): 1801 parallel-beam views of a rock
sample at 54 keV, full-resolution detector rows. A miscalibrated detector
column whose response varies with intensity leaves a large ring, and pixels
near the rotation axis with small constant offsets leave tight rings round it.
``Stripes()`` subtracts the constant offsets and ``Stripes(51, varying=True)``
fits the intensity-dependent pixels to their neighbours. The authors' own
reconstruction (Savu) is shown at the same slice, with each slice's
correlation with it inside the sample.

    uv run python examples/gallery/diamond_rings.py RAW.nxs REFERENCE.tif OUT.png

``RAW.nxs`` is a full-resolution row cut with flats and darks, such as rows
1048-1111; ``REFERENCE.tif`` the authors' slice for the row shown.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from _common import save, style
import matplotlib.pyplot as plt
import numpy as np
import tifffile

import tomojax as tj
from tomojax.corrections import Stripes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--row", type=int, default=2, help="row of the cut to show")
    args = parser.parse_args()

    frames = tj.load_frames(args.raw)
    plain = tj.estimate_center(frames.corrected())
    steps = (Stripes(), Stripes(51, varying=True))
    cleaned = replace(frames.corrected(*steps), geometry=plain.geometry)
    reference = tifffile.imread(args.reference).astype(np.float32)
    n = reference.shape[0]
    centres = np.arange(n) - n / 2 + 0.5
    inside = np.hypot(*np.meshgrid(centres, centres, indexing="ij")) < 0.42 * n

    def correlation(image: np.ndarray) -> float:
        return float(np.corrcoef(image[inside], reference[inside])[0, 1])

    slices = {}
    for title, scan in (
        ("TomoJAX FBP", plain),
        ("with Stripes() and Stripes(51, varying=True)", cleaned),
    ):
        image = np.asarray(tj.reconstruct(scan, "fbp").volume[args.row], np.float32)
        slices[f"{title}\ncorrelation {correlation(image):.3f}"] = image
    slices["authors' reconstruction (Savu)"] = reference

    style()
    # The volume is centred on the rotation axis, where the tight rings are.
    half = 160
    zoom = (slice(n // 2 - half, n // 2 + half),) * 2
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.5), gridspec_kw={"height_ratios": (1, 1)})
    for (title, image), (top_ax, bottom_ax) in zip(slices.items(), axes.T, strict=True):
        low, high = np.percentile(image[zoom], (1, 99.5))
        top_ax.imshow(image, origin="lower", vmin=low, vmax=high)
        top_ax.add_patch(
            plt.Rectangle(
                (zoom[0].start, zoom[1].start),
                2 * half,
                2 * half,
                fill=False,
                color="tab:orange",
                lw=0.8,
            )
        )
        bottom_ax.imshow(image[zoom], origin="lower", vmin=low, vmax=high)
        top_ax.set_title(title)
        for ax in (top_ax, bottom_ax):
            ax.set_axis_off()
    fig.suptitle(
        f"Diamond I12, raw counts to slice: {frames.counts.shape[0]} views, "
        f"{n} x {n} pixels, centre found by tj.estimate_center",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
