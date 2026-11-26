"""A snail shell from a Zeiss Xradia scan: outside, cut open and in two slices.

    uv run python examples/gallery/zeiss_shell.py S_1_tomo-A.txrm OUT.png

The scan is 1601 views of 1024 x 1024 pixels over a full turn, with the center
and per-view image shifts the file records; no alignment is run. It is binned
twice and reconstructed by FDK on 512³ voxels of 30 micrometers. The shell sat
tilted on its mount, so the volume is turned to stand its coiling axis upright
before it is drawn. The surface views are first-hit renderings of the shell,
the largest connected piece of dense material, lit from the upper left.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import tomojax as tj

TILT = ((38.0, (0, 2)), (-25.0, (0, 1)))  # degrees and plane: stands the coiling axis along z
LIGHT = np.array([-0.5, 0.6, 0.62])


def upright(volume: np.ndarray) -> np.ndarray:
    """``volume`` turned so the shell's coiling axis is z, apex at high z."""
    for angle, plane in TILT:
        volume = ndimage.rotate(volume, angle, axes=plane, reshape=False, order=1)
    return volume


def surface(solid: np.ndarray) -> np.ndarray:
    """Shaded first surface of ``solid`` seen along +y: +x to the right, +z up."""
    hit = solid.any(axis=1)
    depth = np.where(hit, solid.argmax(axis=1), solid.shape[1]).astype(np.float32)
    rows, columns = np.gradient(ndimage.gaussian_filter(depth, 1.2))
    normal = np.stack([-columns, -rows, np.ones_like(rows)], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    shade = 0.22 + 0.78 * np.clip(normal @ (LIGHT / np.linalg.norm(LIGHT)), 0, 1)
    return np.where(hit, shade * (1.0 - 0.4 * depth / solid.shape[1]), 0.0)


def main() -> None:
    """Reconstruct the scan and draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scan", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    scan = tj.load_frames(args.scan).corrected(binning=2)
    recon = tj.reconstruct(scan, "fbp")
    volume = upright(np.asarray(recon.volume, np.float32))
    low, high = np.percentile(volume, [50, 99.9])
    labels, count = ndimage.label(volume > low + 0.6 * (high - low))
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, range(1, count + 1))
    shell = labels == 1 + int(np.argmax(sizes))  # the largest piece: the mount is not joined to it
    z, y, x = (slice(i.min() - 6, i.max() + 7) for i in np.nonzero(shell))
    volume, shell = volume[z, y, x], shell[z, y, x]
    center_y = int(ndimage.center_of_mass(shell)[1])
    spire = int(0.7 * shell.shape[0])  # a height where three whorls are cut
    across = shell.shape[2] * recon.grid.vx
    opened = shell.copy()
    opened[:, :center_y] = False

    style()
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.9), constrained_layout=True)
    for ax, image, title in (
        (axes[0], surface(shell), f"The shell, {across:.0f} mm across"),
        (axes[1], surface(opened), "Cut open along its axis"),
    ):
        ax.imshow(image, origin="lower", cmap="bone", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_axis_off()
    panel(axes[2], volume[:, center_y], "Slice through the axis", low, high)
    panel(axes[3], volume[spire], "Slice across the spire, from the apex side", low, high)
    save(fig, args.out)


if __name__ == "__main__":
    main()
