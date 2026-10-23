"""FDK against iterative reconstruction on a wide-cone scan of a lattice part.

An additively manufactured Ti-6Al-4V lattice cylinder
(``bench/phantoms/lattice_part.py``) is scanned in lab cone beam with a cone
of about ten degrees at its top and bottom faces. FDK's approximation shows
there as smeared, darkened horizontal plates; CGLS started from FDK fits the
cone-beam data exactly and restores them, along with the build defects.

    uv run python examples/gallery/lattice_cone.py images/gallery/lattice-cone.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import counts, downsample, panel, phantom, relative_error, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

N = 256
TRUTH_N = 384
VIEWS = 480
PHOTONS = 20000.0
SOURCE_TO_AXIS = 36.0
MAGNIFICATION = 8.0


def simulate(seed: int = 0) -> tuple[tj.Scan, np.ndarray, dict[str, object]]:
    """Return the scan's line integrals, the truth on the reconstruction grid, and metadata."""
    truth, meta = phantom("lattice_part", TRUTH_N)
    fov = float(meta["fov_mm"])
    voxel = fov / N
    grid = tj.Grid(N, N, N, voxel, voxel, voxel)
    fine = tj.Grid(TRUTH_N, TRUTH_N, TRUTH_N, *(3 * (fov / TRUTH_N,)))
    pixels = 384
    detector = tj.Detector(pixels, pixels, voxel * MAGNIFICATION, voxel * MAGNIFICATION)
    beam = tj.ConeBeam(SOURCE_TO_AXIS, SOURCE_TO_AXIS * MAGNIFICATION)
    angles = np.linspace(0.0, 360.0, VIEWS, endpoint=False)
    geometry = tj.ConeGeometry(grid, detector, angles, beam)
    line_integrals = np.asarray(tj.project(geometry, truth, grid=fine))
    rng = np.random.default_rng(seed)
    measured = -np.log(np.maximum(counts(line_integrals, PHOTONS, rng), 1.0) / PHOTONS)
    return tj.Scan(measured.astype(np.float32), geometry), downsample(truth, N), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scan, truth, meta = simulate()
    fdk = np.asarray(tj.reconstruct(scan, "fbp").volume)
    cgls = np.asarray(tj.reconstruct(scan, "cgls", iterations=30, warm_start=True).volume)

    style()
    # Errors are over the part's bounding cylinder, not the air round it.
    centres = (np.arange(N) - N / 2 + 0.5) / N * float(meta["fov_mm"])
    x, y, z = np.meshgrid(centres, centres, centres, indexing="ij")
    mask = (np.hypot(x, y) < 5.2) & (np.abs(z) < 5.2)
    rows = (("xz through the axis", np.s_[:, N // 2, :]), ("xy in the top skin", None))
    top = 0.16
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    columns = (
        ("Phantom", truth),
        (f"FDK, error {relative_error(fdk, truth, mask):.3f}", fdk),
        (f"CGLS from FDK, {relative_error(cgls, truth, mask):.3f}", cgls),
    )
    # The xy slice is the top skin, the solid plane with the largest cone angle.
    skin = N // 2 + int(np.argmax(truth[N // 2 :].mean(axis=(1, 2))))
    for (title, volume), (xz_ax, xy_ax) in zip(columns, axes.T, strict=True):
        panel(xz_ax, volume[:, N // 2, :], title, 0.0, top)
        panel(xy_ax, volume[skin], "", 0.0, top)
    for ax, (label, _) in zip(axes[:, 0], rows, strict=True):
        ax.text(-0.04, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center")
    fig.suptitle(
        f"Wide-cone lab scan of a titanium lattice part: {VIEWS} views, "
        f"{1000 * float(meta['fov_mm']) / N:.0f} µm voxels",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
