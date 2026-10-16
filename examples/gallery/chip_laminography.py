"""Aligning a simulated synchrotron laminography scan of a chip package.

The chip package (``bench/phantoms/chip_package.py``: substrate, copper
ground plane and vias, a silicon die, gold bond wires, solder balls with voids
and a mould cap, as STL meshes) is rendered by gVXR as a DIAD-like scan at
25 keV with a 30° tilt: materials from xraylib, propagation phase contrast, a
detector blur, Poisson noise and flat fields. The detector is 3.2 pixels off
the rotation axis and the sample wobbles and jitters by about two pixels and
0.1° from view to view. ``tj.align`` with ``mode="cor-then-pose"`` finds the
centre, then each view's pose with the volume.

    uv run python bench/phantoms/chip_package.py build DATA
    uv run python examples/gallery/chip_laminography.py DATA images/gallery/chip-laminography.png

Both volumes are FISTA-TV reconstructions. The motion panels compare each
view's recovered and true shifts, on the scale of the motion, after removing a
rigid motion of the whole sample, which a reconstruction cannot detect.
Slices across the sample plane are not shown: laminography's missing cone of
directions blurs them for any method.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

# Total variation keeps the metal's streaks down at laminography's limited angles.
RECONSTRUCTION = {"iterations": 100, "tv_weight": 5e-3, "nonnegative": True}


def without_rigid_motion(values: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Remove the constant and once-per-turn parts that a sample shift produces."""
    theta = np.deg2rad(angles)
    basis = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    return values - basis @ np.linalg.lstsq(basis, values, rcond=None)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, help="the chip_package.py build directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scan = tj.load(args.data / "moving-realistic.nxs")
    with np.load(args.data / "truth.npz") as data:
        truth, true_params, cor_mm = data["volume"], data["true_params"], float(data["cor_mm"])
    start = time.perf_counter()
    result = tj.align(scan, mode="cor-then-pose")
    seconds = time.perf_counter() - start
    nominal = np.asarray(tj.reconstruct(scan, "fista", **RECONSTRUCTION).volume)
    aligned = np.asarray(tj.reconstruct(result.scan, "fista", **RECONSTRUCTION).volume)

    style()
    pixel = scan.geometry.detector.du
    angles = np.asarray(scan.geometry.angles)
    found = result.scan.geometry.detector.center[0] / pixel
    fig = plt.figure(figsize=(8.0, 7.9))
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.62), hspace=0.12, wspace=0.04)
    top = 1.2
    rows = (
        ("solder balls", lambda v: v[:, :, 10]),
        ("die and traces", lambda v: v[:, :, 50]),
    )
    columns = (
        ("Phantom (voxelised meshes)", truth),
        ("Nominal geometry", nominal),
        (f"Aligned, {seconds:.0f} s", aligned),
    )
    for r, (label, cut) in enumerate(rows):
        for c, (title, volume) in enumerate(columns):
            ax = fig.add_subplot(grid[r, c])
            panel(ax, cut(volume), title if r == 0 else "", 0.0, top)
            if c == 0:
                ax.text(
                    -0.03, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center"
                )
    poses = np.asarray(result.poses)
    motion = grid[2, :].subgridspec(1, 3, width_ratios=(1, 1, 2.3), wspace=0.4)
    limit = 1.5
    errors = {}
    for k, (column, name, colour) in enumerate(((3, "dx", "tab:blue"), (4, "dz", "tab:orange"))):
        true = without_rigid_motion(true_params[:, column], angles) / pixel
        estimate = without_rigid_motion(poses[:, column], angles) / pixel
        errors[name] = (estimate - true, colour)
        ax = fig.add_subplot(motion[0, k])
        ax.plot([-limit, limit], [-limit, limit], color="0.6", lw=0.6)
        ax.plot(true, estimate, ".", ms=2, color=colour, alpha=0.6)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_xlabel(f"true {name} (px)")
        ax.set_ylabel("recovered (px)" if k == 0 else "")
        ax.set_title(f"{name}, {np.sqrt(np.mean((estimate - true) ** 2)):.2f} px RMS")
    ax = fig.add_subplot(motion[0, 2])
    for name, (error, colour) in errors.items():
        ax.plot(angles, error, ".", ms=1.5, color=colour, label=name)
    ax.set(xlim=(0, 360), ylim=(-limit, limit), xticks=range(0, 361, 90))
    ax.set_xlabel("rotation angle (°)")
    ax.set_ylabel("recovered \N{MINUS SIGN} true (px)")
    ax.set_title(f"Error per view; centre {found:.2f} px, true {cor_mm / pixel:.2f}")
    ax.legend(frameon=False, fontsize=7, markerscale=4, loc="upper right", ncols=2)
    for ax in fig.axes[-3:]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Laminography of a chip package, 720 views at 30° tilt, 8 µm pixels", y=0.915)
    save(fig, args.output)


if __name__ == "__main__":
    main()
