"""The synthetic-tomography page's two command sequences, drawn.

    uv run python examples/gallery/synthetic_scans.py synthetic-scans.png

Runs the page's ``tomojax simulate`` and ``tomojax reconstruct`` commands in a
temporary directory, at ``--size 128`` rather than the page's 32 so that the
phantom's ellipses are visible, and with FISTA's automatic TV weight rather than
the page's 0.005, set for its 32³ example. It draws for the parallel scan and the 30°
laminography scan: the phantom, one projection, the sinogram of the central
detector row, and the reconstruction across and along the rotation axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from _common import panel, save, style
import h5py
import matplotlib.pyplot as plt
import numpy as np


def tomojax(*arguments: str, cwd: Path) -> None:
    """Run one ``tomojax`` command from this environment."""
    command = Path(sys.executable).with_name("tomojax")
    subprocess.run([str(command), *arguments], cwd=cwd, check=True)


def read(path: Path, name: str) -> np.ndarray:
    """One dataset of a TomoJAX NeXus file."""
    with h5py.File(path) as source:
        return np.asarray(source[name], np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--tv-weight", default="auto", help="a number, or 'auto'")
    args = parser.parse_args()
    size = str(args.size)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        tomojax("simulate", "-o", "synthetic.nxs", "--size", size, "--n-views", "60", cwd=work)
        tomojax("reconstruct", "synthetic.nxs", "-o", "recon.nxs", cwd=work)
        rows.append(("Parallel beam", work / "synthetic.nxs", work / "recon.nxs"))
        tomojax("simulate", "-o", "tilted.nxs", "--geometry", "lamino", "--tilt", "30",
                "--size", size, "--n-views", "60", cwd=work)  # fmt: skip
        weight = [] if args.tv_weight == "auto" else ["--tv-weight", args.tv_weight]
        tomojax("reconstruct", "tilted.nxs", "-o", "tilted-recon.nxs", "--method", "fista",
                "--iterations", "50", *weight, "--nonnegative", "--region", "off",
                cwd=work)  # fmt: skip
        rows.append(("Laminography, 30° tilt", work / "tilted.nxs", work / "tilted-recon.nxs"))
        data = [
            (
                label,
                read(scan, "entry/processing/tomojax/volume"),
                read(scan, "entry/data/projections"),
                read(recon, "entry/processing/tomojax/volume"),
            )
            for label, scan, recon in rows
        ]
    style()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for (label, truth, projections, volume), row in zip(data, axes, strict=True):
        n = truth.shape[0]
        high = float(truth.max())
        panel(row[0], truth[n // 2], "Phantom, central slice", 0.0, high)
        panel(row[1], projections[0], "Projection, first view", 0.0, float(projections.max()))
        panel(row[2], projections[:, projections.shape[1] // 2].T,
              "Sinogram, central row", 0.0, float(projections.max()))  # fmt: skip
        row[2].set_aspect("auto")
        panel(row[3], volume[volume.shape[0] // 2], "Reconstruction, central slice", 0.0, high)
        panel(row[4], volume[:, volume.shape[1] // 2], "Reconstruction, vertical", 0.0, high)
        row[0].text(-0.08, 0.5, label, transform=row[0].transAxes, rotation=90,
                    ha="right", va="center")  # fmt: skip
    fig.suptitle(f"Shepp-Logan phantom, {args.size}³ voxels, 60 views", y=0.995)
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
