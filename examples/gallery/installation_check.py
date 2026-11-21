"""What the installation check's three commands produce.

    uv run python examples/gallery/installation_check.py installation-check.png

Runs the installation page's ``tomojax simulate``, ``tomojax reconstruct`` and
``tomojax inspect`` commands, exactly as written, in a temporary directory and
shows the 32³ phantom's central slice, the central projection and the
reconstruction's central slice that the check produces.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from _common import panel, save, style
import matplotlib.pyplot as plt
from synthetic_scans import read, tomojax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        tomojax("simulate", "-o", "synthetic.nxs", "--size", "32", "--n-views", "60", cwd=work)
        tomojax("reconstruct", "synthetic.nxs", "-o", "recon.nxs", cwd=work)
        tomojax("inspect", "recon.nxs", "--preview", "previews", cwd=work)
        truth = read(work / "synthetic.nxs", "entry/processing/tomojax/volume")
        projections = read(work / "synthetic.nxs", "entry/data/projections")
        volume = read(work / "recon.nxs", "entry/processing/tomojax/volume")
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    high = float(truth.max())
    panel(axes[0], truth[16], "Phantom, central slice", 0.0, high)
    panel(axes[1], projections[30], "Projection, central view", 0.0, float(projections.max()))
    panel(axes[2], volume[16], "Reconstruction, central slice", 0.0, high)
    fig.suptitle("tomojax simulate --size 32 --n-views 60, then tomojax reconstruct", y=0.995)
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
