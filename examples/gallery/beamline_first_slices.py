"""First slices from the beamline-exports page's five recipes.

    uv run python examples/gallery/beamline_first_slices.py DATA beamline-first-slices.png

``DATA`` holds the five files under the names the page uses:
``bamboo_reduced_bliss.nx`` (ESRF, nabu's test file), ``tomo_00069.h5`` (PSI
TOMCAT, TomoBank), ``tomo_00024.h5`` (Elettra SYRMEP, TomoBank),
``porous_glass_dataset.mat`` (PSI cSAXS, Zenodo 3539513) and
``diamond-i13-mg31912-1/raw/159815.nxs`` (Diamond I13). Each recipe runs as the
page writes it: one native detector row, flat/dark correction and the
logarithm (except cSAXS, which holds line integrals), then the default
``tj.reconstruct``. Each slice is shown whole and with one enlarged region.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax.corrections import Step, Stripes

# (row, column, side) of each enlarged region, in slice pixels.
CROPS = {
    "ESRF NXtomo": (300, 300, 200),
    "PSI TOMCAT": (300, 300, 200),
    "Elettra SYRMEP": (700, 700, 400),
    "PSI cSAXS": (500, 500, 250),
    "Diamond I13": (500, 500, 300),
}


# The ring correction each slice needs; panels without rings take none.
RINGS: dict[str, tuple[Step, ...]] = {
    "ESRF NXtomo": (Stripes(9), Stripes(51, varying=True)),
    "Diamond I13": (Stripes(201),),
}


def middle_row(frames: tj.Frames, steps: tuple[Step, ...] = ()) -> np.ndarray:
    """The page's one-row preview: the middle detector row, corrected and reconstructed."""
    row = frames.counts.shape[1] // 2
    scan = frames.cropped(slice(row, row + 1), slice(None)).corrected(*steps)
    return np.asarray(tj.reconstruct(scan).volume, np.float32)[0]


def slices(data: Path) -> dict[str, np.ndarray]:
    """One reconstructed slice per recipe, keyed by facility."""
    out = {
        "ESRF NXtomo": middle_row(
            tj.load_frames(data / "bamboo_reduced_bliss.nx"), RINGS["ESRF NXtomo"]
        )
    }
    pitch = 3.7e-3
    out["PSI TOMCAT"] = middle_row(
        tj.load_frames(
            data / "tomo_00069.h5",
            detector=tj.Detector(1024, 428, pitch, pitch, center=((1023 / 2 - 515.5) * pitch, 0)),
        )
    )
    pitch = 2.04e-3
    out["Elettra SYRMEP"] = middle_row(
        tj.load_frames(
            data / "tomo_00024.h5",
            detector=tj.Detector(2048, 2048, pitch, pitch, center=((2047 / 2 - 972) * pitch, 0)),
        )
    )
    scan = tj.load(data / "porous_glass_dataset.mat")
    d = scan.detector
    grid = tj.Grid(d.nu, d.nu, 1, d.du, d.du, d.dv)
    out["PSI cSAXS"] = np.asarray(tj.reconstruct(scan, grid=grid).volume, np.float32)[0]
    frames = tj.load_frames(
        data / "diamond-i13-mg31912-1/raw/159815.nxs",
        detector=tj.Detector(2560, 2160, 1, 1, center=(120, 0)),
    )
    scan = frames.cropped(slice(1080, 1081), slice(441, 1872)).corrected(*RINGS["Diamond I13"])
    out["Diamond I13"] = np.asarray(tj.reconstruct(scan).volume, np.float32)[0]
    return out


def figure(images: dict[str, np.ndarray], output: Path) -> None:
    """Each slice whole above its enlarged region, sharing that slice's grey range."""
    style()
    fig, axes = plt.subplots(2, len(images), figsize=(13, 5.6))
    for (name, image), (whole, zoom) in zip(images.items(), axes.T, strict=True):
        low, high = np.percentile(image, (0.5, 99.5))
        row, column, side = CROPS[name]
        panel(whole, image, f"{name} · {image.shape[1]} pixels across", low, high)
        whole.add_patch(
            plt.Rectangle((column, row), side, side, fill=False, color="tab:orange", lw=0.8)
        )
        panel(zoom, image[row : row + side, column : column + side], "Enlarged", low, high)
    fig.suptitle("One native detector row from each beamline export, default tj.reconstruct")
    fig.tight_layout()
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    figure(slices(args.data), args.output)


if __name__ == "__main__":
    main()
