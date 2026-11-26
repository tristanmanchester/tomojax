"""The three held-out splits drawn on a real sinogram.

    uv run python examples/gallery/held_out_splits.py k11-54013.nxs held-out-splits.png

One corrected detector row of Diamond K11 laminography scan 54013 (1441 views
of 640 columns) is shown three times, with the observations each
``tomojax.evaluation.Split`` holds out: every sixth view
(``Split.interleaved``), a 30° block of angles with 5° guards
(``Split.angular``), and 32-column blocks with 8-column guards in every view
(``Split.detector``). Held-out observations are orange, guards gray, and
the training observations keep their values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import save, style
import h5py
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax.evaluation import Split

ROW = 270


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="the raw k11-54013.nxs acquisition")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.input) as source:
        sample = np.asarray(source["entry/instrument/EtherCAT/image_key"]) == 0
        angles = np.asarray(source["entry/imaging_sum/smaract_zrot"], np.float64)[sample]
    frames = tj.load_frames(args.input).cropped(slice(ROW, ROW + 1), slice(None))
    sinogram = np.asarray(frames.corrected().projections, np.float32)[:, 0]
    shape = (sinogram.shape[0], 1, sinogram.shape[1])
    splits = {
        "Split.interleaved(stride=6)": Split.interleaved(shape, stride=6),
        "Split.angular(start=60, width=30, guard=5)": Split.angular(
            shape, angles, start=60, width=30, guard=5
        ),
        "Split.detector(start=32, width=32,\nperiod=128, guard=8)": Split.detector(
            shape, start=32, width=32, period=128, guard=8
        ),
    }
    low, high = np.percentile(sinogram, (1, 99))
    gray = np.clip((sinogram - low) / (high - low), 0, 1)
    style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 5.6), sharey=True)
    for ax, (title, split) in zip(axes, splits.items(), strict=True):
        train = np.broadcast_to(split.train, shape)[:, 0]
        test = np.broadcast_to(split.test, shape)[:, 0]
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        rgb[test] = 0.55 * rgb[test] + 0.45 * np.array((1.0, 0.5, 0.05))
        rgb[~train & ~test] = 0.6
        ax.imshow(rgb, aspect="auto", interpolation="nearest",
                  extent=(0, shape[2], angles[-1], angles[0]))  # fmt: skip
        ax.set_title(title)
        ax.set_xlabel("detector column")
    axes[0].set_ylabel("view angle (°)")
    fig.suptitle(f"Held-out observations on K11 scan 54013, detector row {ROW}")
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
