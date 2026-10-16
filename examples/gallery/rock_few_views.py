"""Few-view synchrotron micro-CT of a sandstone plug: FBP, CGLS and FISTA-TV.

A 1 mm sandstone core (``bench/phantoms/porous_rock.py``) is scanned in
parallel beam with only 90 views over a half turn, as in fast or dose-limited
imaging. The sandstone absorbs weakly at 30 keV, so FBP and least squares
(CGLS) are noisy; total-variation regularisation (FISTA-TV, started from FBP)
removes the noise while keeping grain contacts, the cracks and the pyrite.

    uv run python examples/gallery/rock_few_views.py images/gallery/rock-few-views.png
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
VIEWS = 90
PHOTONS = 1e5
TV_WEIGHT = 1e-4


def simulate(
    seed: int = 0, photons: float = PHOTONS
) -> tuple[tj.Scan, np.ndarray, dict[str, object]]:
    """Return the scan's line integrals, the truth on the reconstruction grid, and metadata."""
    truth, meta = phantom("porous_rock", TRUTH_N)
    fov = float(meta["fov_mm"])
    voxel = fov / N
    grid = tj.Grid(N, N, N, voxel, voxel, voxel)
    fine = tj.Grid(TRUTH_N, TRUTH_N, TRUTH_N, *(3 * (fov / TRUTH_N,)))
    detector = tj.Detector(N, N, voxel, voxel)
    angles = np.linspace(0.0, 180.0, VIEWS, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    line_integrals = np.asarray(tj.project(geometry, truth, grid=fine))
    rng = np.random.default_rng(seed)
    measured = -np.log(np.maximum(counts(line_integrals, photons, rng), 1.0) / photons)
    return tj.Scan(measured.astype(np.float32), geometry), downsample(truth, N), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--photons", type=float, default=PHOTONS)
    parser.add_argument("--tv-weight", type=float, default=TV_WEIGHT)
    args = parser.parse_args()
    scan, truth, meta = simulate(photons=args.photons)
    results = {
        "FBP": tj.reconstruct(scan, "fbp").volume,
        "CGLS": tj.reconstruct(scan, "cgls", iterations=20).volume,
        "FISTA-TV": tj.reconstruct(
            scan,
            "fista",
            iterations=150,
            tv_weight=args.tv_weight,
            nonnegative=True,
            warm_start=True,
        ).volume,
    }

    style()
    centres = np.arange(N) - N / 2 + 0.5
    core = np.hypot(*np.meshgrid(centres, centres, indexing="ij")) < 0.44 * N
    mask = np.broadcast_to(core[..., None], truth.shape)
    z, top = N // 2, 0.55
    zoom = (slice(N // 2, N // 2 + 80), slice(N // 2 - 70, N // 2 + 10))
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.9), gridspec_kw={"height_ratios": (1, 1)})
    columns = [("Phantom", truth)] + [
        (f"{name}, error {relative_error(np.asarray(v), truth, mask):.3f}", np.asarray(v))
        for name, v in results.items()
    ]
    for (title, volume), (top_ax, bottom_ax) in zip(columns, axes.T, strict=True):
        image = volume[:, :, z]
        panel(top_ax, image, title, 0.0, top)
        top_ax.add_patch(
            plt.Rectangle(
                (zoom[0].start, zoom[1].start), 80, 80, fill=False, color="tab:orange", lw=0.8
            )
        )
        panel(bottom_ax, image[zoom], "", 0.0, top)
    fig.suptitle(
        f"Sandstone plug, parallel beam: {VIEWS} views over 180°, "
        f"{1000 * float(meta['fov_mm']) / N:.1f} µm voxels",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == "__main__":
    main()
