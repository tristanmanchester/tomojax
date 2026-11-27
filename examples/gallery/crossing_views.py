"""Detecting and excluding a crossing obstruction during joint alignment.

A 128 x 128 x 48 synthetic specimen with internal pores and dense beads drifts
smoothly over a 120-view 360° parallel-beam scan while a vertical support strut
crosses the detector over 10 views (150° to 177°). ``disturbed_views`` flags
that exact interval from the signal shared by the top and bottom quarters of
the detector. Aligning all views pulls the horizontal poses during the crossing
and leaves streaks in the reconstruction; excluding the flagged range with
``AlignConfig(exclude_views=...)`` and reconstructing the retained views
recovers both the poses and the specimen.

    JAX_PLATFORMS=cpu uv run --with matplotlib python \
        examples/gallery/crossing_views.py docs/images/alignment-crossing.png
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from _common import panel, save, style
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj
from tomojax.alignment import AlignConfig
from tomojax.alignment.api import disturbed_views
from tomojax.datasets import blobs

N = 128
NZ = 48
VIEWS = 120


def build_phantom(size: int, nz: int) -> np.ndarray:
    """A cylindrical specimen with pores and bright beads spanning the detector height."""
    volume = blobs(size, size, nz, seed=3).astype(np.float32)
    z, y, x = np.mgrid[:nz, :size, :size]
    cy = (size - 1) / 2.0
    cz = (nz - 1) / 2.0
    radius = np.hypot(y - cy, x - cy)
    volume = np.where(radius < 0.36 * size, volume, 0.0).astype(np.float32)
    scale = size / 128.0
    beads = (
        (cz, cy - 18 * scale, cy + 16 * scale, 4.0 * scale),
        (cz, cy + 20 * scale, cy - 14 * scale, 3.5 * scale),
        (cz, cy + 8 * scale, cy + 24 * scale, 3.0 * scale),
        (cz, cy - 20 * scale, cy - 18 * scale, 3.5 * scale),
        (cz, cy - 2 * scale, cy - 26 * scale, 2.8 * scale),
    )
    for bz, by, bx, br in beads:
        bead = (z - bz) ** 2 + (y - by) ** 2 + (x - bx) ** 2 <= br**2
        volume = np.where(bead, 1.9, volume)
    return volume.astype(np.float32)


def simulate_crossing(
    size: int, nz: int
) -> tuple[tj.Scan, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a drifting 360° parallel-beam scan with a vertical strut crossing 10 views."""
    rng = np.random.default_rng(7)
    angles = np.linspace(0.0, 360.0, VIEWS, endpoint=False)
    phantom = build_phantom(size, nz)
    grid = tj.Grid(size, size, nz, 1.0, 1.0, 1.0)
    detector = tj.Detector(size, nz, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, detector, angles)

    t = np.linspace(0.0, 1.0, VIEWS)
    scale = size / 64.0
    true_poses = np.zeros((VIEWS, 6), dtype=np.float64)
    true_poses[:, 3] = scale * (2.0 * np.sin(2 * np.pi * t) + 0.6 * np.cos(4 * np.pi * t))
    true_poses[:, 4] = scale * (1.0 * np.cos(2 * np.pi * t - 0.5))

    moved = tj.Scan(np.zeros((VIEWS, nz, size), np.float32), geometry).with_poses(true_poses)
    clean = np.asarray(tj.project(moved.geometry, phantom), np.float32)
    noise_sd = 0.015 * float(np.percentile(clean, 99))
    noisy = clean + rng.normal(0.0, noise_sd, clean.shape).astype(np.float32)

    affected = np.arange(50, 60)
    u = np.arange(size, dtype=np.float32)
    centers = np.linspace(0.16 * size, 0.84 * size, len(affected))
    strut = (0.9 * float(np.percentile(clean, 99.5))) * np.exp(
        -(((u[None, :] - centers[:, None]) / (0.05 * size)) ** 2)
    )
    proj = noisy.copy()
    proj[affected] += strut[:, None, :]
    return tj.Scan(proj, geometry), noisy, true_poses, angles, affected


def rigid_residual(
    fitted: np.ndarray, truth: np.ndarray, angles_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-view horizontal and vertical pose error after removing one rigid shift."""
    theta = np.deg2rad(angles_deg)
    dx = fitted[:, 3] - truth[:, 3]
    dz = fitted[:, 4] - truth[:, 4]
    basis = np.column_stack((np.cos(theta), np.sin(theta)))
    dx = dx - basis @ np.linalg.lstsq(basis, dx, rcond=None)[0]
    dz = dz - np.median(dz)
    return dx, dz


def rms(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Root-mean-square of ``values`` (optionally restricted to ``mask``)."""
    subset = values if mask is None else values[mask]
    return float(np.sqrt(np.mean(subset**2)))


def plot_figure(
    output: Path,
    proj: np.ndarray,
    angles: np.ndarray,
    affected: np.ndarray,
    report: dict[str, Any],
    dx_curves: tuple[np.ndarray, np.ndarray],
    slices: tuple[np.ndarray, np.ndarray],
    vmax: float,
) -> None:
    """Draw the projection, score curve, horizontal pose error, and two CGLS slices."""
    dx_all, dx_ex = dx_curves
    recon_all, recon_ex = slices
    nz = proj.shape[1]
    scores = np.asarray([np.nan if s is None else s for s in report["score"]], dtype=float)

    style()
    fig = plt.figure(figsize=(9.6, 5.9))
    gs = fig.add_gridspec(2, 6, height_ratios=(1.0, 1.15), hspace=0.50, wspace=0.38)

    ax_proj = fig.add_subplot(gs[0, :3])
    view_idx = int(affected[len(affected) // 2])
    pmax = float(np.percentile(proj[view_idx], 99.5))
    ax_proj.imshow(proj[view_idx], origin="lower", cmap="gray", vmin=0.0, vmax=pmax, aspect="auto")
    band = max(1, nz // 4)
    for row in (band - 0.5, nz - band - 0.5):
        ax_proj.axhline(row, color="tab:orange", ls="--", lw=0.9)
    label_box = {"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"}
    ax_proj.text(
        3,
        nz - band / 2,
        "top quarter",
        color="tab:orange",
        fontsize=7.5,
        va="center",
        bbox=label_box,
    )
    ax_proj.text(
        3,
        band / 2,
        "bottom quarter",
        color="tab:orange",
        fontsize=7.5,
        va="center",
        bbox=label_box,
    )
    ax_proj.set_title(f"Projection at {angles[view_idx]:.0f}° (vertical strut crossing)")
    ax_proj.set_xlabel("Detector column (px)")
    ax_proj.set_ylabel("Row (px)")

    ax_score = fig.add_subplot(gs[0, 3:])
    ax_score.plot(angles, scores, color="#1f4e79", lw=1.3)
    ax_score.axhline(0.5, color="tab:orange", ls="--", lw=0.9, label="0.5 threshold")
    for a0, a1 in report["angles"]:
        ax_score.axvspan(
            a0, a1, color="tab:orange", alpha=0.18, label=f"flagged {a0:.0f}° to {a1:.0f}°"
        )
    ax_score.set_xlim(0.0, 360.0)
    ax_score.set_ylim(-0.04, 1.08)
    ax_score.set_xlabel("View angle (deg)")
    ax_score.set_ylabel("Disturbance score")
    ax_score.set_title("disturbed_views score per exposure")
    ax_score.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax_score.spines["top"].set_visible(False)
    ax_score.spines["right"].set_visible(False)

    ax_err = fig.add_subplot(gs[1, 0:2])
    for a0, a1 in report["angles"]:
        ax_err.axvspan(a0, a1, color="tab:orange", alpha=0.18)
    ax_err.axhline(0.0, color="#999999", lw=0.6)
    ax_err.plot(angles, dx_all, color="#c0392b", lw=1.1, label="all views")
    ax_err.plot(angles, dx_ex, color="#1f4e79", lw=1.2, label="exclude_views")
    ax_err.set_xlim(0.0, 360.0)
    ax_err.set_xlabel("View angle (deg)")
    ax_err.set_ylabel("Horizontal pose error (px)")
    ax_err.set_title("Horizontal pose error dx")
    ax_err.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)

    panel(
        fig.add_subplot(gs[1, 2:4]),
        recon_all,
        f"All views aligned\ncrossing dx RMS {rms(dx_all, affected):.2f} px",
        0.0,
        vmax,
    )
    panel(
        fig.add_subplot(gs[1, 4:6]),
        recon_ex,
        f"With exclude_views (fit_views)\ncrossing dx RMS {rms(dx_ex, affected):.2f} px",
        0.0,
        vmax,
    )
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=N)
    parser.add_argument("--nz", type=int, default=NZ)
    args = parser.parse_args()

    scan, noisy, true_poses, angles, affected = simulate_crossing(args.size, args.nz)
    report = disturbed_views(scan.projections, scan.geometry)
    ranges = tuple(map(tuple, report["ranges"]))

    base = AlignConfig(levels=(2, 1), iterations=6, image_iterations=12, refine_iterations=3)
    all_fit = tj.align(scan, config=base)
    ex_fit = tj.align(scan, config=replace(base, exclude_views=ranges))

    dx_all, dz_all = rigid_residual(all_fit.poses, true_poses, angles)
    dx_ex, dz_ex = rigid_residual(ex_fit.poses, true_poses, angles)
    print(f"detected ranges: {report['ranges']} angles: {report['angles']}")
    print(
        f"pose RMS (dx, dz) all views: all={rms(dx_all):.3f}/{rms(dz_all):.3f} px, "
        f"excluded={rms(dx_ex):.3f}/{rms(dz_ex):.3f} px"
    )
    print(
        f"pose RMS (dx, dz) crossing:  "
        f"all={rms(dx_all, affected):.3f}/{rms(dz_all, affected):.3f} px, "
        f"excluded={rms(dx_ex, affected):.3f}/{rms(dz_ex, affected):.3f} px"
    )

    mid = args.nz // 2
    recon_all = np.asarray(tj.reconstruct(all_fit.scan, "cgls", iterations=15).volume)[mid]
    recon_ex = np.asarray(
        tj.reconstruct(ex_fit.scan.selected(ex_fit.info["fit_views"]), "cgls", iterations=15).volume
    )[mid]
    recon_true = np.asarray(
        tj.reconstruct(
            tj.Scan(noisy, scan.geometry).with_poses(true_poses), "cgls", iterations=15
        ).volume
    )[mid]
    vmax = float(np.percentile(recon_true, 99.7))
    plot_figure(
        args.output,
        np.asarray(scan.projections),
        angles,
        affected,
        report,
        (dx_all, dx_ex),
        (recon_all, recon_ex),
        vmax,
    )


if __name__ == "__main__":
    main()
