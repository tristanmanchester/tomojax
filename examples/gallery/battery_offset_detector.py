"""A lab cone-beam scan of a cell wider than the detector.

A scaled 18650 lithium-ion cell (``bench/phantoms/battery.py``) is scanned in
cone beam at eight times magnification over a full turn, once with a detector
covering the whole cell and once with a detector 60% as wide, shifted so that
it sees one side of the cell and a little past the rotation axis. FDK weights
the doubly measured rays near the axis (Wang, 2002), so the offset scan
reconstructs the whole cross-section: a field of view nearly twice the
detector's width.

The top row draws each setup from above (not to scale) beside the cell's
cross-section colored by the share of the turn in which the detector sees
each point.

    uv run python examples/gallery/battery_offset_detector.py images/gallery/battery-offset.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import counts, downsample, panel, phantom, relative_error, save, style
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
import matplotlib.pyplot as plt
import numpy as np

import tomojax as tj

N = 256  # reconstruction voxels per side
TRUTH_N = 384  # the phantom is projected from a finer grid
VIEWS = 720
PHOTONS = 5000.0
MAGNIFICATION = 8.0
SOURCE_TO_AXIS = 60.0
CELL_RADIUS = 4.25  # mm
IN_VIEW = LinearSegmentedColormap.from_list("in_view", ["#e8a13a", "#b9b24c", "#5fa98a", "#1f8a8a"])
SHARE = Normalize(0.55, 1.0)  # of the turn


def scan(
    truth: np.ndarray, fov: float, columns: int, offset_mm: float, rng: np.random.Generator
) -> tj.Scan:
    """Scan ``truth`` with ``columns`` detector pixels, centered ``offset_mm`` off the axis."""
    voxel = fov / N
    grid = tj.Grid(N, N, N, voxel, voxel, voxel)
    fine = tj.Grid(TRUTH_N, TRUTH_N, TRUTH_N, *(3 * (fov / TRUTH_N,)))
    pitch = voxel * MAGNIFICATION
    detector = tj.Detector(columns, 300, pitch, pitch, (offset_mm * MAGNIFICATION, 0.0))
    beam = tj.ConeBeam(SOURCE_TO_AXIS, SOURCE_TO_AXIS * MAGNIFICATION)
    angles = np.linspace(0.0, 360.0, VIEWS, endpoint=False)
    geometry = tj.ConeGeometry(grid, detector, angles, beam)
    line_integrals = np.asarray(tj.project(geometry, truth, grid=fine))
    measured = -np.log(np.maximum(counts(line_integrals, PHOTONS, rng), 1.0) / PHOTONS)
    return tj.Scan(measured.astype(np.float32), geometry)


def footprint(geometry: tj.ConeGeometry) -> tuple[float, float]:
    """Where the detector's two edges fall at the rotation axis, in mm from the axis."""
    detector = geometry.detector
    half = detector.nu * detector.du / 2
    return (detector.center[0] - half) / MAGNIFICATION, (detector.center[0] + half) / MAGNIFICATION


def share_in_view(geometry: tj.ConeGeometry, fov: float) -> np.ndarray:
    """The share of the turn in which each point of the central slice reaches the detector."""
    low, high = footprint(geometry)
    centers = (np.arange(N) - N / 2 + 0.5) * fov / N
    y, x = np.meshgrid(centers, centers, indexing="ij")
    seen = np.zeros((N, N))
    angles = np.deg2rad(np.asarray(geometry.angles))
    for angle in angles:
        sideways = x * np.cos(angle) + y * np.sin(angle)
        along = y * np.cos(angle) - x * np.sin(angle)
        u = sideways * SOURCE_TO_AXIS / (SOURCE_TO_AXIS + along)
        seen += (u >= low) & (u <= high)
    return seen / len(angles)


def draw_setup(ax: plt.Axes, cell: np.ndarray, geometry: tj.ConeGeometry, fov: float) -> None:
    """The source, the turning cell and the detector from above, not to scale."""
    low, high = footprint(geometry)
    source, screen = -17.0, 21.5  # drawn distances from the axis; sideways is mm at the axis
    spread = (screen - source) / -source
    edge = lambda u, x: u * (x - source) / -source  # noqa: E731  (the ray through u at the axis)
    ax.add_patch(
        Polygon(
            [(source, 0), (screen, high * spread), (screen, low * spread)],
            facecolor="#e8a13a",
            alpha=0.25,
            edgecolor="#e8a13a",
            lw=0.6,
        )
    )
    half = fov / 2
    image = ax.imshow(
        cell, extent=(-half, half, -half, half), origin="lower", vmin=0.0, vmax=0.3, zorder=2
    )
    disc = Circle((0, 0), CELL_RADIUS + 0.05, transform=ax.transData)
    image.set_clip_path(disc)
    for bound, far in ((low, -half), (high, half)):  # fade what the beam misses in this view
        corners = [
            (-half, edge(bound, -half)),
            (half, edge(bound, half)),
            (half, far),
            (-half, far),
        ]
        ax.add_patch(
            Polygon(corners, facecolor="white", alpha=0.75, lw=0, zorder=2.5)
        ).set_clip_path(disc)
    ax.plot([source, screen], [0, 0], color="0.55", lw=0.6, ls=(0, (4, 3)), zorder=3)
    ax.plot(
        [screen, screen], [-5.25 * spread, 5.25 * spread], color="0.6", lw=2.2, ls=(0, (1.5, 1.2))
    )
    ax.plot(
        [screen, screen], [low * spread, high * spread], color="black", lw=3, solid_capstyle="round"
    )
    ax.plot(source, 0, "o", color="#e8a13a", mec="black", mew=0.5, ms=5)
    ax.plot(0, 0, "o", color="#d1495b", ms=2.6, zorder=4)
    ax.add_patch(
        FancyArrowPatch(
            (-3.2, 5.6), (3.2, 5.6), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
            mutation_scale=7, lw=0.7, color="black",
        )
    )  # fmt: skip
    ax.text(source - 1.0, 1.4, "X-ray\nsource", fontsize=6.5, ha="left", va="bottom")
    ax.text(screen - 0.8, 5.25 * spread, "detector", fontsize=6.5, ha="right", va="top")
    if low > -CELL_RADIUS:  # the detector stops short of the cell's far side
        ax.plot([5.6, 5.6], [0, low], color="#d1495b", lw=1.2, zorder=3)
        ax.plot([5.0, 6.2], [low, low], color="#d1495b", lw=1.2, zorder=3)
        ax.text(
            6.4, low - 1.7, f"{-low:.2f} mm past\nthe axis", fontsize=6, color="#d1495b", va="top"
        )
        ax.text(
            4.4,
            -6.0,
            "outside the beam\nuntil the cell turns",
            fontsize=6.5,
            ha="right",
            va="top",
        )
    ax.set_xlim(source - 1.5, screen + 1.0)
    ax.set_ylim(-5.25 * spread - 0.6, 5.25 * spread + 0.6)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_share(ax: plt.Axes, cell: np.ndarray, share: np.ndarray, fov: float) -> None:
    """The cell's cross-section colored by ``share``, the share of the turn in view."""
    centers = (np.arange(N) - N / 2 + 0.5) * fov / N
    radius = np.hypot(*np.meshgrid(centers, centers, indexing="ij"))
    colors = IN_VIEW(SHARE(share))
    colors[..., :3] *= (0.35 + 0.65 * np.clip(cell / 0.3, 0, 1))[..., None]
    colors[..., 3] = radius <= CELL_RADIUS + 0.05
    ax.imshow(colors, origin="lower")
    ax.plot(N / 2 - 0.5, N / 2 - 0.5, "o", color="#d1495b", ms=2.6)
    rim = share[np.abs(radius - CELL_RADIUS) < fov / N].mean()
    if rim < 0.99:
        ax.text(
            N / 2, N / 2 + 0.16 * N, "100%", color="white", fontsize=6.5, ha="center", weight="bold"
        )
        ax.text(
            N / 2,
            0.1 * N,
            f"{100 * rim:.0f}%",
            color="white",
            fontsize=6.5,
            ha="center",
            weight="bold",
        )
    else:
        ax.text(N / 2, 0.1 * N, "100%", color="white", fontsize=6.5, ha="center", weight="bold")
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    truth, meta = phantom("battery", TRUTH_N)
    fov = float(meta["fov_mm"])
    rng = np.random.default_rng(0)
    full = scan(truth, fov, 300, 0.0, rng)
    # 180 columns of 35 µm at the axis span -1.16 to +5.16 mm: one side of the
    # 8.5 mm cell and 1.16 mm past the axis.
    offset = scan(truth, fov, 180, 2.0, rng)
    volumes = {
        name: np.asarray(tj.reconstruct(s, "fbp").volume)
        for name, s in (("full", full), ("offset", offset))
    }
    truth = downsample(truth, N)

    style()
    centers = (np.arange(N) - N / 2 + 0.5) * fov / N
    x, y = np.meshgrid(centers, centers, indexing="ij")
    mask = np.broadcast_to((np.hypot(x, y) < 4.3)[None], truth.shape)
    z, top = N // 2 - N // 8, 0.3
    # Fixed positions, in inches: a top row of two drawings 1.6 times as wide as tall, each
    # beside a square map; the color bar; then two rows of three square slices.
    width, gap = 7.2, 0.05
    row, side = (width - 5 * gap) / 5.2, (width - 4 * gap) / 3
    slices_top = 0.55 + row + 0.8
    height = slices_top + 2 * side + 2 * gap
    fig = plt.figure(figsize=(width, height))

    def place(left: float, top: float, across: float, down: float) -> plt.Axes:
        return fig.add_axes(
            (left / width, 1 - (top + down) / height, across / width, down / height)
        )

    left = gap
    for title, s in (("Full detector", full), ("Detector 60% as wide, offset", offset)):
        setup = place(left, 0.55, 1.6 * row, row)
        share = place(left + 1.6 * row + gap, 0.55, row, row)
        left += 2.6 * row + 2 * gap
        draw_setup(setup, truth[z], s.geometry, fov)
        setup.set_title(title, loc="left", fontsize=8)
        draw_share(share, truth[z], share_in_view(s.geometry, fov), fov)
    bar = place(width / 2 - 1.3, 0.55 + row + 0.3, 2.6, 0.09)
    fig.colorbar(plt.cm.ScalarMappable(SHARE, IN_VIEW), cax=bar, orientation="horizontal")
    bar.xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    bar.tick_params(labelsize=6.5, length=2)
    bar.set_title("share of the turn in which the detector sees each point", fontsize=7, pad=3)
    axes = np.array(
        [
            [
                place(gap + k * (side + gap), slices_top + r * (side + gap), side, side)
                for k in range(3)
            ]
            for r in range(2)
        ]
    )
    columns = (
        ("Phantom", truth),
        (
            f"full detector, error {relative_error(volumes['full'], truth, mask):.3f}",
            volumes["full"],
        ),
        (
            f"60% detector, offset, {relative_error(volumes['offset'], truth, mask):.3f}",
            volumes["offset"],
        ),
    )
    for (title, volume), (xy_ax, xz_ax) in zip(columns, axes.T, strict=True):
        panel(xy_ax, volume[z], title, 0.0, top)
        panel(xz_ax, volume[:, N // 2, :], "", 0.0, top)
    fig.text(
        0.5,
        1 - 0.14 / height,
        f"Lab cone-beam scan of a lithium-ion cell: {VIEWS} views, {1000 * fov / N:.0f} µm voxels",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    save(fig, args.output, palette=False)


if __name__ == "__main__":
    main()
