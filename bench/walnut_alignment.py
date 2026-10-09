r"""Bring a FIPS walnut's three orbits into register from their uncorrected geometry.

The collection (see ``bench/walnut.py``) ships each orbit's geometry twice:
``scan_geom_original.geom`` as the scanner recorded it and
``scan_geom_corrected.geom``, where the authors moved orbits 2 and 3 up by
about 0.4 and 0.8 mm. This aligns the three orbits from the original record
with ``tj.align``, compares the recovered heights with the authors', and
reconstructs (non-negative least squares) with the original, aligned and
corrected geometries against the published reference.

    python bench/walnut_alignment.py ~/data/walnuts/Walnut1 --figures out/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
from walnut import ROWS, VIEWS, compare, load_orbit, reference, volume_geometry

import tomojax as tj

# check-public-imports: allow-private
from tomojax._scan import record_of, scan_from_record


def _scan(walnut: Path, data: np.ndarray, name: str, every: int, binning: int) -> tj.Scan:
    vectors = np.concatenate(
        [np.loadtxt(walnut / f"Projections/tubeV{o}/{name}")[0:VIEWS:every] for o in (1, 2, 3)]
    )
    proj = {"type": "cone_vec", "DetectorRowCount": ROWS, "DetectorColCount": 768}
    return tj.Scan.from_astra(data, proj | {"Vectors": vectors}, volume_geometry()).binned(binning)


def _orbit_one_fixed(scan: tj.Scan, original: tj.Scan) -> tj.Scan:
    """``scan`` with its translation corrections moved so orbit 1's average none.

    Only the orbits' relative heights are observable; the authors keep orbit 1
    where the scanner recorded it. This expresses an alignment in that
    convention, to compare its volume voxel by voxel with the reference.
    """
    poses, recorded = np.asarray(scan.poses), np.asarray(original.poses)
    per = len(scan.angles) // 3
    offset = (poses - recorded)[:per, 3:].mean(axis=0)
    record = record_of(scan)
    record.align_params = poses - np.concatenate([np.zeros(3), offset])
    return tj.Scan(scan.projections, scan_from_record(record, poses=True).geometry)


def _figures(out: Path, volumes: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    series = ["#2a78d6", "#eb6834", "#1baf7a"]
    plt.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "text.color": ink, "axes.labelcolor": muted, "xtick.color": muted,
        "ytick.color": muted, "axes.edgecolor": grid, "font.size": 10, "axes.titlesize": 11,
    })  # fmt: skip
    names = {
        "reference": "Published reference\n(50 iterations, corrected geometry)",
        "original": "Original geometry",
        "aligned": "TomoJAX-aligned geometry",
        "corrected": "Authors' corrected geometry",
    }
    n = volumes["reference"].shape[0]
    peak = float(np.percentile(volumes["reference"], 99.9))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), constrained_layout=True)
    for col, key in enumerate(names):
        title = names[key]
        if key != "reference":
            title += f"\nerror {summary[key]['relative_l2_in_walnut']:.3f}"
        axes[0, col].imshow(np.rot90(volumes[key][:, n // 2, :]), cmap="gray", vmin=0, vmax=peak)
        axes[0, col].set_title(title)
        axes[1, col].imshow(
            np.rot90(volumes[key][:, :, n // 2 + 60]), cmap="gray", vmin=0, vmax=peak
        )
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    axes[0, 0].set_ylabel("vertical slice")
    axes[1, 0].set_ylabel("axial slice")
    fig.suptitle("FIPS Walnut 1, three orbits: 20 iterations of non-negative least squares")
    fig.savefig(out / "walnut_alignment_slices.png", dpi=110)

    heights = summary["heights_mm"]
    per = heights["views_per_orbit"]
    recovered, authors = np.asarray(heights["tomojax"]), np.asarray(heights["authors"])
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for k in range(3):
        s = slice(k * per, (k + 1) * per)
        views = np.arange(s.start, s.stop)
        ax.plot(views, recovered[s], color=series[k], linewidth=2, label=f"orbit {k + 1}: TomoJAX")
        ax.plot(views, authors[s], color=muted, linewidth=1.2, linestyle=(0, (4, 3)))
        below = k == 0  # orbit 1 sits at the top: label it underneath
        text = (
            f"orbit {k + 1}: {round(recovered[s].mean(), 3) + 0.0:+.3f} mm\n"
            f"(authors {round(authors[s].mean(), 3) + 0.0:+.3f})"
        )
        ax.text(
            views.mean(), recovered[s].mean() + (-0.05 if below else 0.05), text,
            ha="center", va="top" if below else "bottom", color=ink, fontsize=9,
        )  # fmt: skip
    ax.plot([], [], color=muted, linewidth=1.2, linestyle=(0, (4, 3)), label="authors' correction")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_ylim(min(-0.9, recovered.min() - 0.1), 0.1)
    ax.set_xlabel("view (orbits in turn)")
    ax.set_ylabel("vertical shift vs orbit 1 (mm)")
    ax.set_title("Orbit misregistration recovered by tj.align from the uncorrected geometry")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(out / "walnut_alignment_heights.png", dpi=110)


def _save(
    summary: dict[str, Any],
    volumes: dict[str, np.ndarray | None],
    poses: dict[str, np.ndarray | None],
    *,
    output: Path | None,
    slices: Path | None,
) -> None:
    """The record and slices so far, each whole, so a later stage that fails loses nothing."""
    if output is not None:
        partial = output.with_name(f"partial-{output.name}")
        partial.write_text(json.dumps(summary, default=str))
        partial.replace(output)
    if slices is not None:
        arrays = {
            f"{name}_{plane}": np.asarray(cut, np.float16)
            for name, volume in volumes.items()
            if volume is not None
            for plane, cut in _central(volume).items()
        }
        partial = slices.with_name(f"partial-{slices.name}")
        np.savez_compressed(partial, **arrays, **poses)
        partial.replace(slices)


def _central(volume: np.ndarray) -> dict[str, np.ndarray]:
    """The three central orthogonal slices of an ``(x, y, z)`` volume."""
    x, y, z = (n // 2 for n in volume.shape)
    return {"xz": volume[:, y, :], "yz": volume[x], "xy": volume[..., z]}


def main() -> None:
    """Run the alignment comparison and print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("walnut", type=Path, help="A WalnutN directory from the data collection")
    parser.add_argument("--every", type=int, default=4, help="Use every Nth view")
    parser.add_argument("--bin", type=int, default=2, help="Average N x N detector pixels")
    parser.add_argument("--levels", default="4,2", help="Alignment resolution levels")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--figures", type=Path, help="Write PNG figures to this directory")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs to share the views among")
    parser.add_argument("--slices", type=Path, help="Save central slices and poses (.npz)")
    parser.add_argument("--output", type=Path, help="Write the record here after each stage")
    args = parser.parse_args()
    logging.getLogger("tifffile").setLevel(logging.ERROR)

    data = np.concatenate([load_orbit(args.walnut, o, args.every)[0] for o in (1, 2, 3)], axis=1)
    original = _scan(args.walnut, data, "scan_geom_original.geom", args.every, args.bin)
    corrected = _scan(args.walnut, data, "scan_geom_corrected.geom", args.every, args.bin)
    import jax

    # One GPU takes the ordinary one-device path, as a user would run it.
    devices = jax.devices()[: args.gpus] if args.gpus > 1 else None
    start = time.perf_counter()
    levels = tuple(int(f) for f in args.levels.split(","))
    result = tj.align(original, levels=levels, devices=devices)
    summary: dict[str, Any] = {"gpus": args.gpus, "align_seconds": time.perf_counter() - start}
    info = result.info
    summary["alignment"] = {
        k: info.get(k) for k in ("mode", "levels", "factors", "factors_skipped", "loss", "gauge")
    }
    aligned = _orbit_one_fixed(result.scan, original)

    recorded = np.asarray(original.poses)[:, 4]
    per = len(original.angles) // 3
    tomojax = np.asarray(aligned.poses)[:, 4] - recorded
    authors = np.asarray(corrected.poses)[:, 4] - recorded
    summary["heights_mm"] = {
        "views_per_orbit": per,
        "tomojax": tomojax.tolist(),
        "authors": authors.tolist(),
        "orbit_means": {
            "tomojax": tomojax.reshape(3, per).mean(axis=1).tolist(),
            "authors": authors.reshape(3, per).mean(axis=1).tolist(),
        },
    }
    truth = reference(args.walnut)
    volumes = {"reference": truth}
    poses = {
        "original_poses": original.poses,
        "aligned_poses": aligned.poses,
        "corrected_poses": corrected.poses,
    }

    def save() -> None:
        _save(summary, volumes, poses, output=args.output, slices=args.slices)

    save()
    options = {"iterations": args.iterations, "tv_weight": 0.0, "nonnegative": True}
    for name, scan in (("original", original), ("aligned", aligned), ("corrected", corrected)):
        start = time.perf_counter()
        volumes[name] = np.asarray(tj.reconstruct(scan, "fista", devices=devices, **options).volume)
        summary[name] = {"seconds": time.perf_counter() - start, **compare(volumes[name], truth)}
        save()
    if args.figures is not None:
        args.figures.mkdir(parents=True, exist_ok=True)
        _figures(args.figures, volumes, summary)
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
