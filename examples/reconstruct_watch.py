"""Reconstruct a watch movement from few, noisy views with FBP and with FISTA-TV.

A 29 mm skeletonised mechanical watch movement (see ``_watch.py``) is scanned
in a 120 keV parallel beam with 200 views over a half turn and 10^4 photons
per detector pixel. gVXR traces the projections through the movement's
meshes, so they share nothing with TomoJAX's projector, and every pixel
averages the transmission of 2 x 2 rays before Poisson noise and the
logarithm. A 32-slice slab through the plane of the hairspring is
reconstructed on 38.5 µm voxels with filtered backprojection (Hann filter)
and with FISTA-TV (non-negative, warm-started from FBP), and compared with the
exact voxel truth. Writes the README's ``images/reconstruction-example.png``
and a JSON record of settings, errors and run times beside it.

    uv run --no-sync python examples/reconstruct_watch.py
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from _figure import ACCENT, MUTED, Card
import _watch
import tomojax as tj
from tomojax.geometry import stack_view_poses

SIZE, SLAB, FIELD = 768, 32, 29.6  # voxels across, slab thickness in voxels, field in mm
VIEWS, PHOTONS = 200, 1e4
FILTER, TV_WEIGHT, ITERATIONS = "hann", 0.01, 200
ZOOM = 280  # voxels on a side of the balance close-up


def run(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Simulate the scan, reconstruct it both ways and measure the errors."""
    meshes = _watch.meshes(args.blender)
    z = _watch.levels(meshes)["balance"]
    voxel = FIELD / SIZE
    grid = tj.Grid(SIZE, SIZE, SLAB, voxel, voxel, voxel, vol_center=(0.0, 0.0, z))
    detector = tj.Detector(SIZE, SLAB, voxel, voxel, center=(0.0, z))
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, VIEWS, endpoint=False))
    light = _watch.transmission(
        meshes,
        np.asarray(stack_view_poses(geometry, VIEWS)),
        (SLAB, SIZE),
        voxel,
        centre=(0.0, z),
        gvxr_python=args.gvxr_python,
    )
    scan = tj.Scan(_watch.measure(light, PHOTONS, seed=1), geometry)
    truth = _watch.truth(meshes, (SLAB, SIZE, SIZE), voxel, (0.0, 0.0, z))

    start = time.perf_counter()
    fbp = np.asarray(tj.reconstruct(scan, "fbp", filter=FILTER).volume)
    fbp_seconds = time.perf_counter() - start
    start = time.perf_counter()
    fista = tj.reconstruct(
        scan, "fista", iterations=ITERATIONS, tv_weight=TV_WEIGHT, nonnegative=True, warm_start=True
    )
    tv = np.asarray(fista.volume)
    tv_seconds = time.perf_counter() - start

    def error(volume: np.ndarray) -> float:
        return float(np.linalg.norm(volume - truth) / np.linalg.norm(truth))

    middle = SLAB // 2
    slices = {"truth": truth[middle, :, :], "fbp": fbp[middle, :, :], "tv": tv[middle, :, :]}
    metrics = {
        "object": "29 mm watch movement, meshes from bench/phantoms/watch_blender.py",
        "beam": f"parallel, monochromatic {_watch.ENERGY_KEV:g} keV",
        "views": VIEWS,
        "angles_deg": [0, 180],
        "photons_per_pixel": PHOTONS,
        "voxel_mm": voxel,
        "grid": [SIZE, SIZE, SLAB],
        "slice_z_mm": z,
        "fbp": {"filter": FILTER, "relative_l2": error(fbp), "seconds": fbp_seconds},
        "fista_tv": {
            "tv_weight": TV_WEIGHT,
            "iterations": ITERATIONS,
            "nonnegative": True,
            "warm_start": "fbp",
            "relative_l2": error(tv),
            "seconds": tv_seconds,
        },
        "error_region": "the whole reconstructed slab",
        "attenuation_mm": _watch.MU,
        "device": str(__import__("jax").devices()[0]),
        "versions": {name: version(name) for name in ("tomojax", "jax", "numpy", "matplotlib")},
    }
    return slices, metrics


def plot(slices: dict[str, np.ndarray], metrics: dict[str, object], path: Path) -> None:
    """Truth, FBP and FISTA-TV, full and close up, on one grey scale."""
    panel, gap, margin, top = 560, 30, 30, 158
    card = Card(3 * panel + 2 * gap + 2 * margin, top + 2 * panel + 100)
    card.text(margin, 54, "Few-view, noisy scan of a mechanical watch movement", 32, weight="bold")
    card.text(
        margin,
        92,
        f"120 keV parallel beam  ·  {metrics['views']} views over 180°  ·  "
        "10⁴ photons per pixel  ·  slice through the hairspring",
        21,
        MUTED,
    )
    voxel = float(metrics["voxel_mm"])
    centre = (np.array([-4.98, -6.918]) + FIELD / 2) / voxel - 0.5  # the balance
    lo = np.round(centre - ZOOM / 2).astype(int)
    window = (slice(lo[1], lo[1] + ZOOM), slice(lo[0], lo[0] + ZOOM))
    columns = (
        ("truth", "Ground truth", ""),
        ("fbp", "Filtered backprojection", f"error {metrics['fbp']['relative_l2']:.3f}"),
        ("tv", "FISTA-TV", f"error {metrics['fista_tv']['relative_l2']:.3f}"),
    )
    scale = panel / SIZE
    for k, (key, title, note) in enumerate(columns):
        x = margin + k * (panel + gap)
        card.text(x, top - 16, title, 24)
        card.text(x + panel, top - 16, note, 22, MUTED, ha="right")
        card.image(x, top, panel, slices[key], 0.0, 0.36)
        card.rectangle(
            x + lo[0] * scale, top + (SIZE - lo[1] - ZOOM) * scale, ZOOM * scale, ZOOM * scale
        )
        y = top + panel + 22
        card.image(x, y, panel, slices[key][window], 0.0, 0.36)
        card.rectangle(x, y, panel, panel, ACCENT, 1.5)
        if k == 0:
            card.scale_bar(x + 24, top + panel - 24, 5 / voxel * scale, "5 mm")
            card.scale_bar(x + 24, y + panel - 24, 1 / voxel * panel / ZOOM, "1 mm")
    fista = metrics["fista_tv"]
    card.text(
        margin,
        top + 2 * panel + 66,
        f"FBP: Hann filter, {metrics['fbp']['seconds']:.2f} s.   FISTA-TV: weight "
        f"{fista['tv_weight']:g}, non-negative, {fista['iterations']} iterations from FBP, "
        f"{fista['seconds']:.0f} s.   Errors: relative L2 over the whole "
        f"{SIZE} \N{MULTIPLICATION SIGN} {SIZE} \N{MULTIPLICATION SIGN} {SLAB} slab of "
        f"{voxel * 1000:.1f} µm voxels.",
        21,
        MUTED,
    )
    card.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=_watch.ROOT / "images" / "reconstruction-example.png"
    )
    parser.add_argument("--blender", type=Path, default=_watch.BLENDER)
    parser.add_argument("--gvxr-python", type=Path, default=_watch.GVXR_PYTHON)
    args = parser.parse_args()
    start = time.perf_counter()
    slices, metrics = run(args)
    plot(slices, metrics, args.out)
    metrics["total_seconds"] = time.perf_counter() - start
    args.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
