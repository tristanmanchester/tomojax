"""Align a watch-movement scan with an off-centre axis and a wobbling, jittering sample.

The 29 mm movement of ``_watch.py`` is scanned in a 120 keV parallel beam,
400 views over a half turn with 2 x 10^4 photons per pixel, on 77 µm voxels.
The rotation axis projects 5.3 pixels off the detector centre, and every view
has its own pose: a slow wobble of the sample plus jitter, up to 0.25° in each
rotation and 2.4 pixels across and 1.5 pixels along the axis. gVXR traces the
projections through the movement's meshes at those poses. ``tj.align`` with
``mode="cor-then-pose"`` finds the centre, then each view's pose with the
volume; the nominal and the aligned scans are reconstructed with the same
FISTA-TV settings.

Rigidly moving the whole movement, with every pose moved back, predicts the
same data, and so does trading the detector centre against a constant shift of
every view; ``tj.align`` reports the estimate with the least per-view motion.
The truth and its poses are moved to that same estimate before they are
compared. Writes the README's ``images/alignment-example.png`` and a JSON
record beside it.

    uv run --no-sync python examples/align_watch.py
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np

from _figure import ACCENT, MUTED, TEXT, Card
import _watch
import tomojax as tj
from tomojax.alignment.api import apply_pose_updates, least_motion_estimate
from tomojax.geometry import stack_view_poses

SIZE, LAYERS, FIELD, MIDDLE = 384, 82, 29.6, 3.0  # voxels, z layers, field (mm), centre z
VIEWS, PHOTONS = 400, 2e4
CENTRE_PX, ROTATION_DEG = 5.3, 0.25
RECONSTRUCTION = {"iterations": 100, "tv_weight": 0.01, "nonnegative": True, "warm_start": True}
ZOOM = 140  # voxels on a side of the balance close-up


def motion(angles: np.ndarray, voxel: float) -> np.ndarray:
    """Per-view (alpha, beta, phi, dx, dz): a slow wobble plus jitter."""
    rng = np.random.default_rng(3)
    theta = np.deg2rad(angles)
    n = len(angles)
    rotation = np.deg2rad(ROTATION_DEG)
    params = np.zeros((n, 5))
    params[:, 0] = rotation * (0.6 * np.sin(2 * theta + 0.4) + rng.uniform(-0.4, 0.4, n))
    params[:, 1] = rotation * (0.6 * np.cos(2 * theta) + rng.uniform(-0.4, 0.4, n))
    params[:, 2] = rotation * rng.uniform(-0.4, 0.4, n)
    params[:, 3] = voxel * (1.6 * np.sin(2 * theta + 1.1) + rng.uniform(-0.8, 0.8, n))
    params[:, 4] = voxel * (1.0 * np.sin(4 * theta) + rng.uniform(-0.5, 0.5, n))
    return params


def run(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Simulate the scan, align it and compare everything with the truth."""
    meshes = _watch.meshes(args.blender)
    voxel = FIELD / SIZE
    grid = tj.Grid(SIZE, SIZE, LAYERS, voxel, voxel, voxel, vol_center=(0.0, 0.0, MIDDLE))
    detector = tj.Detector(SIZE + 16, LAYERS + 14, voxel, voxel, center=(0.0, MIDDLE))
    angles = np.linspace(0, 180, VIEWS, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    nominal = np.asarray(stack_view_poses(geometry, VIEWS))
    params = motion(angles, voxel)
    poses = apply_pose_updates(
        jnp.asarray(nominal), jnp.asarray(params, jnp.float32), translation_frame="detector"
    )
    centre = CENTRE_PX * voxel
    light = _watch.transmission(
        meshes,
        np.asarray(poses, np.float64),
        (detector.nv, detector.nu),
        voxel,
        centre=(centre, MIDDLE),
        gvxr_python=args.gvxr_python,
    )
    scan = tj.Scan(_watch.measure(light, PHOTONS, seed=2), geometry)
    truth = _watch.truth(meshes, (SIZE, SIZE, LAYERS), voxel, (0.0, 0.0, MIDDLE))

    start = time.perf_counter()
    result = tj.align(scan, mode="cor-then-pose")
    align_seconds = time.perf_counter() - start
    before = np.asarray(tj.reconstruct(scan, "fista", **RECONSTRUCTION).volume)
    after = np.asarray(tj.reconstruct(result.scan, "fista", **RECONSTRUCTION).volume)

    # The truth as the least-motion estimate: a centre offset c predicts the
    # same data as every view shifted by -c, which the gauge moves back.
    true_table = np.pad(params, ((0, 0), (0, 1)))
    true_table[:, 3] -= centre
    moved, true_poses, gauge = least_motion_estimate(
        truth,
        true_table,
        nominal=nominal,
        grid=grid,
        translation_frame="detector",
        active=("alpha", "beta", "phi", "dx", "dz"),
        cone_beam=False,
        detector_offset=True,
    )
    found = np.asarray(result.poses)[:, :5]
    difference = found - true_poses[:, :5]
    true_centre = gauge.detector_offset if gauge is not None else -float(np.mean(true_table[:, 3]))
    found_centre = float(result.scan.geometry.detector.center[0])

    def error(volume: np.ndarray, reference: np.ndarray) -> float:
        return float(np.linalg.norm(volume - reference) / np.linalg.norm(reference))

    k = round((_watch.levels(meshes)["balance"] - MIDDLE) / voxel + LAYERS / 2 - 0.5)
    slices = {
        "before": before[:, :, k],
        "after": after[:, :, k],
        "truth": moved[:, :, k],
        "angles": angles,
        "true_dx": true_poses[:, 3] / voxel,
        "found_dx": found[:, 3] / voxel,
        "true_dz": true_poses[:, 4] / voxel,
        "found_dz": found[:, 4] / voxel,
    }
    metrics = {
        "object": "29 mm watch movement, meshes from bench/phantoms/watch_blender.py",
        "beam": f"parallel, monochromatic {_watch.ENERGY_KEV:g} keV",
        "views": VIEWS,
        "angles_deg": [0, 180],
        "photons_per_pixel": PHOTONS,
        "voxel_mm": voxel,
        "grid": [SIZE, SIZE, LAYERS],
        "slice_z_mm": _watch.levels(meshes)["balance"],
        "true_axis_offset_px": CENTRE_PX,
        "motion": "wobble and jitter, up to 0.25 deg per rotation, 2.4 px dx, 1.5 px dz",
        "mode": "cor-then-pose",
        "align_seconds": align_seconds,
        "reconstruction": {"method": "fista", **RECONSTRUCTION, "warm_start": "fbp"},
        "relative_l2_nominal": error(before, truth),
        "relative_l2_aligned": error(after, moved),
        "least_motion_gauge": gauge.to_dict() if gauge is not None else None,
        "centre_px": {"found": found_centre / voxel, "true": true_centre / voxel},
        "rotation_rmse_deg": np.rad2deg(np.sqrt(np.mean(difference[:, :3] ** 2, 0))).tolist(),
        "shift_rmse_px": (np.sqrt(np.mean(difference[:, 3:] ** 2, 0)) / voxel).tolist(),
        "device": str(__import__("jax").devices()[0]),
        "versions": {name: version(name) for name in ("tomojax", "jax", "numpy", "matplotlib")},
    }
    return slices, metrics


def plot(slices: dict[str, np.ndarray], metrics: dict[str, object], path: Path) -> None:
    """Nominal, aligned and true slices, close-ups, and the recovered motion."""
    panel, gap, margin, top = 560, 30, 30, 158
    strip = 250
    card = Card(3 * panel + 2 * gap + 2 * margin, top + 2 * panel + strip + 168)
    card.text(
        margin,
        54,
        "Aligning a scan with an off-centre axis and a wobbling sample",
        32,
        weight="bold",
    )
    card.text(
        margin,
        92,
        f"{metrics['views']} views over 180°  ·  axis "
        f"{metrics['true_axis_offset_px']:g} px off centre  ·  every view tilted up to "
        f"{ROTATION_DEG:g}° and shifted up to 2.4 px  ·  "
        'tj.align(scan, mode="cor-then-pose")',
        21,
        MUTED,
    )
    voxel = float(metrics["voxel_mm"])
    centre = (np.array([-4.98, -6.918]) + FIELD / 2) / voxel - 0.5
    lo = np.round(centre - ZOOM / 2).astype(int)
    window = (slice(lo[0], lo[0] + ZOOM), slice(lo[1], lo[1] + ZOOM))
    columns = (
        ("before", "Nominal geometry", f"error {metrics['relative_l2_nominal']:.3f}"),
        (
            "after",
            f"Aligned in {metrics['align_seconds']:.0f} s",
            f"error {metrics['relative_l2_aligned']:.3f}",
        ),
        ("truth", "Ground truth", ""),
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
        card.image(x, y, panel, slices[key][window], 0.0, 0.36, smooth=True)
        card.rectangle(x, y, panel, panel, ACCENT, 1.5)
        if k == 2:
            card.scale_bar(
                x + panel - 24 - 5 / voxel * scale, top + panel - 24, 5 / voxel * scale, "5 mm"
            )
            card.scale_bar(
                x + panel - 24 - panel / ZOOM / voxel, y + panel - 24, panel / ZOOM / voxel, "1 mm"
            )
    y = top + 2 * panel + 96
    inner = 3 * panel + 2 * gap
    width = (inner - 3 * 60) // 2
    rmse = metrics["shift_rmse_px"]
    for k, (name, label) in enumerate((("dx", "across the axis"), ("dz", "along the axis"))):
        x = margin + 60 + k * (width + 120)
        ax = card.axes(x, y, width, strip - 40)
        ax.plot(slices["angles"], slices[f"true_{name}"], "o", color="#5d6672", ms=4.2, mew=0)
        ax.plot(slices["angles"], slices[f"found_{name}"], "o", color=ACCENT, ms=1.9, mew=0)
        ax.set_xlim(-2, 182)
        ax.set_xticks([0, 45, 90, 135, 180], ["0°", "45°", "90°", "135°", "180°"])
        ax.set_ylim(-3, 3)
        ax.set_yticks([-2, 0, 2])
        card.text(x, y - 14, f"Shift {label} per view (px)", 22, TEXT)
        card.text(x + width, y - 14, f"recovered to {rmse[k]:.2f} px RMS", 22, MUTED, ha="right")
    rotations = metrics["rotation_rmse_deg"]
    found, true = metrics["centre_px"]["found"], metrics["centre_px"]["true"]
    card.text(
        margin + 60,
        y + strip + 16,
        f"Grey: true, amber: recovered, against rotation angle.  Rotations recovered to "
        f"{max(rotations):.2f}° RMS or better, the detector centre to "
        f"{abs(found - true):.2f} px.",
        21,
        MUTED,
    )
    card.text(
        margin + 60,
        y + strip + 50,
        "Poses, centre and truth are compared in the least-motion frame tj.align reports "
        f"(centre {found:.2f} px found, {true:.2f} px true in that frame).",
        21,
        MUTED,
    )
    card.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=_watch.ROOT / "images" / "alignment-example.png"
    )
    parser.add_argument("--blender", type=Path, default=_watch.BLENDER)
    parser.add_argument("--gvxr-python", type=Path, default=_watch.GVXR_PYTHON)
    args = parser.parse_args()
    start = time.perf_counter()
    slices, metrics = run(args)
    np.savez_compressed(_watch.CACHE / "alignment-slices.npz", **slices)
    plot(slices, metrics, args.out)
    metrics["total_seconds"] = time.perf_counter() - start
    args.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
