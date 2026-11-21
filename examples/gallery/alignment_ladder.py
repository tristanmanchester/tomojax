"""The alignment page's figures: each geometry error, before and after the fit that removes it.

    uv run --no-project --with numpy --with scipy --with scikit-image \\
        python bench/phantoms/spark_plug.py PLUG_MESHES          # the plug's meshes, once
    uv run python examples/gallery/alignment_ladder.py PLUG_MESHES OUT_DIR

Two scans are rendered by gVXR from meshes at 120 keV with 1e4 photons per pixel, so
the data never come from TomoJAX's projector:

- the watch movement of ``examples/_watch.py`` in 35° laminography, 720 views on
  384 x 384 detector pixels of 0.135 mm, reconstructed on a 384³ grid;
- the spark plug of ``bench/phantoms/spark_plug.py`` in parallel beam, 720 views on
  384 x 384 pixels of 0.135 mm, on a 384³ grid.

The errors are a 6-pixel error in the declared rotation centre; drift and five slips,
random per-view rotations of 0.37° RMS, a 0.75° roll of the rotation axis about the
beam, and a 3° tilt of the axis towards the beam with the plug 9 mm off the axis, each
rendered into the data. Every arm is reconstructed by FBP. The lower row of a figure is
each image minus the same data reconstructed with the true poses moved by that arm's
rigid gauge (one rigid motion of the whole object, which no data can fix), so only the
error that remains shows. Pose errors are the RMS detector error of 27 landmarks after
that gauge, in detector pixels. The numbers are printed as JSON at the end.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time
from typing import Any

from _common import save, style
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import tomojax as tj
from tomojax.geometry import stack_view_poses

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _watch

N = 384
VIEWS = 720
PHOTONS = 1e4
WATCH_PITCH = 0.13541666666666666
WATCH_TILT = -35.0
WATCH_Z = 2.5  # the movement's middle height, mm
PLUG_PITCH = 0.135
SCREWS = np.s_[192, 68:224, 88:244]  # the balance wheel's slice and crop
PLUG_HALF = 80  # half-width of the plug crops, pixels


def table(nominal: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The ``Scan.with_poses`` table taking ``nominal`` poses to ``target`` poses."""
    local = np.einsum("nji,njk->nik", nominal[:, :3, :3], target[:, :3, :3])
    out = np.zeros((len(target), 6))
    out[:, [1, 0, 2]] = Rotation.from_matrix(local).as_euler("YXZ", degrees=True)
    out[:, [3, 5, 4]] = target[:, :3, 3] - nominal[:, :3, 3]
    return out


def poses_of(scan: tj.Scan) -> np.ndarray:
    """A scan's ``(views, 4, 4)`` object-to-lab poses."""
    return np.asarray(stack_view_poses(scan.geometry, scan.projections.shape[0]), np.float64)


def landmarks(grid: tj.Grid) -> np.ndarray:
    """27 points over the central +-30% of the grid box."""
    unit = np.stack(np.meshgrid(*[np.linspace(-0.3, 0.3, 3)] * 3, indexing="ij"), -1)
    size = np.array((grid.nx * grid.vx, grid.ny * grid.vy, grid.nz * grid.vz))
    centre = np.asarray(grid.vol_center if grid.vol_center is not None else (0, 0, 0))
    return unit.reshape(-1, 3) * size + centre


def pixels(
    poses: np.ndarray, detector: tj.Detector, points: np.ndarray, roll: float = 0.0
) -> np.ndarray:
    """Detector (u, v) of ``points`` in each view, in pixels; the beam is along y.

    ``roll`` turns the pixel lattice about the detector centre from u towards v, in degrees.
    """
    lab = np.einsum("nij,kj->nki", poses[:, :3, :3], points) + poses[:, None, :3, 3]
    x, z = (lab[..., [0, 2]] - np.asarray(detector.center)).transpose(2, 0, 1)
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    return np.stack((c * x + s * z, c * z - s * x), -1) / detector.du


@dataclasses.dataclass
class Case:
    """Data with the declared geometry, and the poses and detector they were made with."""

    scan: tj.Scan
    true_poses: np.ndarray
    true_detector: tj.Detector

    def gauge(self, fitted: tj.Scan) -> tuple[np.ndarray, float]:
        """The rigid motion G with fitted(G m) closest to true(m), and the RMS error left."""
        points = landmarks(self.scan.grid)
        target = pixels(self.true_poses, self.true_detector, points)
        estimated = poses_of(fitted)

        def residual(p: np.ndarray) -> np.ndarray:
            moved = points @ Rotation.from_rotvec(p[:3]).as_matrix().T + p[3:]
            roll = getattr(fitted.geometry, "detector_roll", 0.0)
            return (pixels(estimated, fitted.detector, moved, roll) - target).ravel()

        p = least_squares(residual, np.zeros(6), xtol=1e-12, ftol=1e-12, gtol=1e-12).x
        g = np.eye(4)
        g[:3, :3], g[:3, 3] = Rotation.from_rotvec(p[:3]).as_matrix(), p[3:]
        error = residual(p).reshape(-1, 2)
        return g, float(np.sqrt((error**2).sum(1).mean()))

    def truth_scan(self, g: np.ndarray | None = None) -> tj.Scan:
        """The data on the true detector and poses, moved by ``g`` into a fit's frame."""
        geometry = dataclasses.replace(self.scan.geometry, detector=self.true_detector)
        start = tj.Scan(self.scan.projections, geometry)
        target = self.true_poses if g is None else self.true_poses @ np.linalg.inv(g)
        moved = start.with_poses(table(poses_of(start), target))
        assert np.abs(poses_of(moved) - target).max() < 1e-4
        return moved


def arm(case: Case, label: str, fit: str | tuple[str, ...] | None = None) -> dict[str, Any]:
    """Fit ``case`` (not at all for ``fit=None``) and reconstruct it and its gauged reference."""
    start = time.perf_counter()
    result = None if fit is None else tj.align(case.scan, fit=fit)
    seconds = None if fit is None else time.perf_counter() - start
    fitted = case.scan if result is None else result.scan
    g, error = case.gauge(fitted)
    print(
        f"{label}: {error:.3f} px" + ("" if seconds is None else f", {seconds:.1f} s"), flush=True
    )
    calibration = {} if result is None else dict(result.info.get("calibration", {}))
    return {
        "label": label,
        "image": np.asarray(tj.reconstruct(fitted, "fbp").volume, np.float32),
        "reference": np.asarray(tj.reconstruct(case.truth_scan(g), "fbp").volume, np.float32),
        "pose_px": error,
        # The fit's frame is the true one moved by g: show the same plane of the object.
        "shift": tuple(int(s) for s in np.rint(g[:3, 3][::-1] / case.scan.grid.vx)),
        "seconds": seconds,
        "calibration": {k: v.get("value") for k, v in calibration.items() if isinstance(v, dict)},
    }


def watch_case(true_table: np.ndarray, *, centre_px: float = 0.0, seed: int) -> Case:
    """The watch rendered with ``true_table`` and declared with a centre ``centre_px`` off."""
    grid = tj.Grid(N, N, N, *(3 * (WATCH_PITCH,)), vol_center=(0.0, 0.0, WATCH_Z))
    v = WATCH_Z * np.cos(np.radians(WATCH_TILT))
    detector = tj.Detector(N, N, WATCH_PITCH, WATCH_PITCH, center=(0.0, v))
    declared = dataclasses.replace(detector, center=(centre_px * WATCH_PITCH, v))
    angles = np.linspace(0.0, 360.0, VIEWS, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, declared, angles, tilt=WATCH_TILT)
    blank = tj.Scan(np.zeros((VIEWS, N, N), np.float32), geometry)
    true_poses = poses_of(
        tj.Scan(blank.projections, dataclasses.replace(geometry, detector=detector)).with_poses(
            true_table
        )
    )
    light = _watch.transmission(
        _watch.meshes(), true_poses, (N, N), WATCH_PITCH, center=detector.center
    )
    return Case(tj.Scan(_watch.measure(light, PHOTONS, seed), geometry), true_poses, detector)


def plug_case(meshes: Path, *, roll: float = 0.0, tilt: float = 0.0, offset: float = 0.0) -> Case:
    """The plug rendered with its axis rolled about the beam or tilted towards it, mm off axis."""
    meta = json.loads((meshes / "spark_plug.json").read_text())
    middle = 0.5 * sum(meta["height_mm"])
    grid = tj.Grid(N, N, N, *(3 * (PLUG_PITCH,)), vol_center=(0.0, 0.0, middle))
    detector = tj.Detector(N, N, PLUG_PITCH, PLUG_PITCH, center=(0.0, middle))
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0.0, 360.0, VIEWS, endpoint=False))
    nominal = poses_of(tj.Scan(np.zeros((VIEWS, N, N), np.float32), geometry))
    turn = np.eye(4)
    turn[:3, :3] = (
        Rotation.from_euler("y", roll, degrees=True) * Rotation.from_euler("x", tilt, degrees=True)
    ).as_matrix()
    pivot, back = np.eye(4), np.eye(4)
    pivot[2, 3], back[2, 3] = middle, -middle
    true_poses = np.einsum("ij,njk->nik", pivot @ turn @ back, nominal)
    mount = np.eye(4)
    mount[0, 3] = offset
    mu = {name: m["mu_mm"] for name, m in meta["materials"].items()}
    light = _watch.transmission(
        meshes, true_poses @ mount, (N, N), PLUG_PITCH, center=detector.center, mu=mu
    )
    scan = tj.Scan(_watch.measure(light, PHOTONS, seed=7), geometry)
    return Case(scan, true_poses, detector)


def drift_table(seed: int = 3) -> np.ndarray:
    """Smooth drift of about a pixel and five slips of 1.5-3 pixels, across and along the axis."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, VIEWS)
    out = np.zeros((VIEWS, 6))
    for column in (3, 4):
        drift = 1.2 * np.sin(2 * np.pi * (t * rng.uniform(0.6, 1.4) + rng.uniform()))
        slips = sum(
            rng.uniform(1.5, 3.0) * rng.choice((-1, 1)) * (np.arange(VIEWS) >= at)
            for at in rng.choice(np.arange(60, VIEWS - 60), 5, replace=False)
        )
        out[:, column] = (drift + slips) * WATCH_PITCH
    return out


def rotation_table(seed: int = 5) -> np.ndarray:
    """Independent per-view rotations, 0.37° RMS in all, with 0.3-pixel translations."""
    rng = np.random.default_rng(seed)
    out = np.zeros((VIEWS, 6))
    out[:, :2] = rng.normal(0.0, 0.37 / np.sqrt(2), (VIEWS, 2))
    out[:, 3:5] = rng.normal(0.0, 0.3 * WATCH_PITCH, (VIEWS, 2))
    return out


def relative(image: np.ndarray, reference: np.ndarray) -> float:
    """Relative L2 difference."""
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def moved(cut: tuple[Any, ...], shift: tuple[int, ...]) -> tuple[Any, ...]:
    """``cut`` (z, y, x) moved by whole voxels; slices keep their length, full ones stay."""
    return tuple(
        c + s if isinstance(c, int) else c if c.start is None else slice(c.start + s, c.stop + s)
        for c, s in zip(cut, shift, strict=True)
    )


def ladder(
    arms: list[dict[str, Any]],
    truth: np.ndarray,
    cut: Any,
    title: str,
    path: Path,
    span: float = 0.45,
) -> None:
    """Each arm's image over its difference from the reference, then the true-geometry image."""
    reference = truth[cut]
    low, high = np.percentile(reference, (0.5, 99.8))
    span *= high - low  # the difference row's range, a share of the grey range
    style()
    columns = len(arms) + 1
    aspect = reference.shape[0] / reference.shape[1]
    fig, axes = plt.subplots(2, columns, figsize=(2.6 * columns, 2 * 2.6 * aspect + 1.0))
    for column, item in enumerate(arms):
        own = moved(cut, item["shift"])
        image = item["image"][own]
        label = item["label"] + f"\n{item['pose_px']:.3f} px"
        if item["seconds"] is not None:
            label += f" · {item['seconds']:.0f} s"
        axes[0, column].imshow(image, origin="lower", vmin=low, vmax=high)
        axes[0, column].set_title(label)
        axes[1, column].imshow(
            image - item["reference"][own], origin="lower", cmap="RdBu_r", vmin=-span, vmax=span
        )
    axes[0, -1].imshow(reference, origin="lower", vmin=low, vmax=high)
    axes[0, -1].set_title("True geometry\n")
    axes[1, 0].set_ylabel("minus true geometry")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, -1].set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, path)


def heights(
    arms: list[dict[str, Any]], truth: np.ndarray, cuts: list[tuple[str, Any]], path: Path
) -> None:
    """Rows of slices at several heights; columns the arms and the true geometry."""
    style()
    columns = len(arms) + 1
    fig, axes = plt.subplots(len(cuts), columns, figsize=(2.3 * columns, 2.3 * len(cuts) + 0.6))
    for row, (name, cut) in enumerate(cuts):
        low, high = np.percentile(truth[cut], (0.5, 99.8))
        for column, item in enumerate(
            [*arms, {"label": "True geometry", "image": truth, "shift": (0, 0, 0)}]
        ):
            ax = axes[row, column]
            ax.imshow(item["image"][moved(cut, item["shift"])], origin="lower", vmin=low, vmax=high)
            if row == 0:
                ax.set_title(item["label"])
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(name)
    fig.suptitle("Spark plug, rotation axis rolled 0.75° about the beam")
    fig.tight_layout()
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("plug_meshes", type=Path, help="bench/phantoms/spark_plug.py's output")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    numbers: dict[str, Any] = {}

    def record(name: str, arms: list[dict[str, Any]], **extra: Any) -> None:
        numbers[name] = {
            "arms": [
                {k: a[k] for k in ("label", "pose_px", "seconds", "calibration")} for a in arms
            ],
            **extra,
        }

    screws = SCREWS
    clean = np.zeros((VIEWS, 6))
    case = watch_case(clean, centre_px=6.0, seed=2)
    truth = np.asarray(tj.reconstruct(case.truth_scan(), "fbp").volume, np.float32)
    arms = [arm(case, "With the error"), arm(case, 'fit="center"', "center")]
    ladder(arms, truth, screws, "The watch's balance wheel, a 6-pixel error in the rotation centre",
           args.out / "alignment-ladder-centre.png")  # fmt: skip
    record("centre", arms)

    case = watch_case(drift_table(), seed=3)
    truth = np.asarray(tj.reconstruct(case.truth_scan(), "fbp").volume, np.float32)
    arms = [
        arm(case, "With the error"),
        arm(case, 'fit="translations"\n(the default)', "translations"),
    ]
    ladder(arms, truth, screws, "Drift and five slips between views",
           args.out / "alignment-ladder-drift.png")  # fmt: skip
    record("drift", arms)

    case = watch_case(rotation_table(), seed=4)
    truth = np.asarray(tj.reconstruct(case.truth_scan(), "fbp").volume, np.float32)
    arms = [
        arm(case, "With the error"),
        arm(case, 'fit="translations"', "translations"),
        arm(case, 'fit=("translations",\n"rotations")', ("translations", "rotations")),
    ]
    ladder(arms, truth, screws, "Per-view rotations, 0.37° RMS",
           args.out / "alignment-ladder-rotations.png", span=0.15)  # fmt: skip
    record("rotations", arms)

    meta = json.loads((args.plug_meshes / "spark_plug.json").read_text())
    middle = 0.5 * sum(meta["height_mm"])
    centre = N // 2
    across = np.s_[centre - PLUG_HALF : centre + PLUG_HALF]

    def level(mm: float) -> int:
        return round((middle + mm - (middle - (N / 2 - 0.5) * PLUG_PITCH)) / PLUG_PITCH)

    case = plug_case(args.plug_meshes, roll=0.75)
    truth = np.asarray(tj.reconstruct(case.truth_scan(), "fbp").volume, np.float32)
    arms = [
        arm(case, "With the error"),
        arm(case, 'fit="translations"', "translations"),
        arm(case, 'fit=("translations",\n"roll")', ("translations", "roll")),
    ]
    section = (np.s_[:], centre, across)
    ladder(arms, truth, section, "Spark plug, rotation axis rolled 0.75° about the beam",
           args.out / "alignment-ladder-roll-vertical.png")  # fmt: skip
    cuts = [(f"{mm:+g} mm", (level(mm), across, across)) for mm in (19.0, 0.0, -11.0, -22.0)]
    heights(arms, truth, cuts, args.out / "alignment-ladder-roll.png")
    record(
        "roll",
        arms,
        slice_error={
            a["label"]: [relative(a["image"][c], a["reference"][c]) for _, c in cuts] for a in arms
        },
    )

    case = plug_case(args.plug_meshes, tilt=3.0, offset=9.0)
    truth = np.asarray(tj.reconstruct(case.truth_scan(), "fbp").volume, np.float32)
    arms = [arm(case, "With the error"), arm(case, 'fit="translations"', "translations")]
    off = centre + round(9.0 / PLUG_PITCH)
    section = (np.s_[:], centre, np.s_[off - PLUG_HALF : off + PLUG_HALF])
    ladder(arms, truth, section, "Spark plug 9 mm off the axis, axis tilted 3° towards the beam",
           args.out / "alignment-ladder-tilt.png")  # fmt: skip
    record(
        "tilt",
        arms,
        section_error={
            a["label"]: relative(a["image"][section], a["reference"][section]) for a in arms
        },
    )
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
