"""Qualification matrix for tj.align: every row scored against independent truth.

Each row is one scan with known geometry and the `fit` it is meant to recover. It runs
twice, each in a fresh process: from the declared (nominal) geometry, and from the true
geometry, which must be a fixed point. A row records the nominal error, both fitted
errors, the pose error after every outer step, cold seconds, peak GPU memory and the
level statuses. Errors are RMS landmark errors over 27 points spanning the central
+-30% of the object, after one common rigid gauge, in the row's stated pixel unit.

    python bench/qualify_alignment.py --list
    python bench/qualify_alignment.py --rows 'analytic/*' --output results.jsonl
    python bench/qualify_alignment.py --row watch/rotations --start truth   # one run

Results stay out of the repository; bench/README.md has the recipe.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

import fit_alignment as analytic
import jax
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import tomojax as tj
from tomojax.alignment._joint import _solver  # check-public-imports: allow-private
from tomojax.geometry import grid_volume_origin

if TYPE_CHECKING:
    from collections.abc import Callable

PACK = Path("/home/tristan/Projects/lab-board/pro-align")
WATCH = PACK / "pack7/data/watch-{}-bin2-fit.npz"
PLUG = PACK / "frames/phantoms/{}.npz"
WALNUTS = Path.home() / "data/walnuts"
WALNUT_CACHE = Path.home() / "Projects/lab-board/walnut-cache"
WATCH_NATIVE_PX = 0.13541666666666666


@dataclass
class Row:
    """A scan, the same data declared with its true geometry, and how to score it."""

    scan: tj.Scan
    true_scan: tj.Scan
    truth: analytic.Rays
    fit: tuple[str, ...]
    points: np.ndarray  # landmarks in object coordinates
    pixel: float  # scoring unit in detector u pitches
    grid: tj.Grid | None  # an explicit working grid, or None for tj.align's own
    # The start scan rebuilt from one reported calibration under its documented
    # convention; it must score like the fit (a round trip of the reported value).
    rebuild: Callable[[dict[str, Any]], tj.Scan] | None = None
    # Injected per-view (offset, row ramp), which tj.align reports as info["offsets"].
    offsets: np.ndarray | None = None


def landmarks(grid: tj.Grid, extent: float = 0.3) -> np.ndarray:
    """Return 27 points over the central +-extent of the grid box, in object coordinates."""
    shape = np.array([grid.nx, grid.ny, grid.nz])
    pitch = np.array([grid.vx, grid.vy, grid.vz])
    center = np.asarray(grid_volume_origin(grid)) + (shape - 1) * pitch / 2
    unit = np.stack(np.meshgrid(*[np.linspace(-extent, extent, 3)] * 3, indexing="ij"), -1)
    return unit.reshape(-1, 3) * shape * pitch + center


BANDS = ((0, 3), (4, 40), (41, None))  # harmonics of the view index


def score(
    estimated: analytic.Rays, truth: analytic.Rays, points: np.ndarray, pixel: float
) -> tuple[float, float]:
    """Return the RMS landmark error after one rigid gauge, and the worst view's."""
    total, worst, _ = score_bands(estimated, truth, points, pixel)
    return total, worst


def score_bands(
    estimated: analytic.Rays, truth: analytic.Rays, points: np.ndarray, pixel: float
) -> tuple[float, float, list[float]]:
    """Also split the RMS error by harmonic of the view index: smooth modes against scatter."""
    target = truth.pixels(points)

    def residual(gauge: np.ndarray) -> np.ndarray:
        moved = points @ Rotation.from_rotvec(gauge[:3]).as_matrix().T + gauge[3:]
        return (estimated.pixels(moved) - target).ravel()

    gauge = least_squares(residual, np.zeros(6), xtol=1e-11, ftol=1e-11, gtol=1e-11).x
    errors = residual(gauge).reshape(-1, len(points), 2) * pixel
    per_view = np.sqrt((errors * errors).sum(-1).mean(1))
    total = float(np.sqrt((per_view * per_view).mean()))
    power = (np.abs(np.fft.rfft(errors, axis=0)) ** 2).sum((1, 2))
    power[1:] *= 2  # one-sided spectrum
    if len(errors) % 2 == 0:
        power[-1] /= 2
    share = [power[lo : None if hi is None else hi + 1].sum() / power.sum() for lo, hi in BANDS]
    return total, float(per_view.max()), [total * float(np.sqrt(s)) for s in share]


def _table(nominal: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Return per-view corrections (alpha, beta, phi, dx, dz, dy) taking nominal to true poses."""
    local = np.einsum("nji,njk->nik", nominal[:, :3, :3], true[:, :3, :3])
    table = np.zeros((len(true), 6))
    table[:, [1, 0, 2]] = Rotation.from_matrix(local).as_euler("YXZ", degrees=True)
    table[:, [3, 5, 4]] = true[:, :3, 3] - nominal[:, :3, 3]
    return table


def _compose(start: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Apply a correction table to (views, 4, 4) poses as the solver does."""
    est = np.asarray(start, np.float64).copy()
    est[:, :3, :3] = est[:, :3, :3] @ Rotation.from_euler("YXZ", table[:, [1, 0, 2]]).as_matrix()
    est[:, :3, 3] += table[:, [3, 5, 4]]
    return est


def _rebuilt(scan: tj.Scan, kind: str, tilt: float = 0.0) -> Callable[[dict[str, Any]], tj.Scan]:
    """Rebuild a scan from one reported calibration under tj.align's documented conventions.

    Center is the detector center in length units, tilt the axis inclination in degrees,
    and positive roll turns detector u towards v.
    """

    def rebuild(calibration: dict[str, Any]) -> tj.Scan:
        ((name, value),) = calibration.items()
        grid, det, angles = scan.grid, scan.detector, np.asarray(scan.angles)
        if name == "center":
            center = tuple(value) if isinstance(value, list | tuple) else (value, det.center[1])
            det = tj.Detector(det.nu, det.nv, det.du, det.dv, center)
        roll = value if name == "roll" else 0.0
        if kind == "cone":
            beam = tj.ConeBeam(192, 288, detector_roll=roll)
            geometry = tj.ConeGeometry(grid, det, angles, beam)
        elif name == "tilt" or tilt:
            geometry = tj.LaminographyGeometry(
                grid,
                det,
                angles,
                tilt=value if name == "tilt" else tilt,
                detector_roll=roll,
            )
        else:
            geometry = tj.ParallelGeometry(grid, det, angles, detector_roll=roll)
        return tj.Scan(scan.projections, geometry)

    return rebuild


def analytic_row(case: str, geometry: str, size: str) -> Row:
    """Build one continuous Gaussian-phantom case from fit_alignment."""
    detector, grid_size, views = {"small": (48, 24, 60), "large": (96, 48, 180)}[size]
    scan, truth, fit, metadata = analytic.make_case(
        case, geometry=geometry, size=detector, views=views, grid_size=grid_size
    )
    rebuild = (
        _rebuilt(scan, geometry, 25.0 if geometry == "lamino" else 0.0)
        if len(fit) == 1
        and fit[0] in {"center", "tilt", "roll"}
        and not (geometry == "cone" and fit[0] == "tilt")
        else None
    )
    offsets = metadata["background"] if case == "background" else None
    return Row(
        scan,
        metadata["true_scan"],
        truth,
        fit,
        landmarks(scan.grid),
        1.0,
        scan.grid,
        rebuild,
        offsets,
    )


def watch_row(case: str) -> Row:
    """Build a run-17 watch laminography case (gVXR, 720 views at 192^2) in native pixels."""
    data = np.load(str(WATCH).format(case))
    g, d = json.loads(str(data["grid"])), json.loads(str(data["detector"]))
    grid = tj.Grid(**g)
    detector = tj.Detector(d["nu"], d["nv"], d["du"], d["dv"], center=tuple(d["center"]))
    views = len(data["projections"])
    geometry = tj.LaminographyGeometry(
        grid, detector, np.linspace(0, 360, views, endpoint=False), tilt=-35
    )
    scan = tj.Scan(np.asarray(data["projections"], np.float32), geometry, name=f"watch {case}")
    nominal = np.asarray([geometry.pose_matrix(i) for i in range(views)])
    if np.abs(nominal - data["nominal_poses"]).max() > 1e-5:
        raise ValueError("the declared watch geometry does not reproduce its nominal poses")
    true = np.asarray(data["true_poses"], np.float64)
    return Row(
        scan,
        scan.with_poses(_table(nominal, true)),
        analytic.physical_rays(true, detector),
        ("translations", "rotations"),
        landmarks(grid),
        detector.du / WATCH_NATIVE_PX,
        None,
    )


def plug_row(name: str, fit: tuple[str, ...], center_px: float = 0.0) -> Row:
    """Build a gVXR spark-plug case (parallel, 720 views of 384^2); roll and tilt are in its poses.

    ``center_px`` declares the detector center that many pixels off the true one.
    """
    data = np.load(str(PLUG).format(name))
    meta = json.loads(str(data["geometry_json"]))
    g, d = meta["grid"], meta["detector"]
    grid = tj.Grid(g["nx"], g["ny"], g["nz"], g["vx"], g["vy"], g["vz"], vol_center=g["vol_center"])
    center = tuple(d["det_center"])
    true_detector = tj.Detector(d["nu"], d["nv"], d["du"], d["dv"], center=center)
    shifted = (center[0] + center_px * d["du"], center[1])
    declared = tj.Detector(d["nu"], d["nv"], d["du"], d["dv"], center=shifted)
    angles = np.asarray(data["angles_deg"])
    projections = np.asarray(data["projections"], np.float32)
    scan = tj.Scan(projections, tj.ParallelGeometry(grid, declared, angles), name=f"plug {name}")
    exact = tj.Scan(projections, tj.ParallelGeometry(grid, true_detector, angles))
    nominal = np.asarray([exact.geometry.pose_matrix(i) for i in range(len(angles))])
    true = np.asarray(data["poses"], np.float64)
    truth = analytic.physical_rays(true, true_detector)
    return Row(
        scan,
        exact.with_poses(_table(nominal, true)),
        truth,
        fit,
        landmarks(grid),
        1.0,
        None,
        _rebuilt(scan, "parallel"),
    )


def walnut_row(number: int) -> Row:
    """Build a FIPS walnut, three cone-beam orbits (every 4th view, 2x binned).

    The start is the scanner's original record and the reference truth is the
    authors' corrected geometry, which moves orbits 2 and 3 by about 0.4 and
    0.8 mm. Only the orbits' relative positions are observable, so the score's
    rigid gauge absorbs the common placement.
    """
    from walnut import load_orbits
    from walnut_alignment import _scan

    walnut = WALNUTS / f"Walnut{number}"
    WALNUT_CACHE.mkdir(parents=True, exist_ok=True)
    data, _ = load_orbits(walnut, [1, 2, 3], 4, cache=WALNUT_CACHE / f"walnut{number}-every4.npz")
    original = _scan(walnut, data, "scan_geom_original.geom", 4, 2)
    corrected = _scan(walnut, data, "scan_geom_corrected.geom", 4, 2)
    truth = analytic.scan_rays(corrected)
    return Row(original, corrected, truth, ("translations",), landmarks(original.grid), 1.0, None)


ROWS: dict[str, tuple[Callable[..., Row], tuple[Any, ...]]] = {
    **{
        f"analytic/{case}/{geometry}/{size}": (analytic_row, (case, geometry, size))
        for case in analytic.CASES
        for geometry in ("parallel", "lamino", "cone")
        for size in ("small", "large")
    },
    "watch/rotations": (watch_row, ("rotations",)),
    "watch/drift-slips": (watch_row, ("drift-slips",)),
    "plug/roll": (plug_row, ("plug2-roll0.75", ("roll",))),
    "plug/tilt": (plug_row, ("plug2-tilt1.5", ("tilt",))),
    "plug/center": (plug_row, ("plug2-roll0", ("center",), 2.0)),
    "walnut/1": (walnut_row, (1,)),
}


def run(name: str, start: str) -> dict[str, Any]:
    """Align one row from 'nominal' or 'truth' in this process and score it."""
    build, args = ROWS[name]
    row = build(*args)
    scan = row.scan if start == "nominal" else row.true_scan
    nominal_error = score(analytic.scan_rays(row.scan), row.truth, row.points, row.pixel)[0]
    start_poses = np.asarray([scan.geometry.pose_matrix(i) for i in range(len(scan.angles))])
    # Per-step poses compose like the final table; global calibrations are not replayed.
    per_step = not set(row.fit) & {"center", "tilt", "roll"} and not analytic.scan_rays(scan).cone
    roll = float(getattr(scan.geometry, "detector_roll", 0))
    steps: list[float] = []
    level = _solver._Level  # noqa: SLF001  (the harness records each outer step's poses)
    original = level.step

    def step(state: Any, *a: Any, **k: Any) -> Any:
        result = original(state, *a, **k)
        if result is not None and per_step:
            ops = state.ops
            table, _ = ops.parameters.physical(
                np.asarray(result[2][2]), np.asarray(ops.scales[: ops.n])
            )
            rays = analytic.physical_rays(_compose(start_poses, table), scan.detector, roll=roll)
            steps.append(round(score(rays, row.truth, row.points, row.pixel)[0], 5))
        return result

    level.step = step
    t = time.perf_counter()
    result = tj.align(scan, fit=row.fit, grid=row.grid)
    np.asarray(result.image)
    seconds = time.perf_counter() - t
    fitted, worst, bands = score_bands(
        analytic.scan_rays(result.scan), row.truth, row.points, row.pixel
    )
    stats = jax.devices()[0].memory_stats() or {}
    levels = result.info.get("levels", [])
    calibration = {k: v["value"] for k, v in result.info.get("calibration", {}).items()}
    round_trip = (
        score(analytic.scan_rays(row.rebuild(calibration)), row.truth, row.points, row.pixel)[0]
        if row.rebuild and start == "nominal" and len(calibration) == 1
        else None
    )
    return {
        "row": name,
        "start": start,
        "fit": list(row.fit),
        "nominal_px": round(nominal_error, 5),
        "fitted_px": round(fitted, 5),
        # Harmonics 0-3 (smooth, near-gauge modes), 4-40 and 41+ (view-to-view scatter).
        "fitted_bands_px": [round(b, 5) for b in bands],
        "worst_view_px": round(worst, 5),
        "per_step_px": steps if per_step else None,
        "seconds": round(seconds, 1),
        "peak_gib": round(stats.get("peak_bytes_in_use", 0) / 2**30, 3),
        "levels": [(lv["status"], len(lv["steps"])) for lv in levels],
        # A linear solve that stops working shows here first.
        "max_cg_residual": [
            round(max((st["cg_relative_residual"] for st in lv["steps"]), default=0.0), 3)
            for lv in levels
        ],
        "first_proposal": round(levels[0]["steps"][0]["proposed_pose_step_rms"], 4)
        if levels and levels[0]["steps"]
        else None,
        "calibration": calibration,
        # A geometry rebuilt from the reported value must score like the fit.
        "round_trip_px": None if round_trip is None else round(round_trip, 5),
        # RMS error of the reported offsets and ramps, relative to the injected RMS.
        "offset_error": None
        if row.offsets is None
        else [
            round(float(np.sqrt(np.mean((fit - true) ** 2) / np.mean(true**2))), 4)
            for fit, true in zip(np.asarray(result.info["offsets"]).T, row.offsets.T, strict=True)
        ],
    }


def main() -> None:
    """Run rows in fresh processes, or one row in this process with --row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--rows", default="*", help="fnmatch pattern over row names")
    parser.add_argument("--row", help="run this row in this process")
    parser.add_argument("--start", choices=("nominal", "truth"), default="nominal")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.list:
        print("\n".join(ROWS))
        return
    if args.row:
        print(json.dumps(run(args.row, args.start), default=str), flush=True)
        return
    for name in (n for n in ROWS if fnmatch.fnmatch(n, args.rows)):
        for start in ("nominal", "truth"):
            command = [sys.executable, __file__, "--row", name, "--start", start]
            try:
                done = subprocess.run(
                    command, capture_output=True, text=True, timeout=args.timeout, check=False
                )
                lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
                record = (
                    json.loads(lines[-1])
                    if lines
                    else {"row": name, "start": start, "error": done.stderr[-2000:]}
                )
            except subprocess.TimeoutExpired:
                record = {"row": name, "start": start, "error": f"timeout {args.timeout} s"}
            line = json.dumps(record, default=str)
            print(line, flush=True)
            if args.output:
                with args.output.open("a") as f:
                    f.write(line + "\n")


if __name__ == "__main__":
    main()
