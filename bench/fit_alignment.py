"""Small independent recovery checks for the named alignment parameters.

Projections are closed-form line integrals of an asymmetric Gaussian mixture,
not output from TomoJAX's projector. Each invocation runs exactly one alignment.
The default is 60 views of a 48-square detector and a 24-cubed working image.
Truth is used only to render observations and to score the completed fit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import tomojax as tj
from tomojax.geometry import grid_volume_origin
from tomojax.geometry.api import is_cone_beam, view_frames

if TYPE_CHECKING:
    from collections.abc import Sequence


CASES = ("clean", "centre", "translations", "tilt", "roll", "rotations", "setup")


def phantom() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centres, precision matrices and amplitudes of fixed smooth ellipsoids."""
    rng = np.random.default_rng(9361)
    centres = rng.uniform(-12, 12, (12, 3))
    radii = rng.uniform(2.0, 4.0, (12, 3))
    orientation = Rotation.random(12, random_state=rng).as_matrix()
    precision = np.einsum("nij,nj,nkj->nik", orientation, radii**-2, orientation)
    return centres, precision, rng.uniform(0.02, 0.06, 12)


def density(points: np.ndarray) -> np.ndarray:
    """Evaluate the continuous phantom in physical xyz coordinates."""
    centres, precision, amplitudes = phantom()
    output = np.zeros(points.shape[:-1], np.float64)
    for centre, matrix, amplitude in zip(centres, precision, amplitudes, strict=True):
        offset = points - centre
        exponent = np.einsum("...i,ij,...j->...", offset, matrix, offset)
        output += amplitude * np.exp(-0.5 * exponent)
    return output


def line_integrals(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Integrate the continuous phantom along unit-direction infinite rays."""
    centres, precision, amplitudes = phantom()
    output = np.zeros(origins.shape[:-1], np.float64)
    for centre, matrix, amplitude in zip(centres, precision, amplitudes, strict=True):
        offset = origins - centre
        a = np.einsum("...i,ij,...j->...", directions, matrix, directions)
        b = np.einsum("...i,ij,...j->...", directions, matrix, offset)
        c = np.einsum("...i,ij,...j->...", offset, matrix, offset)
        output += amplitude * np.sqrt(2 * np.pi / a) * np.exp(-0.5 * (c - b * b / a))
    return output


@dataclass(frozen=True)
class Rays:
    """Object-frame detector centre, pixel steps and beam direction or source."""

    centre: np.ndarray
    u: np.ndarray
    v: np.ndarray
    beam: np.ndarray
    cone: bool

    def pixels(self, points: np.ndarray) -> np.ndarray:
        """Project object landmarks onto detector coordinates, including its centre."""
        points = np.broadcast_to(points, (len(self.centre), *points.shape))
        if self.cone:
            normal = np.cross(self.u, self.v)
            direction = points - self.beam[:, None]
            distance = np.einsum("ni,ni->n", normal, self.centre - self.beam)
            scale = distance[:, None] / np.einsum("ni,nki->nk", normal, direction)
            points = self.beam[:, None] + scale[..., None] * direction
        difference = points - self.centre[:, None]
        return np.stack(
            [
                np.einsum("nki,ni->nk", difference, step) / (step * step).sum(1)[:, None]
                for step in (self.u, self.v)
            ],
            axis=-1,
        )

    def render(self, size: int) -> np.ndarray:
        """Render pixel-centre line integrals without a voxel projector."""
        u, v = np.meshgrid(np.arange(size) - (size - 1) / 2, np.arange(size) - (size - 1) / 2)
        output = []
        for centre, us, vs, beam in zip(self.centre, self.u, self.v, self.beam, strict=True):
            origins = centre + u[..., None] * us + v[..., None] * vs
            if self.cone:
                directions = origins - beam
                directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
                origins = np.broadcast_to(beam, origins.shape)
            else:
                directions = np.broadcast_to(beam, origins.shape)
            output.append(line_integrals(origins, directions))
        return np.asarray(output, np.float32)


def physical_rays(
    poses: np.ndarray, detector: tj.Detector, *, roll: float = 0, cone: bool = False
) -> Rays:
    """Build rays directly from physical lab transforms, independently of fit columns."""
    radians = np.deg2rad(roll)
    u = detector.du * np.array([np.cos(radians), 0, np.sin(radians)])
    v = detector.dv * np.array([-np.sin(radians), 0, np.cos(radians)])
    centre = np.array([detector.center[0], 96 if cone else 0, detector.center[1]])
    rotation, shift = poses[:, :3, :3], poses[:, :3, 3]
    inverse = np.swapaxes(rotation, 1, 2)
    centres = np.einsum("nij,nj->ni", inverse, centre - shift)
    us, vs = (np.einsum("nij,j->ni", inverse, step) for step in (u, v))
    beam = (
        np.einsum("nij,nj->ni", inverse, np.array([0, -192, 0]) - shift)
        if cone
        else inverse[:, :, 1]
    )
    return Rays(centres, us, vs, beam, cone)


def scan_rays(scan: tj.Scan) -> Rays:
    """Read a fitted scan's physical geometry without invoking its projector."""
    if is_cone_beam(scan.geometry):
        frames = view_frames(scan.geometry).reshape(-1, 4, 3)
        return Rays(frames[:, 1], frames[:, 2], frames[:, 3], frames[:, 0], True)
    poses = np.asarray([scan.geometry.pose_for_view(i) for i in range(len(scan.angles))])
    return physical_rays(
        poses, scan.detector, roll=float(getattr(scan.geometry, "detector_roll_deg", 0))
    )


def make_case(
    case: str, *, geometry: str = "parallel", size: int = 48, views: int = 60, grid_size: int = 24
) -> tuple[tj.Scan, Rays, tuple[str, ...], dict[str, object]]:
    """Return observations, separate truth, and the parameter names matching the error."""
    grid = tj.Grid(grid_size, grid_size, grid_size, *([48 / grid_size] * 3))
    pitch = (72 if geometry == "cone" else 48) / size
    detector = tj.Detector(size, size, pitch, pitch)
    angles = np.linspace(0, 360, views, endpoint=False)
    tilt = 25.0 if geometry == "lamino" else 0.0
    if geometry == "cone":
        declared = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(192, 288))
    elif geometry == "lamino":
        declared = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=tilt)
    else:
        declared = tj.ParallelGeometry(grid, detector, angles)
    poses = np.tile(np.eye(4), (views, 1, 1))
    extra_tilt = 0.75 if case in {"tilt", "setup"} else 0.0
    poses[:, :3, :3] = Rotation.from_euler("x", -tilt - extra_tilt, degrees=True).as_matrix() @ (
        Rotation.from_euler("z", angles[:, None], degrees=True).as_matrix()
    )
    roll = 0.75 if case in {"roll", "setup"} else 0.0
    centre = (2 * pitch, -pitch if geometry == "cone" else 0) if case == "centre" else (0, 0)
    true_detector = tj.Detector(size, size, pitch, pitch, centre)
    if case in {"translations", "setup"}:
        t = np.linspace(0, 1, views)
        shifts = np.stack(
            [
                0.8 * np.sin(2 * np.pi * t) + 0.7 * (t > 0.41) - 0.4 * (t > 0.73),
                0.35 * np.sin(3 * np.pi * t),
                0.5 * np.cos(2 * np.pi * t),
            ],
            axis=-1,
        )
        shifts[:, 1] *= geometry == "cone"
        poses[:, :3, 3] = (shifts - shifts.mean(0)) * 48 / size
    if case == "rotations":
        rng = np.random.default_rng(618)
        perturbations = rng.normal(0, 0.37, (views, 3))
        perturbations -= perturbations.mean(0)
        poses[:, :3, :3] = (
            poses[:, :3, :3]
            @ Rotation.from_euler("YXZ", perturbations[:, [1, 0, 2]], degrees=True).as_matrix()
        )
    truth = physical_rays(poses, true_detector, roll=roll, cone=geometry == "cone")
    fit = {
        "clean": ("translations",),
        "centre": ("centre",),
        "translations": ("translations",),
        "tilt": ("tilt",),
        "roll": ("roll",),
        "rotations": ("translations", "rotations"),
        "setup": ("translations", "tilt", "roll"),
    }[case]
    metadata = {"centre_physical": centre, "tilt_correction_deg": extra_tilt, "roll_deg": roll}
    return tj.Scan(truth.render(size), declared, name=f"analytic {case}"), truth, fit, metadata


def score(estimated: Rays, truth: Rays) -> tuple[float, float, np.ndarray]:
    """Score 27 physical landmarks after one rigid gauge, retaining centre offsets."""
    points = np.stack(np.meshgrid(*[np.linspace(-14.4, 14.4, 3)] * 3, indexing="ij"), -1)
    points = points.reshape(-1, 3)
    target = truth.pixels(points)

    def residual(parameters: np.ndarray) -> np.ndarray:
        moved = points @ Rotation.from_rotvec(parameters[:3]).as_matrix().T + parameters[3:]
        return (estimated.pixels(moved) - target).ravel()

    gauge = least_squares(residual, np.zeros(6), xtol=1e-11, ftol=1e-11, gtol=1e-11).x
    errors = residual(gauge).reshape(-1, len(points), 2)
    per_view = np.sqrt((errors * errors).sum(-1).mean(1))
    return float(np.sqrt((per_view * per_view).mean())), float(per_view.max()), gauge


def run_case(
    case: str,
    *,
    geometry: str,
    size: int,
    views: int,
    grid_size: int,
    fit: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Run and score one public alignment; truth is never passed to the fit."""
    scan, truth, matching, metadata = make_case(
        case, geometry=geometry, size=size, views=views, grid_size=grid_size
    )
    before, _, _ = score(scan_rays(scan), truth)
    start = time.perf_counter()
    result = tj.align(scan, fit=matching if fit is None else fit, grid=scan.grid)
    volume = np.asarray(result.volume)
    elapsed = time.perf_counter() - start
    after, worst, gauge = score(scan_rays(result.scan), truth)
    grid = result.scan.grid
    z, y, x = np.indices(volume.shape)
    positions = np.stack([x, y, z], axis=-1) * [grid.vx, grid.vy, grid.vz] + grid_volume_origin(
        grid
    )
    positions = (positions - gauge[3:]) @ Rotation.from_rotvec(gauge[:3]).as_matrix()
    reference = density(positions)
    return {
        "case": case,
        "geometry": geometry,
        "detector": [size, size],
        "views": views,
        "grid": list(volume.shape),
        "fit": list(matching if fit is None else fit),
        "cold_seconds": elapsed,
        "nominal_pose_px": before,
        "fitted_pose_px": after,
        "worst_view_px": worst,
        "image_relative_error": float(
            np.linalg.norm(volume - reference) / np.linalg.norm(reference)
        ),
        "calibration": result.info.get("calibration", {}),
        "truth": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded default case and print its complete accuracy/timing record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--geometry", choices=("parallel", "lamino", "cone"), default="parallel")
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument(
        "--fit", nargs="+", choices=("centre", "translations", "tilt", "roll", "rotations")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    record = run_case(
        args.case,
        geometry=args.geometry,
        size=args.size,
        views=args.views,
        grid_size=args.grid_size,
        fit=None if args.fit is None else tuple(args.fit),
    )
    rendered = json.dumps(record, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
