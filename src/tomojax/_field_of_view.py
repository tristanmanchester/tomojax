"""The part of a scan's grid every view sees.

It is ``region=`` of :func:`tomojax.reconstruct` and :func:`tomojax.align`, and
``--region`` of their commands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np

from tomojax._parameters import RegionName
from tomojax.geometry import Grid, grid_volume_origin
from tomojax.geometry.api import is_cone_beam, view_frames

if TYPE_CHECKING:
    from collections.abc import Iterator

    import jax

    from tomojax._region import RegionEstimate
    from tomojax._scan import Scan

# The regions by name: the box every view sees ("auto", "bbox"), its largest cube or
# cylinder about the axis ("cube", "cyl", which also zeros outside it).
REGIONS = get_args(RegionName.__value__)
REGION_CHOICES = REGIONS
REGION_HELP = (
    "Crop the grid to the detector's field of view: auto, bbox, cube, "
    "or cyl (auto, zeroing outside the cylinder every view sees)"
)

# Samples per grid axis and views tested when finding the voxels every view sees.
_SAMPLES = 49
_VIEWS = 180


@dataclass(frozen=True)
class Cylinder:
    """A cylinder about the rotation axis (object x = y = 0): its radius."""

    radius: float


def region_grid(
    scan: Scan, region: str | RegionEstimate | None, *, shape: tuple[int, int, int] | None = None
) -> tuple[Grid, Cylinder | None]:
    """The grid ``region`` chooses of ``scan``'s, and the cylinder ``"cyl"`` zeros outside.

    The field of view is where the scan's own rays fall: voxel centers that
    every view projects onto the detector, with cone magnification, poses, a
    tilted axis and detector roll. None keeps the scan's grid, and a
    :class:`RegionEstimate` gives its own. ``"auto"`` falls back to the scan's
    grid, with a warning, when no box is seen. ``shape`` (``(nz, ny, nx)`` in
    NumPy order) resamples the region to new voxel counts, preserving its
    physical extent.
    """
    from tomojax._region import RegionEstimate

    if isinstance(region, RegionEstimate):
        _check_estimate_matches_scan(scan, region)
        chosen = region.grid
        return (chosen if shape is None else chosen.resampled(shape=shape)), None
    if region is not None and region not in REGIONS:
        names = ", ".join(REGIONS)
        raise ValueError(f"region is one of {names} or a RegionEstimate, not {region!r}")
    chosen, support = scan.grid, None
    try:
        first, last, cylinder = _seen_voxels(scan) if region is not None else (None, None, None)
    except ValueError as exc:
        if region != "auto":
            raise ValueError(f"region={region!r} could not be applied") from exc
        logging.getLogger(__name__).warning(
            "region='auto' could not be applied; using the scan's grid: %s", exc
        )
        first = last = None
    if first is not None and last is not None:
        if region in {"cube", "cyl"}:
            axes = (0, 1, 2) if region == "cube" else (0, 1)
            side = int(min(last[a] - first[a] + 1 for a in axes))
            for a in axes:
                # Center the shared side in the seen interval, keeping whole voxels.
                first[a] += (last[a] - first[a] + 1 - side) // 2
                last[a] = first[a] + side - 1
        chosen = _window(scan.grid, first, last)
        support = cylinder if region == "cyl" else None
    if shape is not None:
        chosen = chosen.resampled(shape=shape)
    return chosen, support


def _check_estimate_matches_scan(scan: Scan, estimate: RegionEstimate) -> None:
    """Reject a ``RegionEstimate`` built from a different scan or geometry."""
    from tomojax._scan import describe

    ref = estimate.scan
    same_shape = tuple(scan.projections.shape) == tuple(ref.projections.shape)
    same_detector = scan.detector == ref.detector
    same_angles = scan.angles.shape == ref.angles.shape and bool(
        np.allclose(scan.angles, ref.angles, atol=1e-5, rtol=0.0)
    )
    kind_a, meta_a, poses_a, _ = describe(scan.geometry)
    kind_b, meta_b, poses_b, _ = describe(ref.geometry)
    same_geom = kind_a == kind_b and meta_a == meta_b
    same_poses = (poses_a is None and poses_b is None) or (
        poses_a is not None
        and poses_b is not None
        and poses_a[1] == poses_b[1]
        and bool(np.allclose(poses_a[0], poses_b[0], atol=1e-5, rtol=0.0))
    )
    if not (same_shape and same_detector and same_angles and same_geom and same_poses):
        raise ValueError(
            "RegionEstimate does not match this scan; pass the scan it was estimated from "
            "or use estimate.scan"
        )


def cylinder_support(grid: Grid, cylinder: Cylinder) -> np.ndarray:
    """``(nz, ny, nx)`` boolean mask: True inside ``cylinder``, False outside."""
    origin = grid_volume_origin(grid)
    x = origin[0] + np.arange(grid.nx) * grid.vx
    y = origin[1] + np.arange(grid.ny) * grid.vy
    tolerance = 8 * np.finfo(np.float64).eps * cylinder.radius
    # hypot avoids squaring physical lengths, which can underflow or overflow.
    inside = np.hypot(x[None, :], y[:, None]) <= cylinder.radius + tolerance
    return np.broadcast_to(inside, (grid.nz, grid.ny, grid.nx)).copy()


def zero_outside_cylinder(
    volume: np.ndarray | jax.Array,
    grid: Grid,
    cylinder: Cylinder,
    *,
    in_place: bool = False,
) -> np.ndarray:
    """``volume`` with the voxels outside ``cylinder`` set to zero."""
    from tomojax.core.host_volume import host_array, release_planes

    outside = ~cylinder_support(replace(grid, nz=1), cylinder)[0]
    out = (
        volume
        if in_place and isinstance(volume, np.ndarray) and volume.flags.writeable
        else host_array(grid.shape, dtype=volume.dtype)
    )
    for z in range(grid.nz):
        if out is not volume:
            out[z] = volume[z]
        out[z, outside] = 0
        release_planes(out, z, z + 1)
    return out


def _window(grid: Grid, first: np.ndarray, last: np.ndarray) -> Grid:
    """Voxels ``first..last`` (inclusive, x/y/z) of ``grid``, at their own positions."""
    nx, ny, nz = (int(n) for n in last - first + 1)
    if (nx, ny, nz) == (grid.nx, grid.ny, grid.nz):
        return grid
    origin = np.asarray(grid_volume_origin(grid), np.float64)
    pitch = np.array([grid.vx, grid.vy, grid.vz])
    cx, cy, cz = (float(c) for c in origin + (first + last) / 2 * pitch)
    return Grid(nx, ny, nz, grid.vx, grid.vy, grid.vz, vol_center=(cx, cy, cz))


def _seen_voxels(scan: Scan) -> tuple[np.ndarray, np.ndarray, Cylinder]:
    """Voxels every view projects onto the detector: index bounds per axis and their cylinder.

    A lattice of at most ``_SAMPLES`` voxel centers per axis is tested against
    up to ``_VIEWS`` evenly spaced views. Bounds extend to the voxel before the
    next unseen sample, and the cylinder reaches the nearest unseen sample, so
    no measured voxel between samples is cropped or masked.
    """
    grid, detector = scan.grid, scan.detector
    counts = np.array([grid.nx, grid.ny, grid.nz])
    pitch = np.array([grid.vx, grid.vy, grid.vz])
    origin = np.asarray(grid_volume_origin(grid), np.float64)
    indices = [
        np.unique(np.linspace(0, n - 1, min(n, _SAMPLES)).round().astype(int)) for n in counts
    ]
    mesh = np.meshgrid(*(origin[a] + indices[a] * pitch[a] for a in range(3)), indexing="ij")
    points = np.stack(mesh, axis=-1).reshape(-1, 3)
    n_views = len(cast("Any", scan.geometry).angles)
    views = np.unique(np.linspace(0, n_views - 1, min(n_views, _VIEWS)).round().astype(int))
    seen = np.ones(len(points), bool)
    slack = 1e-6  # pixels: a voxel center on the outermost pixel center is seen
    for column, row in _pixel_coordinates(scan, views, points):
        seen &= (column >= -slack) & (column <= detector.nu - 1 + slack)
        seen &= (row >= -slack) & (row <= detector.nv - 1 + slack)
    if not seen.any():
        raise ValueError("no voxel of the grid projects onto the detector in every view")
    seen = seen.reshape(*(len(i) for i in indices))
    first, last = np.empty(3, int), np.empty(3, int)
    for a in range(3):
        hits = np.flatnonzero(seen.any(axis=tuple(b for b in range(3) if b != a)))
        lo, hi = hits[0], hits[-1]
        first[a] = 0 if lo == 0 else indices[a][lo - 1] + 1
        last[a] = counts[a] - 1 if hi == len(indices[a]) - 1 else indices[a][hi + 1] - 1
    # About the rotation axis (x = y = 0), the widest layer's disc: cone-beam
    # caps narrow lower layers, and the disc must mask none of their voxels.
    # It stops just short of the nearest unseen sample, exact at full sampling.
    distance = np.hypot(points[:, 0], points[:, 1]).reshape(seen.shape)
    nearest_unseen = np.where(seen, np.inf, distance).min(axis=(0, 1))
    radius = nearest_unseen[seen.any(axis=(0, 1))].max() * (1 - 1e-9)
    radius = distance.max() if np.isinf(radius) else radius
    return first, last, Cylinder(float(radius))


def _pixel_coordinates(
    scan: Scan, views: np.ndarray, points: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield each view's detector (column, row) pixel coordinates of object ``points``."""
    geometry, detector = scan.geometry, scan.detector
    if is_cone_beam(geometry):
        frames = view_frames(geometry)[views]
        for source, center, u, v in frames.reshape(len(views), 4, 3):
            normal = np.cross(u, v)
            ray = points - source
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (normal @ (center - source)) / (ray @ normal)
            hit = source + t[:, None] * ray - center
            # Solve hit = a u + b v in the detector plane.
            basis = np.array([[u @ u, u @ v], [u @ v, v @ v]])
            a, b = np.linalg.solve(basis, np.stack([hit @ u, hit @ v]))
            behind = ~(t > 0)
            yield (
                np.where(behind, -1.0, a + detector.nu / 2 - 0.5),
                np.where(behind, -1.0, b + detector.nv / 2 - 0.5),
            )
        return
    roll = np.deg2rad(float(getattr(geometry, "detector_roll", 0.0) or 0.0))
    c, s = np.cos(roll), np.sin(roll)
    for view in views:
        pose = np.asarray(cast("Any", geometry).pose_matrix(int(view)), np.float64)
        lab = points @ pose[:3, :3].T + pose[:3, 3]
        x, z = lab[:, 0] - detector.center[0], lab[:, 2] - detector.center[1]
        # The rolled lattice sits at R(roll) @ (x', z') from the detector center.
        yield (
            (c * x + s * z) / detector.du + (detector.nu - 1) / 2,
            (-s * x + c * z) / detector.dv + (detector.nv - 1) / 2,
        )


__all__ = [
    "REGIONS",
    "REGION_CHOICES",
    "REGION_HELP",
    "Cylinder",
    "cylinder_support",
    "region_grid",
    "zero_outside_cylinder",
]
