from __future__ import annotations

"""Field-of-view helpers derived from detector size and center.

These utilities compute ROI grids that fit inside the detector's FOV for
parallel-beam setups (rotation about +z, rays along +y). The ROI is based on
simple, conservative bounds:

- X-Z plane coverage over all rotation angles (around +z) yields a disk of
  radius r_u = (nu/2-0.5)*du reduced by any detector x-center offset |cx|.
- Z coverage is bounded by r_v = (nv/2-0.5)*dv minus |cz|.
- For 3D, reconstruction is conceptually a stack of 2D slices at fixed z. Each
  slice’s in-plane FOV (x–y) is circular with radius r_u. This motivates ROI
  choices where (nx, ny) are chosen from r_u and nz from r_v.
  Optionally apply a cylindrical mask (circle in x–y, extruded along z).

The resulting grid keeps voxel spacings and ny (depth) unchanged, and matches
the original parity (odd/even) along x and z when possible to avoid off-by-half
shifts relative to the default centered origin convention.
"""

import math
from dataclasses import dataclass

from ..core.geometry.base import Grid, Detector


@dataclass(frozen=True)
class RoiInfo:
    """Derived physical FOV half-extents and recommended ROI grid dims."""
    r_u: float  # half-extent along x after angle-invariant reduction (cx)
    r_v: float  # half-extent along z after center offset (cz)
    nx_roi: int
    nz_roi: int


def _half_extent_from_n(n: int, v: float) -> float:
    """Half-extent of an n-sized axis with voxel size v centered at 0.

    Uses TomoJAX's default centered origin convention: ((n/2) - 0.5) * v.
    """
    return ((float(n) / 2.0) - 0.5) * float(v)


def _fit_dim_to_half_extent(H: float, v: float, orig_n: int) -> int:
    """Largest integer n with same parity as orig_n such that half-extent <= H.

    Half-extent for n voxels of size v is ((n/2)-0.5)*v. We solve
      ((n/2) - 0.5) * v <= H  =>  n <= 2*H/v + 1.
    Then, adjust downward by 1 if parity doesn't match the original.
    """
    if H <= 0.0:
        return 1
    n_max = int(math.floor(2.0 * (H / float(v)) + 1.0))
    if n_max < 1:
        n_max = 1
    # Match original parity to preserve centered indexing conventions where possible
    if (n_max % 2) != (orig_n % 2) and n_max > 1:
        n_max -= 1
    return max(1, n_max)


def compute_roi(grid: Grid, detector: Detector) -> RoiInfo:
    """Compute conservative FOV half-extents and grid dims from detector.

    - r_u = max(0, (nu/2-0.5)*du - |cx|) enforces angle-invariant coverage.
    - r_v = max(0, (nv/2-0.5)*dv - |cz|) enforces vertical coverage.
    - nx_roi, nz_roi are the largest dims (matching original parity) whose
      half-extents fit within r_u and r_v respectively, clipped by current grid.
    """
    # Detector pixel-center half extents (world units)
    u_half = ((float(detector.nu) / 2.0) - 0.5) * float(detector.du)
    v_half = ((float(detector.nv) / 2.0) - 0.5) * float(detector.dv)
    cx, cz = float(detector.det_center[0]), float(detector.det_center[1])

    r_u = max(0.0, u_half - abs(cx))
    r_v = max(0.0, v_half - abs(cz))

    # Fit dims; clip to not exceed the current grid dims
    nx_fit = _fit_dim_to_half_extent(r_u, float(grid.vx), int(grid.nx))
    nz_fit = _fit_dim_to_half_extent(r_v, float(grid.vz), int(grid.nz))
    nx_roi = min(int(grid.nx), nx_fit)
    nz_roi = min(int(grid.nz), nz_fit)

    return RoiInfo(r_u=r_u, r_v=r_v, nx_roi=nx_roi, nz_roi=nz_roi)


def grid_from_detector_fov(grid: Grid, detector: Detector) -> Grid:
    """Return a new Grid cropped to the detector FOV bounding box.

    Keeps ny/vy unchanged and centers the cropped grid at the origin using the
    same default centered origin convention.
    """
    info = compute_roi(grid, detector)
    # If detector already covers full current grid, just return the original grid
    if _half_extent_from_n(grid.nx, grid.vx) <= info.r_u + 1e-6 and _half_extent_from_n(grid.nz, grid.vz) <= info.r_v + 1e-6:
        return grid
    return Grid(
        nx=int(info.nx_roi),
        ny=int(grid.ny),
        nz=int(info.nz_roi),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        vol_origin=None,
        vol_center=None,
    )


def grid_from_detector_fov_cube(grid: Grid, detector: Detector) -> Grid:
    """Return a new cubic Grid cropped to the detector FOV (nx=ny=nz).

    Side length is the largest integer not exceeding both x- and z-FOV fits and
    the original ny. This yields a cube that fits inside the detector's FOV
    footprint and the existing volume extent in y.
    """
    info = compute_roi(grid, detector)
    side = min(int(info.nx_roi), int(info.nz_roi), int(grid.ny))
    if side < 1:
        side = 1
    # If the original grid already fits within the FOV and is cubic, return it
    if (
        grid.nx == grid.ny == grid.nz and
        _half_extent_from_n(grid.nx, grid.vx) <= info.r_u + 1e-6 and
        _half_extent_from_n(grid.nz, grid.vz) <= info.r_v + 1e-6
    ):
        return grid
    return Grid(
        nx=side,
        ny=side,
        nz=side,
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        vol_origin=None,
        vol_center=None,
    )


def cylindrical_mask_xy(grid: Grid, detector: Detector):
    """Boolean mask (nx, ny) for cylindrical FOV in the X–Y plane.

    Returns a 2D mask aligned to x and y axes where mask[x,y] is True if the
    voxel center lies within radius r_u in the X–Y plane. This is meant to be
    broadcast along z to mask a cylinder aligned with the z axis.
    """
    info = compute_roi(grid, detector)
    # Build coordinate grids (centered volume convention)
    import numpy as _np

    nx, ny = int(grid.nx), int(grid.ny)
    vx, vy = float(grid.vx), float(grid.vy)
    x = (_np.arange(nx, dtype=_np.float32) - (nx / 2.0 - 0.5)) * vx
    y = (_np.arange(ny, dtype=_np.float32) - (ny / 2.0 - 0.5)) * vy
    X, Y = _np.meshgrid(x, y, indexing="ij")
    mask = (X * X + Y * Y) <= (info.r_u * info.r_u + 1e-6)
    return mask


def grid_from_detector_fov_slices(grid: Grid, detector: Detector) -> Grid:
    """Return an ROI Grid with square x–y slices and z from detector height.

    - (nx, ny) chosen as the largest equal dims that fit within r_u using (vx, vy),
      clipped by current grid dims.
    - nz chosen from r_v using vz, clipped by current nz.
    This matches a common 3D parallel-beam reconstruction: per-z 2D CT slices.
    """
    info = compute_roi(grid, detector)
    nx_fit = _fit_dim_to_half_extent(info.r_u, float(grid.vx), int(grid.nx))
    ny_fit = _fit_dim_to_half_extent(info.r_u, float(grid.vy), int(grid.ny))
    nz_fit = _fit_dim_to_half_extent(info.r_v, float(grid.vz), int(grid.nz))
    side = min(nx_fit, ny_fit)
    if side < 1:
        side = 1
    return Grid(
        nx=min(int(grid.nx), side),
        ny=min(int(grid.ny), side),
        nz=min(int(grid.nz), int(nz_fit)),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        vol_origin=None,
        vol_center=None,
    )
