from __future__ import annotations

"""Field-of-view helpers derived from detector size and center.

These utilities compute ROI grids that fit inside the detector's FOV for
parallel-beam setups (rotation about +z, rays along +y). The ROI is based on
simple, conservative bounds:

- X-Y plane coverage over all rotation angles (around +z) yields a disk of
  radius r_u = (nu/2-0.5)*du reduced by any detector x-center offset |cx|.
- Z coverage is bounded by r_v = (nv/2-0.5)*dv minus |cz|.
- For 3D, reconstruction is conceptually a stack of 2D slices at fixed z. Each
  slice’s in-plane FOV (x–y) is circular with radius r_u. This motivates ROI
  choices where (nx, ny) are chosen from r_u and nz from r_v.
  Optionally apply a cylindrical mask (circle in x–y, extruded along z).

The resulting grid keeps voxel spacings, crops both in-plane axes against r_u,
respects detector height along z, and tries to preserve parity (odd/even)
whenever a shared square/cubic side still allows it.
"""

import math
from dataclasses import dataclass

from ..core.geometry.base import Grid, Detector


@dataclass(frozen=True)
class RoiInfo:
    """Derived physical FOV half-extents and recommended ROI grid dims."""
    r_u: float  # half-extent in the rotation plane after angle-invariant reduction (cx)
    r_v: float  # half-extent along z after center offset (cz)
    nx_roi: int
    ny_roi: int
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


def _match_parity_leq(n: int, parity: int) -> int:
    """Largest integer <= n with the requested parity, clamped to at least 1."""
    if n < 1:
        return 1
    if (n % 2) != (parity % 2) and n > 1:
        n -= 1
    return max(1, n)


def _choose_shared_side(side_max: int, *orig_dims: int) -> int:
    """Pick the largest shared side <= side_max while preserving parity when possible.

    For square/cubic ROI outputs, a single `side` is reused across multiple axes.
    Taking a raw `min()` across already parity-matched fitted dimensions can reintroduce
    a parity flip if the limiting axis has different parity. We instead choose the
    largest side not exceeding `side_max` whose parity preserves the majority of the
    original dimensions, breaking ties in favour of the x-axis parity (the first dim).
    """
    if side_max < 1:
        return 1
    if not orig_dims:
        return max(1, side_max)

    parities = [int(n) % 2 for n in orig_dims]
    even_count = sum(p == 0 for p in parities)
    odd_count = len(parities) - even_count
    if odd_count > even_count:
        target_parity = 1
    elif even_count > odd_count:
        target_parity = 0
    else:
        target_parity = parities[0]
    return _match_parity_leq(int(side_max), target_parity)


def compute_roi(
    grid: Grid,
    detector: Detector,
    *,
    crop_y_to_u: bool = True,
) -> RoiInfo:
    """Compute conservative FOV half-extents and grid dims from detector.

    - r_u = max(0, (nu/2-0.5)*du - |cx|) enforces angle-invariant coverage.
    - r_v = max(0, (nv/2-0.5)*dv - |cz|) enforces vertical coverage.
    - nx_roi is the largest dim (matching original parity) whose half-extent fits
      within the in-plane radius r_u; nz_roi similarly fits r_v.
    - ny_roi is also fit to r_u when `crop_y_to_u=True`, which matches the
      parallel-beam x-y disk interpretation. Laminography callers can set
      `crop_y_to_u=False` to keep the full source y extent.
    - All dimensions are clipped by the current grid.
    """
    # Detector pixel-center half extents (world units)
    u_half = ((float(detector.nu) / 2.0) - 0.5) * float(detector.du)
    v_half = ((float(detector.nv) / 2.0) - 0.5) * float(detector.dv)
    cx, cz = float(detector.det_center[0]), float(detector.det_center[1])

    r_u = max(0.0, u_half - abs(cx))
    r_v = max(0.0, v_half - abs(cz))

    # Fit dims; clip to not exceed the current grid dims
    nx_fit = _fit_dim_to_half_extent(r_u, float(grid.vx), int(grid.nx))
    if crop_y_to_u:
        ny_fit = _fit_dim_to_half_extent(r_u, float(grid.vy), int(grid.ny))
    else:
        ny_fit = int(grid.ny)
    nz_fit = _fit_dim_to_half_extent(r_v, float(grid.vz), int(grid.nz))
    nx_roi = min(int(grid.nx), nx_fit)
    ny_roi = min(int(grid.ny), ny_fit)
    nz_roi = min(int(grid.nz), nz_fit)

    return RoiInfo(r_u=r_u, r_v=r_v, nx_roi=nx_roi, ny_roi=ny_roi, nz_roi=nz_roi)


def grid_from_detector_fov(
    grid: Grid,
    detector: Detector,
    *,
    crop_y_to_u: bool = True,
) -> Grid:
    """Return a new Grid cropped to the detector FOV bounding box.

    All three axes are cropped conservatively: x and y by the in-plane detector
    radius `r_u`, and z by the detector height `r_v`. The cropped grid remains
    centered at the origin using the same default centered-origin convention.
    """
    info = compute_roi(grid, detector, crop_y_to_u=crop_y_to_u)
    # If detector already covers the full current grid, just return the original grid.
    if (
        _half_extent_from_n(grid.nx, grid.vx) <= info.r_u + 1e-6
        and (
            not crop_y_to_u
            or _half_extent_from_n(grid.ny, grid.vy) <= info.r_u + 1e-6
        )
        and _half_extent_from_n(grid.nz, grid.vz) <= info.r_v + 1e-6
    ):
        return grid
    return Grid(
        nx=int(info.nx_roi),
        ny=int(info.ny_roi),
        nz=int(info.nz_roi),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        vol_origin=None,
        vol_center=None,
    )


def grid_from_detector_fov_cube(
    grid: Grid,
    detector: Detector,
    *,
    crop_y_to_u: bool = True,
) -> Grid:
    """Return a new cubic Grid cropped to the detector FOV (nx=ny=nz).

    The shared side is limited by x/y in-plane coverage (`r_u`) and z coverage
    (`r_v`). When multiple fitted dimensions with different parities compete, the
    final side is adjusted downward by at most one voxel so the cubic output keeps
    a consistent, parity-preserving centered-origin convention where possible.
    """
    info = compute_roi(grid, detector, crop_y_to_u=crop_y_to_u)
    side_max = min(int(info.nx_roi), int(info.ny_roi), int(info.nz_roi))
    side = _choose_shared_side(side_max, int(grid.nx), int(grid.ny), int(grid.nz))
    # If the original grid already fits within the FOV and is cubic, return it
    if (
        grid.nx == grid.ny == grid.nz and
        _half_extent_from_n(grid.nx, grid.vx) <= info.r_u + 1e-6 and
        (
            not crop_y_to_u
            or _half_extent_from_n(grid.ny, grid.vy) <= info.r_u + 1e-6
        ) and
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


def grid_from_detector_fov_slices(
    grid: Grid,
    detector: Detector,
    *,
    crop_y_to_u: bool = True,
) -> Grid:
    """Return an ROI Grid with square x–y slices and z from detector height.

    - (nx, ny) chosen as the largest equal dims that fit within r_u using (vx, vy).
    - nz chosen from r_v using vz, clipped by current nz.
    This matches a common 3D parallel-beam reconstruction: per-z 2D CT slices.
    """
    info = compute_roi(grid, detector, crop_y_to_u=crop_y_to_u)
    side_max = min(int(info.nx_roi), int(info.ny_roi))
    side = _choose_shared_side(side_max, int(grid.nx), int(grid.ny))
    return Grid(
        nx=min(int(grid.nx), side),
        ny=min(int(grid.ny), side),
        nz=min(int(grid.nz), int(info.nz_roi)),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        vol_origin=None,
        vol_center=None,
    )
