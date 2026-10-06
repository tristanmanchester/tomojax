"""Calibrate a cone-beam scan's rotation axis from the projections.

Lab CT scanners report the source and detector distances but rarely the exact
position of the rotation axis. An axis offset by ``d`` from the central ray
doubles and blurs every slice, and a detector rolled about the beam makes the
offset change linearly with height. :func:`calibrate_cone_axis` reconstructs
thin FDK slabs at a few heights for trial offsets, keeps the offset that makes
each slab sharpest, and fits the offsets against height for the axis offset
(``ConeBeam.axis_offset``) and the detector roll (``ConeBeam.detector_roll_deg``).

The filtered projections do not depend on the axis, so each slab is filtered
once per resolution level and only backprojected per trial. The search runs
coarse to fine on binned data, so its cost is a few full-resolution slab
backprojections per height.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin
from tomojax.core.geometry.cone import ConeBeam, ConeGeometry, beam_of
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import validate_projection_stack
from tomojax.recon.fdk import (
    FDKConfig,
    _backproject_filtered,
    _detector_window,
    _filter_views,
    _prepare,
    _Prepared,
    _slab_rows,
    _windowable,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConeAxisConfig:
    """Options for :func:`calibrate_cone_axis`.

    ``search_range`` bounds the axis offset searched on either side of the
    geometry's current one, in physical units (default a quarter of the volume
    width). ``estimate_roll`` measures the offset at three heights and fits the
    detector roll; otherwise only the central slab is used. ``slices`` is the
    thickness of each slab in full-resolution voxels (at most an eighth of the
    volume), ``max_roll_deg`` the
    largest roll the slabs leave detector rows for, and ``fdk`` the filter
    (Hann by default, so noise does not swamp the sharpness) and backend.
    """

    search_range: float | None = None
    estimate_roll: bool = True
    slices: int = 32
    max_roll_deg: float = 3.0
    fdk: FDKConfig = field(
        default_factory=lambda: FDKConfig(filter_name="hann", views_per_batch=1024)
    )


@dataclass(frozen=True)
class ConeAxisCalibration:
    """Calibrated axis offset and detector roll, with the per-slab estimates.

    ``axis_offset`` and ``detector_roll_deg`` are absolute ``ConeBeam`` values.
    ``slab_offsets[i]`` is the sharpest axis offset of the slab centred at
    height ``heights[i]`` (physical z), with the calibrated roll applied.
    """

    axis_offset: float
    detector_roll_deg: float
    heights: tuple[float, ...]
    slab_offsets: tuple[float, ...]

    def apply(self, geometry: ConeGeometry) -> ConeGeometry:
        """Return ``geometry`` with the calibrated axis offset and detector roll."""
        beam = replace(
            geometry.beam,
            axis_offset=float(self.axis_offset),
            detector_roll_deg=float(self.detector_roll_deg),
        )
        return replace(geometry, beam=beam)


@dataclass
class _Slab:
    """One height's binned slab grid, detector rows and filtered projections.

    ``window(beam)`` is the binned band of detector rows on ``beam``'s detector,
    which moves with the trial roll.
    """

    grid: Grid
    window: Callable[[ConeBeam], Detector]
    filtered: jax.Array
    mask: jax.Array
    prep: _Prepared


def _binned_rows(
    projections: np.ndarray | jax.Array, r0: int, rows: int, cols: int, f: int, batch: int
) -> np.ndarray:
    """Read detector rows ``[r0, r0 + f * rows)`` of every view, binned ``f x f``."""
    views = int(projections.shape[0])
    out = np.empty((views, rows, cols), np.float32)
    for start in range(0, views, batch):
        stop = min(start + batch, views)
        part = np.asarray(projections[start:stop, r0 : r0 + f * rows, : f * cols], np.float32)
        out[start:stop] = part.reshape(stop - start, rows, f, cols, f).mean(axis=(2, 4))
    return out


def _fov_mask(geometry: ConeGeometry, grid: Grid, detector: Detector) -> np.ndarray:
    """``(nx, ny)`` mask of the voxels inside the cylinder every view sees."""
    beam = geometry.beam
    # A full turn sees the circle reached by the detector edge farther from the axis.
    axis = float(beam.axis_offset) * beam.magnification - float(detector.det_center[0])
    half = detector.nu * detector.du / 2 + abs(axis)
    radius = (
        0.95
        * float(beam.source_to_axis)
        * math.sin(math.atan(half / float(beam.source_to_detector)))
    )
    origin = np.asarray(grid_volume_origin(grid))
    x = origin[0] + np.arange(grid.nx) * grid.vx - float(beam.axis_offset)
    y = origin[1] + np.arange(grid.ny) * grid.vy
    return (x[:, None] ** 2 + y[None, :] ** 2) <= radius**2


@jax.jit
def _sharpness(volume: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean squared in-plane gradient of a slab inside ``mask``."""
    gx = volume[1:, :-1, :] - volume[:-1, :-1, :]
    gy = volume[:-1, 1:, :] - volume[:-1, :-1, :]
    inside = (mask[1:, :-1] & mask[:-1, 1:] & mask[:-1, :-1])[:, :, None]
    energy = jnp.where(inside, gx * gx + gy * gy, 0.0)
    return jnp.sum(energy) / jnp.maximum(jnp.sum(inside) * volume.shape[2], 1)


def _thickness(grid: Grid, cfg: ConeAxisConfig) -> int:
    """Slab thickness in full-resolution voxels: ``cfg.slices``, at most an eighth of the volume."""
    return max(4, min(int(cfg.slices), grid.nz // 8))


def _prepare_slab(
    geometry: ConeGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray | jax.Array,
    height: float,
    f: int,
    cfg: ConeAxisConfig,
) -> _Slab:
    """Bin, filter and hold the detector rows of the slab centred at ``height``."""
    slices = max(1, _thickness(grid, cfg) // f)
    origin = np.asarray(grid_volume_origin(grid))
    nx, ny = grid.nx // f, grid.ny // f
    spacing = (grid.vx * f, grid.vy * f, grid.vz * f)
    slab = Grid(
        nx=nx,
        ny=ny,
        nz=slices,
        vx=spacing[0],
        vy=spacing[1],
        vz=spacing[2],
        vol_origin=(
            float(origin[0] + (f - 1) / 2 * grid.vx),
            float(origin[1] + (f - 1) / 2 * grid.vy),
            float(height - (slices - 1) / 2 * spacing[2]),
        ),
    )
    poses = np.asarray(stack_view_poses(geometry, int(projections.shape[0])), np.float64)
    r0, r1 = _slab_rows(geometry, slab, detector, poses, 0, slices)
    # Rows for the largest roll searched, across half the detector width.
    lift = detector.nu * detector.du / 2 * math.tan(math.radians(cfg.max_roll_deg))
    spare = math.ceil(lift / detector.dv) + f
    r0, r1 = max(0, r0 - spare), min(detector.nv, r1 + spare)
    r0 -= r0 % f
    rows, cols = (r1 - r0) // f, detector.nu // f
    if rows < 2 or cols < 2:
        raise ValueError("calibrate_cone_axis: the slab projects onto too few detector rows")

    def window(beam: ConeBeam) -> Detector:
        return _detector_window(beam, detector, r0, r0 + f * rows, f * cols, f)

    binned = window(geometry.beam)
    data = _binned_rows(projections, r0, rows, cols, f, max(1, cfg.fdk.views_per_batch))
    full = _detector_window(geometry.beam, detector, 0, f * (detector.nv // f), f * cols, f)
    prep = _prepare(geometry, binned, data.shape[0], cfg.fdk, full)
    filtered = _filter_views(prep, data, data.shape[0])
    mask = jnp.asarray(_fov_mask(geometry, slab, detector))
    return _Slab(grid=slab, window=window, filtered=filtered, mask=mask, prep=prep)


def _score(geometry: ConeGeometry, slab: _Slab, offset: float, roll: float) -> float:
    beam = replace(geometry.beam, axis_offset=float(offset), detector_roll_deg=float(roll))
    trial = replace(geometry, beam=beam)
    volume = _backproject_filtered(trial, slab.grid, slab.window(beam), slab.filtered, slab.prep)
    return float(_sharpness(volume, slab.mask))


def _search(
    geometry: ConeGeometry, slab: _Slab, candidates: np.ndarray, roll: float, width: int
) -> float:
    """Sharpest offset among ``candidates``.

    The vertex of a least-squares parabola through the ``2 * width + 1`` scores
    around the best one (noise makes full-resolution scores ripple), kept within
    those candidates.
    """
    scores = np.array([_score(geometry, slab, c, roll) for c in candidates])
    i = int(np.argmax(scores))
    lo, hi = max(0, i - width), min(len(candidates), i + width + 1)
    if hi - lo >= 3:
        a, b, _ = np.polyfit(candidates[lo:hi], scores[lo:hi], 2)
        if a < 0:
            return float(np.clip(-b / (2 * a), candidates[lo], candidates[hi - 1]))
    return float(candidates[i])


class _Slabs:
    """Filtered slabs by (height, binning), prepared on first use."""

    def __init__(
        self,
        geometry: ConeGeometry,
        grid: Grid,
        detector: Detector,
        projections: np.ndarray | jax.Array,
        cfg: ConeAxisConfig,
    ) -> None:
        self.args = (geometry, grid, detector, projections)
        self.cfg = cfg
        self.cache: dict[tuple[float, int], _Slab] = {}

    def get(self, height: float, f: int) -> _Slab:
        key = (float(height), int(f))
        if key not in self.cache:
            self.cache[key] = _prepare_slab(*self.args, height, f, self.cfg)
        return self.cache[key]


def _slab_offset(
    geometry: ConeGeometry,
    slabs: _Slabs,
    height: float,
    start: float,
    radius: float,
    roll: float,
    factors: Sequence[int],
    voxel: float,
) -> float:
    """Sharpest axis offset of one slab, searched coarse to fine around ``start``."""
    best = start
    for level, f in enumerate(factors):
        slab = slabs.get(height, f)
        step = voxel * f
        reach = radius if level == 0 else 3 * step
        count = max(1, round(reach / step))
        best = _search(geometry, slab, best + step * np.arange(-count, count + 1), roll, 2)
        if f == 1:
            # Final quarter-voxel fit over +-1.5 voxels of the full-resolution estimate.
            best = _search(geometry, slab, best + 0.25 * step * np.arange(-6, 7), roll, 6)
    return best


def calibrate_cone_axis(
    geometry: ConeGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray | jax.Array,
    *,
    config: ConeAxisConfig | None = None,
) -> ConeAxisCalibration:
    """Estimate a cone-beam scan's axis offset and detector roll from its projections.

    ``projections`` are ``(views, nv, nu)`` line integrals (NumPy, memmap or
    JAX). The scan should cover at least 180 degrees plus the fan angle; the
    sample should have some in-plane structure at the slab heights. Use
    :meth:`ConeAxisCalibration.apply` to update the geometry.
    """
    cfg = ConeAxisConfig() if config is None else config
    if not isinstance(geometry, ConeGeometry) or beam_of(geometry) is None:
        raise ValueError("calibrate_cone_axis needs a ConeGeometry")
    if not _windowable(geometry.beam):
        raise ValueError("calibrate_cone_axis does not support a pitched or yawed detector")
    validate_projection_stack(
        projections, detector, geometry=geometry, context="calibrate_cone_axis projections"
    )
    beam = geometry.beam
    radius = 0.25 * grid.nx * grid.vx if cfg.search_range is None else float(cfg.search_range)
    radius = max(radius, 2 * grid.vx)
    factors = [f for f in (4, 2) if min(grid.nx, grid.ny) // f >= 48] + [1]
    # Slab heights: the centre and, for the roll, 60% of the way to the top and
    # bottom of the part of the volume that every view sees in full.
    origin = np.asarray(grid_volume_origin(grid))
    z_lo, z_hi = float(origin[2]), float(origin[2] + (grid.nz - 1) * grid.vz)
    coverage = 0.5 * detector.nv * detector.dv / beam.magnification
    centre = float(np.clip(0.0, z_lo, z_hi))
    half = (
        min(coverage, 0.5 * (z_hi - z_lo)) - (_thickness(grid, cfg) / 2 + 2 * factors[0]) * grid.vz
    )
    if cfg.estimate_roll and half > 4 * grid.vz:
        heights = (centre - 0.6 * half, centre, centre + 0.6 * half)
    else:
        heights = (centre,)
    slabs = _Slabs(geometry, grid, detector, projections, cfg)
    offset0, roll = float(beam.axis_offset), float(beam.detector_roll_deg)
    offsets = [
        _slab_offset(geometry, slabs, z, offset0, radius, roll, factors, grid.vx) for z in heights
    ]
    if max(abs(o - offset0) for o in offsets) > radius - 2 * grid.vx * factors[0]:
        LOG.warning(
            "calibrate_cone_axis: the sharpest axis offset is at the edge of the search "
            "range (+-%.4g); widen ConeAxisConfig.search_range",
            radius,
        )
    if len(heights) == 1:
        return ConeAxisCalibration(offsets[0], roll, heights, tuple(offsets))
    intercept = float(np.mean(offsets))
    for _ in range(2):
        # Offsets move as -tan(roll) * z: apply the fitted roll and search again,
        # around the common offset the slabs should then share.
        slope, intercept = np.polyfit(np.asarray(heights), np.asarray(offsets), 1)
        roll += math.degrees(-math.atan(float(slope)))
        offsets = [
            _search(
                geometry, slabs.get(z, 1), intercept + 0.25 * grid.vx * np.arange(-6, 7), roll, 6
            )
            for z in heights
        ]
    slope, intercept = np.polyfit(np.asarray(heights), np.asarray(offsets), 1)
    roll += math.degrees(-math.atan(float(slope)))
    return ConeAxisCalibration(
        axis_offset=float(intercept),
        detector_roll_deg=roll,
        heights=tuple(float(z) for z in heights),
        slab_offsets=tuple(float(o) for o in offsets),
    )


__all__ = ["ConeAxisCalibration", "ConeAxisConfig", "calibrate_cone_axis"]
