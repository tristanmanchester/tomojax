"""Geometry from the data alone: epipolar (Grangeat) consistency between cone-beam views.

Any plane through two views' sources has one plane integral, so both views'
data must give the same derivative of it along the plane's normal: the
integral, over the directions in the plane, of how the ray integral changes
as the ray tilts along the normal (Grangeat's relation). Both sides are
computed numerically from the detector data (a finite tilt and the measured
angles between rays), with no reconstruction; a wrong geometry makes them
disagree. :func:`orbit_heights` uses it to find how far each orbit of a
multi-orbit scan sits above the first.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.ndimage import map_coordinates

from tomojax.core.geometry.cone import ConeSegments, beam_of, view_frames

if TYPE_CHECKING:
    from tomojax.core.geometry import Geometry

LOG = logging.getLogger(__name__)
_PLANES = 90  # epipolar planes per pair of views
_SAMPLES = 800  # points along each plane's line on the detector
_EDGE = 0.02  # a line whose ends read above this fraction of its peak is truncated


@dataclass(frozen=True)
class _View:
    """One view: its source, detector centre and pixel steps (object frame) and image (v, u)."""

    source: np.ndarray
    centre: np.ndarray
    u: np.ndarray
    v: np.ndarray
    image: np.ndarray

    def raised(self, height: float) -> _View:
        lift = np.array([0.0, 0.0, height])
        return _View(self.source + lift, self.centre + lift, self.u, self.v, self.image)

    def _sample(self, points: np.ndarray) -> np.ndarray:
        rel = points - self.centre
        nv, nu = self.image.shape
        col = rel @ self.u / (self.u @ self.u) + nu / 2 - 0.5
        row = rel @ self.v / (self.v @ self.v) + nv / 2 - 0.5
        return map_coordinates(self.image, [row, col], order=1, mode="constant", cval=np.nan)

    def _hit(self, directions: np.ndarray) -> np.ndarray:
        normal = np.cross(self.u, self.v)
        reach = (normal @ (self.centre - self.source)) / (directions @ normal)
        return self.source + reach[:, None] * directions

    def plane_derivative(self, n: np.ndarray) -> float | None:
        """d/ds of the plane integral of the plane through the source with unit normal ``n``.

        None when the plane's line leaves the detector inside the object.
        """
        nv, nu = self.image.shape
        a_n, b_n, rhs = n @ self.u, n @ self.v, n @ (self.source - self.centre)
        along = np.array([-b_n, a_n]) / np.hypot(a_n, b_n)
        foot = rhs * np.array([a_n, b_n]) / (a_n * a_n + b_n * b_n)
        steps = np.linspace(-np.hypot(nu, nv) / 2, np.hypot(nu, nv) / 2, _SAMPLES)
        ab = foot[None] + steps[:, None] * along[None]
        inside = (np.abs(ab[:, 0]) <= nu / 2 - 1) & (np.abs(ab[:, 1]) <= nv / 2 - 1)
        if inside.sum() < 50:
            return None
        ab = ab[inside]
        points = self.centre + ab[:, :1] * self.u + ab[:, 1:] * self.v
        g = self._sample(points)
        peak = float(np.nanmax(np.abs(g))) if np.isfinite(g).any() else 0.0
        if not peak or abs(g[0]) > _EDGE * peak or abs(g[-1]) > _EDGE * peak:
            return None
        rays = points - self.source
        dist = np.linalg.norm(rays, axis=1)
        rays /= dist[:, None]
        tilt = np.linalg.norm(self.u) / dist  # about a pixel
        up = self._sample(self._hit(rays + tilt[:, None] * n))
        down = self._sample(self._hit(rays - tilt[:, None] * n))
        change = (up - down) / (2 * tilt)
        between = np.arccos(np.clip(np.sum(rays[1:] * rays[:-1], axis=1), -1.0, 1.0))
        mid = 0.5 * (change[1:] + change[:-1])
        if not np.isfinite(mid).all():
            return None
        return float(np.sum(mid * between))


def _pair_values(a: _View, b: _View) -> np.ndarray:
    """``(planes, 2)`` plane derivatives from both views, over planes through both sources."""
    base = b.source - a.source
    base /= np.linalg.norm(base)
    e1 = np.cross(base, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(base, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(base, e1)
    out = []
    for k in np.linspace(0.0, np.pi, _PLANES, endpoint=False):
        n = np.cos(k) * e1 + np.sin(k) * e2
        da, db = a.plane_derivative(n), b.plane_derivative(n)
        if da is not None and db is not None:
            out.append((da, db))
    return np.asarray(out).reshape(-1, 2)


def consistency_cost(pairs: list[tuple[_View, _View]]) -> tuple[float, int]:
    """The pairs' relative disagreement, and the number of planes it was measured on."""
    num = den = 0.0
    used = 0
    for a, b in pairs:
        values = _pair_values(a, b)
        num += float(np.sum((values[:, 0] - values[:, 1]) ** 2))
        den += float(np.sum(values**2))
        used += len(values)
    return num / max(den, 1e-30), used


@dataclass(frozen=True)
class OrbitHeights:
    """How far each orbit's source and detector sit above the first orbit's.

    ``heights`` are in the geometry's length unit, one per orbit (the first 0);
    ``contrast`` is each estimate's cost dip, ``1 - minimum / median`` of its
    scan (near 0: the data do not decide the height). ``planes`` counts the
    plane pairs compared.
    """

    heights: np.ndarray
    contrast: np.ndarray
    planes: int


def orbit_heights(
    geometry: Geometry,
    projections: np.ndarray,
    *,
    pairs: int = 32,
    reach: float | None = None,
) -> OrbitHeights | None:
    """Each orbit's height above the first, from the data's consistency alone.

    ``geometry`` is a multi-orbit :class:`ConeSegments` and ``projections`` its
    ``(views, rows, columns)`` data. Each orbit is compared with every orbit
    already placed, in view pairs a quarter and a sixth of a turn apart: its
    height is scanned over +-``reach`` (by default a sixteenth of the
    detector's height at the axis) in steps of two pixels at the axis on a
    quarter of the ``pairs``, then on all of them in quarter steps around the
    lowest, refined by a quadratic through the lowest costs. The estimate is
    good to about half a pixel at the axis: a starting point for alignment.
    None for a single orbit.
    """
    if not isinstance(geometry, ConeSegments) or len(geometry.segments) < 2:
        return None
    frames = view_frames(geometry)
    data = np.asarray(projections, np.float64)
    orbits, start = [], 0
    for segment in geometry.segments:
        angles = np.asarray(cast("Any", segment).angles, np.float64)  # every cone segment has them
        count = len(angles)
        views = [
            _View(frames[i, 0:3], frames[i, 3:6], frames[i, 6:9], frames[i, 9:12], data[i])
            for i in range(start, start + count)
        ]
        orbits.append((views, angles))
        start += count
    first = geometry.segments[0]
    beam = beam_of(first)
    assert beam is not None
    axis_pixel = first.detector.dv / beam.magnification
    step = 2.0 * axis_pixel
    reach = reach if reach is not None else first.detector.nv * axis_pixel / 16
    heights, contrast, planes = [0.0], [1.0], 0
    for k in range(1, len(orbits)):
        chosen = _pairs_with_earlier(orbits, k, heights, pairs)
        few = _pairs_with_earlier(orbits, k, heights, max(2, pairs // 4))
        coarse = np.arange(-reach, reach + step / 2, step)
        costs = np.array([consistency_cost(_raise(few, h))[0] for h in coarse])
        best = float(coarse[int(np.argmin(costs))])
        fine = best + np.arange(-2.0, 2.01, 0.25) * step
        scores = [consistency_cost(_raise(chosen, h)) for h in fine]
        fine_costs = np.array([c for c, _ in scores])
        planes += scores[0][1]
        keep = fine_costs <= 2 * fine_costs.min()
        a2, a1, _ = np.polyfit(fine[keep], fine_costs[keep], 2) if keep.sum() >= 3 else (0, 0, 0)
        height = float(-a1 / (2 * a2)) if a2 > 0 else float(fine[int(np.argmin(fine_costs))])
        height = float(np.clip(height, fine[0], fine[-1]))
        dip = 1.0 - float(costs.min() / np.median(costs))
        heights.append(height)
        contrast.append(dip)
        LOG.info("Orbit %d height %+.4f from data consistency (cost dip %.2f)", k + 1, height, dip)
    return OrbitHeights(np.asarray(heights), np.asarray(contrast), planes)


def _pairs_with_earlier(
    orbits: list[tuple[list[_View], np.ndarray]], k: int, heights: list[float], count: int
) -> list[tuple[_View, _View]]:
    """``count`` view pairs between orbit ``k`` and the orbits before it (raised as placed)."""
    views_k, angles_k = orbits[k]
    chosen = []
    for j in range(k):
        views_j, angles_j = orbits[j]
        picks = np.linspace(0, len(views_j), max(1, count // (2 * k)), endpoint=False).astype(int)
        for gap in (90.0, 60.0):
            for i in picks:
                want = (angles_j[i] + gap) % 360.0
                partner = int(np.argmin(np.abs((angles_k % 360.0 - want + 180.0) % 360.0 - 180.0)))
                chosen.append((views_j[i].raised(heights[j]), views_k[partner]))
    return chosen


def _raise(pairs: list[tuple[_View, _View]], height: float) -> list[tuple[_View, _View]]:
    return [(a, b.raised(height)) for a, b in pairs]
