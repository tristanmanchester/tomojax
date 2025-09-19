"""Geometry base interfaces and grid/detector dataclasses.

These types are used across IO, projector, reconstruction, and alignment. Keep
them lightweight and JAX-friendly (no heavy runtime logic here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Tuple


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    nz: int
    vx: float
    vy: float
    vz: float
    vol_origin: Tuple[float, float, float] | None = None
    vol_center: Tuple[float, float, float] | None = None

    def to_dict(self) -> dict:
        d = {
            "nx": int(self.nx),
            "ny": int(self.ny),
            "nz": int(self.nz),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "vz": float(self.vz),
        }
        if self.vol_origin is not None:
            d["vol_origin"] = list(self.vol_origin)
        if self.vol_center is not None:
            d["vol_center"] = list(self.vol_center)
        return d


@dataclass(frozen=True)
class Detector:
    nu: int
    nv: int
    du: float
    dv: float
    det_center: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    def to_dict(self) -> dict:
        return {
            "nu": int(self.nu),
            "nv": int(self.nv),
            "du": float(self.du),
            "dv": float(self.dv),
            "det_center": list(self.det_center),
        }


class Geometry(Protocol):
    """Common interface for CT, laminography, and custom geometries.

    Implementations describe per-view pose and per-pixel rays.
    Contract: pose_for_view(i) returns a 4x4 world_from_object transform suitable
    for the projector; rays_for_view(i) must define world-frame rays.
    """

    def pose_for_view(self, i: int) -> Tuple[Tuple[float, ...], ...]:
        """Returns a 4x4 homogeneous transform world_from_object (row-major)."""

    def rays_for_view(self, i: int) -> Tuple[
        Callable[[int, int], Tuple[float, float, float]],
        Callable[[int, int], Tuple[float, float, float]],
    ]:
        """Returns (origin_fn, dir_fn) that map detector pixel (u,v) to a ray.

        origin_fn(u,v) -> (x,y,z); dir_fn(u,v) -> (dx,dy,dz) normalized.
        """
