"""Geometry base interfaces and grid/detector dataclasses.

These types are used across IO, projector, reconstruction, and alignment. Keep
them lightweight and JAX-friendly (no heavy runtime logic here).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
PoseMatrix = tuple[tuple[float, ...], ...]
RayFunction = Callable[[int, int], Vec3]
RayPair = tuple[RayFunction, RayFunction]


class GridDictRequired(TypedDict):
    """Required JSON-compatible grid metadata fields."""

    nx: int
    ny: int
    nz: int
    vx: float
    vy: float
    vz: float


class GridDict(GridDictRequired, total=False):
    """Optional JSON-compatible grid placement fields."""

    vol_origin: list[float]
    vol_center: list[float]


class DetectorDict(TypedDict):
    """JSON-compatible detector metadata fields."""

    nu: int
    nv: int
    du: float
    dv: float
    det_center: list[float]


def _coerce_fixed_tuple(name: str, value: Iterable[float], expected_len: int) -> tuple[float, ...]:
    try:
        items = tuple(value)
    except TypeError as e:
        raise TypeError(f"{name} must be an iterable of length {expected_len}") from e
    if len(items) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(items)}")
    try:
        return tuple(float(v) for v in items)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} elements must be real numbers") from e


@dataclass(frozen=True)
class Grid:
    """Regular voxel grid metadata.

    TomoJAX treats ``vol_origin`` as the physical location of voxel ``(0, 0, 0)``'s
    centre. When ``vol_origin`` is omitted, the default centred convention places
    voxel centres on coordinates ``(i - (n/2 - 0.5)) * v`` along each axis.
    ``vol_center`` can override that default centre while still deriving the
    voxel ``(0, 0, 0)`` location from the grid dimensions and voxel sizes. If
    both are provided, ``vol_origin`` is the authoritative placement.
    """

    nx: int
    ny: int
    nz: int
    vx: float
    vy: float
    vz: float
    vol_origin: Vec3 | None = None
    vol_center: Vec3 | None = None

    def __post_init__(self) -> None:
        if self.vol_origin is not None:
            object.__setattr__(
                self,
                "vol_origin",
                _coerce_fixed_tuple("vol_origin", self.vol_origin, 3),
            )
        if self.vol_center is not None:
            object.__setattr__(
                self,
                "vol_center",
                _coerce_fixed_tuple("vol_center", self.vol_center, 3),
            )

    def to_dict(self) -> GridDict:
        """Return JSON-compatible grid metadata."""
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


def grid_volume_origin(grid: Grid) -> Vec3:
    """Return the physical centre of voxel (0, 0, 0) for a grid."""
    if grid.vol_origin is not None:
        return grid.vol_origin

    if grid.vol_center is None:
        cx, cy, cz = (0.0, 0.0, 0.0)
    else:
        cx, cy, cz = tuple(float(v) for v in grid.vol_center)
    return (
        cx - ((grid.nx / 2.0) - 0.5) * float(grid.vx),
        cy - ((grid.ny / 2.0) - 0.5) * float(grid.vy),
        cz - ((grid.nz / 2.0) - 0.5) * float(grid.vz),
    )


@dataclass(frozen=True)
class Detector:
    """Regular detector plane metadata."""

    nu: int
    nv: int
    du: float
    dv: float
    det_center: Vec2 = field(default_factory=lambda: (0.0, 0.0))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "det_center",
            _coerce_fixed_tuple("det_center", self.det_center, 2),
        )

    def to_dict(self) -> DetectorDict:
        """Return JSON-compatible detector metadata."""
        return {
            "nu": int(self.nu),
            "nv": int(self.nv),
            "du": float(self.du),
            "dv": float(self.dv),
            "det_center": list(self.det_center),
        }


def _parallel_detector_rays(
    grid: Grid,
    detector: Detector,
) -> RayPair:
    """Return the standard detector-plane ray model used by parallel-beam setups."""
    nu, nv = int(detector.nu), int(detector.nv)
    du, dv = float(detector.du), float(detector.dv)
    cx, cz = float(detector.det_center[0]), float(detector.det_center[1])

    y0 = grid_volume_origin(grid)[1]

    def origin_fn(u: int, v: int) -> Vec3:
        x = (u - (nu / 2.0 - 0.5)) * du + cx
        z = (v - (nv / 2.0 - 0.5)) * dv + cz
        return float(x), float(y0), float(z)

    def dir_fn(_u: int, _v: int) -> Vec3:
        return (0.0, 1.0, 0.0)

    return origin_fn, dir_fn


class Geometry(Protocol):
    """Common pose interface for projector-compatible geometries.

    Projector contract: ``pose_for_view(i)`` returns a 4x4 world_from_object
    transform. The core projector combines that pose with the supplied
    ``Grid``/``Detector`` to construct the standard parallel-beam detector
    rays; it does not consume custom per-pixel ray callbacks.

    Concrete geometries may expose ``rays_for_view(i)`` as an optional
    metadata/inspection helper, but that helper is not part of the projector
    contract and custom rays are not supported by ``forward_project_view`` or
    ``backproject_view``.
    """

    def pose_for_view(self, i: int) -> PoseMatrix:
        """Returns a 4x4 homogeneous transform world_from_object (row-major)."""


class RayGeometry(Geometry, Protocol):
    """Optional ray-inspection interface implemented by built-in geometries."""

    def rays_for_view(self, i: int) -> RayPair:
        """Returns (origin_fn, dir_fn) that map detector pixel (u,v) to a ray.

        origin_fn(u,v) -> (x,y,z); dir_fn(u,v) -> (dx,dy,dz) normalized.
        """
