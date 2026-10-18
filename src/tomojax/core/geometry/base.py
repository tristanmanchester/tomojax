"""Geometry base interfaces and grid/detector dataclasses.

These types are used across IO, projector, reconstruction, and alignment. Keep
them lightweight and JAX-friendly (no heavy runtime logic here).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import operator
from typing import Any, Protocol, TypedDict

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
    """JSON-compatible detector metadata fields, as written to files.

    ``det_center`` is the file's key for :attr:`Detector.center`.
    """

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


def _plain_numbers(obj: object, counts: tuple[str, ...], pitches: tuple[str, ...]) -> None:
    """Store counts as Python ints and pitches as floats, whatever NumPy type they came as."""
    for name in counts:
        object.__setattr__(obj, name, operator.index(getattr(obj, name)))
    for name in pitches:
        object.__setattr__(obj, name, float(getattr(obj, name)))


@dataclass(frozen=True)
class Grid:
    """Regular voxel grid metadata.

    TomoJAX treats ``vol_origin`` as the physical location of voxel ``(0, 0, 0)``'s
    centre. When ``vol_origin`` is omitted, the default centred convention places
    voxel centres on coordinates ``(i - (n/2 - 0.5)) * v`` along each axis.
    ``vol_center`` can override that default centre while still deriving the
    voxel ``(0, 0, 0)`` location from the grid dimensions and voxel sizes. If
    both are provided, ``vol_origin`` is the authoritative placement.

    Parameters
    ----------
    nx, ny, nz : int
        Voxel counts along x, y and z.
    vx, vy, vz : float
        Voxel pitches in a consistent physical length unit.
    vol_origin : tuple of float or None, optional
        Physical centre of voxel (0, 0, 0); takes precedence over vol_center.
    vol_center : tuple of float or None, optional
        Physical grid centre when vol_origin is absent; defaults to (0, 0, 0).

    Raises
    ------
    TypeError
        If counts are not integers or placement coordinates are not real numbers.
    ValueError
        If a placement vector does not have three components.
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
        _plain_numbers(self, ("nx", "ny", "nz"), ("vx", "vy", "vz"))
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

    def resampled(self, shape: tuple[int, int, int]) -> Grid:
        """Return a grid with new voxel counts and the same physical extent.

        Voxel pitches change to preserve the outer cell edges and centre, not
        the positions of the first and last voxel centres. Explicit origins
        remain authoritative over ``vol_center``. No volume data is resampled.

        Parameters
        ----------
        shape : tuple of int
            Positive voxel counts ``(nx, ny, nz)`` for the returned grid.

        Returns
        -------
        Grid
            Grid covering the same physical box at the requested resolution.

        Raises
        ------
        ValueError
            If ``shape`` does not contain three positive counts.
        TypeError
            If a count is not an integer.

        Examples
        --------
        >>> Grid(100, 80, 60, 0.2, 0.3, 0.4).resampled((50, 40, 30)).vx
        0.4
        """
        nx, ny, nz = (operator.index(n) for n in shape)
        if min(nx, ny, nz) < 1:
            raise ValueError("shape must contain three positive voxel counts")
        if (nx, ny, nz) == (self.nx, self.ny, self.nz):
            return self
        vx, vy, vz = self.nx * self.vx / nx, self.ny * self.vy / ny, self.nz * self.vz / nz
        origin = self.vol_origin
        if origin is not None:
            origin = (
                origin[0] + (vx - self.vx) / 2,
                origin[1] + (vy - self.vy) / 2,
                origin[2] + (vz - self.vz) / 2,
            )
        return replace(self, nx=nx, ny=ny, nz=nz, vx=vx, vy=vy, vz=vz, vol_origin=origin)

    def to_dict(self) -> GridDict:
        """Return JSON-compatible grid metadata.

        Returns
        -------
        dict
            Grid counts, voxel pitches and any explicit placement, ready for JSON.
        """
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
    """Return the physical centre of voxel (0, 0, 0) for a grid.

    Parameters
    ----------
    grid : Grid
        Voxel grid with optional origin or centre.

    Returns
    -------
    tuple of float
        Physical (x, y, z) centre of voxel (0, 0, 0).
    """
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
    """Regular detector plane metadata.

    ``center`` is the detector centre's ``(u, v)`` offset in its plane, in the
    same physical units as ``du`` and ``dv``.

    Parameters
    ----------
    nu, nv : int
        Detector column and row counts.
    du, dv : float
        Pixel pitches along u and v, in the geometry's physical length unit.
    center : tuple of float, optional
        Detector-plane (u, v) centre offset in physical units; defaults to zero.

    Raises
    ------
    TypeError
        If counts are not integers or centre coordinates are not real numbers.
    ValueError
        If center does not have two components.
    """

    nu: int
    nv: int
    du: float
    dv: float
    center: Vec2 = field(default_factory=lambda: (0.0, 0.0))

    def __post_init__(self) -> None:
        _plain_numbers(self, ("nu", "nv"), ("du", "dv"))
        object.__setattr__(self, "center", _coerce_fixed_tuple("center", self.center, 2))

    def to_dict(self) -> DetectorDict:
        """Return JSON-compatible detector metadata.

        Returns
        -------
        dict
            Pixel counts, pitches and centre under the file key det_center.
        """
        return {
            "nu": int(self.nu),
            "nv": int(self.nv),
            "du": float(self.du),
            "dv": float(self.dv),
            "det_center": list(self.center),  # the file's key for ``center``
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Detector:
        """Return the detector that a :meth:`to_dict` mapping describes.

        Parameters
        ----------
        data : mapping
            Complete metadata written by to_dict, including det_center.

        Returns
        -------
        Detector
            Detector described by the metadata.

        Raises
        ------
        KeyError
            If pixel counts or pitches are missing.
        ValueError
            If the centre is missing or has the wrong number of coordinates.
        """
        if "det_center" not in data:
            raise ValueError(
                "Missing detector centre: pass detector=... or geometry=... to tomojax.load_frames"
            )
        return cls(
            nu=int(data["nu"]),
            nv=int(data["nv"]),
            du=float(data["du"]),
            dv=float(data["dv"]),
            center=data["det_center"],
        )


def _parallel_detector_rays(
    grid: Grid,
    detector: Detector,
) -> RayPair:
    """Return the standard detector-plane ray model used by parallel-beam setups."""
    nu, nv = int(detector.nu), int(detector.nv)
    du, dv = float(detector.du), float(detector.dv)
    cx, cz = float(detector.center[0]), float(detector.center[1])

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
        """Return a row-major world-from-object homogeneous transform.

        Parameters
        ----------
        i : int
            Zero-based view index.

        Returns
        -------
        tuple of tuple of float
            Row-major 4 x 4 homogeneous world-from-object transform.
        """


class ScanGeometry(Geometry, Protocol):
    """A geometry that names its reconstruction grid, detector and view angles.

    Every built-in geometry is one; the plain :class:`Geometry` contract needs
    only poses.
    """

    @property
    def grid(self) -> Grid:
        """The reconstruction grid.

        Returns
        -------
        Grid
            Reconstruction voxel grid.
        """
        ...

    @property
    def detector(self) -> Detector:
        """The detector.

        Returns
        -------
        Detector
            Detector dimensions, pixel sizes and physical centre.
        """
        ...

    @property
    def angles(self) -> Sequence[float]:
        """Each view's rotation angle, in degrees.

        Returns
        -------
        sequence of float
            One rotation angle per view, in degrees.
        """
        ...

    def rays_for_view(self, i: int) -> RayPair:
        """Return world-frame ray callbacks for inspection, not projection.

        Parameters
        ----------
        i : int
            Zero-based view index.

        Returns
        -------
        tuple of callable
            Origin and unit-direction callbacks taking detector pixel indices (u, v).
        """
        ...
