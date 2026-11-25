"""Geometry base interfaces and grid/detector dataclasses.

These types are used across IO, projector, reconstruction, and alignment. Keep
them lightweight and JAX-friendly (no heavy runtime logic here).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import functools
import math
import operator
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

import numpy as np

from tomojax.errors import InvalidParameter

if TYPE_CHECKING:
    import jax.numpy as jnp

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
PoseMatrix = tuple[tuple[float, ...], ...]
RayFunction = Callable[[int, int], Vec3]
RayPair = tuple[RayFunction, RayFunction]
# View angles in degrees, as the caller gave them: a sequence or a NumPy array.
type Angles = Sequence[float] | np.ndarray


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
        values = tuple(float(v) for v in items)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} elements must be real numbers") from e
    if not all(math.isfinite(v) for v in values):
        raise InvalidParameter(name, f"{name} must be finite")
    return values


def _plain_numbers(obj: object, counts: tuple[str, ...], pitches: tuple[str, ...]) -> None:
    """Store counts as Python ints and pitches as floats, whatever NumPy type they came as."""
    for name in counts:
        count = operator.index(getattr(obj, name))
        if count < 1:
            raise InvalidParameter(name, f"{name} must be a positive integer")
        object.__setattr__(obj, name, count)
    for name in pitches:
        pitch = float(getattr(obj, name))
        if not math.isfinite(pitch) or pitch <= 0:
            raise InvalidParameter(name, f"{name} must be finite and positive")
        object.__setattr__(obj, name, pitch)


type _State = dict[str, object] | tuple[dict[str, object] | None, dict[str, object] | None]


def _restore(obj: object, state: _State) -> None:
    """Set unpickled or copied fields as stored, keeping the class and its own fields."""
    if isinstance(state, tuple):  # a subclass with __slots__: (instance dict, slot values)
        fields, slots = state
        state = {**(fields or {}), **(slots or {})}
    for name, value in state.items():
        object.__setattr__(obj, name, value)


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
        If a count or pitch is not positive, or a placement vector is not three
        finite numbers.
    """

    nx: int = field(metadata={"unit": "voxel"})
    ny: int = field(metadata={"unit": "voxel"})
    nz: int = field(metadata={"unit": "voxel"})
    vx: float = field(metadata={"unit": "geometry_length"})
    vy: float = field(metadata={"unit": "geometry_length"})
    vz: float = field(metadata={"unit": "geometry_length"})
    vol_origin: Vec3 | None = field(
        default=None, kw_only=True, metadata={"unit": "geometry_length"}
    )
    vol_center: Vec3 | None = field(
        default=None, kw_only=True, metadata={"unit": "geometry_length"}
    )

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

    def __setstate__(self, state: _State) -> None:
        # Run this class's checks, not a subclass's __post_init__: its fields are
        # already initialised, and a subclass's own initialisation need not repeat.
        _restore(self, state)
        Grid.__post_init__(self)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Volume array shape ``(nz, ny, nx)`` in NumPy order.

        Returns
        -------
        tuple of int
            Voxel counts ``(nz, ny, nx)`` matching volume arrays on this grid.
        """
        return (self.nz, self.ny, self.nx)

    def resampled(self, *, shape: tuple[int, int, int]) -> Grid:
        """Return a grid with new voxel counts and the same physical extent.

        Voxel pitches change to preserve the outer cell edges and centre, not
        the positions of the first and last voxel centres. Explicit origins
        remain authoritative over ``vol_center``. No volume data is resampled.

        Parameters
        ----------
        shape : tuple of int
            Positive voxel counts ``(nz, ny, nx)`` in NumPy volume order,
            matching :attr:`shape`.

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
        >>> Grid(100, 80, 60, 0.2, 0.3, 0.4).resampled(shape=(30, 40, 50)).vx
        0.4
        """
        nz, ny, nx = (operator.index(n) for n in shape)
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
        d: GridDict = {
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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Grid:
        """Return the grid that a :meth:`to_dict` mapping describes.

        Parameters
        ----------
        data : mapping
            Grid metadata written by :meth:`to_dict`.

        Returns
        -------
        Grid
            Grid described by the metadata.

        Raises
        ------
        KeyError
            If voxel counts or pitches are missing.
        TypeError
            If counts are not integers or placement coordinates are not real numbers.
        ValueError
            If a placement vector does not have three components.
        """
        return cls(
            nx=int(data["nx"]),
            ny=int(data["ny"]),
            nz=int(data["nz"]),
            vx=float(data["vx"]),
            vy=float(data["vy"]),
            vz=float(data["vz"]),
            vol_origin=data.get("vol_origin"),
            vol_center=data.get("vol_center"),
        )


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

    ``center`` is the detector centre's ``(u, v)`` offset (equivalently lab-frame
    ``(x, z)`` before detector rotations), in the same physical units as ``du``
    and ``dv``. For a rolled or tilted cone-beam detector, rotations are applied
    about this displaced centre.

    Parameters
    ----------
    nu, nv : int
        Detector column and row counts.
    du, dv : float
        Pixel pitches along u and v, in the geometry's physical length unit.
    center : tuple of float, optional
        Detector-centre ``(u, v)`` offset in physical units; defaults to zero.

    Raises
    ------
    TypeError
        If counts are not integers or centre coordinates are not real numbers.
    ValueError
        If a count or pitch is not positive, or center is not two finite numbers.
    """

    nu: int = field(metadata={"unit": "pixel"})
    nv: int = field(metadata={"unit": "pixel"})
    du: float = field(metadata={"unit": "geometry_length"})
    dv: float = field(metadata={"unit": "geometry_length"})
    center: Vec2 = field(default_factory=lambda: (0.0, 0.0), metadata={"unit": "geometry_length"})

    def __post_init__(self) -> None:
        _plain_numbers(self, ("nu", "nv"), ("du", "dv"))
        object.__setattr__(self, "center", _coerce_fixed_tuple("center", self.center, 2))

    def __setstate__(self, state: _State) -> None:
        # As Grid: this class's checks only, on fields that are already initialised.
        _restore(self, state)
        Detector.__post_init__(self)

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
            raise ValueError("Missing detector centre: pass detector=... or geometry=...")
        return cls(
            nu=int(data["nu"]),
            nv=int(data["nv"]),
            du=float(data["du"]),
            dv=float(data["dv"]),
            center=data["det_center"],
        )


@functools.lru_cache(maxsize=8)
def _detector_grid(
    nu: int, nv: int, du: float, dv: float, cu: float, cv: float
) -> tuple[np.ndarray, np.ndarray]:
    # NumPy on the host, so the cache never holds a JAX tracer.
    u = (np.arange(nu, dtype=np.float32) - (nu / 2.0 - 0.5)) * np.float32(du) + np.float32(cu)
    v = (np.arange(nv, dtype=np.float32) - (nv / 2.0 - 0.5)) * np.float32(dv) + np.float32(cv)
    u_grid, v_grid = np.tile(u, nv), np.repeat(v, nu)
    u_grid.setflags(write=False)
    v_grid.setflags(write=False)
    return u_grid, v_grid


def get_detector_grid_device(det: Detector) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The detector's pixel-centre coordinates (u, v), flattened row by row.

    Call it outside jit, grad or scan; the result can be cached and reused.
    """
    import jax.numpy as jnp

    u, v = _detector_grid(
        int(det.nu), int(det.nv), float(det.du), float(det.dv),
        float(det.center[0]), float(det.center[1]),
    )  # fmt: skip
    return jnp.asarray(u, dtype=jnp.float32), jnp.asarray(v, dtype=jnp.float32)


def parallel_detector_rays(
    grid: Grid,
    detector: Detector,
    detector_roll: float = 0.0,
) -> RayPair:
    """Return the standard detector-plane ray model used by parallel-beam setups.

    The pixel lattice turns by ``detector_roll`` about the detector centre,
    from u towards v.
    """
    nu, nv = int(detector.nu), int(detector.nv)
    du, dv = float(detector.du), float(detector.dv)
    cx, cz = float(detector.center[0]), float(detector.center[1])
    angle = math.radians(float(detector_roll))
    c, s = math.cos(angle), math.sin(angle)

    y0 = grid_volume_origin(grid)[1]

    def origin_fn(u: int, v: int) -> Vec3:
        x = (u - (nu / 2.0 - 0.5)) * du
        z = (v - (nv / 2.0 - 0.5)) * dv
        return float(cx + c * x - s * z), float(y0), float(cz + s * x + c * z)

    def dir_fn(_u: int, _v: int) -> Vec3:
        return (0.0, 1.0, 0.0)

    return origin_fn, dir_fn


class Geometry(Protocol):
    """Common pose interface for projector-compatible geometries.

    Projector contract: ``pose_matrix(i)`` returns a 4x4 world_from_object
    transform. The core projector combines that pose with the supplied
    ``Grid``/``Detector`` to construct the standard parallel-beam detector
    rays; it does not consume custom per-pixel ray callbacks.

    Concrete geometries may expose ``rays_for_view(i)`` as an optional
    metadata/inspection helper, but that helper is not part of the projector
    contract and custom rays are not supported by the projectors.
    """

    def pose_matrix(self, i: int) -> PoseMatrix:
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
        ...


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
    def angles(self) -> Angles:
        """Each view's rotation angle, in degrees.

        Returns
        -------
        sequence of float or numpy.ndarray
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
