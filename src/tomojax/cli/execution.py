"""Executor/session boundary. An adapter has no access to the registry or stdout."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

import tomojax as tj
from tomojax.cli.operations import ExecutionInput, Plane
from tomojax.cli.products import DataProduct, HandleProduct, PlaneResult, Provenance, Snapshot
from tomojax.wire.arrays import Array, Axis, Fixed, Member, Npy, Storage, Uniform
from tomojax.wire.progress import Event, Info

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike


class Control(Protocol):
    """Thread-safe job control, owned by the session and shared with the watch."""

    @property
    def state(self) -> Literal["running", "pause-requested", "paused"]:
        """Report the last observed cooperative state while the job is active."""
        ...

    def pause(self) -> None:
        """Request pause."""

    def resume(self) -> None:
        """Wake a paused job."""

    def cancel(self) -> None:
        """Cancel and wake the job."""

    def checkpoint(self) -> None:
        """Raise Cancelled or block while paused, otherwise return."""


class Readable(Protocol):
    """Disk and device arrays can supply a plane without loading the volume."""

    def __getitem__(self, key: tuple[int | slice, ...], /) -> ArrayLike:
        """Read only the indexed region."""
        ...


@dataclass(frozen=True)
class RegisteredArray:
    """A pinned member's pixels and coordinates, valid for the entire admission."""

    data: Readable
    metadata: Array


type HandleValue = (
    tj.Scan
    | tj.Frames
    | tj.Reconstruction
    | tj.Alignment
    | tj.Simulation
    | tj.RegionEstimate
    | RegisteredArray
)


@dataclass(frozen=True)
class Lease:
    """Value and full input lineage pinned before enqueue, even if its handle closes."""

    value: HandleValue
    provenance: Provenance


@dataclass(frozen=True)
class ArrayStage:
    """Private write path and the reference deliverable after owner publication."""

    path: Path
    reference: Npy


class Staging(Protocol):
    """Allocations belong to this admission, including partially written files."""

    def array(self) -> ArrayStage:
        """Reserve a private npy and its eventual public reference."""
        ...

    def output(self, destination: Path, *, force: bool) -> Path:
        """Reserve a private path for one file or directory of a durable output group."""
        ...


@dataclass(frozen=True)
class ExecutionContext:
    """Capabilities restricted to a single request and its pre-admitted inputs."""

    leases: Mapping[Member, Lease]
    control: Control
    event: Callable[[Event], None]
    snapshot: Callable[[Snapshot], None]
    staging: Staging

    def __post_init__(self) -> None:
        object.__setattr__(self, "leases", MappingProxyType(dict(self.leases)))


@dataclass(frozen=True)
class BoundHandle:
    """Product and exact member names available for future leases, including self."""

    product: HandleProduct
    members: Mapping[str, HandleValue]

    def __post_init__(self) -> None:
        if "self" not in self.members:
            raise ValueError("a bound handle needs its self member")
        object.__setattr__(self, "members", MappingProxyType(dict(self.members)))


@dataclass(frozen=True)
class NewHandle:
    """Unpublished root. The owner binds its ID and eventually disposes its resources."""

    logical_bytes: int
    bind: Callable[[str], BoundHandle]
    dispose: Callable[[], None]


@dataclass(frozen=True)
class Success:
    """Return to the session's revision/commit check. Exceptions escape the adapter."""

    value: DataProduct | NewHandle
    info: Info
    manifest: Provenance


type Adapter[T: ExecutionInput] = Callable[[T, ExecutionContext], Success]


def volume_metadata(grid: tj.Grid, *, storage: Storage, dtype: str, length_unit: str) -> Array:
    """Register a (z, y, x) volume using physical voxel-centre coordinates."""
    from tomojax.geometry import grid_volume_origin

    origin = grid_volume_origin(grid)
    axes = tuple(
        Axis(name, "volume", length_unit, Uniform(start, pitch))
        for name, start, pitch in zip(
            "zyx", reversed(origin), (grid.vz, grid.vy, grid.vx), strict=True
        )
    )
    return Array(storage, grid.shape, dtype, axes, f"1/{length_unit}", "attenuation")


def browse_plane(operation: Plane, context: ExecutionContext) -> Success:
    """Stage a single plane using the pinned array and its original registration."""
    context.control.checkpoint()
    lease = context.leases[operation.source]
    if not isinstance(lease.value, RegisteredArray):
        raise ValueError("plane requires a registered volume member")
    array = lease.value.metadata
    if tuple(axis.name for axis in array.axes) != ("z", "y", "x"):
        raise ValueError("plane requires volume axes (z, y, x)")
    dimension = "zyx".index(operation.axis)
    axis = array.axes[dimension]
    if operation.index < 0 or operation.index >= array.shape[dimension]:
        raise ValueError("plane index outside volume")
    if not isinstance(axis.coordinates, Uniform):
        raise ValueError("volume grid requires uniform voxel spacing")
    coordinate = axis.coordinates.origin + operation.index * axis.coordinates.spacing
    fixed = Fixed(axis.name, axis.frame, axis.unit, operation.index, coordinate)
    key = tuple(operation.index if i == dimension else slice(None) for i in range(3))
    pixels = np.asarray(lease.value.data[key])
    stage = context.staging.array()
    with stage.path.open("wb") as stream:
        np.save(stream, pixels, allow_pickle=False)
    context.control.checkpoint()
    plane = Array(
        stage.reference,
        pixels.shape,
        pixels.dtype.str,
        array.axes[:dimension] + array.axes[dimension + 1 :],
        array.unit,
        array.domain,
        (*array.fixed, fixed),
    )
    manifest = Provenance(
        "plane",
        {"axis": axis.name, "index": operation.index, "source": operation.source},
        (lease.provenance,),
    )
    return Success(PlaneResult(plane), Info({}, {}), manifest)


def summarise(
    info: Mapping[str, object],
    *,
    externalise: Callable[[str, object], Array],
) -> Info:
    """Externalise array-valued leaves before scalar JSON conversion.

    Pointers address the original info tree, including lists. The inline position
    of an externalised array is null. A large numeric sequence is also bulk data.
    """
    import math

    import jax

    arrays: dict[str, Array] = {}

    def visit(value: object, pointer: str) -> object:
        if isinstance(value, np.ndarray | jax.Array):
            if value.ndim:
                arrays[pointer] = externalise(pointer, value)
                return None
            value = value.item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Mapping):
            return {
                key: visit(item, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
                for key, item in value.items()
            }
        if isinstance(value, tuple | list):
            if len(value) > 64 and all(isinstance(item, int | float | np.number) for item in value):
                arrays[pointer] = externalise(pointer, np.asarray(value))
                return None
            return [visit(item, f"{pointer}/{index}") for index, item in enumerate(value)]
        return value

    summary = {
        key: visit(value, "/" + key.replace("~", "~0").replace("/", "~1"))
        for key, value in info.items()
    }
    return Info(summary, arrays)
