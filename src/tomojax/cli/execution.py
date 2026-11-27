"""Executor/session boundary. An adapter has no access to the registry or stdout."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import msgspec
import numpy as np

import tomojax as tj
from tomojax._typed_arrays import device_id
from tomojax.cli.operations import ExecutionInput, Plane, input_references
from tomojax.cli.products import DataProduct, HandleProduct, PlaneResult, Provenance, Snapshot
from tomojax.errors import InvalidParameter
from tomojax.wire.arrays import Array, Fixed, Member, Npy, Storage, Uniform
from tomojax.wire.progress import INFO_BYTES, Event, Info

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

    from tomojax._typed_arrays import Device
    from tomojax.io.api import ExportPlan


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

    def boundary(self) -> None:
        """Raise Canceled or block while paused, otherwise return."""


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


@dataclass(frozen=True)
class Feature:
    """A named boolean mask on the grid of the array it was made on.

    Only the mask's bounding box is stored: ``inside`` holds the mask over ``box``,
    one slice per axis of ``registration.shape``, and every voxel outside ``box`` is
    outside the feature. ``registration`` holds the grid's axes and unit, so the
    mask applies unchanged to any other array on the same grid. Index it like an
    array to read a window; ``mask`` builds the whole array (one byte per voxel).
    """

    name: str
    inside: np.ndarray
    box: tuple[slice, ...]
    registration: Array
    recipe: dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        """The grid's shape."""
        return tuple(self.registration.shape)

    @property
    def ndim(self) -> int:
        """The grid's number of axes."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """Masks are boolean."""
        return np.dtype(bool)

    @property
    def mask(self) -> np.ndarray:
        """The whole mask on the grid."""
        whole = np.zeros(self.shape, dtype=bool)
        whole[self.box] = self.inside
        return whole

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """The whole mask, for NumPy functions."""
        return self.mask if dtype is None else self.mask.astype(dtype)

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> np.ndarray:
        """Read a window of the mask, building only that window."""
        key = key if isinstance(key, tuple) else (key,)
        key += (slice(None),) * (self.ndim - len(key))
        picked = [np.atleast_1d(np.arange(n)[k]) for n, k in zip(self.shape, key, strict=True)]
        out = np.zeros([len(i) for i in picked], dtype=bool)
        within = [(i >= b.start) & (i < b.stop) for i, b in zip(picked, self.box, strict=True)]
        if all(w.any() for w in within):
            source = [i[w] - b.start for i, w, b in zip(picked, within, self.box, strict=True)]
            out[np.ix_(*within)] = self.inside[np.ix_(*source)]
        return out.reshape(
            [len(i) for i, k in zip(picked, key, strict=True) if isinstance(k, slice)]
        )


type HandleValue = (
    tj.Scan
    | tj.Frames
    | tj.Reconstruction
    | tj.Alignment
    | tj.Simulation
    | tj.RegionEstimate
    | Feature
    | RegisteredArray
)


@dataclass(frozen=True)
class Lease:
    """Value and full input lineage pinned before enqueue, even if its handle closes."""

    value: HandleValue | Npy
    provenance: Provenance
    metadata: Array | None = None
    owner: HandleValue | None = None


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

    def export(self, plan: ExportPlan, *, force: bool) -> dict[str, object]:
        """Stage a complete interchange export for commit."""
        ...

    def output(self, destination: Path, *, force: bool) -> Path:
        """Reserve a private path for one file or directory of a durable output group."""
        ...


@dataclass(frozen=True)
class ExecutionContext:
    """Capabilities restricted to a single request and its pre-admitted inputs."""

    leases: Mapping[Member | Npy, Lease]
    control: Control
    event: Callable[[Event], None]
    snapshot: Callable[[Snapshot], None]
    staging: Staging
    namespace: dict[str, Any]
    measure: Callable[[int], None]

    def __init__[K: Member | Npy](
        self,
        leases: Mapping[K, Lease],
        control: Control,
        event: Callable[[Event], None],
        snapshot: Callable[[Snapshot], None],
        staging: Staging,
        namespace: dict[str, Any] | None = None,
        measure: Callable[[int], None] = lambda _: None,
    ) -> None:
        object.__setattr__(self, "leases", MappingProxyType(dict(leases)))
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "staging", staging)
        object.__setattr__(self, "namespace", {} if namespace is None else namespace)
        object.__setattr__(self, "measure", measure)

    def lineage(self, command: ExecutionInput) -> tuple[Provenance, ...]:
        """Return each pinned producer in command order, including repeated references."""
        return tuple(self.leases[ref].provenance for ref in input_references(command))


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
    roots: tuple[tuple[str, NewHandle], ...] = ()


type Adapter[T: ExecutionInput] = Callable[[T, ExecutionContext], Success]


def volume_metadata(
    grid: tj.Grid,
    *,
    storage: Storage,
    dtype: str,
    length_unit: str,
    geometry_meta: Mapping[str, Any] | None = None,
) -> Array:
    """Register a (z, y, x) volume using physical voxel-center coordinates."""
    from tomojax.core.registration import volume_axes, volume_values

    axes = volume_axes(grid, length_unit)
    unit, domain = volume_values(np.dtype(dtype), length_unit, geometry_meta or {})
    return Array(storage, grid.shape, dtype, axes, unit, domain)


def plane_source(operation: Plane, context: ExecutionContext) -> tuple[RegisteredArray, int, Fixed]:
    """Check the registered volume and resolve the source index without touching pixels."""
    lease = context.leases[operation.source]
    if not isinstance(lease.value, RegisteredArray):
        raise InvalidParameter("source", "plane requires a registered volume member")
    array = lease.value.metadata
    if tuple(axis.name for axis in array.axes) != ("z", "y", "x"):
        raise InvalidParameter("source", "plane requires volume axes (z, y, x)")
    dimension = "zyx".index(operation.axis)
    axis = array.axes[dimension]
    if operation.index < 0 or operation.index >= array.shape[dimension]:
        raise InvalidParameter("index", "plane index outside volume")
    if not isinstance(axis.coordinates, Uniform):
        raise InvalidParameter("source", "volume grid requires uniform voxel spacing")
    coordinate = axis.coordinates.origin + operation.index * axis.coordinates.spacing
    fixed = Fixed(axis.name, axis.frame, axis.unit, operation.index, coordinate)
    return lease.value, dimension, fixed


def browse_plane(operation: Plane, context: ExecutionContext) -> Success:
    """Stage a single plane using the pinned array and its original registration."""
    context.control.boundary()
    registered, dimension, fixed = plane_source(operation, context)
    array = registered.metadata
    key = tuple(operation.index if i == dimension else slice(None) for i in range(3))
    pixels = np.asarray(registered.data[key])
    stage = context.staging.array()
    with stage.path.open("wb") as stream:
        np.save(stream, pixels, allow_pickle=False)
    context.control.boundary()
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
        {"axis": operation.axis, "index": operation.index, "source": operation.source},
        context.lineage(operation),
    )
    return Success(PlaneResult(plane), Info({}, {}), manifest)


def summarize(
    info: Mapping[str, object],
    *,
    externalize: Callable[[str, object], Array],
) -> Info:
    """Externalize array-valued leaves before scalar JSON conversion.

    Pointers address the original info tree, including lists. The inline position
    of an externalized array is null. A large numeric sequence is also bulk data.
    """
    arrays: dict[str, Array] = {}
    summary = {
        key: _summary_value(
            value, "/" + key.replace("~", "~0").replace("/", "~1"), arrays, externalize
        )
        for key, value in info.items()
    }
    return _bounded_info(summary, arrays)


def _summary_value(
    value: object,
    pointer: str,
    arrays: dict[str, Array],
    externalize: Callable[[str, object], Array],
) -> object:
    import math

    import jax

    value = _config_value(value)
    if pointer.startswith("/disturbed_views/") and isinstance(value, list) and len(value) > 64:
        # Externalize both interval tables and score vectors with missing measurements.
        value = np.asarray(value, dtype=float if pointer.endswith("/score") else None)
    if isinstance(value, np.ndarray | jax.Array):
        if value.ndim:
            arrays[pointer] = externalize(pointer, value)
            return None
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {
            key: _summary_value(
                item, pointer + "/" + key.replace("~", "~0").replace("/", "~1"), arrays, externalize
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        if len(value) > 64 and all(isinstance(item, int | float | np.number) for item in value):
            arrays[pointer] = externalize(pointer, np.asarray(value))
            return None
        return [
            _summary_value(item, f"{pointer}/{index}", arrays, externalize)
            for index, item in enumerate(value)
        ]
    return value if value is None or isinstance(value, str | bool | int | float) else str(value)


def _config_value(value: object) -> object:
    """Walk dataclass settings without asdict's deep copies of support or device arrays."""
    import jax

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, jax.Device):
        return device_id(cast("Device", value))
    return value


def _bounded_info(summary: dict[str, object], arrays: dict[str, Array]) -> Info:
    if len(msgspec.json.encode({"summary": summary, "arrays": arrays})) <= INFO_BYTES:
        return Info(summary, arrays)
    # Keep whole fields and their matching JSON pointers. Prefer the selection
    # decision over bulky solver histories; count every field omitted for size.
    kept: dict[str, object] = {}
    references: dict[str, Array] = {}
    omitted = 0
    priority = {"automatic", "placement", "disturbed_views"}
    for key in sorted(summary, key=lambda k: k not in priority):
        pointer = "/" + key.replace("~", "~0").replace("/", "~1")
        subset = {p: a for p, a in arrays.items() if p == pointer or p.startswith(pointer + "/")}
        record = {
            "summary": {**kept, key: summary[key], "_omitted_fields": len(summary)},
            "arrays": {**references, **subset},
        }
        if len(msgspec.json.encode(record)) <= INFO_BYTES:
            kept[key] = summary[key]
            references.update(subset)
        else:
            omitted += 1
    return Info({**kept, "_omitted_fields": omitted}, references)
