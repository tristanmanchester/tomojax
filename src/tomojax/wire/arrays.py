"""Registration shared by preview, result and persisted arrays."""

from __future__ import annotations

from typing import Annotated, Literal, dataclass_transform, get_args

import msgspec

Id = Annotated[str, msgspec.Meta(min_length=1)]
Count = Annotated[int, msgspec.Meta(ge=0)]
Positive = Annotated[int, msgspec.Meta(gt=0)]
Seconds = Annotated[float, msgspec.Meta(ge=0)]


@dataclass_transform(frozen_default=True)
class Record(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """Closed JSON object."""


class Npy(Record, tag="npy"):
    """A NumPy file. Published session files carry a release token."""

    path: Id
    artifact: Id | None = None


class Hdf5(Record, tag="hdf5"):
    """A dataset inside an HDF5 file."""

    path: Id
    dataset: Id


class Member(Record, tag="member"):
    """A typed member pinned with its root handle."""

    handle: Id
    member: Id


type Storage = Npy | Hdf5 | Member


class Uniform(Record, tag="uniform"):
    """Centre coordinate of element zero and signed step in axis units."""

    origin: float
    spacing: float


class Coordinates(Record, tag="coordinates"):
    """An ordered coordinate vector, length equal to its dimension."""

    values: Storage


class Axis(Record):
    """Coordinates of array elements in one named frame."""

    name: Id
    frame: Id
    unit: Id
    coordinates: Uniform | Coordinates


class Fixed(Record):
    """A dimension removed by indexing in the source array."""

    name: Id
    frame: Id
    unit: Id
    index: Count
    coordinate: float


PoseColumn = Literal["alpha", "beta", "phi", "x", "z", "y"]


class Column(Record):
    """One pose-table column, in acquisition pose convention."""

    name: PoseColumn
    unit: Id


class Array(Record):
    """A registered array. Axes and shape follow storage dimension order."""

    storage: Storage
    shape: tuple[Count, ...]
    dtype: Id
    axes: tuple[Axis, ...]
    unit: Id
    domain: Id
    fixed: tuple[Fixed, ...] = ()
    columns: tuple[Column, ...] = ()

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.axes):
            raise ValueError("array shape and axes differ in rank")
        if self.domain == "poses" or self.columns:
            names = tuple(column.name for column in self.columns)
            if (
                names != get_args(PoseColumn)
                or len(self.shape) != 2
                or self.shape[-1] != len(names)
            ):
                raise ValueError("poses require six columns in acquisition order")
            if any(col.unit != "degree" for col in self.columns[:3]):
                raise ValueError("pose rotations use degrees")
            if len({col.unit for col in self.columns[3:]}) != 1:
                raise ValueError("pose translations share a length unit")
