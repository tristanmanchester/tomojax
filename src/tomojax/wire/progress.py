"""Version two of the progress stream shared by CLI and worker clients."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import msgspec

from tomojax.wire.arrays import Array, Count, Id, Positive, Record, Seconds

type ErrorKind = Literal[
    "input", "invalid_parameter", "out_of_memory", "canceled", "superseded", "internal"
]
type Scalar = str | int | float | bool | None


INFO_BYTES = 65536


class Info(Record):
    """Bounded JSON summary, with bulk members referenced separately by JSON Pointer."""

    summary: dict[str, Any]
    arrays: dict[str, Array]

    def __post_init__(self) -> None:
        if len(msgspec.json.encode(self)) > INFO_BYTES:
            raise ValueError(f"info exceeds {INFO_BYTES} UTF-8 JSON bytes")


class Progress(Record, tag_field="event"):
    """Standalone progress records explicitly identify the schema version."""

    schema: Literal[2]


class Timed(Progress):
    """Time is monotonic seconds since this watch began."""

    task: Positive
    time: Seconds


class Started(Timed, tag="started"):
    """An operation entered on the owning thread."""

    parent: Positive | None
    name: Id
    label: str
    total: Count | None
    unit: str


class Placed(Timed, tag="placed"):
    """Participating devices in coordinator order."""

    devices: Annotated[tuple[Id, ...], msgspec.Meta(min_length=1)]


class Advanced(Timed, tag="advanced"):
    """One coherent checkpoint, ETA in active seconds."""

    done: Count
    total: Count | None
    metrics: dict[str, float | str | None]
    eta: Seconds | None


class Finished(Timed, tag="finished"):
    """End of a task."""

    status: Literal["done", "failed", "canceled"]


type NoteLevel = Literal["debug", "info", "warning", "error", "critical"]


class Note(Progress, tag="note"):
    """A message attributed to the active watch."""

    task: Positive | None
    time: Seconds
    level: NoteLevel
    message: str


class Paused(Timed, tag="paused"):
    """The operation has reached a checkpoint and is blocked."""


class Resumed(Timed, tag="resumed"):
    """The operation has left its paused checkpoint."""


class Preview(Timed, tag="preview"):
    """Coherent named arrays for this checkpoint."""

    arrays: dict[str, Array]


class Result(Progress, tag="result"):
    """Successful CLI terminal."""

    outputs: tuple[str, ...]
    method: str | None
    arrays: dict[str, Array]
    info: Info


class Error(Progress, tag="error"):
    """Operational failure with an optional nested parameter path."""

    kind: ErrorKind
    message: str
    path: str | None = None


type Event = Started | Placed | Advanced | Finished | Note | Paused | Resumed | Preview
type ProgressRecord = Event | Result | Error
