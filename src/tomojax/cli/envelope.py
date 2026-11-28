"""Version one of the local JSON Lines worker protocol."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import msgspec

from tomojax.cli.operations import ExecutionInput, Window
from tomojax.cli.products import Output, Snapshot
from tomojax.wire.arrays import Count, Hdf5, Id, Npy, Positive, Record
from tomojax.wire.progress import Error, Event

INLINE_LIMIT = 8 << 20  # Leave space in the worker outbox for framing and queued replies.


class Describe(Record, tag="describe"):
    """Return cached form descriptions."""


class Devices(Record, tag="devices"):
    """Return cached device inventory."""


class Close(Record, tag="close"):
    """Prevent subsequent admission against a handle."""

    handle: Id


class Release(Record, tag="release"):
    """Release a session-issued artifact token."""

    artifact: Id


class ControlRequest(Record, tag="control"):
    """Request a cooperative transition."""

    target: Id
    action: Literal["pause", "resume", "cancel"]


class Status(Record, tag="status"):
    """Read the owner's latest snapshots."""


class Shutdown(Record, tag="shutdown"):
    """Cancel outstanding work and close the session."""


PixelEncoding = Literal["jpeg", "float32", "float64", "uint16"]


class Fetch(Window, tag="fetch", kw_only=True):
    """Fetch a plane from registered storage, independently of the executor."""

    storage: Npy | Hdf5
    shape: tuple[Positive, ...]
    encoding: PixelEncoding
    window: tuple[float, float] | None = None


class Credit(Record):
    """Return consumed pixel bytes to the shared flow-control window."""

    credit: Positive


PixelBytes = Annotated[int, msgspec.Meta(ge=65536, le=1048576)]
INITIAL_PIXEL_CREDIT = 262144


class PixelWindow(Record):
    """Set the shared limit on outstanding pixel bytes."""

    window: PixelBytes


class Block(Record, tag="block"):
    """Read this many raw bytes immediately after the newline."""

    request: Id
    bytes: Positive


class Pixels(Record, tag="pixels"):
    """Description of the complete payload in the preceding blocks."""

    shape: tuple[Positive, ...]
    encoding: PixelEncoding
    bytes: Positive
    range: tuple[float, float] | None = None


type Command = (
    ExecutionInput | Describe | Devices | Close | Release | ControlRequest | Status | Shutdown
)


class Request(Record):
    """One client line. A non-null channel enables superseding."""

    id: Id
    command: Fetch | Command
    channel: Id | None = None


class ExecutorRequest(Request):
    """The subset of worker commands forwarded to Python."""

    command: Command


class Versions(Record):
    """Runtime versions, with unavailable optional runtimes represented by null."""

    tomojax: str
    jax: str
    cuda: str | None
    cupy: str | None


class Device(Record):
    """Stable session identifier used in placement and parameters."""

    id: Id
    platform: Id
    name: str


class Ready(Record, tag="ready"):
    """First stdout line."""

    protocol: Literal[1]
    progress: Literal[2]
    versions: Versions
    capabilities: tuple[Id, ...]
    devices: tuple[Device, ...]
    schema: str = ""
    root: str = ""


class WorkerReady(Record, tag="ready"):
    """The native worker can serve files immediately on connection."""

    protocol: Literal[1]
    progress: Literal[2]
    capabilities: tuple[Id, ...]
    host: str
    home: str
    directory: str
    jobs: Literal["inline", "process", "slurm"]
    pixel_credit: Positive


class ExecutorEvent(Record, tag="executor"):
    """A generation of the numerical executor started, became ready or ended."""

    state: Literal["starting", "ready", "gone"]
    detail: Ready | str | None
    generation: Positive


class ChannelRevision(Record):
    """Atomic channel name and revision pair."""

    name: Id
    revision: Positive


class Reply(Record):
    """Each admitted reply repeats the owner-assigned revision, null off-channel."""

    request: Id
    channel: ChannelRevision | None


class EventReply(Reply, tag="progress"):
    """Nonterminal watch event."""

    record: Event


class Description(Record, tag="description"):
    """Parameter description document."""

    parameters: dict[str, Any]


class DeviceList(Record, tag="devices"):
    """Cached inventory."""

    devices: tuple[Device, ...]


class StatusReport(Record, tag="status"):
    """Logical handle bytes are independent of allocator measurements."""

    requests: dict[str, Literal["queued", "running", "pause-requested", "paused"]]
    handles: dict[str, Count]
    allocators: tuple[Snapshot, ...]
    python_namespace_bytes: Count = 0
    executor: ExecutorEvent | None = None


class Ack(Record, tag="ack"):
    """Owner acknowledged the command."""


class ResultReply(Reply, tag="result"):
    """Exactly one terminal on success."""

    result: Pixels | Output | Description | DeviceList | StatusReport | Ack


class ErrorReply(Reply, tag="error"):
    """Exactly one terminal on execution failure."""

    error: Error


class ProtocolError(Record, tag="protocol_error"):
    """Rejected input never owns a terminal or affects a prior admission."""

    kind: Literal["invalid_request"]
    message: str
    request: str | None = None


type Response = Block | Ready | EventReply | ResultReply | ErrorReply | ProtocolError
