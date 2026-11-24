"""Version one of the local JSON Lines worker protocol."""

from __future__ import annotations

from typing import Any, Literal

from tomojax.cli.operations import ExecutionInput
from tomojax.cli.products import Output, Snapshot
from tomojax.wire.arrays import Count, Id, Positive, Record
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


type Command = (
    ExecutionInput | Describe | Devices | Close | Release | ControlRequest | Status | Shutdown
)


class Request(Record):
    """One client line. A non-null channel enables superseding."""

    id: Id
    command: Command
    channel: Id | None = None


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
    recovery: dict[str, list[str]] | None = None


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


class Ack(Record, tag="ack"):
    """Owner acknowledged the command."""


class ResultReply(Reply, tag="result"):
    """Exactly one terminal on success."""

    result: Output | Description | DeviceList | StatusReport | Ack


class ErrorReply(Reply, tag="error"):
    """Exactly one terminal on execution failure."""

    error: Error


class ProtocolError(Record, tag="protocol_error"):
    """Rejected input never owns a terminal or affects a prior admission."""

    kind: Literal["invalid_request"]
    message: str
    request: str | None = None


type Response = Ready | EventReply | ResultReply | ErrorReply | ProtocolError
