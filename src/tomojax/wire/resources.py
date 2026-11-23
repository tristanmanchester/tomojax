"""Device memory samples and estimated task placement."""

from typing import Literal

from tomojax.wire.arrays import Count, Id, Record, Seconds


class Snapshot(Record):
    """Immutable executor sample. Bytes refer to the named allocator and device."""

    time: Seconds
    device: Id
    scope: Id
    used: Count | None
    peak: Count | None
    available: Count | None


class Placement(Record):
    """Resource estimate at planning time."""

    devices: tuple[Id, ...]
    strategy: Literal["resident", "slabbed"]
    host_bytes: Count
    device_bytes: Count
    snapshot: tuple[Snapshot, ...]
