"""Pixel reads and block arbitration under the session's admission lock."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Condition
from typing import Literal

import msgspec

from tomojax.cli import envelope as e
from tomojax.cli._pixels import fetch
from tomojax.errors import classify
from tomojax.wire.progress import Error

CREDIT = 262144
BLOCK = 65536


@dataclass(eq=False)
class Read:
    id: str
    channel: e.ChannelRevision | None
    command: e.Fetch
    pins: ExitStack


class PixelLane:
    def __init__(
        self, changed: Condition, emit: Callable[[e.Response], None], wake: Callable[[], None]
    ) -> None:
        self.changed = changed
        self.emit = emit
        self.wake = wake
        self.active: dict[str, Read] = {}
        self.waiting: dict[Read, None] = {}
        self.ready: dict[Read, tuple[e.Pixels, memoryview]] = {}
        self.credit = CREDIT
        self.closed = False

    def admit(self, read: Read) -> None:
        self.active[read.id] = read
        self.waiting[read] = None
        self.changed.notify_all()

    def cancel(self, read: Read, kind: Literal["superseded", "canceled"]) -> None:
        del self.active[read.id]
        if read in self.waiting:
            del self.waiting[read]
            read.pins.close()
        if self.ready.pop(read, None) is not None:
            read.pins.close()
        self.emit(e.ErrorReply(read.id, read.channel, Error(2, kind, kind)))

    def stop(self) -> None:
        self.closed = True
        for read in tuple(self.active.values()):
            self.cancel(read, "canceled")
        self.changed.notify_all()

    def work(self) -> None:
        while True:
            with self.changed:
                _ = self.changed.wait_for(lambda: self.closed or self.waiting)
                if self.closed:
                    return
                read = next(iter(self.waiting))
                del self.waiting[read]
            try:
                result, payload = fetch(read.command)
            except Exception as exc:
                with self.changed:
                    if self.active.get(read.id) is read:
                        del self.active[read.id]
                        self.emit(e.ErrorReply(read.id, read.channel, classify(exc)))
            else:
                with self.changed:
                    if self.active.get(read.id) is read:
                        self.ready[read] = result, memoryview(payload)
                        self.wake()
                del payload
            finally:
                with self.changed:
                    if self.active.get(read.id) is not read:
                        read.pins.close()

    def take(self) -> bytes:
        if not self.ready or not self.credit:
            return b""
        read = min(self.ready, key=lambda read: len(self.ready[read][1]))
        result, data = self.ready[read]
        size = min(self.credit, BLOCK, len(data))
        self.credit -= size
        if size == len(data):
            del self.ready[read], self.active[read.id]
            read.pins.close()
            self.emit(e.ResultReply(read.id, read.channel, result))
        else:
            self.ready[read] = result, data[size:]
        return msgspec.json.encode(e.Block(read.id, size)) + b"\n" + data[:size]

    def acknowledge(self, n: int) -> None:
        with self.changed:
            if n > CREDIT - self.credit:
                raise ValueError("credit exceeds outstanding pixel bytes")
            self.credit += n
