"""Local JSON Lines transport for a single GUI session."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from importlib import metadata, resources
import os
from pathlib import Path
import select
import socket
import sys
import tempfile
from threading import Lock
from typing import TYPE_CHECKING

import msgspec

import tomojax
from tomojax.cli import envelope as e, execution as ex, operations as op
from tomojax.cli._session import Session
from tomojax.wire import progress as w

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

_LINE = 1024 * 1024
_BACKLOG = 16 * 1024 * 1024
_DRAIN = 2.0


class _Outbox:
    """One consumer owns the partial line. Producers never write or wait on stdout."""

    def __init__(self, wake: socket.socket) -> None:
        self.wake = wake
        self.lock = Lock()
        self.queue: deque[tuple[bytes, tuple[str, int] | None]] = deque()
        self.size = 0
        self.lost = False

    def send(self, reply: e.Response) -> None:
        data = msgspec.json.encode(reply) + b"\n"
        key = None
        if isinstance(reply, e.EventReply) and isinstance(reply.record, w.Advanced):
            key = (reply.request, reply.record.task)
        with self.lock:
            if self.lost:
                return
            if key and self.queue and self.queue[-1][1] == key:
                self.size -= len(self.queue.pop()[0])
            self.size += len(data)
            if self.size > _BACKLOG:
                self.lost = True
            else:
                self.queue.append((data, key))
            with suppress(BlockingIOError):  # An unread wakeup already covers this change.
                self.wake.send(b".")

    def take(self) -> memoryview:
        with self.lock:
            if not self.queue:
                return memoryview(b"")
            data, _ = self.queue.popleft()
            self.size -= len(data)
            return memoryview(data)

    def drain(self, stdout: int, pending: memoryview) -> int:
        poll = select.poll()
        poll.register(stdout, select.POLLOUT)
        while pending or (pending := self.take()):
            if not poll.poll(int(1000 * _DRAIN)):
                return 1
            pending = pending[os.write(stdout, pending) :]
        return 0


class _Input:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.discard = False
        self.decode = msgspec.json.Decoder(e.Request)

    def feed(self, data: bytes, session: Session) -> None:
        for index, chunk in enumerate(data.split(b"\n")):
            if index:
                if not self.discard:
                    try:
                        request = self.decode.decode(self.buffer)
                    except (msgspec.DecodeError, UnicodeDecodeError, RecursionError) as err:
                        session.emit(e.ProtocolError("invalid_request", str(err)))
                    else:
                        session.admit(request)
                self.buffer.clear()
                self.discard = False
                if session.closing:
                    return
            if not self.discard:
                self.buffer.extend(chunk)
                if len(self.buffer) > _LINE:
                    self.discard = True
                    self.buffer.clear()
                    session.emit(e.ProtocolError("invalid_request", "line exceeds 1 MiB"))

    def read(self, fd: int, session: Session) -> None:
        data = os.read(fd, 65536)
        if data:
            self.feed(data, session)
        else:
            if self.buffer and not self.discard:
                self.feed(b"\n", session)
            session.stop()


def _transport(
    session: Session,
    running: Future[None],
    box: _Outbox,
    wake: socket.socket,
    stdin: int,
    stdout: int,
) -> int:
    poll = select.poll()
    poll.register(stdin, select.POLLIN)
    poll.register(wake, select.POLLIN)
    blocking = os.get_blocking(stdout)
    os.set_blocking(stdout, False)
    pending = memoryview(b"")
    lines = _Input()
    try:
        while not session.closing and not box.lost:
            if not pending:
                pending = box.take()
            poll.register(stdout, select.POLLOUT if pending else 0)
            for fd, mask in poll.poll():
                if fd == stdout:
                    if mask & (select.POLLHUP | select.POLLERR | select.POLLNVAL):
                        return 1
                    with suppress(BlockingIOError):
                        pending = pending[os.write(stdout, pending) :]
                elif fd == stdin:
                    lines.read(stdin, session)
                else:
                    wake.recv(65536)
        session.stop()
        running.result()
        if box.lost:
            return 1
        return box.drain(stdout, pending)
    except BrokenPipeError:
        return 1
    finally:
        os.set_blocking(stdout, blocking)


def serve(
    execute: ex.Adapter[op.ExecutionInput],
    *,
    ready: e.Ready | Callable[[], e.Ready],
    description: e.Description,
) -> int:
    """Serve the process pipes until shutdown, EOF or loss of the output connection."""
    reader, writer = socket.socketpair()
    with (
        reader,
        writer,
        tempfile.TemporaryDirectory(prefix="tomojax-") as root,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tomojax-executor") as executor,
    ):
        writer.setblocking(False)
        box = _Outbox(writer)
        inventory = executor.submit(ready).result() if callable(ready) else ready
        session = Session(Path(root), execute, box.send, inventory, description)
        box.send(inventory)
        running = executor.submit(session.work)
        try:
            return _transport(
                session, running, box, reader, sys.stdin.fileno(), sys.stdout.fileno()
            )
        finally:
            session.stop()
            running.result()
            session.close()


def _cpu_adapter(command: op.ExecutionInput, context: ex.ExecutionContext) -> ex.Success:
    if isinstance(command, op.Plane):
        return ex.browse_plane(command, context)
    raise ValueError("operation unavailable in this worker")


def build_parser() -> argparse.ArgumentParser:
    """Describe the stdin/stdout command."""
    return argparse.ArgumentParser(prog="tomojax serve", description=__doc__)


def _ready() -> e.Ready:
    import jax

    devices = jax.devices()
    inventory = tuple(e.Device(f"{d.platform}:{d.id}", d.platform, d.device_kind) for d in devices)
    cuda = next(
        (
            d.client.platform_version
            for d in devices
            if d.platform == "gpu" and "cuda" in d.client.platform_version.lower()
        ),
        None,
    )
    cupy = next(
        (
            dist.version
            for dist in metadata.distributions()
            if (dist.metadata.get("Name") or "").lower().split("-")[0] == "cupy"
        ),
        None,
    )
    return e.Ready(
        1,
        2,
        e.Versions(tomojax.__version__, jax.__version__, cuda, cupy),
        ("describe", "devices", "status", "control", "close", "release", "shutdown", "plane"),
        inventory,
    )


def main() -> int:
    """Start the worker with the installed plane adapter."""
    build_parser().parse_args()
    return serve(
        _cpu_adapter,
        ready=_ready,
        description=e.Description(
            msgspec.json.decode(
                resources.files("tomojax.cli").joinpath("parameters.json").read_bytes()
            )
        ),
    )
