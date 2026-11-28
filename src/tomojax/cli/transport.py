"""Local JSON Lines transport for a single GUI session."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from importlib import metadata, resources
import os
from pathlib import Path
import select
import signal
import socket
import tempfile
import threading
from threading import Condition, Event
from typing import TYPE_CHECKING

import msgspec

import tomojax
from tomojax.cli import envelope as e, execution as ex, operations as op
from tomojax.cli._pixel_lane import PixelLane
from tomojax.cli._session import Session
from tomojax.wire import progress as w

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

_LINE = 1024 * 1024
_BACKLOG = 2 * e.INLINE_LIMIT
_DRAIN = 2.0


class _Outbox:
    """One consumer owns the partial line. Producers never write or wait on stdout."""

    def __init__(self, wake: socket.socket) -> None:
        self.wake = wake
        self.lock = Condition()
        self.pixels = PixelLane(self.lock, self.send, self.notify)
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
            self.notify()

    def notify(self) -> None:
        with suppress(BlockingIOError):  # An unread wakeup already covers this change.
            self.wake.send(b".")

    def take(self) -> memoryview:
        with self.lock:
            if not self.queue:
                return memoryview(self.pixels.take())
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


class _RequestId(msgspec.Struct):
    id: str


class _Input:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.discard = False
        self.decode = msgspec.json.Decoder(e.Request)

    def feed(self, data: bytes, session: Session) -> None:
        for index, chunk in enumerate(data.split(b"\n")):
            if index:
                if not self.discard:
                    self.admit_line(session)
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

    def admit_line(self, session: Session) -> None:
        try:
            request = self.decode.decode(self.buffer)
        except (msgspec.DecodeError, UnicodeDecodeError, RecursionError) as err:
            try:
                credit = msgspec.json.decode(self.buffer, type=e.Credit)
            except (msgspec.DecodeError, UnicodeDecodeError, RecursionError):
                self.reject(session, err)
            else:
                try:
                    session.pixels.acknowledge(credit.credit)
                except ValueError as invalid:
                    self.reject(session, invalid)
        else:
            session.admit(request)

    def reject(self, session: Session, err: Exception) -> None:
        rid = None
        with suppress(msgspec.DecodeError, UnicodeDecodeError, RecursionError):
            rid = msgspec.json.decode(self.buffer, type=_RequestId).id
        session.emit(e.ProtocolError("invalid_request", str(err), rid))

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
    terminated: Event,
) -> int:
    """Poll the pipes. Darwin requires POLLHUP to be requested for idle stdout."""
    poll = select.poll()
    poll.register(stdin, select.POLLIN)
    poll.register(wake, select.POLLIN)
    blocking = os.get_blocking(stdout)
    os.set_blocking(stdout, False)
    pending = memoryview(b"")
    lines = _Input()
    try:
        while not session.closing and not box.lost and not terminated.is_set():
            if not pending:
                pending = box.take()
            poll.register(stdout, select.POLLHUP | (select.POLLOUT if pending else 0))
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
    stdin: int,
    stdout: int,
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
        ThreadPoolExecutor(max_workers=4, thread_name_prefix="tomojax-io") as io,
    ):
        writer.setblocking(False)
        box = _Outbox(writer)
        terminated = Event()
        _stop_on_sigterm(terminated, writer)
        inventory = executor.submit(ready).result() if callable(ready) else ready
        session = Session(Path(root), execute, box.send, inventory, description, box.pixels)
        for _ in range(4):
            io.submit(box.pixels.work).add_done_callback(_executor_done)
        box.send(inventory)
        running = executor.submit(session.work)
        running.add_done_callback(_executor_done)
        try:
            return _transport(session, running, box, reader, stdin, stdout, terminated)
        finally:
            session.stop()
            running.result()
            io.shutdown()
            session.close()


def _stop_on_sigterm(terminated: Event, wake: socket.socket) -> None:
    """On SIGTERM (a scheduler's kill), shut down as at EOF and exit once the jobs end.

    Shutdown cancels every job: a checkpointed one saves its state at its next
    boundary first, and its ``canceled`` terminal names the checkpoint. The handler
    only flags the transport, whose loop then runs the shutdown.
    """
    if threading.current_thread() is not threading.main_thread():
        return

    def terminate(_signum: int, _frame: object) -> None:
        terminated.set()
        with suppress(BlockingIOError):
            _ = wake.send(b".")

    _ = signal.signal(signal.SIGTERM, terminate)


def _executor_done(future: Future[None]) -> None:
    # The transport may be blocked in poll while the only executor has died.
    if future.exception() is not None:
        os._exit(1)


def _ready() -> e.Ready:
    import jax

    from tomojax.cli._adapters import EXECUTABLE

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
        (
            "fetch",
            "describe",
            "devices",
            "status",
            "control",
            "close",
            "release",
            "shutdown",
            *(str(kind.__struct_config__.tag) for kind in EXECUTABLE),
        ),
        inventory,
    )


def run(stdin: int, stdout: int) -> int:
    """Start the worker with its production operation adapters."""
    from tomojax.cli._adapters import execute

    return serve(
        execute,
        stdin=stdin,
        stdout=stdout,
        ready=_ready,
        description=e.Description(
            msgspec.json.decode(
                resources.files("tomojax.cli").joinpath("parameters.json").read_bytes()
            )
        ),
    )
